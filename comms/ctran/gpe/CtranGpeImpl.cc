// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <chrono>
#include <optional>
#include <stdexcept>

#include <folly/dynamic.h>

#include "comms/ctran/algos/AllToAll/AllToAllPImpl.h"
#include "comms/ctran/algos/AllToAll/AllToAllvDynamicPImpl.h"
#include "comms/ctran/algos/common/GpeKernel.h"
#include "comms/ctran/gpe/CtranChecksum.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/gpe/CtranGpeDev.h"
#include "comms/ctran/gpe/CtranGpeImpl.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/tracing/CollTraceWrapper.h"
#include "comms/ctran/tracing/MapperTrace.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/ctran/utils/Debug.h"
#include "comms/ctran/utils/Exception.h"
#include "comms/ctran/utils/ExtUtils.h"

#include "comms/utils/colltrace/CollRecord.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

using namespace ctran;
using namespace ncclx::colltrace;
using meta::comms::colltrace::CollTraceHandleTriggerState;

static std::unordered_map<KernelConfig::KernelType, const std::string>
    kernelTypeToName = {
        {KernelConfig::KernelType::ALLGATHER, "AllGather"},
        {KernelConfig::KernelType::ALLREDUCE, "AllReduce"},
        {KernelConfig::KernelType::SEND, "Send"},
        {KernelConfig::KernelType::RECV, "Recv"},
        {KernelConfig::KernelType::SENDRECV, "SendRecv"},
        {KernelConfig::KernelType::ALLTOALL, "AllToAll"},
        {KernelConfig::KernelType::ALLTOALLV, "AllToAllv"},
        {KernelConfig::KernelType::ALLTOALLV_DYNAMIC, "AllToAllvDynamic"},
        {KernelConfig::KernelType::ALLTOALLV_DYNAMIC_SPLIT,
         "AllToAllvDynamicSplit"},
        {KernelConfig::KernelType::ALLTOALLV_DYNAMIC_SPLIT_NON_CONTIG,
         "AllToAllvDynamicSplitNonContig"},
};

CtranGpe::Impl::Impl() {
  this->kernelFlagPool = std::unique_ptr<KernelFlagPool>(
      new KernelFlagPool(NCCL_CTRAN_NUM_KERNEL_FLAGS));

  this->kernelElemPool = std::unique_ptr<KernelElemPool>(
      new KernelElemPool(NCCL_CTRAN_NUM_KERNEL_ELEMS));

  this->checksumPool =
      std::unique_ptr<ChecksumPool>(new ChecksumPool(NCCL_CTRAN_NUM_CHECKSUMS));

  this->gpeKernelSyncPool =
      std::make_unique<GpeKernelSyncPool>(NCCL_CTRAN_NUM_GPE_KERNEL_SYNCS);

  FB_CUDACHECKTHROW(
      cudaEventCreateWithFlags(&execEvent_, cudaEventDisableTiming));
  FB_CUDACHECKTHROW(
      cudaStreamCreateWithFlags(&execOrderStream_, cudaStreamNonBlocking));

  return;
}

CtranGpe::Impl::~Impl() {
  FB_CUDACHECKTHROW(cudaEventDestroy(execEvent_));
  FB_CUDACHECKTHROW(cudaStreamDestroy(execOrderStream_));
}

struct cmdCbPlan {
  CtranGpeCmd* cmd{nullptr};
  CtranGpe* gpe{nullptr};
};

void CUDART_CB CtranGpe::Impl::cmdCb(void* data) {
  struct cmdCbPlan* plan = reinterpret_cast<struct cmdCbPlan*>(data);
  plan->gpe->pimpl->cmdEnqueue(plan->cmd);
}

void CUDART_CB CtranGpe::Impl::cmdDestroy(void* data) {
  CtranGpeCmd* cmd = reinterpret_cast<CtranGpeCmd*>(data);
  if (!cmd->persistent) {
    CLOGF(WARN, "CTranGPE: cmd desctructor called for non-persistent cmd");
  }
  for (const auto& x : cmd->coll.opGroup) {
    x->setStatus(KernelElem::ElemStatus::RESET);
  }
  delete cmd;
}

commResult_t CtranGpe::Impl::preKernelLaunch(cudaStream_t curStream) {
  // check if we can skip stream wait when posting kernels on the same stream
  if (curStream == lastUserStream_) {
    return commSuccess;
  }
  lastUserStream_ = curStream;
  // wait on previously enqueued CTRAN kernels
  return streamWaitStream(curStream, execOrderStream_, execEvent_);
}

commResult_t CtranGpe::Impl::postKernelLaunch(cudaStream_t curStream) {
  // add sync point execOrderStream_ to block future CTRAN kernels
  return streamWaitStream(execOrderStream_, curStream, execEvent_);
}

commResult_t CtranGpe::Impl::submit(
    CtranGpeCmd::TypeEnum type,
    std::vector<std::unique_ptr<struct OpElem>> opGroup,
    opFunc func,
    KernelConfig& kernelConfig,
    const void* ncclKernel,
    std::optional<std::chrono::milliseconds> timeout,
    PreLaunchGraphPrepareFn graphPrepareFn) {
  commResult_t res = commSuccess;

  // Reclaim once to gain back available flags
  if (this->kernelFlagPool->size() == 0) {
    this->kernelFlagPool->reclaim();
  }

  if (this->checksumPool->size() == 0) {
    this->checksumPool->reclaim();
  }

  // We do not expect such high amount of inuse flags, return error here to
  // avoid hang. If there can be really such a high usage case, either
  // increase the pool size or set a timeout here to reclaim multiple times.
  // Avoid timeout logic for now to avoid complexity.
  if (this->kernelFlagPool->size() == 0) {
    CLOGF(
        ERR,
        "CTRAN-GPE: Internal KernelFlag pool has unexpected high usage (capacity: {}, available: {}). It is likely that some COMM kernels are not released properly",
        kernelFlagPool->capacity(),
        kernelFlagPool->size());
    return commInternalError;
  }

  // Error checking before GPE cmd and kernel submission
  if (kernelConfig.args.devState_d == nullptr) {
    CLOGF(
        ERR,
        "COMM internally passed invalid devState_d (nullptr) to kernel {}",
        kernelTypeToName[kernelConfig.type]);
    return commInternalError;
  }

  bool ifchecksum = checksumIsSampled(kernelConfig.type, kernelConfig.opCount);

  // Get kernelFlag from the pool
  auto kernelFlag = opGroup.size() ? this->kernelFlagPool->pop() : nullptr;
  volatile int* flag = nullptr;
  if (kernelFlag != nullptr) {
    // TODO: remove this allowlist once the per-block flag is enabled in all
    // kernels. This helps to reduce blast radius of the larger fix.
    static const std::unordered_set<KernelConfig::KernelType>
        perBlockFlagEnabledKernels = {
            KernelConfig::KernelType::ALLREDUCE,
            KernelConfig::KernelType::SEND,
            KernelConfig::KernelType::RECV,
            KernelConfig::KernelType::SENDRECV,
            KernelConfig::KernelType::RECV_UNPACK,
            KernelConfig::KernelType::SENDRECV_UNPACK,
        };
    if (perBlockFlagEnabledKernels.contains(kernelConfig.type)) {
      kernelFlag->numGroups_ = kernelConfig.numBlocks;
    } else {
      kernelFlag->numGroups_ = 1;
    }
    flag = kernelFlag->flag_;
  }

  // Set first kernel argument as kernelFlag if GPE op is not empty.
  // Check it before passing opGroup to cmd
  std::array<void*, 3> kernelArgs;
  kernelArgs.at(0) = (void*)&flag;
  kernelArgs.at(1) = (void*)&kernelConfig.args.devState_d;
  if (kernelConfig.algoArgs) {
    // Use pointer to algoArgs if specified; otherwise, pass default
    // args.collective
    // TODO: refactor existing collective args following algoArgs approach, to
    // avoid aggregating algo specific args in a single struct
    kernelArgs.at(2) = kernelConfig.algoArgs;
  } else {
    kernelArgs.at(2) = (void*)&kernelConfig.args.collective;
  }

  // Record CollTrace event. Must be called before moving opGroup to cmd
  auto colltraceHandle = meta::comms::colltrace::getCollTraceHandle(
      comm, opGroup, kernelConfig, ifchecksum);

  utils::cudagraph::StreamCaptureInfo streamCaptureInfo;
  FB_CUDACHECK(
      utils::cudagraph::getStreamCaptureInfo(
          kernelConfig.stream, streamCaptureInfo));

  size_t opGroupSize = 0;
  // Enqueue op to gpeThread if any op is appended
  if (!opGroup.empty()) {
    // record opGroup size before moving the object
    opGroupSize = opGroup.size();
    class CtranGpeCmd* cmd = new class CtranGpeCmd;
    cmd->type = type;
    cmd->kernelFlag = kernelFlag;
    cmd->timeout = timeout;

    if (type == CtranGpeCmd::TypeEnum::GRAPH_ENQUEUE) {
      cmd->coll.opGroup = std::move(opGroup);
      cmd->coll.func = func;
      if (colltraceHandle != nullptr) {
        cmd->coll.collHandle = colltraceHandle;
      }
      cmd->coll.comm = comm;
    }
    if (streamCaptureInfo.status == cudaStreamCaptureStatusActive) {
      FB_COMMCHECK(preLaunchGraphPrepare(cmd, graphPrepareFn));
      struct cmdCbPlan* plan = new struct cmdCbPlan;
      plan->cmd = cmd;
      plan->gpe = this->gpe;
      cmd->persistent = true;

      FB_COMMCHECKGOTO(
          utils::cudagraph::addHostNode(
              cmd,
              reinterpret_cast<void*>(plan),
              cmdCb,
              cmdDestroy,
              streamCaptureInfo),
          res,
          fail);
    } else {
      cmdEnqueue(cmd);
    }
  }

  // FIXME: the multi-stream order enforcement is not compatible with cuda graph
  // capture; disable it under cuda graph capture as a workaround. We'd need
  // proper fix to support the compatibility.
  if (streamCaptureInfo.status != cudaStreamCaptureStatusActive) {
    FB_COMMCHECK(preKernelLaunch(kernelConfig.stream));
  }

  if (NCCL_CTRAN_ENALBE_CLUSTER_KERNEL_LAUNCH) {
#if CUDART_VERSION >= 11080

    dim3 grid = {kernelConfig.numBlocks, 1, 1};
    dim3 blocks = {kernelConfig.numThreads, 1, 1};
    unsigned int clusterSize = NCCL_CTRAN_CGA_CLUSTER_SIZE;
    CUlaunchConfig launchConfig = {0};
    CUlaunchAttribute launchAttrs[3];
    int attrs = 0;
    if (clusterSize) {
      // Grid dimension must be divisible by clusterSize
      if (grid.x % clusterSize) {
        clusterSize = 1;
      }
      launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
      launchAttrs[attrs++].value.clusterDim = {clusterSize, 1, 1};
      launchAttrs[attrs].id =
          CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE;
      launchAttrs[attrs++].value.clusterSchedulingPolicyPreference =
          CU_CLUSTER_SCHEDULING_POLICY_SPREAD;
    }
#if CUDART_VERSION >= 12000
    // Set the NCCL Mem Sync domain on CUDA 12.0 and later (sm90)
    launchAttrs[attrs].id = CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN;
    launchAttrs[attrs++].value.memSyncDomain =
        (CUlaunchMemSyncDomain)NCCL_MEM_SYNC_DOMAIN;
#endif
    launchConfig.gridDimX = grid.x;
    launchConfig.gridDimY = grid.y;
    launchConfig.gridDimZ = grid.z;
    launchConfig.blockDimX = blocks.x;
    launchConfig.blockDimY = blocks.y;
    launchConfig.blockDimZ = blocks.z;
    launchConfig.attrs = launchAttrs;
    launchConfig.numAttrs = attrs;
    launchConfig.hStream = kernelConfig.stream;
    CUfunction cuFn;
    FB_CUDACHECKGOTO(cudaGetFuncBySymbol(&cuFn, ncclKernel), res, fail);
    CLOGF_TRACE(COLL, "CTranGPE: submit {}", kernelConfig.toString());

    if (colltraceHandle != nullptr) {
      colltraceHandle->trigger(
          CollTraceHandleTriggerState::BeforeEnqueueKernel);
    }
    FB_CUCHECK_GOTO(
        cuLaunchKernelEx(&launchConfig, cuFn, kernelArgs.data(), nullptr),
        res,
        fail);
#endif
  } else {
    // Enqueue the kernel with arguments.  It will not start till all other
    // operations on this stream have completed
    dim3 grid = {kernelConfig.numBlocks, 1, 1};
    dim3 blocks = {kernelConfig.numThreads, 1, 1};

    CLOGF_TRACE(COLL, "CTranGPE: submit {}", kernelConfig.toString());

    if (colltraceHandle != nullptr) {
      colltraceHandle->trigger(
          CollTraceHandleTriggerState::BeforeEnqueueKernel);
    }

    // Set the maximum dynamic shared memory size since CtranAlgoDeviceState
    // (~67KB) exceeds the default limit (48KB)
    FB_CUDACHECKGOTO(
        cudaFuncSetAttribute(
            ncclKernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            sizeof(CtranAlgoDeviceState)),
        res,
        fail);
    FB_CUDACHECKGOTO(
        cudaLaunchKernel(
            ncclKernel,
            grid,
            blocks,
            kernelArgs.data(),
            sizeof(CtranAlgoDeviceState),
            kernelConfig.stream),
        res,
        fail);
  }

  if (ifchecksum) {
    ChecksumItem* checksumItem = this->checksumPool->pop();
    std::optional<ChecksumArgs> checksumArgs =
        ctranFillChecksumArgs(kernelConfig, checksumItem, comm);
    if (checksumArgs.has_value()) {
      std::array<void*, 3> args;
      args.at(0) = &checksumArgs.value().buf;
      args.at(1) = &checksumArgs.value().size;
      args.at(2) = &checksumArgs.value().out;
      void* kernelFn =
          reinterpret_cast<void*>(checksumKernel<CHECKSUM_NUM_THREAD>);
      auto checksumGrid = getChecksumGrid(checksumArgs.value().size);
      auto res = cudaLaunchKernel(
          kernelFn,
          checksumGrid,
          CHECKSUM_NUM_THREAD,
          args.data(),
          0,
          kernelConfig.stream);
      if (res != cudaSuccess && checksumItem != nullptr) {
        // Do not return error if the internal checksum fails
        CLOGF(WARN, "CTranGPE: Failed to launch checksum kernel");
        checksumItem->reset();
      } else if (colltraceHandle != nullptr) {
        folly::dynamic dynamicObj = folly::dynamic::object();
        dynamicObj["checksumItem"] =
            folly::toDynamic(reinterpret_cast<int64_t>(checksumItem));
        colltraceHandle->triggerPlugin("ctranChecksum", std::move(dynamicObj));
      }
    }
  }

  // FIXME: the multi-stream order enforcement is not compatible with cuda graph
  // capture; disable it under cuda graph capture as a workaround. We'd need
  // proper fix to support the compatibility.
  if (streamCaptureInfo.status != cudaStreamCaptureStatusActive) {
    FB_COMMCHECK(postKernelLaunch(kernelConfig.stream));
  }

  if (colltraceHandle != nullptr) {
    colltraceHandle->trigger(CollTraceHandleTriggerState::AfterEnqueueKernel);
  }

  CLOGF_SUBSYS(
      INFO,
      COLL,
      "CTRAN-GPE: Launched kernelType {} (opGroup size {}) with kernelFlag={}",
      kernelConfig.type,
      opGroupSize,
      (void*)kernelFlag);

  comm->ctran_->updateOpCount();

  return commSuccess;

fail:
  if (kernelFlag != nullptr) {
    kernelFlag->reset();
  }
  return res;
}

commResult_t CtranGpe::Impl::submitHost(
    CtranGpeCmd::TypeEnum type,
    std::vector<std::unique_ptr<struct OpElem>> opGroup,
    opFunc func,
    KernelConfig& kernelConfig,
    CtranExRequestImpl* exReq,
    bool allowNullReq) {
  // Enqueue op to gpeThread if any op is appended
  if (!opGroup.empty()) {
    class CtranGpeCmd* cmd = new class CtranGpeCmd;
    cmd->type = type;
    cmd->kernelFlag = nullptr;
    cmd->exReq = exReq;
    cmd->allowNullReq = allowNullReq;

    if (type == CtranGpeCmd::TypeEnum::GRAPH_ENQUEUE) {
      cmd->coll.opGroup = std::move(opGroup);
      cmd->coll.func = func;
      cmd->coll.comm = comm;
      auto colltraceHandle = meta::comms::colltrace::getCollTraceHandle(
          comm, cmd->coll.opGroup, kernelConfig, false /* ifChecksum */);
      if (colltraceHandle != nullptr) {
        cmd->coll.collHandle = colltraceHandle;
        // We don't have before/after enqueue kernel for cpu collectives
        // But we still need to trigger the before/after enqueue kernel for
        // colltrace to function.
        colltraceHandle->trigger(
            CollTraceHandleTriggerState::BeforeEnqueueKernel);
        colltraceHandle->trigger(
            CollTraceHandleTriggerState::AfterEnqueueKernel);
      }
    }

    cmdEnqueue(cmd);
  }

  comm->ctran_->updateOpCount();

  return commSuccess;
}

void CtranGpe::Impl::start() {
  thread_ = std::thread([this] { gpeThreadFn(); });
}

void CtranGpe::Impl::terminate() {
  class CtranGpeCmd* cmd = new class CtranGpeCmd;
  cmd->type = CtranGpeCmd::TypeEnum::TERMINATE;

  cmdEnqueue(cmd);
  thread_.join();

  const auto& statex = comm->statex_;
  CLOGF_SUBSYS(
      INFO,
      INIT,
      "CTranGPE thread joined on rank {} commHash {:x} commDesc {}",
      statex->rank(),
      statex->commHash(),
      statex->commDesc());
}

void CtranGpe::Impl::gpeThreadFn() {
  const auto& statex = comm->statex_;
  assert(statex != nullptr);
  commNamedThreadStart(
      "CTranGPE",
      statex->rank(),
      statex->commHash(),
      statex->commDesc(),
      __func__);

  CTRAN_ASYNC_ERR_GUARD(comm->getAsyncError(), {
    FB_CUDACHECKTHROW(cudaSetDevice(cudaDev));

    while (1) {
      auto cmd = cmdDequeue();

      if (cmd->timeout.has_value()) {
        comm->setTimeout(cmd->timeout.value());
      }
      SCOPE_EXIT {
        // if comm is aborted for any reason, we mark it as aborted to avoid
        // resetting the state.
        if (comm->testAbort()) {
          comm->setAbort();
        }
        comm->cancelTimeout();
      };

      if (cmd->type == CtranGpeCmd::TypeEnum::TERMINATE) {
        CLOGF_SUBSYS(
            INFO,
            INIT,
            "[COMM THREAD] CTranGPE thread terminated on rank {} commHash {:x} commDesc {}",
            statex->rank(),
            statex->commHash(),
            statex->commDesc());
        delete cmd;
        return;
      }

      // If kernelFlag is set, indicates it is a device memory communication
      // thus, wait for the kernel to launch
      KernelFlagItem* kernelFlag = cmd->kernelFlag;
      if (kernelFlag) {
        volatile int* flag_d = kernelFlag->flag_;
        // Here we check just flag_d[0]. This is ok because Kernel Start signal
        // is only used for tracing purposes. Before the flags are freed below
        // with reset, all block flags are checked.
        while (flag_d[0] != KERNEL_STARTED &&
               flag_d[0] != KERNEL_STARTED_AND_EXIT && !comm->testAbort()) {
          std::this_thread::yield();
        }
      }

      if (cmd->coll.collHandle != nullptr) {
        cmd->coll.collHandle->trigger(
            CollTraceHandleTriggerState::KernelStarted);
      }

      auto comm = cmd->coll.comm;
      if (comm != nullptr) {
        ncclx::colltrace::recordMapperEvent(
            comm,
            ncclx::colltrace::CollStart{
                .coll = cmd->coll.collHandle != nullptr
                    ? cmd->coll.collHandle->getCollRecord().value_or(
                          std::make_unique<meta::comms::colltrace::CollRecord>(
                              -1, nullptr))
                    : nullptr,
            });
      }

      {
        // comm may be dummy in GPE UT, although never happens in real.
        // Pass in nullptr mapper to skip lock.
        auto mapper =
            ctranInitialized(comm) ? comm->ctran_->mapper.get() : nullptr;
        CtranMapperEpochRAII epochRAII(mapper);

        /* run collective */
        // TODO: lost peerRank info which would be useful for some errors (e.g.,
        // commRemoteError). We may want to enrich commResult_t to contain such
        // info at failure or throw exception from bottom.
        CTRAN_ASYNC_ERR_GUARD_FAULT_TOLERANCE(comm, {
          FB_COMMCHECKTHROW_EX(
              cmd->coll.func(cmd->coll.opGroup),
              statex->rank(),
              statex->commHash());
        });

        if (cmd->persistent) {
          for (const auto& x : cmd->coll.opGroup) {
            x->setStatus(KernelElem::ElemStatus::INUSE);
          }
        } else {
          cmd->coll.opGroup.clear();
        }
      }

      if (kernelFlag) {
        volatile int* flag_d = kernelFlag->flag_;
        if (flag_d[0] == KERNEL_STARTED_AND_EXIT) {
          // Indicate kernel would exit without the terminate signal, thus free
          // the flag now
          while (!kernelFlag->testFlagAllGroups(KERNEL_STARTED_AND_EXIT)) {
            std::this_thread::yield();
          }
          // After all blocks exited, we can safely reset.
          kernelFlag->reset();
        } else {
          // Stop kernel and kernel will free up the flag after confirmed the
          // termination
          kernelFlag->setFlagPerGroup(
              comm->testAbort() ? KERNEL_HOST_ABORT : KERNEL_TERMINATE);

          if (comm->abortEnabled()) {
            // Wait for kernel to exit, only necessary for Abort enabled
            // case
            while (!kernelFlag->testFlagAllGroups(KERNEL_UNSET) &&
                   !comm->testAbort()) {
              std::this_thread::yield();
            }
            if (comm->testAbort()) {
              for (int i = 0; i < kernelFlag->numGroups_; i++) {
                if (kernelFlag->flag_[i] == KERNEL_TERMINATE) {
                  kernelFlag->flag_[i] = KERNEL_HOST_ABORT;
                }
              }
            }
          }
        }
      } else {
        // This is a host memory communication, so we can complete it here
        if (!cmd->allowNullReq) {
          FB_CHECKABORT(
              cmd->exReq,
              "CTranGPE: ExReq is nullptr but required for host memory communication");
          cmd->exReq->complete();
        }
      }

      if (cmd->coll.collHandle != nullptr) {
        cmd->coll.collHandle->trigger(
            CollTraceHandleTriggerState::KernelFinished);
      }

      if (cmd->coll.comm != nullptr) {
        ncclx::colltrace::recordMapperEvent(
            cmd->coll.comm, ncclx::colltrace::CollEnd{});
      }

      if (!cmd->persistent) {
        delete cmd;
      }
    }
    return;
  });
}

#define CHECK_KELEM_NGROUPS(e)                                    \
  FB_CHECKABORT(                                                  \
      (e)->ngroups > 0 && (e)->ngroups < MAX_NGROUPS,             \
      "Invalid ngroups {} in nelem {} (expect 0 < ngroups < {})", \
      ngroups,                                                    \
      (void*)(e),                                                 \
      MAX_NGROUPS);

#define CHECK_KELEM_GROUPID(e, groupId)                           \
  FB_CHECKABORT(                                                  \
      groupId > 0 && groupId < MAX_NGROUPS,                       \
      "Invalid groupId {} in nelem {} (expect 0 < groupId < {})", \
      groupId,                                                    \
      (void*)(e),                                                 \
      MAX_NGROUPS);

void KernelElem::unuse() {
  CHECK_KELEM_NGROUPS(this);
  for (int i = 0; i < this->ngroups; i++) {
    this->status[i] = KernelElem::ElemStatus::RESET;
  }
  CLOGF_TRACE(
      COLL,
      "CTRAN-GPE: elem {} set to unuse with ngroups {}",
      (void*)this,
      this->ngroups);
}

void KernelElem::setStatus(KernelElem::ElemStatus s) {
  CHECK_KELEM_NGROUPS(this);
  for (int i = 0; i < this->ngroups; i++) {
    this->status[i] = s;
  }
  CLOGF_TRACE(
      COLL,
      "CTRAN-GPE: elem {} set to {} with ngroups {}",
      (void*)this,
      s,
      this->ngroups);
}

void KernelElem::free() {
  CHECK_KELEM_NGROUPS(this);

  bool canFree = true;
  for (int i = 0; i < this->ngroups; i++) {
    // OK to free
    if (this->status[i] == KernelElem::ElemStatus::DONE ||
        this->status[i] == KernelElem::ElemStatus::RESET) {
      continue;
      // Skip and let kernel free it
    } else if (this->status[i] == KernelElem::ElemStatus::REVOKED) {
      canFree = false;
      break;
    }

    // Let elem free if in any other state, because it likely indicates the
    // collective already fails
    // TODO: clearly set FATAL state at algorithm layers
  }

  if (!canFree) {
    return;
  }

  // Free
  for (int i = 0; i < this->ngroups; i++) {
    this->status[i] = KernelElem::ElemStatus::RESET;
  }
  CLOGF_TRACE(
      COLL,
      "CTRAN-GPE: elem {} freed with ngroups {}",
      (void*)this,
      this->ngroups);
}

bool KernelElem::isFree() {
  CHECK_KELEM_NGROUPS(this);
  bool allFree = true;
  for (int i = 0; i < this->ngroups && allFree; i++) {
    allFree &= (this->status[i] == KernelElem::ElemStatus::RESET);
  }
  if (allFree) {
    CLOGF_TRACE(
        COLL,
        "CTRAN-GPE: elem {} isFree = true with ngroups {}",
        (void*)this,
        this->ngroups);
  }
  return allFree;
}

void KernelElem::post(int groupId) {
  // ensure in-order exec of status store after any argument update before
  // post
  wcStoreFence();

  if (groupId == -1) {
    CHECK_KELEM_NGROUPS(this);

    // Ring doorbell to each thread block on kernel side
    for (int i = 0; i < this->ngroups; i++) {
      this->status[i] = KernelElem::ElemStatus::POSTED;
    }
    CLOGF_TRACE(
        COLL,
        "CTRAN-GPE: elem {} posted with ngroups {}",
        (void*)this,
        this->ngroups);
  } else {
    CHECK_KELEM_GROUPID(this, groupId);

    // Ring doorbell to each thread block on kernel side
    this->status[groupId] = KernelElem::ElemStatus::POSTED;
    CLOGF_TRACE(
        COLL,
        "CTRAN-GPE: elem {} posted with groupId {}",
        (void*)this,
        groupId);
  }
}

void KernelElem::revoke() {
  CHECK_KELEM_NGROUPS(this);

  // ensure in-order exec of status store after any argument update before post
  wcStoreFence();
  // Ring doorbell to each thread block on kernel side
  for (int i = 0; i < this->ngroups; i++) {
    this->status[i] = KernelElem::ElemStatus::REVOKED;
  }
  CLOGF_TRACE(
      COLL,
      "CTRAN-GPE: elem {} revoked with ngroups {}",
      (void*)this,
      this->ngroups);
}

bool KernelElem::isComplete(int groupId) {
  bool complete = true;
  if (groupId == -1) {
    CHECK_KELEM_NGROUPS(this);

    // check all thread blocks have finished
    for (int i = 0; i < this->ngroups && complete; i++) {
      complete &= (this->status[i] == KernelElem::ElemStatus::DONE);
    }
  } else {
    CHECK_KELEM_GROUPID(this, groupId);
    complete = (this->status[groupId] == KernelElem::ElemStatus::DONE);
  }
  if (complete) {
    CLOGF_TRACE(
        COLL,
        "CTRAN-GPE: elem {} completed at step {}",
        (void*)this,
        (int)this->stepDone);
  }
  return complete;
}

void KernelElem::wait(int groupId) {
  // wait for all thread blocks to complete
  while (!this->isComplete(groupId)) {
    // friendly spin so we don't hog CPU
    std::this_thread::yield();
  }
}

KernelElemPool::KernelElemPool(size_t capacity) : capacity_(capacity) {
  FB_CUDACHECKTHROW(cudaHostAlloc(
      &this->memPtr_,
      this->capacity_ * sizeof(struct KernelElem),
      cudaHostAllocDefault));

  for (int i = 0; i < capacity_; ++i) {
    KernelElem* workElem = reinterpret_cast<KernelElem*>(this->memPtr_) + i;
    this->resetWorkElem(workElem);
    this->freeWorkElems_.push(workElem);
  }
  return;
}

KernelElemPool::~KernelElemPool() {
  this->reclaim();
  if (this->inuseWorkElems_.size()) {
    CLOGF(
        WARN,
        "CTRAN-GPE: Internal KernelElem pool has {} inuse elements",
        this->inuseWorkElems_.size());
  }
  FB_CUDACHECKIGNORE(cudaFreeHost(this->memPtr_));

  // Dot not throw exception in destructor to avoid early termination in stack
  // unwind. See discussion in
  // https://stackoverflow.com/questions/130117/if-you-shouldnt-throw-exceptions-in-a-destructor-how-do-you-handle-errors-in-i
}

void KernelElemPool::resetWorkElem(KernelElem* workElem) {
  workElem->~KernelElem();
  new (workElem) KernelElem();

  for (int i = 0; i < CTRAN_ALGO_MAX_THREAD_BLOCKS; ++i) {
    workElem->status[i] = KernelElem::ElemStatus::RESET;
  }
}

size_t KernelElemPool::size() {
  return this->freeWorkElems_.size();
}

size_t KernelElemPool::capacity() {
  return capacity_;
}

KernelElem* KernelElemPool::pop(int ngroups) {
  if (ngroups > CTRAN_ALGO_MAX_THREAD_BLOCKS) {
    CLOGF(
        WARN,
        "CTRAN-GPE: ngroups {} exceeds max thread blocks {}",
        ngroups,
        CTRAN_ALGO_MAX_THREAD_BLOCKS);
    return nullptr;
  }

  KernelElem* workElem = this->freeWorkElems_.top();
  this->freeWorkElems_.pop();
  workElem->ngroups = ngroups;
  for (int i = 0; i < ngroups; i++) {
    workElem->status[i] = KernelElem::ElemStatus::INUSE;
  }

  CLOGF_TRACE(
      COLL,
      "CTRAN-GPE: elem {} popped with ngroups {}",
      (void*)workElem,
      ngroups);

  this->inuseWorkElems_.push_back(workElem);
  return workElem;
}

void KernelElemPool::reclaim() {
  // Iterate inuseWorkElems_ and reclaim any unused workElems
  auto it = this->inuseWorkElems_.begin();
  while (it != this->inuseWorkElems_.end()) {
    auto workElem = *it;

    // Each kernel thread block resets inuse flag when finished
    // If no more thread block is using, reclaim the workElem
    if (workElem->isFree()) {
      it = this->inuseWorkElems_.erase(it);

      this->resetWorkElem(workElem);
      this->freeWorkElems_.push(workElem);
    } else {
      it++;
    }
  }
}

bool checksumIsSampled(KernelConfig::KernelType kernelType, int opCount) {
  using KT = KernelConfig::KernelType;
  switch (kernelType) {
    case KT::ALLGATHER:
      return ChecksumHandler<KT::ALLGATHER>::isSampled(opCount);
    case KT::SEND:
      return ChecksumHandler<KT::SEND>::isSampled(opCount);
    case KT::RECV:
      return ChecksumHandler<KT::RECV>::isSampled(opCount);
    default:
      return false;
  }
}

std::optional<ChecksumArgs> ctranFillChecksumArgs(
    KernelConfig& kernelConfig,
    ChecksumItem* checksumItem,
    const CtranComm* comm) {
  using KT = KernelConfig::KernelType;
  switch (kernelConfig.type) {
    case KT::ALLGATHER:
      return ChecksumHandler<KT::ALLGATHER>::ctranFillChecksumArgs(
          kernelConfig, checksumItem, comm);
    case KT::SEND:
      return ChecksumHandler<KT::SEND>::ctranFillChecksumArgs(
          kernelConfig, checksumItem, comm);
    case KT::RECV:
      return ChecksumHandler<KT::RECV>::ctranFillChecksumArgs(
          kernelConfig, checksumItem, comm);
    default:
      CLOGF(
          WARN,
          "CTRAN-GPE: Unsupported kernel type {} for checksum",
          kernelConfig.type);
      return std::nullopt;
  }
}

commResult_t allocGpeKernelSyncs(
    GpeKernelSyncPool* gpeKernelSyncPool,
    size_t count,
    int nworkers,
    std::vector<ctran::algos::GpeKernelSync*>& gpeKernelSyncs) {
  // reclaim from outstanding kernels once if pool items are insufficient
  if (gpeKernelSyncPool->size() < count) {
    gpeKernelSyncPool->reclaim();

    // We do not expect such high amount of inuse pool items, return error here
    // to avoid hang. If there can be really such a high usage case, either
    // increase the pool size or set a timeout here to reclaim multiple times.
    // Avoid timeout logic for now to avoid complexity.
    if (count > gpeKernelSyncPool->size()) {
      CLOGF(
          WARN,
          "CTRAN-GPE: Internal KernelSync pool has unexpected high usage (capacity: {}, available: {}, current request: {}). "
          "It is likely that some COMM kernels are not released properly",
          gpeKernelSyncPool->capacity(),
          gpeKernelSyncPool->size(),
          count);
      return ErrorStackTraceUtil::log(commInternalError);
    }
  }

  for (int i = 0; i < count; i++) {
    auto* g = gpeKernelSyncPool->pop();
    if (!g) {
      return ErrorStackTraceUtil::log(commInternalError);
    }
    // essentially the constructor (note resetStatus() is needed since we didn't
    // set nworkers before this point)
    g->nworkers = nworkers;
    g->resetStatus();
    gpeKernelSyncs.emplace_back(g);
  }
  return commSuccess;
}
