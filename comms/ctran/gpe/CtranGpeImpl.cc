// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <chrono>
#include <optional>
#include <stdexcept>

#include <folly/dynamic.h>

#include "comms/ctran/algos/AllToAll/AllToAllPImpl.h"
#include "comms/ctran/algos/AllToAll/AllToAllvDynamicPImpl.h"
#include "comms/ctran/algos/common/GpeKernel.h"
#include "comms/ctran/colltrace/CollTraceWrapper.h"
#include "comms/ctran/colltrace/MapperTrace.h"
#include "comms/ctran/gpe/CtranChecksum.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/gpe/CtranGpeDev.h"
#include "comms/ctran/gpe/CtranGpeImpl.h"
#include "comms/ctran/mapper/CtranMapper.h"
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
        {KernelConfig::KernelType::DEVICE_ALLTOALLV, "DeviceAllToAllvPipes"},
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
}

CtranGpe::Impl::~Impl() = default;

void OrderedWorkStreamGuard::init(const CommLogData& logMetaData) {
  logMetaData_ = &logMetaData;
  FB_CUDACHECKTHROW_EX(
      cudaEventCreateWithFlags(&execModeSyncEvent_, cudaEventDisableTiming),
      logMetaData);
  sideStream_ = std::make_unique<meta::comms::GraphSideStream>();
}

OrderedWorkStreamGuard::~OrderedWorkStreamGuard() {
  FB_CHECKABORT(
      logMetaData_ != nullptr,
      "OrderedWorkStreamGuard destroyed without init()");
  FB_CUDACHECKTHROW_EX(cudaEventDestroy(execModeSyncEvent_), *logMetaData_);
}

OrderedWorkStreamGuard::Scope::Scope(
    OrderedWorkStreamGuard& guard,
    cudaStream_t userStream,
    const ctran::utils::cudagraph::StreamCaptureInfo& captureInfo)
    : guard_(&guard), userStream_(userStream), captureInfo_(captureInfo) {
  status_ = guard_->doAcquire(userStream_, captureInfo_);
}

OrderedWorkStreamGuard::Scope::~Scope() {
  if (guard_) {
    guard_->doRelease(userStream_, captureInfo_);
  }
}

OrderedWorkStreamGuard::Scope::Scope(Scope&& other) noexcept
    : guard_(other.guard_),
      userStream_(other.userStream_),
      captureInfo_(other.captureInfo_),
      status_(other.status_) {
  other.guard_ = nullptr;
}

OrderedWorkStreamGuard::Scope& OrderedWorkStreamGuard::Scope::operator=(
    Scope&& other) noexcept {
  if (this != &other) {
    if (guard_) {
      guard_->doRelease(userStream_, captureInfo_);
    }
    guard_ = other.guard_;
    userStream_ = other.userStream_;
    captureInfo_ = other.captureInfo_;
    status_ = other.status_;
    other.guard_ = nullptr;
  }
  return *this;
}

OrderedWorkStreamGuard::Scope OrderedWorkStreamGuard::acquire(
    cudaStream_t userStream,
    const ctran::utils::cudagraph::StreamCaptureInfo& captureInfo) {
  return Scope(*this, userStream, captureInfo);
}

void CUDART_CB CtranGpe::Impl::cmdCb(void* data) {
  CtranGpeCmd* cmd = reinterpret_cast<CtranGpeCmd*>(data);
  if (cmd->persistent) {
    cmd->inFlight.fetch_add(1, std::memory_order_release);
  }
  cmd->gpe->pimpl->cmdEnqueue(cmd);
}

CtranGpeCmd::~CtranGpeCmd() {
  // Release kernelFlag back to pool for persistent (graph) cmds.
  // clearPersistent() removes the reclaim guard. reset() clears flags
  // to KERNEL_UNSET — normally a no-op since the kernel already wrote
  // UNSET (both TERMINATE and HOST_ABORT paths), but needed if the
  // graph was never replayed (flags still KERNEL_SCHEDULED from onPop).
  if (persistent && kernelFlag) {
    kernelFlag->clearPersistent();
    kernelFlag->reset();
  }

  // For persistent (graph) cmds, postKernelCleanup is deliberately skipped
  // during replay (the resources must persist across replays). Run it here
  // on destruction so resources like device-allocated sendsList/recvsList
  // are freed when the graph is destroyed.
  if (postKernelCleanup) {
    postKernelCleanup();
  }
}

void CUDART_CB CtranGpe::Impl::cmdDestroy(void* data) {
  CtranGpeCmd* cmd = reinterpret_cast<CtranGpeCmd*>(data);
  if (!cmd->persistent) {
    CLOGF(WARN, "CTranGPE: cmd desctructor called for non-persistent cmd");
  }
  // Wait for the GPE thread to finish processing any queued instances of
  // this cmd before deleting. With KERNEL_STARTED_AND_EXIT persistent cmds,
  // the GPE processes cmds instantly (stale flag), so cmdCb enqueues from
  // graph replays may still be in the GPE queue when the graph is destroyed.
  while (cmd->inFlight.load(std::memory_order_acquire) > 0) {
    std::this_thread::yield();
  }
  delete cmd;
}

commResult_t OrderedWorkStreamGuard::doAcquire(
    cudaStream_t userStream,
    const utils::cudagraph::StreamCaptureInfo& captureInfo) {
  const bool isCapturing = captureInfo.status == cudaStreamCaptureStatusActive;

  bool isNewCapture = isCapturing && captureInfo.id != lastCaptureId_;
  if (isNewCapture) {
    lastCaptureId_ = captureInfo.id;
    everCaptured_ = true;
  }

  auto doWait = [&]() -> commResult_t {
    FB_CUDACHECK(cudaStreamWaitEvent(
        userStream,
        execModeSyncEvent_,
        isCapturing ? cudaEventWaitExternal : cudaEventWaitDefault));
    return commSuccess;
  };

  if (lastUserStream_ == nullptr) {
    if (isCapturing) {
      return doWait();
    }
    return commSuccess; // first submit ever
  }

  if (!isCapturing) {
    if (everCaptured_) {
      // Graph replays bypass submit(), so we cannot know for certain whether
      // the previous operation was a graph replay or eager. CPU-side sync
      // ensures any in-flight graph host node (which enqueues a GPE command)
      // has fired before the caller can cmdEnqueue. Without this, the eager
      // command lands in the GPE queue first and the single-threaded GPE
      // deadlocks.
      FB_CUDACHECK(cudaEventSynchronize(execModeSyncEvent_));
    } else if (userStream != lastUserStream_) {
      // Cross-stream eager, no graphs: GPU-side ordering only.
      // We don't make any thread-safety guarantees for submit()
      // so this is sufficient.
      FB_COMMCHECK(doWait());
    }
    return commSuccess;
  }

  if (!isNewCapture) {
    // Intra-capture cross-stream: add the RECORD node from the previous
    // doRelease as a capture dependency of this stream. This creates an
    // explicit graph edge, since cudaStreamWaitEvent cannot see RECORD
    // nodes added via cudaGraphAddEventRecordNode.
#if defined(__HIP_PLATFORM_AMD__)
    FB_CUDACHECK(cudaStreamUpdateCaptureDependencies(
        userStream, &lastRecordNode_, 1, hipStreamAddCaptureDependencies));
#elif CUDART_VERSION >= 13000
    FB_CUDACHECK(cudaStreamUpdateCaptureDependencies(
        userStream,
        &lastRecordNode_,
        nullptr,
        1,
        cudaStreamAddCaptureDependencies));
#else
    FB_CUDACHECK(cudaStreamUpdateCaptureDependencies(
        userStream, &lastRecordNode_, 1, cudaStreamAddCaptureDependencies));
#endif
  }

  return doWait();
}

commResult_t OrderedWorkStreamGuard::doRelease(
    cudaStream_t userStream,
    const utils::cudagraph::StreamCaptureInfo& captureInfo) {
  const bool isCapturing = captureInfo.status == cudaStreamCaptureStatusActive;

  if (!isCapturing) {
    FB_CUDACHECK(cudaEventRecord(execModeSyncEvent_, userStream));
  } else {
    // Route the external EVENT_RECORD node onto a side stream so its
    // release fence doesn't stall unrelated work on userStream between
    // ctran submissions. The next doAcquire on userStream still sees
    // lastRecordNode_ via cudaStreamUpdateCaptureDependencies, which
    // reinstates the explicit DAG edge ordering the next ctran op after
    // this record. Non-ctran work on userStream is not serialized
    // behind the record.
    commResult_t innerRes = commSuccess;
    FB_CUDACHECK(
        sideStream_->fork_from(userStream, [&](cudaStream_t sideStream) {
          innerRes = utils::cudagraph::addEventRecordNodeToCapture(
              sideStream, captureInfo.g, execModeSyncEvent_, &lastRecordNode_);
        }));
    if (innerRes != commSuccess) {
      return innerRes;
    }
  }

  lastUserStream_ = userStream;

  return commSuccess;
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

  // Error checking before GPE cmd and kernel submission
  if (kernelConfig.args.devState_d == nullptr) {
    CLOGF(
        ERR,
        "COMM internally passed invalid devState_d (nullptr) to kernel {}",
        kernelTypeToName[kernelConfig.type]);
    return commInternalError;
  }

  bool ifchecksum = checksumIsSampled(kernelConfig.type, kernelConfig.opCount);

  utils::cudagraph::StreamCaptureInfo streamCaptureInfo;
  FB_CUDACHECK(
      utils::cudagraph::getStreamCaptureInfo(
          kernelConfig.stream, streamCaptureInfo));
  bool isCapturing = streamCaptureInfo.status == cudaStreamCaptureStatusActive;

  // For eager (non-capture) submits with empty opGroup but a
  // postKernelCleanup, we still need a cmd + kernelFlag so the GPE thread
  // can synchronize with the kernel before running cleanup. During graph
  // capture the cleanup is retained directly on the graph via
  // retainUserObject, avoiding host-node overhead.
  bool needsKernelFlag =
      !opGroup.empty() || (kernelConfig.postKernelCleanup && !isCapturing);

  auto kernelFlag = needsKernelFlag ? this->kernelFlagPool->pop() : nullptr;
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
            KernelConfig::KernelType::BROADCAST_UNPACK,
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

  cudaStream_t launchStream = kernelConfig.stream;
  std::optional<OrderedWorkStreamGuard::Scope> wsScope;

  // Acquire the work-stream baton before adding the host node so that
  // during graph replay the host node (which enqueues a GPE command) only
  // fires after the previous operation's kernel has completed. Without this
  // ordering, a subsequent eager submit could enqueue its GPE command before
  // the graph's host node fires, causing the single-threaded GPE to deadlock
  // (spinning on the eager kernel's KERNEL_STARTED while the graph's command
  // is stuck behind it in the queue).
  auto maybeAcquireWorkStreamScope = [&]() {
    if (!kernelConfig.canConcurrent) {
      wsScope = ws_.acquire(kernelConfig.stream, streamCaptureInfo);
      FB_COMMCHECK(wsScope->status());
      launchStream = wsScope->stream();
    }
    return commSuccess;
  };

  size_t opGroupSize = 0;
  // Enqueue op to gpeThread if any op is appended, or if there is a
  // postKernelCleanup that needs to run after the kernel completes.
  if (needsKernelFlag) {
    // record opGroup size before moving the object
    opGroupSize = opGroup.size();
    class CtranGpeCmd* cmd = new class CtranGpeCmd;
    cmd->type = type;
    cmd->kernelFlag = kernelFlag;
    cmd->timeout = timeout;
    cmd->unpackPool = kernelConfig.unpackPool;
    cmd->postKernelCleanup = std::move(kernelConfig.postKernelCleanup);

    if (type == CtranGpeCmd::TypeEnum::GRAPH_ENQUEUE) {
      cmd->coll.opGroup = std::move(opGroup);
      cmd->coll.func = func;
      if (colltraceHandle != nullptr) {
        cmd->coll.collHandle = colltraceHandle;
      }
      cmd->coll.comm = comm;
    }

    maybeAcquireWorkStreamScope();

    if (isCapturing) {
      FB_COMMCHECK(preLaunchGraphPrepare(cmd, graphPrepareFn));
      cmd->persistent = true;
      // Mark the flag as persistent so reclaim() won't steal it between
      // graph replays (the kernel writes KERNEL_UNSET after each replay).
      if (kernelFlag) {
        kernelFlag->setPersistent();
      }
      cmd->gpe = this->gpe;

      FB_COMMCHECKGOTO(
          utils::cudagraph::addHostNode(
              /*data=*/cmd,
              /*execCallback=*/cmdCb,
              /*destroyCallback=*/cmdDestroy,
              kernelConfig.stream,
              streamCaptureInfo),
          res,
          fail);
    } else {
      cmdEnqueue(cmd);
    }
  } else {
    maybeAcquireWorkStreamScope();
  }

  // For the no-cmd path during graph capture, retain cleanup on the graph.
  if (isCapturing && !needsKernelFlag) {
    if (kernelConfig.postKernelCleanup) {
      FB_COMMCHECKGOTO(
          utils::cudagraph::retainUserObject(
              /*obj=*/
              new std::function<void()>(
                  std::move(kernelConfig.postKernelCleanup)),
              /*destroyCallback=*/
              [](void* p) {
                auto* fn = static_cast<std::function<void()>*>(p);
                (*fn)();
                delete fn;
              },
              streamCaptureInfo),
          res,
          fail);
    }

    // Mark KernelElems as persistent and release on graph destruction.
    // In the cmd path, ~OpElem handles free() (which also clears persistent).
    if (!kernelConfig.persistentKernelElems.empty()) {
      for (auto* elem : kernelConfig.persistentKernelElems) {
        elem->setPersistent();
      }
      auto* elems = new std::vector<KernelElem*>(
          std::move(kernelConfig.persistentKernelElems));
      FB_COMMCHECKGOTO(
          utils::cudagraph::retainUserObject(
              /*obj=*/elems,
              /*destroyCallback=*/
              [](void* p) {
                auto* v = static_cast<std::vector<KernelElem*>*>(p);
                for (auto* elem : *v) {
                  elem->clearPersistent();
                  elem->free();
                }
                delete v;
              },
              streamCaptureInfo),
          res,
          fail);
    }
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
    launchConfig.hStream = launchStream;
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
    size_t sharedMemBytes = kernelConfig.dynamicSharedMemBytes > 0
        ? kernelConfig.dynamicSharedMemBytes
        : sizeof(CtranAlgoDeviceState);
    FB_CUDACHECKGOTO(
        cudaFuncSetAttribute(
            ncclKernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            sharedMemBytes),
        res,
        fail);
    FB_CUDACHECKGOTO(
        cudaLaunchKernel(
            ncclKernel,
            grid,
            blocks,
            kernelArgs.data(),
            sharedMemBytes,
            launchStream),
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
          launchStream);
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

  // early release
  wsScope.reset();

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
    std::shared_ptr<std::atomic_flag> cpuFlag) {
  // postKernelCleanup is not supported for host submits (no kernel launched).
  DCHECK(!kernelConfig.postKernelCleanup);

  // Enqueue op to gpeThread if any op is appended
  if (!opGroup.empty()) {
    class CtranGpeCmd* cmd = new class CtranGpeCmd;
    cmd->type = type;
    cmd->kernelFlag = nullptr;
    cmd->cpuFlag = std::move(cpuFlag);
    cmd->unpackPool = kernelConfig.unpackPool;

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
  ws_.init(comm->logMetaData_);
  thread_ = std::thread([this] { gpeThreadFn(); });
}

void CtranGpe::Impl::terminate() {
  class CtranGpeCmd* cmd = new class CtranGpeCmd;
  cmd->type = CtranGpeCmd::TypeEnum::TERMINATE;

  cmdEnqueue(cmd);
  thread_.join();

  // Pool elements are released by CUDA's async cmdDestroy callback
  // (cudaUserObjectNoDestructorSync). Spin until all pools drain before
  // returning, to avoid freeing pinned memory from under an in-flight callback.
  const auto& statex = comm->statex_;
  const auto start = std::chrono::steady_clock::now();
  auto nextLog = start + std::chrono::seconds(5);
  while (true) {
    this->kernelFlagPool->reclaim();
    this->kernelElemPool->reclaim();
    this->gpeKernelSyncPool->reclaim();
    if (this->kernelFlagPool->capacity() == this->kernelFlagPool->size() &&
        this->kernelElemPool->capacity() == this->kernelElemPool->size() &&
        this->gpeKernelSyncPool->capacity() ==
            this->gpeKernelSyncPool->size()) {
      break;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now >= nextLog) {
      const auto elapsedSec =
          std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
      CLOGF_SUBSYS(
          WARNING,
          INIT,
          "terminate() spin-wait: pools still draining after {}s on rank {} commHash {:x}"
          " -- kernelFlag {}/{} kernelElem {}/{} gpeKernelSync {}/{}."
          " Most likely cudaGraphDestroy() was not called on all CUDA graphs"
          " that captured CTranGPE operations.",
          elapsedSec,
          statex->rank(),
          statex->commHash(),
          this->kernelFlagPool->size(),
          this->kernelFlagPool->capacity(),
          this->kernelElemPool->size(),
          this->kernelElemPool->capacity(),
          this->gpeKernelSyncPool->size(),
          this->gpeKernelSyncPool->capacity());
      nextLog = now + std::chrono::seconds(5);
    }
    std::this_thread::yield();
  }

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
    FB_CUDACHECKTHROW_EX(cudaSetDevice(cudaDev), comm->logMetaData_);

    while (1) {
      auto cmd = cmdDequeue();

      if (cmd->timeout.has_value()) {
        comm->setTimeout(cmd->timeout.value());
      } else if (auto d = comm->getAbort()->GetDefaultTimeoutDuration();
                 d.has_value()) {
        // Fall back to comm-level default (CUDA-graph replay path).
        comm->setTimeout(*d);
      }
      SCOPE_EXIT {
        // if comm is aborted for any reason, we mark it as aborted to avoid
        // resetting the state.
        if (comm->testAbort()) {
          auto abort = comm->getAbort();
          if (abort->TimedOut()) {
            CLOGF(
                ERR,
                "Communicator aborted due to timeout on rank {} commHash {:x}",
                statex->rank(),
                statex->commHash());
          } else {
            CLOGF(
                ERR,
                "Communicator aborted (explicit) on rank {} commHash {:x}",
                statex->rank(),
                statex->commHash());
          }
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
               flag_d[0] != KERNEL_STARTED_AND_EXIT) {
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

        // Ensure the host communication request completes, irrespective of
        // outcome of collective function - success, failure, or an exception
        SCOPE_EXIT {
          if (cmd->cpuFlag) {
            cmd->cpuFlag->test_and_set();
          }
        };

        /* run collective */
        if (comm->testAbort()) {
          // Comm already aborted — skip collective to prevent
          // progressInternal() from accessing stale VC queue entries
          // left by a previously aborted collective (double-complete bug).
          CLOGF(
              WARN,
              "Communicator aborted, skipping collective (opType={}, opCount={}) on rank {} commHash {:x}",
              cmd->coll.opGroup.empty()
                  ? -1
                  : static_cast<int>(cmd->coll.opGroup.front()->type),
              cmd->coll.opGroup.empty() ? 0UL
                                        : cmd->coll.opGroup.front()->opCount,
              statex->rank(),
              statex->commHash());
          // Ensure async error is set so callers see a non-success result
          // via getResult(). The abort flag may have been set externally
          // (e.g. comm->abort()) without setting the async exception.
          if (comm->getAsyncError()->getAsyncResult() == commSuccess) {
            comm->getAsyncError()->setAsyncException(
                ctran::utils::Exception(
                    "collective skipped: communicator aborted",
                    commRemoteError));
          }
        } else if (!cmd->coll.opGroup.empty() /* skip when opGroup is empty, i.e,. we are only here for post-kernel cmd destruction/cleanup */) {
          CTRAN_ASYNC_ERR_GUARD_FAULT_TOLERANCE(comm, {
            FB_COMMCHECKTHROW_EX(
                cmd->coll.func(cmd->coll.opGroup), comm->logMetaData_);
          });
        }

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
          if (!cmd->persistent) {
            kernelFlag->reset();
          }
        } else {
          // In case of aborted comm, wait for kernel to start
          while (comm->testAbort() &&
                 !kernelFlag->testFlagAllGroups(KERNEL_STARTED)) {
            std::this_thread::yield();
          }
          // Stop kernel and kernel will free up the flag after confirmed the
          // termination
          kernelFlag->setFlagPerGroup(
              comm->testAbort() ? KERNEL_HOST_ABORT : KERNEL_TERMINATE);
        }
        // Teardown unpack queue if it was allocated for this operation (TcpDM
        // backend). Don't wait for kernel to finish, the pool manages
        // the allocations in the round robin fashion to avoid immediate
        // reuse.
        if (cmd->unpackPool != nullptr) {
          FB_COMMCHECKTHROW_EX(
              comm->ctran_->mapper->teardownUnpackConsumer(cmd->unpackPool),
              comm->logMetaData_);
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

      if (cmd->persistent) {
        cmd->inFlight.fetch_sub(1, std::memory_order_release);
      } else {
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
  // Clear persistence so reclaim() can pick up this elem after free.
  persistent_ = false;
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
  if (persistent_) {
    return false;
  }
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

void KernelElem::wait(std::shared_ptr<ctran::utils::Abort> abort, int groupId) {
  // wait for all thread blocks to complete
  while (!this->isComplete(groupId) && !abort->Test()) {
    // friendly spin so we don't hog CPU
    std::this_thread::yield();
  }
}

KernelElemPool::KernelElemPool(size_t capacity) : capacity_(capacity) {
  FB_CUDACHECKTHROW_EX_NOCOMM(cudaHostAlloc(
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
  for (size_t i = 0; i < count; i++) {
    auto* g = gpeKernelSyncPool->pop();
    // essentially the constructor (note resetStatus() is needed since we didn't
    // set nworkers before this point)
    g->nworkers = nworkers;
    g->resetStatus();
    gpeKernelSyncs.emplace_back(g);
  }
  return commSuccess;
}
