// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <chrono>
#include <deque>
#include <optional>

#if !defined(ENABLE_PRIMS)
#include <mutex> // std::once_flag/std::call_once for the prims-off fallback warn
#endif

#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/SendRecv/SendRecvImpl.h"
#include "comms/ctran/gpe/CtranGpe.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/ctran/utils/CudaGraphUtils.h"
#include "comms/ctran/utils/ExtUtils.h"

namespace {
thread_local std::deque<OpElem*> CtranOpGroup;
thread_local std::optional<enum NCCL_SENDRECV_ALGO> CtranOpGroupAlgo;

commResult_t checkAndCacheAlgo(enum NCCL_SENDRECV_ALGO algo) {
  if (CtranOpGroupAlgo.has_value()) {
    if (*CtranOpGroupAlgo != algo) {
      FB_ERRORRETURN(
          commInvalidUsage,
          "Mixed sendrecv algos in CtranOpGroup: {} vs {}",
          static_cast<int>(*CtranOpGroupAlgo),
          static_cast<int>(algo));
    }
  } else {
    CtranOpGroupAlgo = algo;
  }
  return commSuccess;
}
} // namespace

std::unordered_map<KernelConfig::KernelType, void*> kernelFns = {
    {KernelConfig::KernelType::SEND, reinterpret_cast<void*>(ncclKernelSend)},
    {KernelConfig::KernelType::RECV,
     reinterpret_cast<void*>(ncclKernelRecv</*UNPACK=*/false>)},
    {KernelConfig::KernelType::SENDRECV,
     reinterpret_cast<void*>(ncclKernelSendRecv</*UNPACK=*/false>)},
    {KernelConfig::KernelType::RECV_UNPACK,
     reinterpret_cast<void*>(ncclKernelRecv</*UNPACK=*/true>)},
    {KernelConfig::KernelType::SENDRECV_UNPACK,
     reinterpret_cast<void*>(ncclKernelSendRecv</*UNPACK=*/true>)},
#if defined(ENABLE_PRIMS)
    {KernelConfig::KernelType::SENDRECV_P2P,
     reinterpret_cast<void*>(ncclKernelSendRecvP2p)},
#endif // defined(ENABLE_PRIMS)
};

static const auto myAlgo = NCCL_SENDRECV_ALGO::ctran;

bool ctranSendRecvSupport(
    int peer,
    CtranComm* comm,
    enum NCCL_SENDRECV_ALGO algo,
    cudaStream_t stream) {
  if (!ctranInitialized(comm)) {
    return false;
  }

#if !defined(ENABLE_PRIMS)
  // ctp2p/ctgraph are prims-backed and not compiled in; decline so the caller
  // falls back to baseline send/recv.
  if (algo == NCCL_SENDRECV_ALGO::ctp2p ||
      algo == NCCL_SENDRECV_ALGO::ctgraph) {
    static std::once_flag warnedOnce;
    std::call_once(warnedOnce, []() {
      CLOGF_SUBSYS(
          WARN,
          COLL,
          "CTRAN SendRecv: ctp2p/ctgraph requires comms::prims device transports "
          "(ENABLE_PRIMS), which are not compiled in this build; falling back to "
          "baseline send/recv.");
    });
    return false;
  }
#endif // !defined(ENABLE_PRIMS)

  const auto statex = comm->statex_.get();

  if (algo == NCCL_SENDRECV_ALGO::ctgraph) {
    // TODO: skip NVL peers for now; will add support based on window
    if (peer != statex->rank() &&
        comm->ctran_->mapper->getBackend(peer) != CtranMapperBackend::IB) {
      return false;
    }
    if (stream == nullptr) {
      return false;
    }
    ctran::utils::cudagraph::StreamCaptureInfo captureInfo;
    auto err =
        ctran::utils::cudagraph::getStreamCaptureInfo(stream, captureInfo);
    if (err != cudaSuccess ||
        captureInfo.status != cudaStreamCaptureStatusActive) {
      CLOGF_SUBSYS(
          INFO,
          COLL,
          "SendRecv ctgraph: not in capture mode. "
          "Falling back to baseline. "
          "commHash {:x} commDesc {} peer {} nRanks {} nLocalRanks {} nNodes {}",
          statex->commHash(),
          statex->commDesc(),
          peer,
          statex->nRanks(),
          statex->nLocalRanks(),
          statex->nNodes());
      return false;
    }
    return true;
  }

  // Self peer is handled by CE directly, other peers require a valid ctran
  // backend
  if (peer == statex->rank() ||
      comm->ctran_->mapper->getBackend(peer) != CtranMapperBackend::UNSET) {
    return true;
  }
  return false;
}

commResult_t ctranSend(
    const void* sendbuff,
    size_t count,
    commDataType_t datatype,
    int peer,
    CtranComm* comm,
    cudaStream_t stream,
    enum NCCL_SENDRECV_ALGO algo) {
  auto result = checkAndCacheAlgo(algo);
  if (result != commSuccess) {
    return result;
  }
  auto opCount = comm->ctran_->getOpCount();
  CTRAN_COLL_INFO(
      "CtranSend", sendbuff, nullptr, count, datatype, peer, comm, stream);

  auto op = new OpElem(OpElem::opType::SEND, stream, comm, opCount);
  op->send.sendbuff = sendbuff;
  op->send.count = count;
  op->send.datatype = datatype;
  op->send.peerRank = peer;
  CtranOpGroup.push_back(op);

  return commSuccess;
}

commResult_t ctranRecv(
    void* recvbuff,
    size_t count,
    commDataType_t datatype,
    int peer,
    CtranComm* comm,
    cudaStream_t stream,
    enum NCCL_SENDRECV_ALGO algo) {
  auto result = checkAndCacheAlgo(algo);
  if (result != commSuccess) {
    return result;
  }
  auto opCount = comm->ctran_->getOpCount();
  CTRAN_COLL_INFO(
      "CtranRecv", nullptr, recvbuff, count, datatype, peer, comm, stream);

  auto op = new OpElem(OpElem::opType::RECV, stream, comm, opCount);
  op->recv.recvbuff = recvbuff;
  op->recv.count = count;
  op->recv.datatype = datatype;
  op->recv.peerRank = peer;

  CtranOpGroup.push_back(op);

  return commSuccess;
}

commResult_t ctranGroupEndHookImpl(
    std::deque<OpElem*>& opGroup,
    enum NCCL_SENDRECV_ALGO algo,
    std::optional<std::chrono::milliseconds> timeout) {
  // By default, use zero-copy kernel for sendrecv.
  if (algo == NCCL_SENDRECV_ALGO::ctran) {
    algo = NCCL_SENDRECV_ALGO::ctzcopy;
  }
  while (!opGroup.empty()) {
    // TODO: clean up duplicate info in allops, nvlOps and ibOps
    std::vector<OpElem*> allOps;
    std::vector<OpElem*> selfSends, selfRecvs, nvlOps, ibOps;
    std::deque<OpElem*> pending;
    bool hasSend = false;
    bool hasRecv = false;
    bool hasTcpDmRecv = false;

    // Submit ops with the same comm and stream in a single batch
    CtranComm* comm = opGroup.front()->comm_;
    cudaStream_t stream = opGroup.front()->stream;
    const auto statex = comm->statex_.get();
    auto mapper = comm->ctran_->mapper.get();

    while (!opGroup.empty()) {
      auto op = dequeFront(opGroup);

      if (op->comm_ == comm && op->stream == stream) {
        if (op->type == OpElem::opType::SEND) {
          hasSend = true;
          if (op->send.peerRank == statex->rank()) {
            selfSends.push_back(op);
            continue;
          }

          // Async buffer registration for send and recv buffers to hide
          // registration cost.
          // - If the buffer has already been registered, regAsync will return
          //   immediately.
          // - If the buffer is not yet registered at regAsync internal query, a
          //   request will be enqueued to asyncReg thread.
          // - A first-used buffer will be registered either by asyncReg thread
          //   or GPE thread (see RegCache::regRange).
          // - regAsync is a no-op if NCCL_CTRAN_REGISTER is not async mode.
          //
          // Expected performance improvement for communication involving
          // first-time registration:
          // - [Improved case] If the buffer is registered by asyncReg thread
          //   ahead of time, it hides registration cost.
          // - If the buffer is registered by GPE thread, e.g., due to too busy
          //   asyncReg thread or not-advanced CPU schedule, the registration
          //   cost has to be exposed similar to lazy registatration mode.
          size_t nbytes = op->send.count * commTypeSize(op->send.datatype);
          FB_COMMCHECK(mapper->regAsync(op->send.sendbuff, nbytes));
          if (comm->ctran_->mapper->getBackend(op->send.peerRank) ==
              CtranMapperBackend::NVL) {
            nvlOps.push_back(op);
          } else {
            ibOps.push_back(op);
          }
        } else if (op->type == OpElem::opType::RECV) {
          hasRecv = true;
          if (op->recv.peerRank == statex->rank()) {
            selfRecvs.push_back(op);
            continue;
          }

          // For TCP Device Memory, if we have peers we are going to receive
          // from, we need to unpack the data from the bounce buffer.
          if (comm->ctran_->mapper->getBackend(op->recv.peerRank) ==
              CtranMapperBackend::TCPDM) {
            hasTcpDmRecv = true;
          }

          size_t nbytes = op->recv.count * commTypeSize(op->recv.datatype);
          FB_COMMCHECK(mapper->regAsync(op->recv.recvbuff, nbytes));
          if (comm->ctran_->mapper->getBackend(op->recv.peerRank) ==
              CtranMapperBackend::NVL) {
            nvlOps.push_back(op);
          } else {
            ibOps.push_back(op);
          }
        }

        allOps.push_back(op);
      } else {
        // If not belong to this batch, put to pending and handle in next batch
        pending.push_back(op);
      }
    }

    // Handle self sends and recvs via CE-based icopy
    FB_COMMCHECK(selfSendRecvImpl(selfSends, selfRecvs, comm));

    // For non-self sends and recvs: decide the kernel function and submit to
    // GPE
    if (!allOps.empty()) {
      // host side
      std::vector<std::unique_ptr<struct OpElem>> gpeOpGroup;
      FB_COMMCHECK(
          ctran::sendrecv::setupGpeOp(
              comm, allOps, nvlOps, ibOps, gpeOpGroup, algo));

      // device side
      KernelConfig::KernelType kernelType =
          ctran::sendrecv::getKernelType(hasSend, hasRecv, hasTcpDmRecv, algo);
      auto config = KernelConfig(
          kernelType,
          stream,
          sendRecvAlgoName(myAlgo, allOps),
          allOps.front()->opCount);

      ctran::sendrecv::KernArgs kernArgs;
      FB_COMMCHECK(
          ctran::sendrecv::setupKernelConfig(
              comm, allOps, nvlOps, config, kernArgs));

      FB_COMMCHECK(comm->ctran_->gpe->submit(
          std::move(gpeOpGroup),
          sendRecvImpl,
          config,
          kernelFns.at(kernelType),
          timeout));
    }

    // No kernel would be submitted if only self sendrecv is called, update op
    // count here
    if (allOps.empty() && !selfSends.empty()) {
      comm->ctran_->updateOpCount();
    }
    allOps.clear();

    comm->ctran_->numGroupedDefaultOps = 0;

    // handle next batch
    opGroup = std::move(pending);
  }

  return commSuccess;
}

commResult_t ctranGroupEndHook(
    std::optional<std::chrono::milliseconds> timeout) {
  // Default to ctran when no ops were enqueued (dead path due to empty-group
  // early return below, but kept for defensive safety).
  auto algo = CtranOpGroupAlgo.value_or(NCCL_SENDRECV_ALGO::ctran);
  CtranOpGroupAlgo.reset();
  if (CtranOpGroup.empty()) {
    return commSuccess;
  }
  if (algo == NCCL_SENDRECV_ALGO::ctgraph) {
    cudaStream_t stream = CtranOpGroup.front()->stream;
    ctran::utils::cudagraph::StreamCaptureInfo captureInfo;
    FB_CUDACHECK(
        ctran::utils::cudagraph::getStreamCaptureInfo(stream, captureInfo));
    if (captureInfo.status == cudaStreamCaptureStatusActive) {
      return ctranSendRecvCudagraphAware(
          CtranOpGroup, CtranOpGroup.front()->comm_, stream, timeout);
    }
    FB_ERRORRETURN(
        commInvalidUsage,
        "SendRecv ctgraph called outside CUDA graph capture. "
        "ctranSendRecvSupport should have returned false.");
  } else {
    return ctranGroupEndHookImpl(CtranOpGroup, algo, timeout);
  }
}

void ctranGroupTrackDefaultOp(CtranComm* comm) {
  if (ctranInitialized(comm)) {
    comm->ctran_->numGroupedDefaultOps++;
  }
}
