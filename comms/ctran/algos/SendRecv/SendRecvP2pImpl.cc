// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/SendRecv/SendRecvP2pImpl.h"
#include "comms/ctran/utils/Alloc.h"
#include "comms/ctran/utils/PinnedHostPool.h"
#include "comms/utils/cvars/nccl_cvars.h"

namespace {

inline size_t getNumGroups(size_t nbytes) {
  // TODO: copied the same config from NCCL baseline, need further tune for
  // CTRAN
  size_t nGroups = 1;
  if (nbytes <= 131072) {
    nGroups = std::max(nGroups, nbytes / 16384);
  } else if (nbytes <= 268435456) {
    nGroups = 8;
  } else {
    nGroups = 16;
  }
  return nGroups;
}

using SendRecvOpHostPool = PinnedHostPool<ctran::sendrecv::SendRecvOpHostBuf>;

SendRecvOpHostPool& getSendRecvOpHostPool() {
  static SendRecvOpHostPool pool(/*startCapacity=*/1);
  return pool;
}

} // namespace

namespace ctran::sendrecv {

commResult_t setupP2pKernelConfig(
    CtranComm* comm,
    const std::vector<OpElem*>& nvlOps,
    KernelConfig& config,
    ctran::sendrecv::KernArgs& kernArgs) {
  const auto statex = comm->statex_.get();

  kernArgs.numSends = 0;
  kernArgs.numRecvs = 0;
  kernArgs.numSendBlocks = 0;
  kernArgs.numRecvBlocks = 0;
  kernArgs.sendsList = nullptr;
  kernArgs.recvsList = nullptr;
  kernArgs.useList = false;

  // Count send/recv ops first to determine if we need list format
  size_t numSendOps = 0;
  size_t numRecvOps = 0;
  for (auto op : nvlOps) {
    if (op->type == OpElem::opType::SEND && op->send.count > 0) {
      numSendOps++;
    } else if (op->type == OpElem::opType::RECV && op->recv.count > 0) {
      numRecvOps++;
    }
  }

  // Fallback to list format if exceeding kCtranMaxNvlSendRecvOps.
  // This is slower but handles arbitrary number of ops.
  bool needUseList = numSendOps > kCtranMaxNvlSendRecvOps ||
      numRecvOps > kCtranMaxNvlSendRecvOps;

  // Skip the pool during graph capture: it's a one-time cost
  // anyway, and there is no limit on the number of ops.
  bool isCapturing = false;
  if (needUseList && config.stream) {
    cudaStreamCaptureStatus status;
    FB_CUDACHECK(cudaStreamIsCapturing(config.stream, &status));
    isCapturing = status == cudaStreamCaptureStatusActive;
  }

  if (kernArgs.useList = needUseList; needUseList) {
    // use pool allocation if...
    // 1. pool items are large enough to hold numSendOps and numRecvOps
    // 2. we aren't capturing
    if (!isCapturing && numSendOps + numRecvOps <= kMaxSendRecvOpsPerPoolBuf) {
      auto* poolBuf = getSendRecvOpHostPool().pop();
      if (numSendOps > 0) {
        kernArgs.sendsList = poolBuf->ops;
      }
      if (numRecvOps > 0) {
        kernArgs.recvsList = poolBuf->ops + numSendOps;
      }
      config.postKernelCleanup = [poolBuf]() { poolBuf->inUse_ = false; };
    } else {
      if (numSendOps > 0) {
        FB_COMMCHECK(
            ctran::utils::commCudaHostAlloc(
                &kernArgs.sendsList,
                numSendOps,
                cudaHostAllocDefault,
                &comm->logMetaData_,
                "setupP2pKernelConfig"));
      }
      if (numRecvOps > 0) {
        FB_COMMCHECK(
            ctran::utils::commCudaHostAlloc(
                &kernArgs.recvsList,
                numRecvOps,
                cudaHostAllocDefault,
                &comm->logMetaData_,
                "setupP2pKernelConfig"));
      }
      config.postKernelCleanup = [sendsList = kernArgs.sendsList,
                                  recvsList = kernArgs.recvsList]() {
        if (sendsList) {
          ctran::utils::commCudaFreeHost(sendsList);
        }
        if (recvsList) {
          ctran::utils::commCudaFreeHost(recvsList);
        }
      };
    }
  }

  // Set base pointer to pre-allocated transport array
  // Kernel will use peerLocalRank to index into this array
  kernArgs.nvlTransportsBase = comm->ctran_->algo->getNvlTransportsBase();

  size_t sendIdx = 0;
  size_t recvIdx = 0;
  for (auto op : nvlOps) {
    if (op->type == OpElem::opType::SEND && op->send.count > 0) {
      ctran::sendrecv::SendRecvOp sendOp;
      sendOp.buff = const_cast<void*>(op->send.sendbuff);
      sendOp.nbytes = op->send.count * commTypeSize(op->send.datatype);
      sendOp.peerLocalRank = statex->localRank(op->send.peerRank);
      size_t nGroups = getNumGroups(sendOp.nbytes);
      sendOp.nGroups = nGroups;
      (kernArgs.useList ? kernArgs.sendsList : kernArgs.sends)[sendIdx] =
          sendOp;
      kernArgs.numSendBlocks = std::max(kernArgs.numSendBlocks, nGroups);
      sendIdx++;
    } else if (op->type == OpElem::opType::RECV && op->recv.count > 0) {
      ctran::sendrecv::SendRecvOp recvOp;
      recvOp.buff = op->recv.recvbuff;
      recvOp.nbytes = op->recv.count * commTypeSize(op->recv.datatype);
      recvOp.peerLocalRank = statex->localRank(op->recv.peerRank);
      size_t nGroups = getNumGroups(recvOp.nbytes);
      recvOp.nGroups = nGroups;
      (kernArgs.useList ? kernArgs.recvsList : kernArgs.recvs)[recvIdx] =
          recvOp;
      kernArgs.numRecvBlocks = std::max(kernArgs.numRecvBlocks, nGroups);
      recvIdx++;
    }
  }

  kernArgs.numSends = sendIdx;
  kernArgs.numRecvs = recvIdx;

  // If no kernel ops, still need to launch kernel for GPE to start: so at least
  // 1 block needed.
  config.numBlocks =
      std::max((size_t)1, kernArgs.numSendBlocks + kernArgs.numRecvBlocks);
  // TODO: tunning needed
  kernArgs.useBlockGroup = true;
  config.numThreads = NCCL_CTRAN_NVL_SENDRECV_STAGED_COPY_THREAD_BLOCK_SIZE;
  config.algoArgs = &kernArgs;
  return commSuccess;
}

} // namespace ctran::sendrecv
