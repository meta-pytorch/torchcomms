// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/ctran/algos/SendRecv/SendRecvP2pImpl.h"
#include "comms/ctran/algos/CtranAlgoDev.h"
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
  if (numSendOps > kCtranMaxNvlSendRecvOps ||
      numRecvOps > kCtranMaxNvlSendRecvOps) {
    // TODO: the `useList` path has ms-level overhead, need to optimize by using
    // pre-allocated host-pinned memory pool
    kernArgs.useList = true;
    if (numSendOps > 0) {
      FB_CUDACHECK(cudaMalloc(
          &kernArgs.sendsList,
          numSendOps * sizeof(ctran::sendrecv::SendRecvOp)));
    }
    if (numRecvOps > 0) {
      FB_CUDACHECK(cudaMalloc(
          &kernArgs.recvsList,
          numRecvOps * sizeof(ctran::sendrecv::SendRecvOp)));
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

      if (kernArgs.useList) {
        FB_CUDACHECK(cudaMemcpy(
            &kernArgs.sendsList[sendIdx],
            &sendOp,
            sizeof(ctran::sendrecv::SendRecvOp),
            cudaMemcpyHostToDevice));
      } else {
        kernArgs.sends[sendIdx] = sendOp;
      }
      kernArgs.numSendBlocks = std::max(kernArgs.numSendBlocks, nGroups);
      sendIdx++;
    } else if (op->type == OpElem::opType::RECV && op->recv.count > 0) {
      ctran::sendrecv::SendRecvOp recvOp;
      recvOp.buff = op->recv.recvbuff;
      recvOp.nbytes = op->recv.count * commTypeSize(op->recv.datatype);
      recvOp.peerLocalRank = statex->localRank(op->recv.peerRank);
      size_t nGroups = getNumGroups(recvOp.nbytes);
      recvOp.nGroups = nGroups;

      if (kernArgs.useList) {
        FB_CUDACHECK(cudaMemcpy(
            &kernArgs.recvsList[recvIdx],
            &recvOp,
            sizeof(ctran::sendrecv::SendRecvOp),
            cudaMemcpyHostToDevice));
      } else {
        kernArgs.recvs[recvIdx] = recvOp;
      }
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
