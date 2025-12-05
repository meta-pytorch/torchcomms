// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/ctran/algos/SendRecv/SendRecvStagedCopyImpl.h"
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

commResult_t setupStagedCopyKernelConfig(
    CtranComm* comm,
    const std::vector<OpElem*>& nvlOps,
    KernelConfig& config,
    ctran::sendrecv::KernArgs& kernArgs) {
  const auto statex = comm->statex_.get();

  kernArgs.numSends = 0;
  kernArgs.numRecvs = 0;
  kernArgs.numSendBlocks = 0;
  kernArgs.numRecvBlocks = 0;

  size_t sendIdx = 0;
  size_t recvIdx = 0;
  for (auto op : nvlOps) {
    if (op->type == OpElem::opType::SEND && op->send.count > 0) {
      // TODO: currently only support max CTRAN_MAX_NVL_PEERS sends/recvs in a
      // group
      if (sendIdx >= CTRAN_MAX_NVL_PEERS) {
        CLOGF(
            ERR,
            "Too many send ops for SENDRECV_STAGED: {} > {}",
            sendIdx + 1,
            CTRAN_MAX_NVL_PEERS);
        return commInvalidUsage;
      }
      kernArgs.sends[sendIdx].buff = const_cast<void*>(op->send.sendbuff);
      kernArgs.sends[sendIdx].nbytes =
          op->send.count * commTypeSize(op->send.datatype);
      kernArgs.sends[sendIdx].peerLocalRank =
          statex->localRank(op->send.peerRank);
      size_t nGroups = getNumGroups(kernArgs.sends[sendIdx].nbytes);
      kernArgs.sends[sendIdx].nGroups = nGroups;
      kernArgs.numSendBlocks = std::max(kernArgs.numSendBlocks, nGroups);
      sendIdx++;
    } else if (op->type == OpElem::opType::RECV && op->recv.count > 0) {
      if (recvIdx >= CTRAN_MAX_NVL_PEERS) {
        CLOGF(
            ERR,
            "Too many recv ops for SENDRECV_STAGED: {} > {}",
            recvIdx + 1,
            CTRAN_MAX_NVL_PEERS);
        return commInvalidUsage;
      }
      kernArgs.recvs[recvIdx].buff = op->recv.recvbuff;
      kernArgs.recvs[recvIdx].nbytes =
          op->recv.count * commTypeSize(op->recv.datatype);
      kernArgs.recvs[recvIdx].peerLocalRank =
          statex->localRank(op->recv.peerRank);
      size_t nGroups = getNumGroups(kernArgs.recvs[recvIdx].nbytes);
      kernArgs.recvs[recvIdx].nGroups = nGroups;
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
  config.numThreads = NCCL_CTRAN_NVL_SENDRECV_STAGED_COPY_THREAD_BLOCK_SIZE;
  config.algoArgs = &kernArgs;
  return commSuccess;
}

} // namespace ctran::sendrecv
