// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/ReduceScatter/ReduceScatterImpl.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/utils/cvars/nccl_cvars.h"

bool ctranReduceScatterSupport(
    CtranComm* comm,
    enum NCCL_REDUCESCATTER_ALGO algo) {
  const int nRanks = comm->statex_->nRanks();
  const int rank = comm->statex_->rank();
  const int nNodes = comm->statex_->nNodes();
  const int nLocalRanks = comm->statex_->nLocalRanks();

  // Check if ctran is initialized
  if (!ctranInitialized(comm)) {
    CLOGF_EVERY_MS(
        WARN,
        60000,
        "ctranReduceScatterSupport: CTRAN not initialized, falling back to baseline");
    return false;
  }

  // Check backend availability (except for single rank case)
  if (nRanks > 1 && !comm->ctran_->mapper->hasBackend()) {
    CLOGF_EVERY_MS(
        WARN,
        60000,
        "ctranReduceScatterSupport: no backend available, falling back to baseline");
    for (int peer = 0; peer < nRanks; peer++) {
      if (peer != rank &&
          comm->ctran_->mapper->getBackend(peer) == CtranMapperBackend::UNSET) {
        CLOGF(
            WARN,
            "ctranReduceScatterSupport: rank {} peer {} has unset backend",
            rank,
            peer);
      }
    }
    return false;
  }

  // Check algorithm-specific requirements
  switch (algo) {
    case NCCL_REDUCESCATTER_ALGO::ctdirect:
      if (nNodes > 1) {
        CLOGF_EVERY_MS(
            WARN,
            60000,
            "ctranReduceScatterSupport: ctdirect only supports nNodes=1. Falling back to baseline.");
        return false;
      }
      break;
    case NCCL_REDUCESCATTER_ALGO::ctring:
      if (nLocalRanks > 1) {
        CLOGF_EVERY_MS(
            WARN,
            60000,
            "ctranReduceScatterSupport: ctring only supports nLocalRanks=1. Falling back to baseline.");
        return false;
      }
      break;
    case NCCL_REDUCESCATTER_ALGO::ctrhd:
      // ctrhd can run when nLocalRanks=1, nNodes is a power of 2, and data
      // buffer size is smaller than NCCL_CTRAN_INTERNODE_TMPBUF_SIZE.
      // Currently, we return false here since we can't check the data buffer
      // size.
      CLOGF_EVERY_MS(
          WARN,
          60000,
          "ctranReduceScatterSupport returns false for all cases of algo=ctrhd. Falling back to baseline.");
      return false;
    case NCCL_REDUCESCATTER_ALGO::ctran:
      // currently no ctran algo that supports nNodes > 1 and nLocalRanks > 1 so
      // we return false here. If a new ctran algo is added that supports this
      // case, we can remove this check.
      if (nNodes > 1 && nLocalRanks > 1) {
        CLOGF_EVERY_MS(
            WARN,
            60000,
            "ctranReduceScatterSupport: ctran only supports nLocalRanks=1 or nNodes=1. Falling back to baseline.");
        return false;
      }
      break;
    case NCCL_REDUCESCATTER_ALGO::orig: // invalid query
      return false;
  }

  return true;
}

commResult_t ctranReduceScatter(
    const void* sendbuff,
    void* recvbuff,
    size_t recvcount,
    commDataType_t datatype,
    commRedOp_t redOp,
    CtranComm* comm,
    cudaStream_t stream,
    enum NCCL_REDUCESCATTER_ALGO algo) {
  if (comm->statex_->nRanks() == 1) {
    return reduceScatterSingleRankImpl(
        sendbuff, recvbuff, recvcount, datatype, redOp, comm, stream);
  }

  const int nNodes = comm->statex_->nNodes();
  const int nLocalRanks = comm->statex_->nLocalRanks();
  // Only ctdirect supports nNodes == 1 case.
  // nLocalRanks>1 case is currently unsupported.
  if (algo == NCCL_REDUCESCATTER_ALGO::ctran) {
    if (nNodes == 1) {
      CLOGF_SUBSYS(
          INFO, COLL, "Running ReduceScatter ctdirect algorithm for nNodes=1");
      algo = NCCL_REDUCESCATTER_ALGO::ctdirect;
    } else if (nLocalRanks == 1) {
      // Only ctring for nLocalRanks=1 && nNodes >1 case
      CLOGF_SUBSYS(
          INFO,
          COLL,
          "Running ReduceScatter ctring algorithm for nLocalRanks=1 and nNodes>1");
      algo = NCCL_REDUCESCATTER_ALGO::ctring;
    }
  }

  switch (algo) {
    case NCCL_REDUCESCATTER_ALGO::ctdirect:
      return ctranReduceScatterDirect(
          sendbuff, recvbuff, recvcount, datatype, redOp, comm, stream);
    case NCCL_REDUCESCATTER_ALGO::ctring:
      return ctranReduceScatterRing(
          sendbuff, recvbuff, recvcount, datatype, redOp, comm, stream);
    case NCCL_REDUCESCATTER_ALGO::ctrhd:
      return ctranReduceScatterRHD(
          sendbuff, recvbuff, recvcount, datatype, redOp, comm, stream);
    default:
      CLOGF(
          WARN,
          "ctranReduceScatter: no valid algorithm to support nLocalRanks {} nNodes {}",
          nLocalRanks,
          nNodes);
      return ErrorStackTraceUtil::log(commInternalError);
  }
}
