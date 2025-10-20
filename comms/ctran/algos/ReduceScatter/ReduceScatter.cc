// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/algos/ReduceScatter/ReduceScatterImpl.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/utils/cvars/nccl_cvars.h"

bool ctranReduceScatterSupport(CtranComm* comm) {
  // CTran supports either single node case or nLocalRanks == 1 multi-node case
  // for now.
  bool topoSupport =
      comm->statex_->nNodes() == 1 || comm->statex_->nLocalRanks() == 1;
  bool ctranInited = ctranInitialized(comm);
  bool supported = false;
  if (ctranInited) {
    supported =
        (comm->statex_->nRanks() == 1 || comm->ctran_->mapper->hasBackend()) &&
        topoSupport;
  }

  // Print details of unsupport reason
  if (!supported) {
    CLOGF(
        WARN,
        "ctranReduceScatterSupport: not supported. Likely because "
        "unsupported topology ({}): nNodes {} nLocalRanks {}, ctranInitialized ({}), or hasBackend ({})",
        topoSupport ? "supported" : "unsupported",
        comm->statex_->nNodes(),
        comm->statex_->nLocalRanks(),
        ctranInited ? "true" : "false",
        ctranInited && comm->ctran_->mapper->hasBackend() ? "true" : "false");

    // Print details of unsupported peers
    if (ctranInited && !comm->ctran_->mapper->hasBackend()) {
      const auto nRanks = comm->statex_->nRanks();
      const auto rank = comm->statex_->rank();
      for (int peer = 0; peer < nRanks; peer++) {
        if (peer != rank &&
            comm->ctran_->mapper->getBackend(peer) ==
                CtranMapperBackend::UNSET) {
          CLOGF(
              WARN,
              "ctranReduceScatterSupport: rank {} peer {} has unset backend",
              rank,
              peer);
        }
      }
    }
  }
  return supported;
}

commResult_t ctranReduceScatter(
    const void* sendbuff,
    void* recvbuff,
    size_t recvcount,
    commDataType_t datatype,
    commRedOp_t redOp,
    CtranComm* comm,
    cudaStream_t stream) {
  if (comm->statex_->nRanks() == 1) {
    return reduceScatterSingleRankImpl(
        sendbuff, recvbuff, recvcount, datatype, redOp, comm, stream);
  }
  auto algo = NCCL_REDUCESCATTER_ALGO;

  const int nNodes = comm->statex_->nNodes();
  const int nLocalRanks = comm->statex_->nLocalRanks();
  // Only ctdirect supports nNodes == 1 case.
  // nLocalRanks>1 case is currently unsupported.
  if (algo == NCCL_REDUCESCATTER_ALGO::ctran) {
    if (nNodes == 1) {
      algo = NCCL_REDUCESCATTER_ALGO::ctdirect;
    } else if (nLocalRanks == 1) {
      // Only ctring for nLocalRanks=1 && nNodes >1 case
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
