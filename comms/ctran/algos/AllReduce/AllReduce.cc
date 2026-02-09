// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <chrono>
#include <optional>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllReduce/AllReduceImpl.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

bool ctranAllReduceSupport(CtranComm* comm, enum NCCL_ALLREDUCE_ALGO algo) {
  if (!ctranInitialized(comm) || !comm->ctran_->mapper->hasBackend()) {
    return false;
  }

  switch (algo) {
    case NCCL_ALLREDUCE_ALGO::ctring:
      // TODO(T240133674): remove this check and return true once ctring is
      // supported for all topologies
      if (comm->statex_->nLocalRanks() == 1) {
        return true;
      }
      CLOGF(
          WARN,
          "ctring algo currently only supported for nLocalRanks=1 for ctranAllReduce, falling back to baseline");
      return false;
    case NCCL_ALLREDUCE_ALGO::ctran:
    case NCCL_ALLREDUCE_ALGO::ctdirect:
      return true;
    default: // invalid query
      return false;
  }
}

commResult_t ctranAllReduce(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    commDataType_t datatype,
    commRedOp_t redOp,
    CtranComm* comm,
    cudaStream_t stream,
    enum NCCL_ALLREDUCE_ALGO algo,
    std::optional<std::chrono::milliseconds> timeout) {
  switch (algo) {
    case NCCL_ALLREDUCE_ALGO::ctring:
      if (comm->statex_->nRanks() == 1) {
        // TODO(T242570177): this is a temp workaround for nRanks == 1. Remove
        // the warning below if fixed.
        CLOGF(
            WARN,
            "AllReduce ctring currently requires nRanks > 1, fallback to ctdirect");
        return ctranAllReduceDirect(
            sendbuff, recvbuff, count, datatype, redOp, comm, stream, timeout);
      }
      if (count < comm->statex_->nRanks()) {
        CLOGF(
            WARN,
            "AllReduce ctring requires count {} > nRanks {}, fallback to ctdirect",
            count,
            comm->statex_->nRanks());
        return ctranAllReduceDirect(
            sendbuff, recvbuff, count, datatype, redOp, comm, stream, timeout);
      }
      return ctranAllReduceRing(
          sendbuff, recvbuff, count, datatype, redOp, comm, stream, timeout);
    case NCCL_ALLREDUCE_ALGO::ctdirect:
    default:
      return ctranAllReduceDirect(
          sendbuff, recvbuff, count, datatype, redOp, comm, stream, timeout);
  }
}
