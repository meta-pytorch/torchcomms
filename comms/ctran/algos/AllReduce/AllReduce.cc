// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <chrono>
#include <optional>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllReduce/AllReduceImpl.h"
#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

bool ctranAllReduceSupport(CtranComm* comm) {
  return ctranInitialized(comm) && comm->ctran_->mapper->hasBackend();
}

commResult_t ctranAllReduce(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    commDataType_t datatype,
    commRedOp_t redOp,
    CtranComm* comm,
    cudaStream_t stream,
    std::optional<const enum NCCL_ALLREDUCE_ALGO> algoSpecified,
    std::optional<std::chrono::milliseconds> timeout) {
  // Use global config if user doesn't provide specific algo per collective
  auto algo = algoSpecified.value_or(NCCL_ALLREDUCE_ALGO);

  switch (algo) {
    case NCCL_ALLREDUCE_ALGO::ctarg:
      if (timeout != std::nullopt) {
        CLOGF(WARN, "timeout is ignored for AllReduce ctarg algorithm");
      }
      return ctranAllReduceARG(
          sendbuff, recvbuff, count, datatype, redOp, comm, stream, timeout);
    case NCCL_ALLREDUCE_ALGO::ctring:
      if (comm->statex_->nRanks() == 1) {
        // TODO(T242570177): this is a temp workaround for nRanks == 1.
        return ctranAllReduceDirect(
            sendbuff, recvbuff, count, datatype, redOp, comm, stream, timeout);
      }
      return ctranAllReduceRing(
          sendbuff, recvbuff, count, datatype, redOp, comm, stream, timeout);
    case NCCL_ALLREDUCE_ALGO::ctdirect:
    default:
      if (timeout != std::nullopt) {
        CLOGF(WARN, "timeout is ignored for AllReduce ctdirect algorithm");
      }
      return ctranAllReduceDirect(
          sendbuff, recvbuff, count, datatype, redOp, comm, stream, timeout);
  }
}
