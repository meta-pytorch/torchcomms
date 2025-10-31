// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CTRAN_ALLREDUCE_IMPL_H_
#define CTRAN_ALLREDUCE_IMPL_H_

#include <chrono>

#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/utils/cvars/nccl_cvars.h"

commResult_t ctranAllReduceDirect(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    commDataType_t datatype,
    commRedOp_t redOp,
    CtranComm* comm,
    cudaStream_t stream,
    std::optional<std::chrono::milliseconds> timeout = std::nullopt);
commResult_t ctranAllReduceARG(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    commDataType_t datatype,
    commRedOp_t redOp,
    CtranComm* comm,
    cudaStream_t stream,
    std::optional<std::chrono::milliseconds> timeout = std::nullopt);
commResult_t ctranAllReduceRing(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    commDataType_t datatype,
    commRedOp_t redOp,
    CtranComm* comm,
    cudaStream_t stream,
    std::optional<std::chrono::milliseconds> timeout = std::nullopt);

static inline const std::string allReduceAlgoName(
    enum NCCL_ALLREDUCE_ALGO algo) {
  switch (algo) {
    case NCCL_ALLREDUCE_ALGO::ctdirect:
      return "CtranAllReduceDirect";
    case NCCL_ALLREDUCE_ALGO::ctran:
      return "CtranAuto";
    case NCCL_ALLREDUCE_ALGO::orig:
      return "Baseline";
    case NCCL_ALLREDUCE_ALGO::ctarg:
      return "CtranAllReduceA2ARAG";
    case NCCL_ALLREDUCE_ALGO::ctring:
      return "CtranAllReduceRing";
    default:
      return "Unknown";
  }
}

#endif
