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
commResult_t ctranAllReduceRing(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    commDataType_t datatype,
    commRedOp_t redOp,
    CtranComm* comm,
    cudaStream_t stream,
    std::optional<std::chrono::milliseconds> timeout = std::nullopt);
/**
 * Run the Prims-backed tree AllReduce implementation.
 *
 * The implementation supports `commSum` over `commFloat32` and `commFloat16`.
 * It relies on Prims NVL and IBGDA transport staging for transient receives and
 * does not allocate message-size-dependent AllReduce staging.
 */
commResult_t ctranAllReduceTree(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    commDataType_t datatype,
    commRedOp_t redOp,
    CtranComm* comm,
    cudaStream_t stream,
    std::optional<std::chrono::milliseconds> timeout = std::nullopt);

/**
 * Run the Prims-backed hierarchical ring AllReduce implementation.
 *
 * Three phases: NVL ReduceScatter (Phase 1), inter-node IBGDA ring (Phase 2),
 * NVL AllGather (Phase 3), reusing the shared NVL phases with the ring as the
 * cross-node Phase 2. Implemented in a stacked diff; this declaration plus the
 * dispatch wiring land first.
 */
commResult_t ctranAllReduceHierarchicalRing(
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
    case NCCL_ALLREDUCE_ALGO::ctring:
      return "CtranAllReduceRing";
    case NCCL_ALLREDUCE_ALGO::ctree:
      return "CtranAllReduceTree";
    case NCCL_ALLREDUCE_ALGO::cthierarchical_ring:
      return "CtranAllReduceHierarchicalRing";
    default:
      return "Unknown";
  }
}

#endif
