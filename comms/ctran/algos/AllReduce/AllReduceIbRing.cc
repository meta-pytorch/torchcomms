// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <chrono>
#include <optional>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/AllReduce/AllReduceImpl.h"
#include "comms/utils/logger/LogUtils.h"

// Stub for the Pipes-backed hierarchical ring AllReduce (NVL ReduceScatter +
// inter-node IBGDA ring + NVL AllGather). The wiring (enum, AlgoStrConv,
// dispatch) lands in this diff; the real implementation lands in the next
// stacked diff. Until then `ctranAllReduceSupport` returns false for
// `cthierarchical_ring`, so this path is not reached in practice.
commResult_t ctranAllReduceHierarchicalRing(
    const void* /* sendbuff */,
    void* /* recvbuff */,
    size_t /* count */,
    commDataType_t /* datatype */,
    commRedOp_t /* redOp */,
    CtranComm* /* comm */,
    cudaStream_t /* stream */,
    std::optional<std::chrono::milliseconds> /* timeout */) {
  CLOGF(ERR, "AllReduce cthierarchical_ring is not yet implemented");
  return commInvalidArgument;
}
