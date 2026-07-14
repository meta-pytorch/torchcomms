// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <chrono>
#include <optional>

#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/utils/commSpecs.h"

/*
 * Note: the Prims-backed tree AllReduce ("ctree", ctranAllReduceTree) was moved
 * into MCCL (comms/mccl/collectives/allreduce) as the single owner; CTRAN no
 * longer declares or implements it.
 */

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
