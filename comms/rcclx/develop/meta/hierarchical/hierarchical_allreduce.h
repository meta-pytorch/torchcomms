/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

#include "nccl.h"

namespace rcclx::hier {

/**
 * Hierarchical allreduce for multi-host communicators whose inter-node network
 * is the Socket transport (front-end NICs).
 *
 * Per pipeline tile: intra-node reduce-scatter over xGMI, inter-node allreduce
 * of each local rank's shard with the same local rank on every other node
 * (exchange for 2 nodes, reduce-scatter + all-gather otherwise), intra-node
 * all-gather. Each host puts the minimum 2*(N-1)/N*S bytes on the network,
 * spread evenly over every local GPU and so over every NIC. Stages of
 * consecutive tiles share one ncclGroup, so xGMI and network transfers
 * overlap. Every element is reduced by exactly one rank, so all ranks receive
 * bit-identical results. Graph capture is supported.
 *
 * Off by default. With NCCL_HIER_ALLREDUCE=1 it engages when the communicator
 * spans 2 to NCCL_HIER_ALLREDUCE_MAX_NODES nodes (default 2) and has more than
 * 8 ranks (NCCL_HIER_ALLREDUCE_MIN_RANKS, default 9). It is tuned for the
 * Socket net transport.
 */

// True when ncclAllReduce should take the hierarchical path. Every input is
// collective-consistent, so all ranks reach the same answer.
bool hierAllReduceEligible(
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm);

ncclResult_t hierAllReduce(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    cudaStream_t stream);

// Number of calls that ran on the hierarchical path in this process.
uint64_t hierAllReduceEngageCount();

} // namespace rcclx::hier
