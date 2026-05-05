/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "nccl.h"

// Maximum number of groups supported in multi-group allreduce
#define SHARDED_RELAY_MAX_GROUPS 8

/**
 * Fused Multi-Group Sharded Relay AllReduce for 2D Sparse Parallelism.
 *
 * This API performs multiple sharded relay allreduces in a single fused call,
 * coordinating phases across all groups to prevent XGMI link contention.
 *
 * Problem with Separate Calls:
 * ============================
 * When 4 separate sharded relay allreduces run in parallel (one per sparse
 * group), different groups may be in different phases:
 *   - Group [0,1] doing active→helpers (sending on links 3→0, 3→1)
 *   - Group [2,3] doing helpers→active (receiving on links 3→0, 3→1)
 * This causes bidirectional contention on shared XGMI links, degrading
 * bandwidth by up to 10x.
 *
 * Solution - Phase-Synchronized Execution with Passthrough Helpers:
 * =================================================================
 * This fused API executes ALL groups in lockstep phases:
 *
 *   Phase 1: ALL groups scatter (active→helpers) simultaneously.
 *            Each helper receives one chunk per active rank into a
 *            two-slot helper buffer (slot 0 from a0, slot 1 from a1).
 *
 *   Phase 2: ALL groups forward (helpers→other active) simultaneously.
 *            Helpers act as PURE PASSTHROUGH (no local compute):
 *            they forward slot 0 to a1 and slot 1 to a0.  Active ranks
 *            receive into a per-group recv-scratch slot.
 *
 *   Phase 3: ALL active ranks reduce the numHelpers helper-relayed chunks
 *            into their own active buffer (pipelined with Phase 2 recvs)
 *            and apply the AVG scaling once across the reduced region.
 *
 *   Phase 4: ALL groups direct-exchange the last (chunk N) data
 *            simultaneously between the two active ranks.
 *
 *   Phase 5: Active ranks perform the final reduction on the directly
 *            exchanged chunk.
 *
 * Since all groups are in the same phase at any time, XGMI links carry
 * unidirectional traffic only, eliminating contention.
 *
 * Helper-Buffer Contract (passthrough-at-helper):
 * ===============================================
 * Helpers no longer perform local reductions; they simply forward each
 * received chunk to the OTHER active rank.  Two slots are needed per
 * helper group so that the recv from a0 (slot 0) and the recv from a1
 * (slot 1) can proceed concurrently — without two slots, a helper would
 * have to serialize the two directions and halve its instantaneous
 * network bandwidth.
 *
 * Caller MUST supply at least
 *   nActiveRanksPerGroup × chunkSize_aligned
 * elements per helper group, where:
 *   chunkSize_aligned = (per_group_count / numChunks) rounded down to
 *                       CHUNK_ALIGN_ELEMENTS (128 elements).
 * Returns ncclInvalidArgument when per_group_count < numChunks × 128
 * (the buffer is too small to scatter); callers should fall back to a
 * regular allreduce in that case.
 *
 * Across the (nGroups - 1) helper groups per rank, that totals:
 *   (nGroups - 1) × nActiveRanksPerGroup × chunkSize
 * For the BM-FM 4-group / 2-active topology this is:
 *   3 × 2 × chunkSize = 6 × chunkSize per rank.
 *
 * Memory Model:
 * =============
 * Each rank is ACTIVE for exactly ONE group (has real tensor data).
 * For other groups, the rank is a HELPER (uses provided two-slot scratch).
 * The caller must provide:
 *   - sendBuffs[nGroups]: One buffer per group
 *   - recvBuffs[nGroups]: One buffer per group
 *   - For the group where rank is active: full per_group_counts[g] elements
 *   - For other groups: two-slot scratch (>= nActiveRanks * chunkSize)
 *   - Each helper group MUST have its own buffer (no aliasing across groups)
 *     because all groups are processed simultaneously under phase-sync
 *
 * 2D Sparse Parallelism Example (8 GPUs, 4 groups):
 * =================================================
 *   Group 0: activeRanks = {0, 1}, helpers = {2,3,4,5,6,7}
 *   Group 1: activeRanks = {2, 3}, helpers = {0,1,4,5,6,7}
 *   Group 2: activeRanks = {4, 5}, helpers = {0,1,2,3,6,7}
 *   Group 3: activeRanks = {6, 7}, helpers = {0,1,2,3,4,5}
 *
 * @param sendBuffs Array of send buffer pointers (one per group)
 * @param recvBuffs Array of receive buffer pointers (one per group)
 * @param counts Array of element counts (one per group, allows different sizes)
 * @param datatype NCCL data type
 * @param op Reduction operation (only ncclSum and ncclAvg supported)
 * @param comm NCCL communicator
 * @param stream CUDA stream
 * @param allActiveRanks 2D array of active ranks
 * [nGroups][nActiveRanksPerGroup]
 * @param nActiveRanksPerGroup Number of active ranks per group (typically 2)
 * @param nGroups Number of groups (typically 4 for 8-GPU node)
 * @return ncclResult_t Success or error code
 */
ncclResult_t ncclShardedRelayMultiGroupAllReduceImpl(
    const void* const* sendBuffs,
    void* const* recvBuffs,
    const size_t* counts,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    cudaStream_t stream,
    const int* const* allActiveRanks,
    int nActiveRanksPerGroup,
    int nGroups);
