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
 * Active ranks per group (nActiveRanksPerGroup) must be a power of two — 2 or 4
 * are supported (on an 8-GPU node: 2 active + 6 helpers, or 4 active + 4
 * helpers). A=2 uses the original single-pass flow. A=4 uses a flat
 * helper-reduce-AND-broadcast: count is split into a DIRECT region (allreduced
 * among the active ranks over the 1-hop intra links via a direct reduce-scatter
 * + all-gather) and an OFFLOAD region (allreduced 2-hop through the otherwise-
 * idle helpers, which SUM all active ranks' chunk and BROADCAST the result
 * back); the AVG divisor is nActiveRanks. Both the direct owned-shard reduce
 * and the helper SUM use a single fused multi-input reduce pass (read the owned
 * shard plus all peer contributions once, apply the AVG divisor, write once)
 * rather than a loop of per-contribution add + scale passes. The two paths live
 * in separate internal functions selected by nActiveRanksPerGroup.
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
 * For A=4 the helper (reduce-and-broadcast) holds A source chunks plus one
 * reduced chunk per offload slice, so the caller must supply a larger scratch:
 * at least 2 × counts[g] elements per helper group (covers (A+1) × oChunk for
 * any offload fraction).
 *
 * Memory Model:
 * =============
 * Each rank is ACTIVE for exactly ONE group (has real tensor data).
 * For other groups, the rank is a HELPER (uses provided two-slot scratch).
 * The caller must provide:
 *   - sendBuffs[nGroups]: one contiguous input buffer per group. For the active
 *     group it is the real input tensor (counts[g] elements); helper groups may
 *     pass the same pointer as recvBuffs.
 *   - recvBuffs[nGroups]: one base buffer per group. For helper groups this is
 *     the two-slot passthrough scratch (>= nActiveRanks * chunkSize); for the
 *     active group it is the working/output base (counts[g] elements).
 *   - Allreduce may be in-place (recvBuffs[g] aliases sendBuffs[g]) or
 *     out-of-place (distinct buffers), keyed off sendBuffs[g]==recvBuffs[g].
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
 * @param sendBuffs Array of per-group contiguous input buffer pointers (one per
 *        group); the active group holds counts[g] elements
 * @param recvBuffs Array of per-group base buffer pointers (one per group);
 *        helper groups pass two-slot passthrough scratch
 * @param counts Array of element counts (one per group, allows different
 *        sizes); the active group's input/output each hold counts[g] elements
 * @param datatype NCCL data type
 * @param op Reduction operation (only ncclSum and ncclAvg supported)
 * @param comm NCCL communicator
 * @param stream CUDA stream
 * @param allActiveRanks 2D array of active ranks
 * [nGroups][nActiveRanksPerGroup]
 * @param nActiveRanksPerGroup Number of active ranks per group (2 or 4)
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
