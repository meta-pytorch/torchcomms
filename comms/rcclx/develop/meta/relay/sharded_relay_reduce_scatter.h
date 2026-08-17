/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "nccl.h"

// Maximum number of groups supported in multi-group reduce-scatter.
// Mirrors SHARDED_RELAY_MAX_GROUPS in sharded_relay_allreduce.h; redefined
// here so this header is self-contained (the two values must stay in sync).
#ifndef SHARDED_RELAY_MAX_GROUPS
#define SHARDED_RELAY_MAX_GROUPS 8
#endif

/**
 * Fused Multi-Group Sharded Relay Reduce-Scatter for 2D Sparse Parallelism.
 *
 * This API performs multiple sharded relay reduce-scatters in a single fused
 * call, coordinating phases across all groups to prevent XGMI link contention.
 * It is the reduce-scatter analogue of
 * `ncclShardedRelayMultiGroupAllReduceImpl` and reuses the same
 * phase-synchronized passthrough-helper scheme; helpers perform no local
 * compute and all reductions happen on the active ranks per group.
 *
 * nActiveRanksPerGroup must be a power of two (2 or 4). A==2 uses the original
 * 2-active passthrough path (block-restricted 5-phase relay); A>2 uses the
 * bandwidth-optimal recursive path (recursive-halving relay woven with a direct
 * all-to-all, mirroring the merged allreduce). The direct all-to-all's owned
 * shard is reduced with a single fused multi-input reduce pass (owned shard +
 * all A-1 peer contributions read once, AVG divisor applied, written once)
 * instead of a loop of per-contribution add + scale passes. Both keep the same
 * per-helper buffer contract (nActiveRanksPerGroup x chunkSize).
 *
 * Reduce-Scatter Semantics (per group, 2 active ranks a0, a1):
 * ===========================================================
 * NCCL reduce-scatter among the 2 active ranks:
 *   - Each active rank's sendBuff holds nActiveRanksPerGroup × recvCounts[g]
 *     (= 2 × recvCounts[g]) elements: logically block[i] is the slice
 *     destined for active index i.
 *   - Each active rank's recvBuff holds recvCounts[g] elements: active index
 *     i receives sum_over_active_ranks( sendBuff[block i] ).
 *
 * So each active rank keeps its own block[myActiveIndex] and must receive the
 * other rank's block[myActiveIndex]; equivalently each active rank ships
 * block[otherActiveIndex] to the other rank. This is a single-block
 * (recvCounts[g]-element) sharded exchange + local reduce — structurally
 * identical to the allreduce relay but restricted to one block.
 *
 * Block offsets per active rank (in elements):
 *   - ownBlockOffset  = myActiveIndex    × recvCounts[g] (local contribution)
 *   - sendBlockOffset = otherActiveIndex × recvCounts[g] (shipped to other)
 *
 * Five Phases (mirroring allreduce, restricted to the relevant block):
 * ===================================================================
 *   Phase 1 (active→helpers): each active rank scatters numHelpers chunks of
 *            its sendBlockOffset block to helpers; each helper stores two
 *            slots (one per active rank) — same passthrough contract as
 *            allreduce.
 *   Phase 2 (helpers→active, batched): each helper forwards slot-from-a0 → a1
 *            and slot-from-a1 → a0; active rank receives numHelpers chunks
 *            into relay scratch (numHelpers × chunkSize).
 *   Phase 3 (active reduce): fused add (+scale for AVG) of relay scratch into
 *            the output block (recvBuff), seeded with the local ownBlockOffset
 *            contribution.
 *   Phase 4 (active↔active): direct exchange of the last (direct) chunk.
 *   Phase 5 (active reduce): final reduction of the direct chunk.
 *
 * Chunking:
 * =========
 *   chunkSize  = recvCounts[g] / numChunks rounded down to 128 elements,
 *   numChunks  = numHelpers + 1 (last chunk is the direct-exchange chunk).
 * Returns ncclInvalidArgument when recvCounts[g] < numChunks × 128 (too small
 * to scatter); callers should fall back to a plain reduce-scatter.
 *
 * Helper-Buffer Contract (passthrough-at-helper):
 * ===============================================
 * Identical to allreduce. Caller MUST supply at least
 *   nActiveRanksPerGroup × chunkSize_aligned
 * elements per helper group, where chunkSize_aligned is computed from
 * recvCounts[g] (NOT the doubled send count).
 *
 * Memory Model:
 * =============
 * Each rank is ACTIVE for exactly ONE group (has real tensor data).
 * For other groups, the rank is a HELPER (uses provided two-slot scratch).
 * The caller must provide:
 *   - sendBuffs[nGroups]: one contiguous input buffer per group. For the active
 *     group it holds nActiveRanksPerGroup × recvCounts[g] elements (one
 *     contribution block per active index); helper groups may pass the same
 *     pointer as recvBuffs.
 *   - recvBuffs[nGroups]: one base buffer per group. For helper groups this is
 *     the two-slot passthrough scratch (>= nActiveRanks × chunkSize); for the
 *     active group it is the output base (recvCounts[g] elements).
 *   - In-place is detected when the active group's output aliases the owned
 *     input block (recvBuffs[g] == sendBuffs[g] + ownBlockOffset).
 *   - Each helper group MUST have its own buffer (no aliasing across groups)
 *     because all groups are processed simultaneously under phase-sync.
 *
 * Op support: ncclSum and ncclAvg only (AVG divisor = nActiveRanksPerGroup).
 *
 * @param sendBuffs Array of per-group contiguous input buffer pointers (one per
 *        group); the active group holds nActiveRanksPerGroup × recvCounts[g]
 *        elements (one contribution block per active index)
 * @param recvBuffs Array of per-group base buffer pointers (one per group);
 *        helper groups pass two-slot passthrough scratch
 * @param recvCounts Array of per-group OUTPUT element counts (one per group)
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
ncclResult_t ncclShardedRelayMultiGroupReduceScatterImpl(
    const void* const* sendBuffs,
    void* const* recvBuffs,
    const size_t* recvCounts,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    cudaStream_t stream,
    const int* const* allActiveRanks,
    int nActiveRanksPerGroup,
    int nGroups);
