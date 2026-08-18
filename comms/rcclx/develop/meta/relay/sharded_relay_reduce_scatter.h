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
 * nActiveRanksPerGroup must be a power of two (2 or 4). A==2 uses the 2-active
 * passthrough relay described below. A>2 uses a two-group flat relay with
 * REDUCE-AT-HELPER: each helper owns one position slice of every block,
 * collects that slice from the A-1 non-owner sources, sums them, and forwards a
 * single reduced chunk to the owner, woven with a direct all-to-all
 * reduce-scatter over the intra links. Both the direct region and the offload
 * region are folded into the output with fused multi-input reduce passes rather
 * than a loop of per-contribution add + scale passes.
 *
 * Because the A>2 helper reduces rather than forwards, it needs one chunk per
 * (owner, source) pair -- A x (A-1) x chunk, i.e. 1.5 x recvCounts[g] on an
 * 8-GPU node -- so its helper contract is larger than the A==2 two-slot one
 * (see the Helper-Buffer Contract below).
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
 * Two Comm Groups (mirroring allreduce, restricted to the relevant block):
 * ========================================================================
 *   Group 1: each active rank scatters numHelpers chunks of its sendBlockOffset
 *            block to helpers (each helper stores two slots, one per active
 *            rank) AND the two active ranks directly exchange one chunk over
 *            the otherwise-idle active<->active link.
 *   Group 2: each helper forwards slot-from-a0 -> a1 and slot-from-a1 -> a0
 * into the destination's scratch AND the active ranks directly exchange a
 *            second chunk over the same idle link.
 *   Reduce:  one fused pass folds the local ownBlockOffset contribution into
 * the whole foreign-contribution scratch, which mirrors the output layout, so
 * relayed and directly exchanged chunks reduce together.
 *
 * Unlike allreduce, the helper CANNOT reduce: its slot 0 holds a0's
 * contribution to a1's output and slot 1 holds a1's contribution to a0's output
 * -- different outputs. Helpers stay pure passthrough and the active rank
 * reduces.
 *
 * Chunking:
 * =========
 *   chunkSize = recvCounts[g] / numChunks rounded down to 128 elements,
 *   numChunks = numHelpers + 2. The active<->active link is idle while the
 * relay scatter and forward run on the cross links, so instead of a third comm
 * group for one direct chunk, one direct chunk rides along with each relay
 * group. Every link then carries exactly one chunk per direction per group and
 * the critical path is 2 x chunkSize instead of 3 x
 * recvCounts[g]/(numHelpers+1). Returns ncclInvalidArgument when recvCounts[g]
 * < numChunks x 128 (too small to scatter); callers should fall back to a plain
 * reduce-scatter.
 *
 * Helper-Buffer Contract:
 * =======================
 * A==2 path: identical to allreduce. Caller MUST supply at least
 *   nActiveRanksPerGroup × chunkSize_aligned
 * elements per helper group, where chunkSize_aligned is computed from
 * recvCounts[g] (NOT the doubled send count).
 *
 * A>2 path: the helper reduces rather than forwards, so it holds one chunk per
 * (owner, contributing source) pair: A × (A-1) × cs, where cs is recvCounts[g]
 * divided by (A + numHelpers) and 128-aligned. That is 1.5 × recvCounts[g] on
 * an 8-GPU node, so allocating 2 × recvCounts[g] per helper group covers every
 * supported (A, numHelpers) split.
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
