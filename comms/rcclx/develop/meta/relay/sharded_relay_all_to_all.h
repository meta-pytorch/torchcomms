/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "nccl.h"

// Maximum number of groups supported in multi-group all-to-all.
// Mirrors SHARDED_RELAY_MAX_GROUPS in sharded_relay_allreduce.h; redefined
// here so this header is self-contained (the two values must stay in sync).
#ifndef SHARDED_RELAY_MAX_GROUPS
#define SHARDED_RELAY_MAX_GROUPS 8
#endif

/**
 * Fused Multi-Group Sharded Relay All-to-All for 2D Sparse Parallelism.
 *
 * This API performs multiple sharded relay all-to-alls in a single fused call,
 * coordinating phases across all groups to prevent XGMI link contention. It is
 * the all-to-all analogue of the sharded relay allreduce/reduce-scatter and
 * reuses the same phase-synchronized passthrough-helper scheme; helpers perform
 * no compute and simply relay sharded chunks.
 *
 * nActiveRanksPerGroup must be a power of two (2 or 4). A==2 uses the original
 * 2-active passthrough relay (which wins there with 6 dedicated helpers per
 * group); A>2 uses a PURE-DIRECT all-to-all among the active ranks over the
 * 1-hop intra links -- NO helper relay, because the 2-hop offload is a net loss
 * at A>2 (only A helpers, serial source->helper->owner) and the fused direct
 * exchange already beats NCCL's P2P-bound 4-GPU all_to_all by ~1.4x. Helper
 * ranks do no work for a group they are not active in. OUT-OF-PLACE ONLY in
 * both cases.
 *
 * Unlike allreduce/reduce-scatter, all-to-all performs NO reduction — it is
 * pure data movement — so no reduction kernels are used and there is no
 * reduction op parameter.
 *
 * All-to-All Semantics (per group, A active ranks):
 * =======================================================
 * Each active rank's sendBuff and recvBuff hold nActiveRanksPerGroup x
 * segmentCounts[g] (= 2 x segmentCounts[g]) elements:
 *   - sendBuff = [sendSeg[0] | sendSeg[1]]; sendSeg[j] is the segment destined
 *     for active index j.
 *   - recvBuff = [recvSeg[0] | recvSeg[1]]; recvSeg[i] receives the segment
 *     from active index i.
 *
 * For the active rank with myActiveIndex = m, otherActiveIndex = o = 1 - m:
 *   - Diagonal (self -> self): recvSeg[m] = sendSeg[m]; both at offset
 *     m x segmentCount. A local copy.
 *   - Exchange: rank m ships sendSeg[o] (offset o x segmentCount) to the other
 *     rank and receives the other rank's sendSeg[m] into recvSeg[o] (offset
 *     o x segmentCount). This single-segment exchange is sharded across
 * helpers.
 *
 * IN-PLACE IS NOT SUPPORTED. Like the native RCCL ncclAllToAll, sendBuff and
 * recvBuff MUST be distinct; passing sendBuff == recvBuff returns
 * ncclInvalidArgument.
 *
 * Five Phases (no reduction; pure relay + placement copies):
 * ==========================================================
 *   Phase 1 (active->helpers): each active rank scatters numHelpers chunks of
 *            its sendSeg[o] segment to helpers; each helper stores two slots
 *            (one per active rank).
 *   Phase 2 (helpers->active, batched): each helper forwards slot-from-a0 -> a1
 *            and slot-from-a1 -> a0; the active rank receives numHelpers chunks
 *            DIRECTLY into recvSeg[o] (offset o x segmentCount + h x
 * chunkSize). No relay scratch and no reduction are needed. Phase 3 (diagonal
 * copy): recvSeg[m] = sendSeg[m] (cudaMemcpyAsync). Phase 4 (active<->active):
 * direct exchange of the last (direct) chunk of the exchange segment, received
 * directly into recvSeg[o]. Phase 5: none (no reduction).
 *
 * Chunking:
 * =========
 *   chunkSize  = segmentCounts[g] / numChunks rounded down to 128 elements,
 *   numChunks  = numHelpers + 1 (last chunk is the direct-exchange chunk).
 * Returns ncclInvalidArgument when segmentCounts[g] < numChunks x 128 (too
 * small to scatter); callers should fall back to a plain all-to-all.
 *
 * Helper-Buffer Contract (passthrough-at-helper):
 * ===============================================
 * A==2: caller MUST supply at least nActiveRanksPerGroup x chunkSize_aligned
 * elements per helper group, where chunkSize_aligned is computed from
 * segmentCounts[g].
 * A>2: PURE-DIRECT (no helper relay), so helper ranks do no work and need NO
 * helper buffer; the caller may pass any small placeholder tensor for the
 * groups it is a helper for.
 *
 * Memory Model:
 * =============
 * Each rank is ACTIVE for exactly ONE group (has real tensor data).
 * For other groups, the rank is a HELPER (uses provided two-slot scratch).
 *   - sendBuffs[nGroups]: one contiguous input buffer per group. For the active
 *     group it holds nActiveRanksPerGroup x segmentCounts[g] elements; helper
 *     groups may pass a small placeholder.
 *   - recvBuffs[nGroups]: one base buffer per group. For helper groups this is
 *     the two-slot passthrough scratch (>= nActiveRanks x chunkSize); for the
 *     active group it is the output base (nActiveRanksPerGroup x
 *     segmentCounts[g] elements).
 *   - IN-PLACE IS NOT SUPPORTED: sendBuffs[g] and recvBuffs[g] MUST be distinct
 *     (matching native ncclAllToAll); aliasing returns ncclInvalidArgument.
 *   - Each helper group MUST have its own buffer (no aliasing across groups)
 *     because all groups are processed simultaneously under phase-sync.
 *
 * @param sendBuffs Array of per-group contiguous input buffer pointers (one per
 *        group); the active group holds nActiveRanksPerGroup x segmentCounts[g]
 *        elements
 * @param recvBuffs Array of per-group base buffer pointers (one per group);
 *        helper groups pass two-slot passthrough scratch
 * @param segmentCounts Array of per-group per-segment element counts (one per
 *        group); the active group's input/output together hold
 *        nActiveRanksPerGroup x segmentCounts[g] elements
 * @param datatype NCCL data type
 * @param comm NCCL communicator
 * @param stream CUDA stream
 * @param allActiveRanks 2D array of active ranks
 * [nGroups][nActiveRanksPerGroup]
 * @param nActiveRanksPerGroup Number of active ranks per group (2 or 4)
 * @param nGroups Number of groups (typically 4 for 8-GPU node)
 * @return ncclResult_t Success or error code
 */
ncclResult_t ncclShardedRelayMultiGroupAllToAllImpl(
    const void* const* sendBuffs,
    void* const* recvBuffs,
    const size_t* segmentCounts,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream,
    const int* const* allActiveRanks,
    int nActiveRanksPerGroup,
    int nGroups);
