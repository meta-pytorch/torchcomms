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
 * nActiveRanksPerGroup must be a power of two (2 or 4). A==2 selects between
 * the exact direct exchange at small sizes and the original 2-active
 * passthrough relay with 6 dedicated helpers. A==4 with 4
 * helpers uses the no-pack XOR/Latin relay once the common maximum per-active-
 * rank input size, A * max(segmentCounts) * elementSize, reaches 27 MiB (fused)
 * or 9 MiB (independent) and every group count is positive. Smaller A==4 calls
 * use the exact pure-direct all-to-all. OUT-OF-PLACE ONLY in all cases.
 *
 * Unlike allreduce/reduce-scatter, all-to-all performs NO reduction — it is
 * pure data movement — so no reduction kernels are used and there is no
 * reduction op parameter.
 *
 * All-to-All Semantics (per group, A active ranks):
 * =======================================================
 * Each active rank's sendBuff and recvBuff hold nActiveRanksPerGroup x
 * segmentCounts[g] elements:
 *   - sendBuff = [sendSeg[0] | ... | sendSeg[A-1]]; sendSeg[j] is destined for
 *     active index j.
 *   - recvBuff = [recvSeg[0] | ... | recvSeg[A-1]]; recvSeg[i] receives from
 *     active index i.
 *
 * For active index m, recvSeg[m] = sendSeg[m] is a local diagonal copy. Every
 * off-diagonal sendSeg[j] is transferred to active index j and placed in its
 * recvSeg[m].
 *
 * IN-PLACE IS NOT SUPPORTED. Like the native RCCL ncclAllToAll, sendBuff and
 * recvBuff MUST be distinct; passing sendBuff == recvBuff returns
 * ncclInvalidArgument.
 *
 * Schedules (no reduction; data movement + a placement copy):
 * ============================================================
 *   Diagonal: recvSeg[m] = sendSeg[m] (cudaMemcpyAsync).
 *   A==2: group 1 scatters helper chunks and exchanges directA; group 2
 *         forwards helper slots and exchanges directB.
 *   Routed A==4: each off-diagonal segment is split contiguously into directA,
 *         relay, and directB. Group 1 transfers directA to the destination and
 *         relay to helper index source XOR p[destination], p={0,2,3,1}. Group 2
 *         transfers directB and forwards each compact helper slot directly to
 *         recvSeg[source] + directA. Each helper receives and forwards exactly
 *         three tasks to three distinct destinations.
 *   Direct A==4: one grouped active-to-active exchange; helpers are idle.
 *
 * Chunking:
 * =========
 * A==2 relay: chunkSize = segmentCounts[g] / (numHelpers + 2), rounded down to
 * 128 elements. One direct chunk rides with each relay group; if alignment
 * makes chunkSize zero, directA/directB cover the whole segment.
 * Routed A==4: directA = floor(segmentCounts[g] / 3), relayCount is directA
 * rounded down to 128 elements, and directB absorbs every alignment and
 * division tail. The three contiguous regions cover every element once.
 *
 * Helper-Buffer Contract (passthrough-at-helper):
 * ===============================================
 * A==2: caller MUST supply at least nActiveRanksPerGroup x chunkSize_aligned
 * elements per helper group, where chunkSize_aligned is computed from
 * segmentCounts[g].
 * A==4 routed window: caller MUST supply at least 3 x relayCount elements per
 * helper group, where relayCount = alignDown(segmentCounts[g] / 3, 128). This
 * is bounded by one segment, so allocating segmentCounts[g] elements suffices.
 * A==4 direct routes: helpers do no work and need no scratch, though callers
 * may retain the one-segment allocation used for the routed window.
 *
 * Memory Model:
 * =============
 * Each rank is ACTIVE for exactly ONE group (has real tensor data).
 * For other groups, the rank is a HELPER (uses the provided scratch when the
 * selected route relays data).
 *   - sendBuffs[nGroups]: one contiguous input buffer per group. For the active
 *     group it holds nActiveRanksPerGroup x segmentCounts[g] elements; helper
 *     groups provide scratch sized by the selected route's contract above.
 *   - recvBuffs[nGroups]: one base buffer per group. For helper groups this is
 *     the passthrough scratch; for the active group it is the output base
 *     (nActiveRanksPerGroup x segmentCounts[g] elements).
 *   - IN-PLACE IS NOT SUPPORTED: sendBuffs[g] and recvBuffs[g] MUST be distinct
 *     (matching native ncclAllToAll); aliasing returns ncclInvalidArgument.
 *   - Each helper group MUST have its own buffer (no aliasing across groups)
 *     because all groups are processed simultaneously under phase-sync.
 *
 * @param sendBuffs Array of per-group contiguous input buffer pointers (one per
 *        group); the active group holds nActiveRanksPerGroup x segmentCounts[g]
 *        elements
 * @param recvBuffs Array of per-group base buffer pointers (one per group);
 *        helper groups pass scratch satisfying the helper-buffer contract
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
