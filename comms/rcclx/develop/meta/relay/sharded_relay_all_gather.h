/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "nccl.h"

// Maximum number of groups supported in multi-group all-gather.
// Mirrors SHARDED_RELAY_MAX_GROUPS in sharded_relay_allreduce.h; redefined
// here so this header is self-contained (the two values must stay in sync).
#ifndef SHARDED_RELAY_MAX_GROUPS
#define SHARDED_RELAY_MAX_GROUPS 8
#endif

/**
 * Fused Multi-Group Sharded Relay All-Gather for 2D Sparse Parallelism.
 *
 * This API performs multiple sharded relay all-gathers in a single fused call,
 * coordinating phases across all groups to prevent XGMI link contention. It is
 * the all-gather analogue of the sharded relay allreduce/reduce-scatter and
 * reuses the same phase-synchronized passthrough-helper scheme; helpers perform
 * no compute and simply relay sharded chunks.
 *
 * nActiveRanksPerGroup must be a power of two (2 or 4). A==2 uses the 2-active
 * passthrough relay described below; A>2 uses a two-group flat
 * scatter->forward relay (the dual of the reduce-scatter path): a direct intra
 * exchange woven with a 2-hop offload through the idle helper GPUs. Group 1's
 * cross links carry one chunk and group 2's carry A-1 (the helper fans each
 * source's slice out to every other dest), so the direct region is split
 * 1:(A-1) across the two groups to keep both balanced.
 *
 * All-gather performs NO reduction (pure data movement), so no reduction
 * kernels are used and there is no reduction op parameter. It is the dual of
 * reduce-scatter.
 *
 * All-Gather Semantics (per group, A active ranks):
 * =======================================================
 * Each active rank's sendBuff holds sendCounts[g] elements (its own
 * contribution); each active recvBuff holds nActiveRanksPerGroup x
 * sendCounts[g] (= 2 x sendCounts[g]) elements, where recvBuff[i x sendCount]
 * receives the contribution from active index i.
 *
 * For the active rank with myActiveIndex = m, otherActiveIndex = o = 1 - m:
 *   - Diagonal (self): recvBuff[m x sendCount] = sendBuff (my own data in my
 *     slot).
 *   - Gather: ship sendBuff to the other rank and receive the other rank's
 *     sendBuff into recvBuff[o x sendCount]. This single-segment exchange is
 *     sharded across helpers.
 *
 * Both IN-PLACE and OUT-OF-PLACE are supported (like reduce-scatter and unlike
 * all-to-all). In-place is detected when sendBuff == recvBuff + m x sendCount
 * (the standard NCCL all-gather in-place convention: each rank's send data
 * already sits in its slot of recvBuff). No scratch buffers are required in
 * either mode: the gather destination (slot o) never overlaps the send source
 * (slot m), so there is no send/recv aliasing hazard.
 *
 * Two Comm Groups (no reduction; pure relay + a placement copy):
 * ==============================================================
 *   Diagonal: recvBuff[m x sendCount] = sendBuff (a no-op when in-place).
 *   Group 1:  each active rank scatters numHelpers chunks of its sendBuff to
 *             helpers (each helper stores two slots) AND the two active ranks
 *             directly exchange one chunk over the otherwise-idle
 *             active<->active link, straight into recvBuff[o x sendCount].
 *   Group 2:  each helper forwards slot-from-a0 -> a1 and slot-from-a1 -> a0,
 *             landing DIRECTLY in recvBuff[o x sendCount + h x chunkSize], AND
 *             the active ranks directly exchange a second chunk over the same
 *             idle link.
 *
 * Chunking:
 * =========
 *   chunkSize = sendCounts[g] / numChunks rounded down to
 *   kRelayChunkAlignElements (512) elements,
 *   numChunks = numHelpers + 2. The active<->active link is idle while the
 * relay scatter and forward run on the cross links, so instead of a third comm
 * group for one direct chunk, one direct chunk rides along with each relay
 * group. Every link then carries exactly one chunk per direction per group and
 * the critical path is 2 x chunkSize instead of 3 x
 * sendCounts[g]/(numHelpers+1). Returns ncclInvalidArgument when sendCounts[g]
 * < numChunks x 128 (too small to scatter); callers should fall back to a plain
 * all-gather.
 *
 * Helper-Buffer Contract (passthrough-at-helper):
 * ===============================================
 * A==2 path: caller MUST supply at least nActiveRanksPerGroup x
 * chunkSize_aligned elements per helper group, where chunkSize_aligned is
 * computed from sendCounts[g].
 *
 * A>2 (flat) path: the helper stores one offload chunk of `cs` elements per
 * active source, so the caller MUST supply at least A*cs elements per helper
 * group, where cs is the 128-aligned offload fraction of sendCounts[g] divided
 * by numHelpers. This is bounded above by sendCounts[g], so allocating A x
 * sendCounts[g] per helper group (as the C++ tests do) always satisfies it.
 *
 * Memory Model:
 * =============
 * Each rank is ACTIVE for exactly ONE group (has real tensor data).
 * For other groups, the rank is a HELPER (uses provided two-slot scratch).
 *   - sendBuffs[nGroups]: one contiguous input buffer per group. For the active
 *     group it holds this rank's contribution (sendCounts[g] elements); helper
 *     groups may pass the same pointer as recvBuffs.
 *   - recvBuffs[nGroups]: one base buffer per group. For helper groups this is
 *     the two-slot passthrough scratch (>= nActiveRanks x chunkSize); for the
 *     active group it is the gathered-output base (nActiveRanksPerGroup x
 *     sendCounts[g] elements).
 *   - In-place is detected when the input aliases the active rank's own output
 *     slot (sendBuffs[g] == recvBuffs[g] + myActiveIndex x sendCount).
 *   - Each helper group MUST have its own buffer (no aliasing across groups)
 *     because all groups are processed simultaneously under phase-sync.
 *
 * @param sendBuffs Array of per-group contiguous input buffer pointers (one per
 *        group); the active group holds this rank's contribution (sendCounts[g]
 *        elements)
 * @param recvBuffs Array of per-group base buffer pointers (one per group);
 *        helper groups pass two-slot passthrough scratch
 * @param sendCounts Array of per-group per-rank contribution element counts
 *        (one per group); the active output together holds
 *        nActiveRanksPerGroup x sendCounts[g] elements
 * @param datatype NCCL data type
 * @param comm NCCL communicator
 * @param stream CUDA stream
 * @param allActiveRanks 2D array of active ranks
 * [nGroups][nActiveRanksPerGroup]
 * @param nActiveRanksPerGroup Number of active ranks per group (2 or 4)
 * @param nGroups Number of groups (typically 4 for 8-GPU node)
 * @param lowPrecision Non-zero to use the low-precision (fp8e4m3) wire format
 *        where it pays; an internal size-only gate declines to full precision
 *        silently otherwise (see sharded_relay_lp.h). COLLECTIVE -- it must be
 *        identical on every rank of the call, like datatype and the counts.
 *        Ranks that disagree disagree on how many bytes cross each link, so the
 *        call hangs or corrupts rather than degrading. Documented rather than
 *        validated, because a per-call check would cost an allreduce; datatype
 *        is already treated the same way.
 * @return ncclResult_t Success or error code
 */
ncclResult_t ncclShardedRelayMultiGroupAllGatherImpl(
    const void* const* sendBuffs,
    void* const* recvBuffs,
    const size_t* sendCounts,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream,
    const int* const* allActiveRanks,
    int nActiveRanksPerGroup,
    int nGroups,
    int lowPrecision = 0);
