/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "sharded_relay_all_gather.h"
#include "comm.h"

// GPU memory access alignment in elements for chunk size rounding.
// Distinct from the CPU CACHE_LINE_SIZE (64 bytes) defined in comm.h
// which is used for struct padding.
static constexpr size_t CHUNK_ALIGN_ELEMENTS = 128;

// The rank-config builder below is a deliberate copy of the file-local helper
// in sharded_relay_allreduce.cc (also mirrored in the reduce-scatter and
// all-to-all relays). It is file-local there, so it cannot be linked across
// translation units; this TU re-declares its own copy in an anonymous namespace
// to keep it internal and ODR-safe. All-gather performs no reduction, so NO
// reduction kernels are used, and the relay paths land every byte directly in
// recvBuff / the helper passthrough buffers (no working-buffer scratch).
namespace {

// Maximum number of helper ranks supported per group.
constexpr int SHARDED_RELAY_MAX_HELPERS = 8;

// Maximum number of active ranks per group. The relay schedule requires
// nActiveRanks to be a power of two; supported values are 2 and 4 (on an 8-GPU
// node this leaves 6 or 4 helpers respectively).
constexpr int SHARDED_RELAY_MAX_ACTIVE = 8;

// Returns true if v is a power of two (v >= 1).
inline bool isPowerOfTwo(int v) {
  return v > 0 && (v & (v - 1)) == 0;
}

/**
 * Rank Configuration for Sharded Relay All-Gather
 *
 * Holds parsed active and helper rank information for a single group.
 * Supports a power-of-two number of active ranks per group (2 or 4).
 */
struct ShardedRelayRankConfig {
  int activeRanks[SHARDED_RELAY_MAX_ACTIVE]; // Active rank IDs (power of two)
  int nActiveRanks; // Number of active ranks (2 or 4)
  int helperRanks[SHARDED_RELAY_MAX_HELPERS]; // Helper rank IDs
  int numHelpers; // Number of helper ranks
  bool isActiveRank; // Is current rank active?
  int myActiveIndex; // Index in activeRanks array (-1 if helper)
  int myHelperIndex; // Index in helperRanks array (-1 if active)
};

/**
 * Build rank configuration from provided active ranks array.
 *
 * Requires a power-of-two active-rank count in [2, SHARDED_RELAY_MAX_ACTIVE]
 * (supported values: 2 and 4).
 */
bool buildShardedRelayRankConfig(
    int nRanks,
    int rank,
    const int* activeRanksInput,
    int nActiveRanksInput,
    ShardedRelayRankConfig& config) {
  config.nActiveRanks = 0;
  config.numHelpers = 0;
  config.isActiveRank = false;
  config.myActiveIndex = -1;
  config.myHelperIndex = -1;

  // Validate input - require a power-of-two active-rank count in
  // [2, SHARDED_RELAY_MAX_ACTIVE] (supported values: 2 and 4).
  if (activeRanksInput == nullptr || nActiveRanksInput < 2 ||
      nActiveRanksInput > SHARDED_RELAY_MAX_ACTIVE ||
      !isPowerOfTwo(nActiveRanksInput)) {
    return false;
  }

  // Copy active ranks and validate
  for (int i = 0; i < nActiveRanksInput; i++) {
    int rankId = activeRanksInput[i];
    if (rankId >= 0 && rankId < nRanks) {
      config.activeRanks[config.nActiveRanks++] = rankId;
    }
  }

  // Validate: need exactly nActiveRanksInput valid active ranks
  if (config.nActiveRanks != nActiveRanksInput) {
    return false;
  }

  // Build list of helper ranks (all ranks NOT in activeRanks).
  for (int r = 0; r < nRanks; r++) {
    bool isActive = false;
    for (int a = 0; a < config.nActiveRanks; a++) {
      if (r == config.activeRanks[a]) {
        isActive = true;
        break;
      }
    }
    if (!isActive) {
      if (config.numHelpers >= SHARDED_RELAY_MAX_HELPERS) {
        return false;
      }
      config.helperRanks[config.numHelpers++] = r;
    }
  }

  // Validate: need at least 1 helper
  if (config.numHelpers < 1) {
    return false;
  }

  // Determine if this rank is active
  for (int a = 0; a < config.nActiveRanks; a++) {
    if (rank == config.activeRanks[a]) {
      config.isActiveRank = true;
      config.myActiveIndex = a;
      break;
    }
  }

  // For helpers, determine which chunk index this rank handles
  if (!config.isActiveRank) {
    for (int i = 0; i < config.numHelpers; i++) {
      if (config.helperRanks[i] == rank) {
        config.myHelperIndex = i;
        break;
      }
    }
  }
  return true;
}

// All-gather only moves bytes, so it has no reduce kernels to dispatch on --
// but the relay presents ONE supported type set across all four collectives,
// and only these ten are exercised anywhere in it. Two concrete hazards make
// this worth rejecting up front rather than trusting the caller:
//   - ncclTypeSize() returns int -1 for a type it does not know, and the
//     callers below store it in a `size_t elementSize`, turning it into
//     SIZE_MAX rather than 0. Every subsequent count * elementSize and
//     offset * elementSize then overflows into wild addresses.
//   - ncclFloat8e4m3 / ncclFloat8e5m2 are valid NCCL types that size cleanly
//     at 1 byte, so a range check against ncclNumTypes would admit them even
//     though no relay collective has ever been exercised with them.
bool isSupportedRelayDataType(ncclDataType_t datatype) {
  switch (datatype) {
    case ncclInt8:
    case ncclUint8:
    case ncclInt32:
    case ncclUint32:
    case ncclInt64:
    case ncclUint64:
    case ncclFloat16:
    case ncclFloat:
    case ncclDouble:
    case ncclBfloat16:
      return true;
    default:
      return false;
  }
}

} // namespace

/**
 * Two-active sharded relay all-gather (original path), the dual of the 2-active
 * reduce-scatter relay.
 *
 * Each group has exactly 2 active ranks; the logical collective is a 2-rank
 * all-gather between them, accelerated by passthrough helpers that relay
 * sharded chunks of each active rank's sendBuff (sendCounts[g] elements). There
 * is NO reduction: helpers and active ranks only move data, so no kernels and
 * no scratch buffers are required. This is byte-for-byte unchanged from the
 * original implementation; the A>2 path lives in
 * shardedRelayAllGatherFlat.
 *
 * Per active rank (index m, o = 1 - m), with sendCount = sendCounts[g]:
 *   - sendBuff holds sendCount elements; recvBuff holds 2 x sendCount elements
 *     (recvBuff[i x sendCount] receives the contribution from active index i).
 *   - Diagonal: recvBuff[m x sendCount] = sendBuff.
 *   - Gather: ship sendBuff to the other rank; receive the other rank's
 *     sendBuff into recvBuff[o x sendCount].
 *
 * Both in-place and out-of-place are supported. In-place is detected when
 * sendBuff == recvBuff + m x sendCount; in that case the diagonal copy is a
 * no-op. No scratch is needed in either mode because the gather destination
 * (slot o) never overlaps the send source (slot m).
 */
static ncclResult_t shardedRelayAllGather2Active(
    const void* const* sendBuffs,
    void* const* recvBuffs,
    const size_t* sendCounts,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream,
    const ShardedRelayRankConfig* configs,
    int myActiveGroup,
    int numHelpers,
    int nGroups,
    size_t elementSize) {
  // =========================================================================
  // CHUNK GEOMETRY: numHelpers relayed chunks + TWO direct chunks
  // =========================================================================
  // The active<->active link is idle while the relay scatter and forward run on
  // the cross links, so instead of a third comm group for a single direct
  // chunk, one direct chunk rides along with each relay group. With numChunks =
  // numHelpers + 2 every link carries exactly one chunk per direction per
  // group, making the critical path 2*sendCount/numChunks instead of the
  // 3*sendCount/(numHelpers+1) of a separate direct phase.
  const int numChunks = numHelpers + 2;

  size_t chunkSizes[SHARDED_RELAY_MAX_GROUPS];
  size_t dirAOffsets[SHARDED_RELAY_MAX_GROUPS];
  size_t dirASizes[SHARDED_RELAY_MAX_GROUPS];
  size_t dirBOffsets[SHARDED_RELAY_MAX_GROUPS];
  size_t dirBSizes[SHARDED_RELAY_MAX_GROUPS]; // absorbs the remainder

  for (int g = 0; g < nGroups; g++) {
    size_t count = sendCounts[g];

    // Zero-count groups are skipped by every loop below.
    if (count == 0) {
      chunkSizes[g] = 0;
      dirAOffsets[g] = 0;
      dirASizes[g] = 0;
      dirBOffsets[g] = 0;
      dirBSizes[g] = 0;
      continue;
    }

    size_t chunkSize = count / numChunks;
    chunkSize = (chunkSize / CHUNK_ALIGN_ELEMENTS) * CHUNK_ALIGN_ELEMENTS;
    chunkSizes[g] = chunkSize;
    if (chunkSize == 0) {
      dirAOffsets[g] = 0;
      dirASizes[g] = count / 2;
      dirBOffsets[g] = dirASizes[g];
      dirBSizes[g] = count - dirASizes[g];
      continue;
    }

    dirAOffsets[g] = static_cast<size_t>(numHelpers) * chunkSize;
    dirASizes[g] = chunkSize;
    dirBOffsets[g] = dirAOffsets[g] + dirASizes[g];
    dirBSizes[g] = count - dirBOffsets[g];
  }

  // For an active rank, compute its slot offsets and detect in-place.
  size_t gatherSlotOffset = 0; // recvBuff slot for the other rank (o x count)
  size_t diagSlotOffset = 0; // recvBuff slot for my own data (m x count)
  bool isInPlace = false;
  if (myActiveGroup >= 0 && sendCounts[myActiveGroup] > 0) {
    const ShardedRelayRankConfig& cfg = configs[myActiveGroup];
    size_t sendCount = sendCounts[myActiveGroup];
    diagSlotOffset = static_cast<size_t>(cfg.myActiveIndex) * sendCount;
    gatherSlotOffset = static_cast<size_t>(1 - cfg.myActiveIndex) * sendCount;

    // In-place when sendBuff aliases its own slot of recvBuff
    // (sendBuff == recvBuff + m x sendCount).
    const char* mySlotStart =
        static_cast<const char*>(recvBuffs[myActiveGroup]) +
        diagSlotOffset * elementSize;
    isInPlace =
        (static_cast<const void*>(sendBuffs[myActiveGroup]) ==
         static_cast<const void*>(mySlotStart));
  }

  // =========================================================================
  // DIAGONAL COPY: recvBuff[m x sendCount] = sendBuff
  // =========================================================================
  // Issued before the comm groups so it overlaps nothing that reads it.
  // In-place already has it (sendBuff aliases recvBuff[m x sendCount]).
  if (myActiveGroup >= 0 && sendCounts[myActiveGroup] > 0 && !isInPlace) {
    cudaMemcpyAsync(
        static_cast<char*>(recvBuffs[myActiveGroup]) +
            diagSlotOffset * elementSize,
        sendBuffs[myActiveGroup],
        sendCounts[myActiveGroup] * elementSize,
        cudaMemcpyDeviceToDevice,
        stream);
  }

  // =========================================================================
  // GROUP 1: relay scatter (active->helpers) || direct chunk A
  // (active<->active)
  // =========================================================================
  NCCLCHECK(ncclGroupStart());

  for (int g = 0; g < nGroups; g++) {
    if (sendCounts[g] == 0)
      continue;
    const ShardedRelayRankConfig& cfg = configs[g];
    size_t chunkSize = chunkSizes[g];

    if (cfg.isActiveRank) {
      const char* sendbuff = static_cast<const char*>(sendBuffs[g]);
      char* recvbuff = static_cast<char*>(recvBuffs[g]);

      // Scatter: chunk h of my sendBuff goes to helper h.
      for (int h = 0; h < cfg.numHelpers && chunkSize > 0; h++) {
        NCCLCHECK(ncclSend(
            sendbuff + static_cast<size_t>(h) * chunkSize * elementSize,
            chunkSize,
            datatype,
            cfg.helperRanks[h],
            comm,
            stream));
      }

      // Direct chunk A over the otherwise-idle active<->active link. The
      // gather slot never overlaps the send source, so this is safe in-place.
      int partner = cfg.activeRanks[1 - cfg.myActiveIndex];
      NCCLCHECK(ncclSend(
          sendbuff + dirAOffsets[g] * elementSize,
          dirASizes[g],
          datatype,
          partner,
          comm,
          stream));
      NCCLCHECK(ncclRecv(
          recvbuff + (gatherSlotOffset + dirAOffsets[g]) * elementSize,
          dirASizes[g],
          datatype,
          partner,
          comm,
          stream));
    } else if (chunkSize > 0) {
      // Helper: receive active rank a's chunk into slot a.
      char* helperBuf = static_cast<char*>(recvBuffs[g]);
      for (int a = 0; a < cfg.nActiveRanks; a++) {
        NCCLCHECK(ncclRecv(
            helperBuf + static_cast<size_t>(a) * chunkSize * elementSize,
            chunkSize,
            datatype,
            cfg.activeRanks[a],
            comm,
            stream));
      }
    }
  }

  NCCLCHECK(ncclGroupEnd());

  // =========================================================================
  // GROUP 2: relay forward (helpers->active) || direct chunk B
  // (active<->active)
  // =========================================================================
  // Helpers are pure passthrough: slot a goes to the OTHER active rank, landing
  // directly in that rank's gather slot. No reduction, no relay scratch.
  NCCLCHECK(ncclGroupStart());

  for (int g = 0; g < nGroups; g++) {
    if (sendCounts[g] == 0)
      continue;
    const ShardedRelayRankConfig& cfg = configs[g];
    size_t chunkSize = chunkSizes[g];

    if (cfg.isActiveRank) {
      char* recvbuff = static_cast<char*>(recvBuffs[g]);

      for (int h = 0; h < cfg.numHelpers && chunkSize > 0; h++) {
        NCCLCHECK(ncclRecv(
            recvbuff +
                (gatherSlotOffset + static_cast<size_t>(h) * chunkSize) *
                    elementSize,
            chunkSize,
            datatype,
            cfg.helperRanks[h],
            comm,
            stream));
      }

      // Direct chunk B, again over the idle active<->active link.
      int partner = cfg.activeRanks[1 - cfg.myActiveIndex];
      NCCLCHECK(ncclSend(
          static_cast<const char*>(sendBuffs[g]) + dirBOffsets[g] * elementSize,
          dirBSizes[g],
          datatype,
          partner,
          comm,
          stream));
      NCCLCHECK(ncclRecv(
          recvbuff + (gatherSlotOffset + dirBOffsets[g]) * elementSize,
          dirBSizes[g],
          datatype,
          partner,
          comm,
          stream));
    } else if (chunkSize > 0) {
      const char* helperBuf = static_cast<const char*>(recvBuffs[g]);
      for (int a = 0; a < cfg.nActiveRanks; a++) {
        NCCLCHECK(ncclSend(
            helperBuf + static_cast<size_t>(a) * chunkSize * elementSize,
            chunkSize,
            datatype,
            cfg.activeRanks[1 - a],
            comm,
            stream));
      }
    }
  }

  NCCLCHECK(ncclGroupEnd());

  return ncclSuccess;
}

/**
 * All-gather for > 2 active ranks (two-group scatter/forward relay).
 *
 * Every active SOURCE must deliver its sc elements to the A-1 other active
 * DESTs. Part of sc goes DIRECT over the 1-hop intra links (active<->active)
 * and part is OFFLOADED 2-hop through the otherwise-idle helper GPUs
 * (active->helper, then the helper forwards to every dest).
 *
 * Link accounting, per direction, in units of the per-(source,helper) chunk cs.
 * Note that on this topology a rank's helpers ARE the active ranks of another
 * group, so a rank's scatter and its helper-forward duty are egress on the SAME
 * cross links -- they add, they do not overlap. Hence two groups, not a
 * pipeline:
 *
 *   group 1:  cross = cs (scatter one slice per helper)   intra = direct part 1
 *   group 2:  cross = (A-1)*cs (forward every source's    intra = direct part 2
 *                      slice to every other dest)
 *
 * Balancing each group's cross and intra load gives direct part 1 = cs and
 * direct part 2 = (A-1)*cs, so the direct region is A*cs and the offload region
 * is H*cs: cs = sc/(A+H) -- eight equal units on the 8-GPU node, exactly as in
 * the 2-active path. The critical path is then cs + (A-1)*cs = A*cs = sc/2
 * against a pure-direct sc, a 2x ceiling.
 *
 * Buffer layout of sendBuff [0, sc): direct region [0, A*cs) whose first cs go
 * out in group 1 and whose remaining (A-1)*cs go out in group 2, then the
 * offload region [A*cs, sc) as H slices of cs, slice h owned by helper h.
 *
 * Helper scratch = recvBuffs[g], holding one cs slice per active source
 * (A*cs <= sc elements). Both in-place (sendBuff == recvBuff + m*sc) and
 * out-of-place are supported.
 *
 * At small sizes the 2-hop offload only adds a helper hop and a group boundary,
 * so below a threshold the offload is disabled and this degenerates to a
 * single-group pure-direct all-gather with the helpers idle.
 */
static ncclResult_t shardedRelayAllGatherFlat(
    const void* const* sendBuffs,
    void* const* recvBuffs,
    const size_t* sendCounts,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream,
    const ShardedRelayRankConfig* configs,
    int myActiveGroup,
    int numHelpers,
    int nActiveRanksPerGroup,
    int nGroups,
    size_t elementSize) {
  const int A = nActiveRanksPerGroup;
  const int H = numHelpers;

  // All groups march through the same schedule to keep XGMI traffic
  // phase-synced, so tune off the largest per-group message.
  size_t maxSc = 0;
  for (int g = 0; g < nGroups; g++) {
    if (sendCounts[g] > maxSc) {
      maxSc = sendCounts[g];
    }
  }
  const size_t maxBytes = maxSc * elementSize;

  // An independent call has the cross links to itself, so the offload pays off
  // earlier than in a fused call where every group is driving them at once.
  // Measured crossover (MI350X, A=4, bf16, 8 GPUs): a fused call has every
  // group driving the cross links at once so the offload needs ~12 MB to pay
  // off, while an independent call has them to itself and turns a profit from
  // ~8 MB.
  const size_t kOffloadMinBytes = (nGroups > 1)
      ? (static_cast<size_t>(12) << 20) // fused: >= 12 MB
      : (static_cast<size_t>(8) << 20); // independent: >= 8 MB
  // A == 2 only reaches this function in the small-message regime, where the
  // dedicated 2-active relay has already been ruled out; running the offload
  // there just re-adds the hop it was routed here to avoid.
  const bool useOffload = (H > 0) && (A > 2) && (maxBytes >= kOffloadMinBytes);

  // Per-group geometry. cs = sc/(A+H) aligned down; the direct region absorbs
  // the remainder so directSz + H*cs == sc.
  size_t csArr[SHARDED_RELAY_MAX_GROUPS];
  size_t directArr[SHARDED_RELAY_MAX_GROUPS];
  size_t d1Arr[SHARDED_RELAY_MAX_GROUPS]; // direct bytes sent in group 1
  for (int g = 0; g < nGroups; g++) {
    size_t sc = sendCounts[g];
    if (sc == 0) {
      csArr[g] = 0;
      directArr[g] = 0;
      d1Arr[g] = 0;
      continue;
    }
    size_t cs = useOffload ? (sc / static_cast<size_t>(A + H)) : 0;
    cs = (cs / CHUNK_ALIGN_ELEMENTS) * CHUNK_ALIGN_ELEMENTS;
    csArr[g] = cs;
    directArr[g] = sc - static_cast<size_t>(H) * cs;
    // Group 1's cross links carry one cs, group 2's carry (A-1)*cs, so split
    // the direct region the same way to keep both groups balanced.
    d1Arr[g] = (cs > 0) ? cs : directArr[g];
  }

  // In-place detection for the diagonal copy.
  bool isInPlace = false;
  size_t myDiagOffset = 0;
  if (myActiveGroup >= 0 && sendCounts[myActiveGroup] > 0) {
    const ShardedRelayRankConfig& cfg = configs[myActiveGroup];
    myDiagOffset =
        static_cast<size_t>(cfg.myActiveIndex) * sendCounts[myActiveGroup];
    const char* mySlot = static_cast<const char*>(recvBuffs[myActiveGroup]) +
        myDiagOffset * elementSize;
    isInPlace =
        (static_cast<const void*>(sendBuffs[myActiveGroup]) ==
         static_cast<const void*>(mySlot));
  }

  // Diagonal: recvBuff[m*sc] = sendBuff.
  if (myActiveGroup >= 0 && sendCounts[myActiveGroup] > 0 && !isInPlace) {
    cudaMemcpyAsync(
        static_cast<char*>(recvBuffs[myActiveGroup]) +
            myDiagOffset * elementSize,
        sendBuffs[myActiveGroup],
        sendCounts[myActiveGroup] * elementSize,
        cudaMemcpyDeviceToDevice,
        stream);
  }

  // =========================================================================
  // GROUP 1: direct part 1 (active<->active) || offload scatter
  // (active->helper)
  // =========================================================================
  NCCLCHECK(ncclGroupStart());
  for (int g = 0; g < nGroups; g++) {
    if (sendCounts[g] == 0)
      continue;
    const ShardedRelayRankConfig& cfg = configs[g];
    size_t sc = sendCounts[g];
    size_t cs = csArr[g];
    size_t directSz = directArr[g];
    size_t d1 = d1Arr[g];

    if (cfg.isActiveRank) {
      const char* sendbuff = static_cast<const char*>(sendBuffs[g]);
      char* recvbuff = static_cast<char*>(recvBuffs[g]);
      int m = cfg.myActiveIndex;

      if (d1 > 0) {
        for (int d = 0; d < A; d++) {
          if (d == m)
            continue;
          NCCLCHECK(ncclSend(
              sendbuff, d1, datatype, cfg.activeRanks[d], comm, stream));
        }
        for (int s = 0; s < A; s++) {
          if (s == m)
            continue;
          NCCLCHECK(ncclRecv(
              recvbuff + static_cast<size_t>(s) * sc * elementSize,
              d1,
              datatype,
              cfg.activeRanks[s],
              comm,
              stream));
        }
      }

      // Offload scatter: slice h of my offload region goes to helper h.
      for (int h = 0; h < cfg.numHelpers && cs > 0; h++) {
        NCCLCHECK(ncclSend(
            sendbuff + (directSz + static_cast<size_t>(h) * cs) * elementSize,
            cs,
            datatype,
            cfg.helperRanks[h],
            comm,
            stream));
      }
    } else if (cs > 0) {
      // Helper: receive each source's slice into slot s.
      char* hbuff = static_cast<char*>(recvBuffs[g]);
      for (int s = 0; s < cfg.nActiveRanks; s++) {
        NCCLCHECK(ncclRecv(
            hbuff + static_cast<size_t>(s) * cs * elementSize,
            cs,
            datatype,
            cfg.activeRanks[s],
            comm,
            stream));
      }
    }
  }
  NCCLCHECK(ncclGroupEnd());

  // When the offload is disabled the whole direct region went out in group 1,
  // so there is nothing left to do.
  bool anyGroup2 = false;
  for (int g = 0; g < nGroups; g++) {
    if (sendCounts[g] > 0 && csArr[g] > 0) {
      anyGroup2 = true;
      break;
    }
  }
  if (!anyGroup2) {
    return ncclSuccess;
  }

  // =========================================================================
  // GROUP 2: direct part 2 (active<->active) || offload forward
  // (helper->active)
  // =========================================================================
  NCCLCHECK(ncclGroupStart());
  for (int g = 0; g < nGroups; g++) {
    if (sendCounts[g] == 0 || csArr[g] == 0)
      continue;
    const ShardedRelayRankConfig& cfg = configs[g];
    size_t sc = sendCounts[g];
    size_t cs = csArr[g];
    size_t directSz = directArr[g];
    size_t d1 = d1Arr[g];
    size_t d2 = directSz - d1;

    if (cfg.isActiveRank) {
      const char* sendbuff = static_cast<const char*>(sendBuffs[g]);
      char* recvbuff = static_cast<char*>(recvBuffs[g]);
      int m = cfg.myActiveIndex;

      if (d2 > 0) {
        for (int d = 0; d < A; d++) {
          if (d == m)
            continue;
          NCCLCHECK(ncclSend(
              sendbuff + d1 * elementSize,
              d2,
              datatype,
              cfg.activeRanks[d],
              comm,
              stream));
        }
        for (int s = 0; s < A; s++) {
          if (s == m)
            continue;
          NCCLCHECK(ncclRecv(
              recvbuff + (static_cast<size_t>(s) * sc + d1) * elementSize,
              d2,
              datatype,
              cfg.activeRanks[s],
              comm,
              stream));
        }
      }

      // Offload forward: helper h delivers every other source's slice h.
      for (int h = 0; h < cfg.numHelpers; h++) {
        for (int s = 0; s < A; s++) {
          if (s == m)
            continue;
          NCCLCHECK(ncclRecv(
              recvbuff +
                  (static_cast<size_t>(s) * sc + directSz +
                   static_cast<size_t>(h) * cs) *
                      elementSize,
              cs,
              datatype,
              cfg.helperRanks[h],
              comm,
              stream));
        }
      }
    } else {
      // Helper: forward source s's slice to every dest other than s.
      const char* hbuff = static_cast<const char*>(recvBuffs[g]);
      for (int s = 0; s < cfg.nActiveRanks; s++) {
        for (int d = 0; d < A; d++) {
          if (d == s)
            continue;
          NCCLCHECK(ncclSend(
              hbuff + static_cast<size_t>(s) * cs * elementSize,
              cs,
              datatype,
              cfg.activeRanks[d],
              comm,
              stream));
        }
      }
    }
  }
  NCCLCHECK(ncclGroupEnd());

  return ncclSuccess;
}

/**
 * Fused Multi-Group Sharded Relay All-Gather.
 *
 * Executes multiple sharded relay all-gathers in one fused call, phase-synced
 * across all groups so XGMI links carry unidirectional traffic. Helpers are
 * pure passthrough. Each rank is ACTIVE for exactly one group and a HELPER for
 * the others. Both in-place and out-of-place are supported.
 *
 * nActiveRanksPerGroup must be a power of two (2 or 4): A==2 uses the original
 * 2-active path; A>2 uses the bandwidth-optimal flat scatter->forward path.
 */
HOT ncclResult_t ncclShardedRelayMultiGroupAllGatherImpl(
    const void* const* sendBuffs,
    void* const* recvBuffs,
    const size_t* sendCounts,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream,
    const int* const* allActiveRanks,
    int nActiveRanksPerGroup,
    int nGroups) {
  int nRanks, rank;
  NCCLCHECK(ncclCommCount(comm, &nRanks));
  NCCLCHECK(ncclCommUserRank(comm, &rank));

  // Validate every argument before touching sendCounts: the all-zero scan
  // below indexes sendCounts[0..nGroups), so a null pointer or an out-of-range
  // nGroups has to be rejected first. Bounds-checking nGroups up here also
  // means nGroups <= 0 reports ncclInvalidArgument rather than skipping the
  // scan entirely and returning ncclSuccess.
  if (nGroups < 1 || nGroups > SHARDED_RELAY_MAX_GROUPS) {
    return ncclInvalidArgument;
  }

  if (recvBuffs == nullptr || allActiveRanks == nullptr ||
      sendCounts == nullptr || sendBuffs == nullptr) {
    return ncclInvalidArgument;
  }

  // Require a power-of-two active-rank count (>= 2) for the XOR schedule.
  if (nActiveRanksPerGroup < 2 || !isPowerOfTwo(nActiveRanksPerGroup)) {
    return ncclInvalidArgument;
  }

  if (!isSupportedRelayDataType(datatype)) {
    return ncclInvalidArgument;
  }

  // Check if all sendCounts are zero
  bool allZero = true;
  for (int g = 0; g < nGroups; g++) {
    if (sendCounts[g] != 0) {
      allZero = false;
      break;
    }
  }
  if (allZero) {
    return ncclSuccess;
  }

  size_t elementSize = ncclTypeSize(datatype);

  // =========================================================================
  // BUILD RANK CONFIGURATION FOR ALL GROUPS
  // =========================================================================
  ShardedRelayRankConfig configs[SHARDED_RELAY_MAX_GROUPS];
  int myActiveGroup = -1; // Which group is this rank active for?

  for (int g = 0; g < nGroups; g++) {
    if (!buildShardedRelayRankConfig(
            nRanks,
            rank,
            allActiveRanks[g],
            nActiveRanksPerGroup,
            configs[g])) {
      return ncclInvalidArgument;
    }
    if (configs[g].isActiveRank) {
      myActiveGroup = g;
    }
  }

  // All groups should have the same number of helpers (same chunk structure)
  int numHelpers = configs[0].numHelpers;

  // Build per-group buffer arrays for the unchanged kernels. Helper groups use
  // their scratch (recvBuffs[g]); the active group uses the caller's contiguous
  // input/output buffers directly. All-gather may be in-place (sendBuffs[g]
  // aliases recvBuffs[g] + myActiveIndex*sendCount) or out-of-place.
  const void* sendBuffs2[SHARDED_RELAY_MAX_GROUPS];
  void* recvBuffs2[SHARDED_RELAY_MAX_GROUPS];
  for (int g = 0; g < nGroups; g++) {
    sendBuffs2[g] = sendBuffs[g];
    recvBuffs2[g] = recvBuffs[g];
  }

  ncclResult_t r;
  // Size-adaptive routing for A==2. The 2Active relay wins on bandwidth at
  // large sizes, but at small sizes it is latency-bound; the A-generic Flat
  // path (with its own small-size pure-direct mode) does a single-group direct
  // shard exchange with minimal latency. Route small A==2 to Flat, large to
  // 2Active. maxBytes (sendCount*elemSize) equals the bench per-rank input
  // shard label.
  size_t maxScAg = 0;
  for (int g = 0; g < nGroups; g++) {
    if (sendCounts[g] > maxScAg) {
      maxScAg = sendCounts[g];
    }
  }
  const size_t maxBytesAg = maxScAg * elementSize;
  // Crossover measured MI350X (bf16, 8 GPUs): fused 2Active relay overtakes
  // Flat at ~4.5 MB (Flat wins the small end, 576 KB 0.75x->1.31x), independent
  // at ~13.5 MB (0.34->0.51 @4KB). Independent has no cross-group contention so
  // direct holds on longer. Cross over below each.
  const size_t kAgA2PureDirectMaxBytes = (nGroups > 1)
      ? (static_cast<size_t>(2) << 20) // fused: < 2 MB
      : (static_cast<size_t>(9) << 20); // independent: < 9 MB
  const bool agA2UseFlat =
      (nActiveRanksPerGroup == 2) && (maxBytesAg < kAgA2PureDirectMaxBytes);
  if (nActiveRanksPerGroup == 2 && !agA2UseFlat) {
    r = shardedRelayAllGather2Active(
        sendBuffs2,
        recvBuffs2,
        sendCounts,
        datatype,
        comm,
        stream,
        configs,
        myActiveGroup,
        numHelpers,
        nGroups,
        elementSize);
  } else {
    // A>2, or small A==2: flat scatter->forward relay (dual of reduce-scatter)
    // with a size-adaptive pure-direct small-size mode.
    r = shardedRelayAllGatherFlat(
        sendBuffs2,
        recvBuffs2,
        sendCounts,
        datatype,
        comm,
        stream,
        configs,
        myActiveGroup,
        numHelpers,
        nActiveRanksPerGroup,
        nGroups,
        elementSize);
  }

  return r;
}
