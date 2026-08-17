/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "sharded_relay_all_to_all.h"
#include "comm.h"

// GPU memory access alignment in elements for chunk size rounding.
// Distinct from the CPU CACHE_LINE_SIZE (64 bytes) defined in comm.h
// which is used for struct padding.
static constexpr size_t CHUNK_ALIGN_ELEMENTS = 128;

// The rank-config builder below is a deliberate copy of the file-local helper
// in sharded_relay_allreduce.cc (also mirrored in
// sharded_relay_reduce_scatter.cc). It is file-local there, so it cannot be
// linked across translation units; this TU re-declares its own copy in an
// anonymous namespace to keep it internal and ODR-safe. All-to-all performs no
// reduction, so NO reduction kernels are used.
namespace {

// Maximum number of helper ranks supported per group.
constexpr int SHARDED_RELAY_MAX_HELPERS = 8;

// Maximum number of active ranks per group. The flat all-to-all uses an XOR
// round schedule (round-r partner = myActiveIndex XOR r) that requires
// nActiveRanks to be a power of two; supported values are 2 and 4 (on an 8-GPU
// node this leaves 6 or 4 helpers respectively).
constexpr int SHARDED_RELAY_MAX_ACTIVE = 8;

// Returns true if v is a power of two (v >= 1).
inline bool isPowerOfTwo(int v) {
  return v > 0 && (v & (v - 1)) == 0;
}

/**
 * Rank Configuration for Sharded Relay All-to-All
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
 * Requires a power-of-two active-rank count in [2, SHARDED_RELAY_MAX_ACTIVE];
 * the XOR round schedule of the A>2 flat path depends on it.
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
  // [2, SHARDED_RELAY_MAX_ACTIVE]. The XOR round schedule depends on it.
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

// All-to-all only moves bytes, so it has no reduce kernels to dispatch on --
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
 * Two-active sharded relay all-to-all (original path).
 *
 * Each group has exactly 2 active ranks; the logical collective is a 2-rank
 * all-to-all between them, accelerated by passthrough helpers that relay
 * sharded chunks of a single exchange segment (segmentCounts[g] elements).
 * There is NO reduction: helpers and active ranks only move data, so no kernels
 * and no relay scratch are required. This is byte-for-byte unchanged from the
 * original implementation; the A>2 path lives in shardedRelayAllToAllFlat.
 *
 * Per active rank (index m, o = 1 - m), with segmentCount = segmentCounts[g]:
 *   - sendBuff/recvBuff hold 2 x segmentCount elements; segment i is at offset
 *     i x segmentCount.
 *   - Diagonal: recvBuff[m x segmentCount] = sendBuff[m x segmentCount].
 *   - Exchange: ship sendBuff[o x segmentCount] to the other rank; receive the
 *     other rank's segment into recvBuff[o x segmentCount].
 *
 * In-place is NOT supported (matches native RCCL ncclAllToAll): sendBuff and
 * recvBuff must be distinct (validated by the caller).
 */
static ncclResult_t shardedRelayAllToAll2Active(
    const void* const* sendBuffs,
    void* const* recvBuffs,
    const size_t* segmentCounts,
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
  // group, making the critical path 2*segmentCount/numChunks instead of the
  // 3*segmentCount/(numHelpers+1) of a separate direct phase.
  const int numChunks = numHelpers + 2;

  size_t chunkSizes[SHARDED_RELAY_MAX_GROUPS];
  size_t relayTotals[SHARDED_RELAY_MAX_GROUPS]; // == direct chunk A's offset
  size_t dirASizes[SHARDED_RELAY_MAX_GROUPS];
  size_t dirBOffsets[SHARDED_RELAY_MAX_GROUPS];
  size_t dirBSizes[SHARDED_RELAY_MAX_GROUPS]; // absorbs the remainder

  for (int g = 0; g < nGroups; g++) {
    size_t count = segmentCounts[g];

    // Zero-count groups are skipped by every loop below.
    if (count == 0) {
      chunkSizes[g] = 0;
      relayTotals[g] = 0;
      dirASizes[g] = 0;
      dirBOffsets[g] = 0;
      dirBSizes[g] = 0;
      continue;
    }

    size_t chunkSize = count / numChunks;
    chunkSize = (chunkSize / CHUNK_ALIGN_ELEMENTS) * CHUNK_ALIGN_ELEMENTS;
    chunkSizes[g] = chunkSize;
    relayTotals[g] = static_cast<size_t>(numHelpers) * chunkSize;
    if (chunkSize == 0) {
      dirASizes[g] = count / 2;
      dirBOffsets[g] = dirASizes[g];
    } else {
      dirASizes[g] = chunkSize;
      dirBOffsets[g] = relayTotals[g] + chunkSize;
    }
    dirBSizes[g] = count - dirBOffsets[g];
  }

  // For an active rank, compute its segment offsets up-front. The exchange
  // segment shares the same offset (o x segmentCount) in both buffers.
  size_t exchangeSegOffset = 0;
  size_t diagOffset = 0;
  if (myActiveGroup >= 0 && segmentCounts[myActiveGroup] > 0) {
    const ShardedRelayRankConfig& cfg = configs[myActiveGroup];
    size_t segmentCount = segmentCounts[myActiveGroup];
    diagOffset = static_cast<size_t>(cfg.myActiveIndex) * segmentCount;
    exchangeSegOffset =
        static_cast<size_t>(1 - cfg.myActiveIndex) * segmentCount;
  }

  // =========================================================================
  // DIAGONAL COPY: recvSeg[m] = sendSeg[m]
  // =========================================================================
  if (myActiveGroup >= 0 && segmentCounts[myActiveGroup] > 0) {
    cudaMemcpyAsync(
        static_cast<char*>(recvBuffs[myActiveGroup]) + diagOffset * elementSize,
        static_cast<const char*>(sendBuffs[myActiveGroup]) +
            diagOffset * elementSize,
        segmentCounts[myActiveGroup] * elementSize,
        cudaMemcpyDeviceToDevice,
        stream);
  }

  // =========================================================================
  // GROUP 1: relay scatter (active->helpers) || direct chunk A
  // (active<->active)
  // =========================================================================
  NCCLCHECK(ncclGroupStart());

  for (int g = 0; g < nGroups; g++) {
    if (segmentCounts[g] == 0)
      continue;
    const ShardedRelayRankConfig& cfg = configs[g];
    size_t chunkSize = chunkSizes[g];

    if (cfg.isActiveRank) {
      const char* sendSeg = static_cast<const char*>(sendBuffs[g]) +
          exchangeSegOffset * elementSize;
      char* recvSeg =
          static_cast<char*>(recvBuffs[g]) + exchangeSegOffset * elementSize;

      // Scatter: chunk h of my exchange segment goes to helper h.
      if (chunkSize > 0) {
        for (int h = 0; h < cfg.numHelpers; h++) {
          NCCLCHECK(ncclSend(
              sendSeg + static_cast<size_t>(h) * chunkSize * elementSize,
              chunkSize,
              datatype,
              cfg.helperRanks[h],
              comm,
              stream));
        }
      }

      // Direct chunk A over the otherwise-idle active<->active link.
      int partner = cfg.activeRanks[1 - cfg.myActiveIndex];
      NCCLCHECK(ncclSend(
          sendSeg + relayTotals[g] * elementSize,
          dirASizes[g],
          datatype,
          partner,
          comm,
          stream));
      NCCLCHECK(ncclRecv(
          recvSeg + relayTotals[g] * elementSize,
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
  // directly in that rank's exchange segment.
  NCCLCHECK(ncclGroupStart());

  for (int g = 0; g < nGroups; g++) {
    if (segmentCounts[g] == 0)
      continue;
    const ShardedRelayRankConfig& cfg = configs[g];
    size_t chunkSize = chunkSizes[g];

    if (cfg.isActiveRank) {
      const char* sendSeg = static_cast<const char*>(sendBuffs[g]) +
          exchangeSegOffset * elementSize;
      char* recvSeg =
          static_cast<char*>(recvBuffs[g]) + exchangeSegOffset * elementSize;

      if (chunkSize > 0) {
        for (int h = 0; h < cfg.numHelpers; h++) {
          NCCLCHECK(ncclRecv(
              recvSeg + static_cast<size_t>(h) * chunkSize * elementSize,
              chunkSize,
              datatype,
              cfg.helperRanks[h],
              comm,
              stream));
        }
      }

      // Direct chunk B, again over the idle active<->active link.
      int partner = cfg.activeRanks[1 - cfg.myActiveIndex];
      NCCLCHECK(ncclSend(
          sendSeg + dirBOffsets[g] * elementSize,
          dirBSizes[g],
          datatype,
          partner,
          comm,
          stream));
      NCCLCHECK(ncclRecv(
          recvSeg + dirBOffsets[g] * elementSize,
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
 * All-to-all for > 2 active ranks (flat, pure-direct 1-hop).
 *
 * All-to-all is a permutation, not a reduction. Every (source m, owner j) pair
 * carries a distinct segment; each active rank's sendBuff/recvBuff hold A
 * segments of segmentCount (sendSeg[j] -> owner j, recvSeg[s] <- source s).
 *
 * For A>2 this is a plain direct all-to-all among the active ranks over the
 * 1-hop intra links: each active sends sendSeg[j] straight to owner j and recvs
 * source s's segment into recvSeg[s]; the diagonal recvSeg[m] = sendSeg[m] is
 * copied locally. NO helper relay is used.
 *
 * The 2-hop helper offload that pays off for the other A>2 collectives was
 * implemented and measured here, and it loses. Its link model promises 1.67x
 * (direct 2*(A-1)*cs + offload H*cs per segment, both groups balanced at
 * (A-1)*cs per link), but a permutation gives the helper A*(A-1) = 12 distinct
 * (dest, source) chunks per group with no reduction to amortize them, and the
 * resulting p2p op count costs more than the extra link-spreading buys:
 * measured on MI350X (A=4, bf16, 8 GPUs) it ran 0.75-0.97x against pure-direct
 * from 13.5 MB to 135 MB and only reached 1.03-1.07x at 256 MB-1 GB. Coalescing
 * the helper's sends per dest (its slots are already grouped by dest) would
 * need a gather kernel on the send side and a scatter kernel on the recv side,
 * since both ends are strided by segmentCount; that is the open lead if this is
 * revisited.
 *
 * OUT-OF-PLACE ONLY (sendBuff != recvBuff, validated by the caller). Requires a
 * power-of-two active count. (The 2-active path still uses the helper relay,
 * which wins there thanks to 6 dedicated helpers per group.)
 */
static ncclResult_t shardedRelayAllToAllFlat(
    const void* const* sendBuffs,
    void* const* recvBuffs,
    const size_t* segmentCounts,
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
  (void)numHelpers; // pure-direct: no helper relay for A>2 all-to-all.

  // Diagonal copy recvSeg[m] = sendSeg[m] (active group only, out-of-place).
  if (myActiveGroup >= 0 && segmentCounts[myActiveGroup] > 0) {
    const ShardedRelayRankConfig& cfg = configs[myActiveGroup];
    size_t sc = segmentCounts[myActiveGroup];
    size_t diagOffset = static_cast<size_t>(cfg.myActiveIndex) * sc;
    cudaMemcpyAsync(
        static_cast<char*>(recvBuffs[myActiveGroup]) + diagOffset * elementSize,
        static_cast<const char*>(sendBuffs[myActiveGroup]) +
            diagOffset * elementSize,
        sc * elementSize,
        cudaMemcpyDeviceToDevice,
        stream);
  }

  // Direct all-to-all: each active sends sendSeg[j] to owner j and recvs source
  // s's segment into recvSeg[s] over the 1-hop intra links, in one ncclGroup
  // (send AND recv together so the exchange cannot deadlock). A rank that is a
  // helper (not active) for a group does nothing for it.
  NCCLCHECK(ncclGroupStart());
  for (int g = 0; g < nGroups; g++) {
    if (segmentCounts[g] == 0)
      continue;
    const ShardedRelayRankConfig& cfg = configs[g];
    if (!cfg.isActiveRank)
      continue;
    size_t sc = segmentCounts[g];
    const char* sendbuff = static_cast<const char*>(sendBuffs[g]);
    char* recvbuff = static_cast<char*>(recvBuffs[g]);
    int m = cfg.myActiveIndex;
    for (int j = 0; j < A; j++) {
      if (j == m)
        continue;
      NCCLCHECK(ncclSend(
          sendbuff + static_cast<size_t>(j) * sc * elementSize,
          sc,
          datatype,
          cfg.activeRanks[j],
          comm,
          stream));
    }
    for (int s = 0; s < A; s++) {
      if (s == m)
        continue;
      NCCLCHECK(ncclRecv(
          recvbuff + static_cast<size_t>(s) * sc * elementSize,
          sc,
          datatype,
          cfg.activeRanks[s],
          comm,
          stream));
    }
  }
  NCCLCHECK(ncclGroupEnd());

  return ncclSuccess;
}

/**
 * Fused Multi-Group Sharded Relay All-to-All.
 *
 * Executes multiple sharded relay all-to-alls in one fused call. Each rank is
 * ACTIVE for exactly one group and a HELPER for the others. In-place is NOT
 * supported (matches native RCCL ncclAllToAll).
 *
 * nActiveRanksPerGroup must be a power of two (2 or 4): A==2 uses the dedicated
 * 2-active relay (6 helpers per group), A>2 a pure-direct all-to-all among the
 * active ranks -- see shardedRelayAllToAllFlat for why the helper offload loses
 * for a permutation.
 */
HOT ncclResult_t ncclShardedRelayMultiGroupAllToAllImpl(
    const void* const* sendBuffs,
    void* const* recvBuffs,
    const size_t* segmentCounts,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream,
    const int* const* allActiveRanks,
    int nActiveRanksPerGroup,
    int nGroups) {
  int nRanks, rank;
  NCCLCHECK(ncclCommCount(comm, &nRanks));
  NCCLCHECK(ncclCommUserRank(comm, &rank));

  // Validate every argument before touching segmentCounts: the all-zero scan
  // below indexes segmentCounts[0..nGroups), so a null pointer or an
  // out-of-range nGroups has to be rejected first. Bounds-checking nGroups up
  // here also means nGroups <= 0 reports ncclInvalidArgument rather than
  // skipping the scan entirely and returning ncclSuccess.
  if (nGroups < 1 || nGroups > SHARDED_RELAY_MAX_GROUPS) {
    return ncclInvalidArgument;
  }

  if (recvBuffs == nullptr || allActiveRanks == nullptr ||
      segmentCounts == nullptr || sendBuffs == nullptr) {
    return ncclInvalidArgument;
  }

  // Require a power-of-two active-rank count (>= 2) for the XOR schedule.
  if (nActiveRanksPerGroup < 2 || !isPowerOfTwo(nActiveRanksPerGroup)) {
    return ncclInvalidArgument;
  }

  if (!isSupportedRelayDataType(datatype)) {
    return ncclInvalidArgument;
  }

  // Check if all segmentCounts are zero
  bool allZero = true;
  for (int g = 0; g < nGroups; g++) {
    if (segmentCounts[g] != 0) {
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

  // In-place is NOT supported for all-to-all: reject when the active group's
  // input and output alias (matches native ncclAllToAll).
  if (myActiveGroup >= 0 && segmentCounts[myActiveGroup] > 0 &&
      sendBuffs[myActiveGroup] == recvBuffs[myActiveGroup]) {
    return ncclInvalidArgument;
  }

  // Build per-group buffer arrays for the unchanged kernels. Helper groups use
  // their scratch (recvBuffs[g]); the active group uses the caller's contiguous
  // input/output buffers directly (out-of-place).
  const void* sendBuffs2[SHARDED_RELAY_MAX_GROUPS];
  void* recvBuffs2[SHARDED_RELAY_MAX_GROUPS];
  for (int g = 0; g < nGroups; g++) {
    sendBuffs2[g] = sendBuffs[g];
    recvBuffs2[g] = recvBuffs[g];
  }

  ncclResult_t r;
  // Size-adaptive routing for A==2. The 2Active relay (scatter/forward/direct =
  // 3 group boundaries + a helper HBM round trip) wins on bandwidth at large
  // sizes, but at small sizes it is latency-bound; the A-generic Flat path does
  // a single-group pure-direct exchange (the two active ranks swap their off-
  // diagonal segment directly, helpers idle) with minimal latency. Route small
  // A==2 to Flat, large to the relay. maxBytes (A*segment*elemSize) equals the
  // bench per-rank input label; crossover set nGroups-aware below.
  size_t maxSeg = 0;
  for (int g = 0; g < nGroups; g++) {
    if (segmentCounts[g] > maxSeg) {
      maxSeg = segmentCounts[g];
    }
  }
  const size_t maxBytes =
      static_cast<size_t>(nActiveRanksPerGroup) * maxSeg * elementSize;
  // Crossover measured MI350X (bf16, 8 GPUs): fused relay overtakes Flat at
  // ~9 MB (Flat wins the small end big, e.g. 576 KB 0.92x->1.45x), independent
  // at ~27 MB (0.38->0.53 @4KB). Independent has no cross-group contention so
  // direct holds on longer. Cross over below each.
  const size_t kA2PureDirectMaxBytes = (nGroups > 1)
      ? (static_cast<size_t>(2) << 20) // fused: < 2 MB
      : (static_cast<size_t>(6) << 20); // independent: < 6 MB
  const bool a2UseFlat =
      (nActiveRanksPerGroup == 2) && (maxBytes < kA2PureDirectMaxBytes);
  if (nActiveRanksPerGroup == 2 && !a2UseFlat) {
    r = shardedRelayAllToAll2Active(
        sendBuffs2,
        recvBuffs2,
        segmentCounts,
        datatype,
        comm,
        stream,
        configs,
        myActiveGroup,
        numHelpers,
        nGroups,
        elementSize);
  } else {
    // A>2, or small A==2: flat single-group pure-direct all-to-all.
    r = shardedRelayAllToAllFlat(
        sendBuffs2,
        recvBuffs2,
        segmentCounts,
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
