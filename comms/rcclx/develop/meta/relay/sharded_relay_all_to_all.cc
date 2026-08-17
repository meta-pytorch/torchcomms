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
  int numChunks = numHelpers + 1;

  // =========================================================================
  // CALCULATE PER-GROUP CHUNK SIZES (from the per-segment segmentCount)
  // =========================================================================
  size_t chunkSizes[SHARDED_RELAY_MAX_GROUPS];
  size_t lastChunkSizes[SHARDED_RELAY_MAX_GROUPS];
  size_t directChunkOffsets[SHARDED_RELAY_MAX_GROUPS];
  size_t directChunkSizes[SHARDED_RELAY_MAX_GROUPS];

  for (int g = 0; g < nGroups; g++) {
    size_t count = segmentCounts[g];

    // Skip groups with segmentCount == 0; the per-phase loops below already
    // check segmentCounts[g] == 0 and bypass NCCL ops for those groups.
    if (count == 0) {
      chunkSizes[g] = 0;
      lastChunkSizes[g] = 0;
      directChunkOffsets[g] = 0;
      directChunkSizes[g] = 0;
      continue;
    }

    // Calculate chunk size (aligned to CHUNK_ALIGN_ELEMENTS). When the
    // per-chunk size rounded down to CHUNK_ALIGN_ELEMENTS is zero, the segment
    // is too small to scatter and the caller should fall back to a regular
    // all-to-all.
    size_t chunkSize = count / numChunks;
    chunkSize = (chunkSize / CHUNK_ALIGN_ELEMENTS) * CHUNK_ALIGN_ELEMENTS;
    if (chunkSize == 0) {
      return ncclInvalidArgument;
    }
    chunkSizes[g] = chunkSize;

    // Calculate the size of the last chunk (absorbs the remainder)
    size_t totalChunkedElements = static_cast<size_t>(numChunks) * chunkSize;
    size_t lastChunkSize = chunkSize;
    if (totalChunkedElements < count) {
      lastChunkSize = chunkSize + (count - totalChunkedElements);
    }
    lastChunkSizes[g] = lastChunkSize;

    // Direct exchange chunk info (within the single exchange segment)
    int directChunkIndex = numHelpers;
    directChunkOffsets[g] = static_cast<size_t>(directChunkIndex) * chunkSize;
    directChunkSizes[g] = lastChunkSize;
  }

  // For an active rank, compute its segment offsets up-front.
  size_t recvSegOffset = 0; // exchange segment in recvBuff (recvSeg[o])
  size_t sendSegOffset = 0; // exchange segment in sendBuff (sendSeg[o])
  size_t diagOffset = 0; // diagonal segment offset (m x segmentCount)
  if (myActiveGroup >= 0 && segmentCounts[myActiveGroup] > 0) {
    const ShardedRelayRankConfig& cfg = configs[myActiveGroup];
    size_t segmentCount = segmentCounts[myActiveGroup];
    int otherActiveIndex = 1 - cfg.myActiveIndex;
    diagOffset = static_cast<size_t>(cfg.myActiveIndex) * segmentCount;
    sendSegOffset = static_cast<size_t>(otherActiveIndex) * segmentCount;
    recvSegOffset = sendSegOffset; // exchange segment shares offset o x count
  }

  // =========================================================================
  // PHASE 1 (active->helpers): active ranks scatter their sendSeg[o] chunks
  // =========================================================================
  // Helpers receive from each active rank into offset-based slots:
  //   slot a at offset (a x chunkSize) holds data from active rank a.
  NCCLCHECK(ncclGroupStart());

  for (int g = 0; g < nGroups; g++) {
    if (segmentCounts[g] == 0)
      continue;
    const ShardedRelayRankConfig& cfg = configs[g];
    const void* sendbuff = sendBuffs[g];
    void* recvbuff = recvBuffs[g];
    size_t chunkSize = chunkSizes[g];

    if (cfg.isActiveRank) {
      // Active rank: send chunk h of my sendSeg[o] to helper h
      for (int h = 0; h < cfg.numHelpers; h++) {
        int helperRank = cfg.helperRanks[h];
        size_t chunkOffset = sendSegOffset + static_cast<size_t>(h) * chunkSize;

        NCCLCHECK(ncclSend(
            static_cast<const char*>(sendbuff) + chunkOffset * elementSize,
            chunkSize,
            datatype,
            helperRank,
            comm,
            stream));
      }
    } else {
      // Helper rank: receive from each active rank into slot a
      for (int a = 0; a < cfg.nActiveRanks; a++) {
        int activeRank = cfg.activeRanks[a];
        size_t helperOffset = static_cast<size_t>(a) * chunkSize;

        NCCLCHECK(ncclRecv(
            static_cast<char*>(recvbuff) + helperOffset * elementSize,
            chunkSize,
            datatype,
            activeRank,
            comm,
            stream));
      }
    }
  }

  NCCLCHECK(ncclGroupEnd());

  // =========================================================================
  // PHASE 2 (helpers->active, batched): Passthrough forward
  // =========================================================================
  // Each helper sends slot 0 (a0's data) -> a1 and slot 1 (a1's data) -> a0.
  // The active rank receives all numHelpers chunks DIRECTLY into the exchange
  // segment of recvBuff (recvSeg[o]) at offset recvSegOffset + h x chunkSize.
  // No reduction and no relay scratch are required.
  NCCLCHECK(ncclGroupStart());

  for (int g = 0; g < nGroups; g++) {
    if (segmentCounts[g] == 0)
      continue;
    const ShardedRelayRankConfig& cfg = configs[g];
    void* recvbuff = recvBuffs[g];
    size_t chunkSize = chunkSizes[g];

    if (!cfg.isActiveRank) {
      // Helper for group g: forward each slot to the OTHER active rank.
      for (int a = 0; a < cfg.nActiveRanks; a++) {
        int targetActive = cfg.activeRanks[1 - a];
        NCCLCHECK(ncclSend(
            static_cast<const char*>(recvbuff) +
                static_cast<size_t>(a) * chunkSize * elementSize,
            chunkSize,
            datatype,
            targetActive,
            comm,
            stream));
      }
    } else {
      // Active rank: receive forwarded data from each helper directly into the
      // exchange segment of recvBuff at offset recvSegOffset + h x chunkSize.
      for (int h = 0; h < cfg.numHelpers; h++) {
        int helperRank = cfg.helperRanks[h];
        size_t dstOffset = recvSegOffset + static_cast<size_t>(h) * chunkSize;
        NCCLCHECK(ncclRecv(
            static_cast<char*>(recvbuff) + dstOffset * elementSize,
            chunkSize,
            datatype,
            helperRank,
            comm,
            stream));
      }
    }
  }

  NCCLCHECK(ncclGroupEnd());

  // =========================================================================
  // PHASE 3 (diagonal copy): recvSeg[m] = sendSeg[m]
  // =========================================================================
  // The self segment is copied locally (out-of-place); both reside at offset
  // m x segmentCount in their respective buffers.
  if (myActiveGroup >= 0 && segmentCounts[myActiveGroup] > 0) {
    const void* sendbuff = sendBuffs[myActiveGroup];
    void* recvbuff = recvBuffs[myActiveGroup];
    size_t segmentBytes = segmentCounts[myActiveGroup] * elementSize;
    cudaMemcpyAsync(
        static_cast<char*>(recvbuff) + diagOffset * elementSize,
        static_cast<const char*>(sendbuff) + diagOffset * elementSize,
        segmentBytes,
        cudaMemcpyDeviceToDevice,
        stream);
  }

  // =========================================================================
  // PHASE 4 (active<->active): Direct exchange of the last chunk
  // =========================================================================
  // Active ranks exchange the direct chunk of the exchange segment; it is
  // received directly into recvSeg[o]'s direct-chunk slot (out-of-place).
  NCCLCHECK(ncclGroupStart());

  for (int g = 0; g < nGroups; g++) {
    if (segmentCounts[g] == 0)
      continue;
    const ShardedRelayRankConfig& cfg = configs[g];

    if (cfg.isActiveRank) {
      const void* sendbuff = sendBuffs[g];
      void* recvbuff = recvBuffs[g];
      size_t directChunkOffset = directChunkOffsets[g];
      size_t directChunkSize = directChunkSizes[g];
      size_t sendDirectOffset = sendSegOffset + directChunkOffset;
      size_t recvDirectOffset = recvSegOffset + directChunkOffset;

      // Send my direct chunk to all other active ranks
      for (int a = 0; a < cfg.nActiveRanks; a++) {
        if (a == cfg.myActiveIndex)
          continue;
        int otherActiveRank = cfg.activeRanks[a];

        NCCLCHECK(ncclSend(
            static_cast<const char*>(sendbuff) + sendDirectOffset * elementSize,
            directChunkSize,
            datatype,
            otherActiveRank,
            comm,
            stream));
      }

      // Receive direct chunks from all other active ranks directly into the
      // exchange segment of recvBuff (out-of-place, so no aliasing hazard).
      for (int a = 0; a < cfg.nActiveRanks; a++) {
        if (a == cfg.myActiveIndex)
          continue;
        int otherActiveRank = cfg.activeRanks[a];

        NCCLCHECK(ncclRecv(
            static_cast<char*>(recvbuff) + recvDirectOffset * elementSize,
            directChunkSize,
            datatype,
            otherActiveRank,
            comm,
            stream));
      }
    }
  }

  NCCLCHECK(ncclGroupEnd());

  // Phase 5: none. All-to-all performs no reduction; the exchange and diagonal
  // segments are already in place in recvBuff.
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
 * copied locally. NO helper relay is used -- the 2-hop offload that helps the
 * other collectives is a net loss for A>2 all-to-all (its serial
 * source->helper->owner path plus the helper HBM round-trip costs more than the
 * extra link-spreading buys, with only A helpers), and the fused multi-group
 * direct exchange already outruns NCCL's own P2P-bound 4-GPU all_to_all by a
 * wide margin (~1.4x). Helper ranks do no work for a group they are not active
 * in.
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
 * nActiveRanksPerGroup must be a power of two (2 or 4): A==2 uses the original
 * 2-active path (helper relay, which wins there); A>2 uses a pure-direct
 * all-to-all among the active ranks (no helper relay -- the 2-hop offload is a
 * net loss at A>2, see shardedRelayAllToAllFlat), so helper ranks do no work.
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
      ? (static_cast<size_t>(6) << 20) // fused: < 6 MB
      : (static_cast<size_t>(16) << 20); // independent: < 16 MB
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
