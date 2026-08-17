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
  int numChunks = numHelpers + 1;

  // =========================================================================
  // CALCULATE PER-GROUP CHUNK SIZES (from the per-rank sendCount)
  // =========================================================================
  size_t chunkSizes[SHARDED_RELAY_MAX_GROUPS];
  size_t lastChunkSizes[SHARDED_RELAY_MAX_GROUPS];
  size_t directChunkOffsets[SHARDED_RELAY_MAX_GROUPS];
  size_t directChunkSizes[SHARDED_RELAY_MAX_GROUPS];

  for (int g = 0; g < nGroups; g++) {
    size_t count = sendCounts[g];

    // Skip groups with sendCount == 0; the per-phase loops below already check
    // sendCounts[g] == 0 and bypass NCCL ops for those groups.
    if (count == 0) {
      chunkSizes[g] = 0;
      lastChunkSizes[g] = 0;
      directChunkOffsets[g] = 0;
      directChunkSizes[g] = 0;
      continue;
    }

    // Calculate chunk size (aligned to CHUNK_ALIGN_ELEMENTS). When the
    // per-chunk size rounded down to CHUNK_ALIGN_ELEMENTS is zero, the send
    // buffer is too small to scatter and the caller should fall back to a
    // regular all-gather.
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

    // Direct exchange chunk info (within the sendCount-element send buffer)
    int directChunkIndex = numHelpers;
    directChunkOffsets[g] = static_cast<size_t>(directChunkIndex) * chunkSize;
    directChunkSizes[g] = lastChunkSize;
  }

  // For an active rank, compute its slot offsets and detect in-place.
  size_t gatherSlotOffset = 0; // recvBuff slot for the other rank (o x count)
  size_t diagSlotOffset = 0; // recvBuff slot for my own data (m x count)
  bool isInPlace = false;
  if (myActiveGroup >= 0 && sendCounts[myActiveGroup] > 0) {
    const ShardedRelayRankConfig& cfg = configs[myActiveGroup];
    size_t sendCount = sendCounts[myActiveGroup];
    int otherActiveIndex = 1 - cfg.myActiveIndex;
    diagSlotOffset = static_cast<size_t>(cfg.myActiveIndex) * sendCount;
    gatherSlotOffset = static_cast<size_t>(otherActiveIndex) * sendCount;

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
  // PHASE 1 (active->helpers): active ranks scatter their sendBuff chunks
  // =========================================================================
  // Helpers receive from each active rank into offset-based slots:
  //   slot a at offset (a x chunkSize) holds data from active rank a.
  NCCLCHECK(ncclGroupStart());

  for (int g = 0; g < nGroups; g++) {
    if (sendCounts[g] == 0)
      continue;
    const ShardedRelayRankConfig& cfg = configs[g];
    const void* sendbuff = sendBuffs[g];
    void* recvbuff = recvBuffs[g];
    size_t chunkSize = chunkSizes[g];

    if (cfg.isActiveRank) {
      // Active rank: send chunk h of my sendBuff to helper h
      for (int h = 0; h < cfg.numHelpers; h++) {
        int helperRank = cfg.helperRanks[h];
        size_t chunkOffset = static_cast<size_t>(h) * chunkSize;

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
  // The active rank receives all numHelpers chunks DIRECTLY into the OTHER
  // rank's slot of recvBuff (recvBuff[o x sendCount + h x chunkSize]). No
  // reduction and no relay scratch are required.
  NCCLCHECK(ncclGroupStart());

  for (int g = 0; g < nGroups; g++) {
    if (sendCounts[g] == 0)
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
      // other rank's slot of recvBuff at offset gatherSlotOffset + h x
      // chunkSize.
      for (int h = 0; h < cfg.numHelpers; h++) {
        int helperRank = cfg.helperRanks[h];
        size_t dstOffset =
            gatherSlotOffset + static_cast<size_t>(h) * chunkSize;
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
  // PHASE 3 (diagonal copy): recvBuff[m x sendCount] = sendBuff
  // =========================================================================
  // Place this rank's own contribution into its slot. In-place already has it
  // (sendBuff aliases recvBuff[m x sendCount]), so the copy is skipped.
  if (myActiveGroup >= 0 && sendCounts[myActiveGroup] > 0 && !isInPlace) {
    const void* sendbuff = sendBuffs[myActiveGroup];
    void* recvbuff = recvBuffs[myActiveGroup];
    size_t sendBytes = sendCounts[myActiveGroup] * elementSize;
    cudaMemcpyAsync(
        static_cast<char*>(recvbuff) + diagSlotOffset * elementSize,
        sendbuff,
        sendBytes,
        cudaMemcpyDeviceToDevice,
        stream);
  }

  // =========================================================================
  // PHASE 4 (active<->active): Direct exchange of the last chunk
  // =========================================================================
  // Active ranks exchange the direct chunk of their sendBuff; it is received
  // directly into the other rank's slot of recvBuff. The gather destination
  // (slot o) never overlaps the send source (slot m), so this is safe in-place.
  NCCLCHECK(ncclGroupStart());

  for (int g = 0; g < nGroups; g++) {
    if (sendCounts[g] == 0)
      continue;
    const ShardedRelayRankConfig& cfg = configs[g];

    if (cfg.isActiveRank) {
      const void* sendbuff = sendBuffs[g];
      void* recvbuff = recvBuffs[g];
      size_t directChunkOffset = directChunkOffsets[g];
      size_t directChunkSize = directChunkSizes[g];
      size_t recvDirectOffset = gatherSlotOffset + directChunkOffset;

      // Send my direct chunk to all other active ranks
      for (int a = 0; a < cfg.nActiveRanks; a++) {
        if (a == cfg.myActiveIndex)
          continue;
        int otherActiveRank = cfg.activeRanks[a];

        NCCLCHECK(ncclSend(
            static_cast<const char*>(sendbuff) +
                directChunkOffset * elementSize,
            directChunkSize,
            datatype,
            otherActiveRank,
            comm,
            stream));
      }

      // Receive direct chunks from all other active ranks directly into the
      // other rank's slot of recvBuff.
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

  // Phase 5: none. All-gather performs no reduction; the gathered and diagonal
  // slots are already in place in recvBuff.
  return ncclSuccess;
}

/**
 * All-gather for > 2 active ranks (flat scatter->forward relay, pipelined; the
 * dual of the reduce-scatter). Every active SOURCE delivers its sc to every
 * other active DEST's recvBuff slot: a DIRECT share over the 1-hop intra links
 * (active<->active), and an OFFLOAD share 2-hop through the idle helper GPUs
 * (active->helper, then helper broadcasts to the dests). The offload is sliced
 * into P stages and software-pipelined so the helper-forward of stage i
 * (helper->active) overlaps the active-send of stage i+1 (active->helper)
 * within one ncclGroup, recruiting the otherwise-idle helper links.
 *
 * Superstep schedule (one ncclGroup each; P+1 supersteps):
 *   k = 0:        S_0                  (send stage for slice 0 only)
 *   k = 1..P-1:   F_{k-1}  ||  S_k     (forward slice k-1 + send slice k)
 *   k = P:        F_{P-1}              (forward stage for last slice)
 * The group boundary between superstep k-1 and k guarantees S_{k-1} completed
 * before F_{k-1} reads it. Within a superstep, F_{k-1} reads helper slot
 * sub-slice k-1 while S_k writes sub-slice k (disjoint), so there is no hazard,
 * and every superstep is one ncclGroup with a fully matched send/recv set (no
 * deadlock). Slicing is plain even division (last part absorbs the remainder);
 * all slice sizes derive only from sendCounts[g], so they are identical across
 * ranks.
 *
 * Helper scratch = recvBuffs[g] for helper groups; it holds one offload chunk
 * of `cs` per active source (A*cs <= sc elements). Both in-place (sendBuff ==
 * recvBuff + m*sc) and out-of-place are supported.
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

  // Largest per-group message drives the size-adaptive tuning below. All groups
  // march through the same superstep count to keep XGMI traffic phase-synced,
  // so tune off the max.
  size_t maxSc = 0;
  for (int g = 0; g < nGroups; g++) {
    if (sendCounts[g] > maxSc) {
      maxSc = sendCounts[g];
    }
  }
  const size_t maxBytes = maxSc * elementSize;

  // Size-adaptive helper offload. The offload (2-hop active->helper->active)
  // share spreads traffic across otherwise-idle cross links and is a bandwidth
  // win only at large sizes, but it costs an extra group boundary + a hop of
  // latency. Below the threshold, disable it: the kernel degenerates to a
  // single-group pure-direct all-gather (each active rank exchanges its shard
  // directly with the other A-1 active ranks, helpers idle) -- the same
  // minimal-latency shape as all-to-all.
  //
  // The crossover depends on cross-link contention: a fused multi-group call
  // runs all groups phase-synced, so the offload cross links contend with the
  // other groups and only pay off at the very top end (>=256 MB); an
  // independent single-group call has those cross links free, so offload pays
  // off ~2x earlier (>=128 MB). Measured on MI350X (A=4, bf16, 8 GPUs): fused
  // offload ties/loses to pure-direct until 256 MB (1.03x @512MB), while
  // independent offload wins from 135 MB (1.10x) -- so cross over per-scenario.
  // Above the threshold the direct share is 429/1000 (~3/7), balancing the
  // intra direct link against the helper->dest link.
  const size_t kOffloadMinBytes = (nGroups > 1)
      ? (static_cast<size_t>(256) << 20) // fused: >= 256 MB
      : (static_cast<size_t>(128) << 20); // independent: >= 128 MB
  const bool useOffload = (H > 0) && (maxBytes >= kOffloadMinBytes);
  const size_t kOffPermille = useOffload ? (1000 - 429) : 0;

  // Pipeline depth. Only the offload path has a helper->active forward hop to
  // overlap against the next active->helper scatter hop; the pure-direct path
  // is a single group. In the offload regime the superstep loop runs P+1
  // ncclGroup boundaries (each ~25us of fixed launch+handshake tax); at these
  // large sizes the deep pipeline (P=16; P=32 regressed) amortizes that tax
  // against the per-slice transfer.
  const int P = useOffload ? 16 : 1;

  // Per-group geometry (same as the flat path): cs = per-source helper chunk
  // (128-aligned); directSz absorbs the remainder so directSz + H*cs == sc.
  size_t csArr[SHARDED_RELAY_MAX_GROUPS];
  size_t directArr[SHARDED_RELAY_MAX_GROUPS];
  for (int g = 0; g < nGroups; g++) {
    size_t sc = sendCounts[g];
    if (sc == 0) {
      csArr[g] = 0;
      directArr[g] = 0;
      continue;
    }
    size_t offTarget = (sc * kOffPermille) / 1000;
    size_t cs = (H > 0) ? (offTarget / static_cast<size_t>(H)) : 0;
    cs = (cs / CHUNK_ALIGN_ELEMENTS) * CHUNK_ALIGN_ELEMENTS;
    csArr[g] = cs;
    directArr[g] = sc - static_cast<size_t>(H) * cs;
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

  // Even-division slice geometry (last slice absorbs the remainder).
  auto sliceOff = [P](size_t total, int k) -> size_t {
    return static_cast<size_t>(k) * (total / static_cast<size_t>(P));
  };
  auto sliceLen = [P](size_t total, int k) -> size_t {
    size_t base = total / static_cast<size_t>(P);
    return (k == P - 1) ? (total - static_cast<size_t>(k) * base) : base;
  };

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

  // When offload is disabled (small messages), csArr is all zero and every
  // forward stage is a no-op, so the loop only needs the P direct-send
  // supersteps -- drop the trailing forward-only superstep to save one
  // boundary.
  bool anyOffload = false;
  for (int g = 0; g < nGroups; g++) {
    if (csArr[g] > 0) {
      anyOffload = true;
      break;
    }
  }
  const int lastK = anyOffload ? P : (P - 1);

  // Supersteps 0..lastK (one ncclGroup each).
  for (int k = 0; k <= lastK; k++) {
    NCCLCHECK(ncclGroupStart());
    for (int g = 0; g < nGroups; g++) {
      if (sendCounts[g] == 0)
        continue;
      const ShardedRelayRankConfig& cfg = configs[g];
      size_t sc = sendCounts[g];
      size_t cs = csArr[g];
      size_t directSz = directArr[g];

      // ---- S_k: send stage for slice k (active->helper + active<->active)
      // ----
      if (k < P) {
        size_t dOff = sliceOff(directSz, k);
        size_t dLen = sliceLen(directSz, k);
        size_t cOff = sliceOff(cs, k);
        size_t cLen = sliceLen(cs, k);
        if (cfg.isActiveRank) {
          const char* sendbuff = static_cast<const char*>(sendBuffs[g]);
          char* recvbuff = static_cast<char*>(recvBuffs[g]);
          int m = cfg.myActiveIndex;
          // Direct slice k: send to each dest, recv from each source (same
          // group -> csz==0-safe).
          if (dLen > 0) {
            for (int d = 0; d < A; d++) {
              if (d == m)
                continue;
              NCCLCHECK(ncclSend(
                  sendbuff + dOff * elementSize,
                  dLen,
                  datatype,
                  cfg.activeRanks[d],
                  comm,
                  stream));
            }
            for (int s = 0; s < A; s++) {
              if (s == m)
                continue;
              NCCLCHECK(ncclRecv(
                  recvbuff + (static_cast<size_t>(s) * sc + dOff) * elementSize,
                  dLen,
                  datatype,
                  cfg.activeRanks[s],
                  comm,
                  stream));
            }
          }
          // Offload sub-slice k: send to each helper once.
          if (cs > 0 && cLen > 0) {
            for (int h = 0; h < cfg.numHelpers; h++) {
              NCCLCHECK(ncclSend(
                  sendbuff +
                      (directSz + static_cast<size_t>(h) * cs + cOff) *
                          elementSize,
                  cLen,
                  datatype,
                  cfg.helperRanks[h],
                  comm,
                  stream));
            }
          }
        } else if (cs > 0 && cLen > 0) {
          // Helper: recv offload sub-slice k from each source into slot s.
          char* hbuff = static_cast<char*>(recvBuffs[g]);
          for (int s = 0; s < cfg.nActiveRanks; s++) {
            NCCLCHECK(ncclRecv(
                hbuff + (static_cast<size_t>(s) * cs + cOff) * elementSize,
                cLen,
                datatype,
                cfg.activeRanks[s],
                comm,
                stream));
          }
        }
      }

      // ---- F_{k-1}: forward stage for slice k-1 (helper->active) ----
      if (k >= 1 && cs > 0) {
        int kf = k - 1;
        size_t cOffF = sliceOff(cs, kf);
        size_t cLenF = sliceLen(cs, kf);
        if (cLenF > 0) {
          if (cfg.isActiveRank) {
            char* recvbuff = static_cast<char*>(recvBuffs[g]);
            int m = cfg.myActiveIndex;
            for (int h = 0; h < cfg.numHelpers; h++) {
              for (int s = 0; s < A; s++) {
                if (s == m)
                  continue;
                NCCLCHECK(ncclRecv(
                    recvbuff +
                        (static_cast<size_t>(s) * sc + directSz +
                         static_cast<size_t>(h) * cs + cOffF) *
                            elementSize,
                    cLenF,
                    datatype,
                    cfg.helperRanks[h],
                    comm,
                    stream));
              }
            }
          } else {
            const char* hbuff = static_cast<const char*>(recvBuffs[g]);
            for (int s = 0; s < cfg.nActiveRanks; s++) {
              for (int d = 0; d < A; d++) {
                if (d == s)
                  continue;
                NCCLCHECK(ncclSend(
                    hbuff + (static_cast<size_t>(s) * cs + cOffF) * elementSize,
                    cLenF,
                    datatype,
                    cfg.activeRanks[d],
                    comm,
                    stream));
              }
            }
          }
        }
      }
    }
    NCCLCHECK(ncclGroupEnd());
  }

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
      : (static_cast<size_t>(12) << 20); // independent: < 12 MB
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
