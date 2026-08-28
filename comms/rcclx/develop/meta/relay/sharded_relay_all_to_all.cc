/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "sharded_relay_all_to_all.h"
#include "comm.h"
#include "sharded_relay_route.h"

#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>

// GPU memory access alignment in elements for chunk size rounding.
// Distinct from the CPU CACHE_LINE_SIZE (64 bytes) defined in comm.h
// which is used for struct padding.
static constexpr size_t CHUNK_ALIGN_ELEMENTS =
    rcclx::relay::kRelayChunkAlignElements;

// The rank-config builder below is a deliberate copy of the file-local helper
// in sharded_relay_allreduce.cc (also mirrored in
// sharded_relay_reduce_scatter.cc). It is file-local there, so it cannot be
// linked across translation units; this TU re-declares its own copy in an
// anonymous namespace to keep it internal and ODR-safe. All-to-all performs no
// reduction, so NO reduction kernels are used.
namespace {

// Cached scratch buffer pool for kernel-owned relay helper staging (keyed by
// (device, stream, key), never shrinks, mutex-protected). A file-local copy of
// the pool in sharded_relay_allreduce.cc / _reduce_scatter.cc so callers can
// pass placeholder buffers for the groups where they are a helper.
class ScratchBufferCache {
 public:
  static ScratchBufferCache& getInstance() {
    static ScratchBufferCache instance;
    return instance;
  }

  void* get(int key, size_t requiredBytes, cudaStream_t stream) {
    if (requiredBytes == 0) {
      return nullptr;
    }
    int device;
    cudaGetDevice(&device);
    std::lock_guard<std::mutex> lock(mutex_);
    // Keyed by (device, stream, key). The stream is part of the key because two
    // relay collectives can run concurrently on one device on different streams
    // (independent communicators do exactly this): sharing one staging buffer
    // between them corrupts both. It also makes the stream-ordered free below
    // safe -- an entry is only ever read or written by the stream that owns it.
    auto& entry = buffers_[std::make_tuple(
        device, static_cast<const void*>(stream), key)];
    if (entry.buffer == nullptr || entry.size < requiredBytes) {
      if (entry.buffer != nullptr) {
        cudaFreeAsync(entry.buffer, stream);
      }
      size_t allocSize = requiredBytes;
      if (allocSize >= 1024 * 1024) {
        allocSize =
            ((requiredBytes + 64 * 1024 * 1024 - 1) / (64 * 1024 * 1024)) *
            (64 * 1024 * 1024);
      }
      cudaError_t err = cudaMallocAsync(&entry.buffer, allocSize, stream);
      if (err != cudaSuccess) {
        entry.buffer = nullptr;
        entry.size = 0;
        return nullptr;
      }
      entry.size = allocSize;
    }
    return entry.buffer;
  }

  ScratchBufferCache(const ScratchBufferCache&) = delete;
  ScratchBufferCache& operator=(const ScratchBufferCache&) = delete;

 private:
  ScratchBufferCache() = default;
  ~ScratchBufferCache() = default;
  struct BufferEntry {
    void* buffer = nullptr;
    size_t size = 0;
  };
  std::mutex mutex_;
  // (device, stream, group) -> grow-only staging buffer.
  std::map<std::tuple<int, const void*, int>, BufferEntry> buffers_;
};

static constexpr int kHelperScratchKeyBase = SHARDED_RELAY_MAX_GROUPS + 1;

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

struct A4RelayTask {
  int sourceIndex;
  int destIndex;
  int helperIndex;
  int helperSlot;
};

constexpr int A4_RELAY_TASKS = 12;
constexpr int A4_RELAY_TASKS_PER_HELPER = 3;

bool buildA4RelayTasks(
    const ShardedRelayRankConfig& config,
    A4RelayTask (&tasks)[A4_RELAY_TASKS]) {
  if (config.nActiveRanks != 4 || config.numHelpers != 4) {
    return false;
  }

  // p is a Latin permutation: h = source XOR p[dest]. Excluding the diagonal
  // leaves each helper with three distinct (source, destination) tasks and
  // three distinct destinations. helperIndex indexes config.helperRanks, so
  // every assignment is also checked against the actual helper set.
  constexpr int p[4] = {0, 2, 3, 1};
  int helperTaskCounts[4] = {0, 0, 0, 0};
  bool helperDestSeen[4][4] = {};
  int taskIndex = 0;
  for (int source = 0; source < 4; source++) {
    for (int dest = 0; dest < 4; dest++) {
      if (source == dest) {
        continue;
      }
      const int helperIndex = source ^ p[dest];
      if (helperIndex < 0 || helperIndex >= config.numHelpers ||
          config.helperRanks[helperIndex] < 0 ||
          helperDestSeen[helperIndex][dest]) {
        return false;
      }
      const int helperSlot = helperTaskCounts[helperIndex]++;
      if (helperSlot >= A4_RELAY_TASKS_PER_HELPER) {
        return false;
      }
      helperDestSeen[helperIndex][dest] = true;
      tasks[taskIndex++] = {source, dest, helperIndex, helperSlot};
    }
  }

  if (taskIndex != A4_RELAY_TASKS) {
    return false;
  }
  for (int helper = 0; helper < 4; helper++) {
    if (helperTaskCounts[helper] != A4_RELAY_TASKS_PER_HELPER) {
      return false;
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

  // Helper staging is kernel-owned scratch: callers pass a placeholder buffer
  // for groups where they are a helper. Each helper group holds nActiveRanks
  // chunks (one per active source) to receive and forward.
  void* helperScratch[SHARDED_RELAY_MAX_GROUPS] = {nullptr};
  for (int g = 0; g < nGroups; g++) {
    const ShardedRelayRankConfig& cfg = configs[g];
    if (!cfg.isActiveRank && segmentCounts[g] > 0 && chunkSizes[g] > 0) {
      size_t needBytes =
          static_cast<size_t>(cfg.nActiveRanks) * chunkSizes[g] * elementSize;
      helperScratch[g] = ScratchBufferCache::getInstance().get(
          kHelperScratchKeyBase + g, needBytes, stream);
      if (helperScratch[g] == nullptr) {
        return ncclInternalError;
      }
    }
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
      char* helperBuf = static_cast<char*>(helperScratch[g]);
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
      const char* helperBuf = static_cast<const char*>(helperScratch[g]);
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
 * Software-pipelined single-group 2-active sharded relay all-to-all.
 *
 * Same logical collective and the same helper roles as
 * shardedRelayAllToAll2Active, but for nGroups == 1 -- where the active ranks
 * and the helpers are disjoint sets -- the relay is tiled and pipelined so both
 * directions of every cross link stay busy. See relayPipelineTiles() for why
 * the two-group schedule cannot do that and what it costs.
 *
 * With T tiles and unit u = align(segmentCount / ((H+1)*T + 1)), the exchange
 * segment splits as:
 *   [0, H*T*u)   offload region; helper h owns [h*T*u, (h+1)*T*u), its tile t
 * at h*T*u + t*u [H*T*u, sc)  direct region as T+1 chunks of u, the last
 * absorbing the
 *                /((H+1)*T + 1) remainder and the alignment loss
 *
 * Group k, for k in [0, T]: the active rank scatters tile k (k < T) to every
 * helper, receives the partner's tile k-1 (k > 0) from every helper straight
 * into its receive segment, and exchanges direct chunk k over the
 * active<->active link; helper h receives tile k of each active's chunk into
 * ping-pong buffer k%2 and forwards buffer (k-1)%2 to the OTHER active rank.
 *
 * Every rank posts exactly one send and one recv per link per group. The
 * helper's staging is two units per active rather than the whole chunk, so a
 * forwarded tile is still cache-resident when it is read back.
 *
 * OUT-OF-PLACE ONLY, like every other all-to-all route.
 */
static ncclResult_t shardedRelayAllToAll2ActivePipelined(
    const void* const* sendBuffs,
    void* const* recvBuffs,
    const size_t* segmentCounts,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream,
    const ShardedRelayRankConfig* configs,
    int myActiveGroup,
    int numHelpers,
    int nTiles,
    size_t elementSize) {
  const ShardedRelayRankConfig& cfg = configs[0];
  const size_t sc = segmentCounts[0];
  const int H = numHelpers;
  const int T = nTiles;
  const size_t u =
      ((sc / (static_cast<size_t>(H + 1) * T + 1)) / CHUNK_ALIGN_ELEMENTS) *
      CHUNK_ALIGN_ELEMENTS;
  if (u == 0) {
    return ncclInvalidArgument;
  }
  const size_t tileStride = static_cast<size_t>(T) * u;
  const size_t directBase = static_cast<size_t>(H) * tileStride;
  const size_t lastDirect = sc - directBase - tileStride;

  // Diagonal: recvSeg[m] = sendSeg[m].
  size_t exchangeSegOffset = 0;
  if (myActiveGroup == 0) {
    const size_t diagOffset = static_cast<size_t>(cfg.myActiveIndex) * sc;
    exchangeSegOffset = static_cast<size_t>(1 - cfg.myActiveIndex) * sc;
    cudaMemcpyAsync(
        static_cast<char*>(recvBuffs[0]) + diagOffset * elementSize,
        static_cast<const char*>(sendBuffs[0]) + diagOffset * elementSize,
        sc * elementSize,
        cudaMemcpyDeviceToDevice,
        stream);
  }

  // Helper staging: two ping-pong units per active source.
  char* hbuff = nullptr;
  if (!cfg.isActiveRank) {
    hbuff = static_cast<char*>(ScratchBufferCache::getInstance().get(
        kHelperScratchKeyBase,
        static_cast<size_t>(cfg.nActiveRanks) * 2 * u * elementSize,
        stream));
    if (hbuff == nullptr) {
      return ncclInternalError;
    }
  }

  for (int k = 0; k <= T; k++) {
    NCCLCHECK(ncclGroupStart());
    if (cfg.isActiveRank) {
      const char* sendSeg = static_cast<const char*>(sendBuffs[0]) +
          exchangeSegOffset * elementSize;
      char* recvSeg =
          static_cast<char*>(recvBuffs[0]) + exchangeSegOffset * elementSize;
      const int partner = cfg.activeRanks[1 - cfg.myActiveIndex];
      const size_t directOffset = directBase + static_cast<size_t>(k) * u;
      const size_t directSize = (k < T) ? u : lastDirect;

      if (k < T) {
        for (int h = 0; h < H; h++) {
          NCCLCHECK(ncclSend(
              sendSeg +
                  (static_cast<size_t>(h) * tileStride +
                   static_cast<size_t>(k) * u) *
                      elementSize,
              u,
              datatype,
              cfg.helperRanks[h],
              comm,
              stream));
        }
      }
      NCCLCHECK(ncclSend(
          sendSeg + directOffset * elementSize,
          directSize,
          datatype,
          partner,
          comm,
          stream));
      if (k > 0) {
        for (int h = 0; h < H; h++) {
          NCCLCHECK(ncclRecv(
              recvSeg +
                  (static_cast<size_t>(h) * tileStride +
                   static_cast<size_t>(k - 1) * u) *
                      elementSize,
              u,
              datatype,
              cfg.helperRanks[h],
              comm,
              stream));
        }
      }
      NCCLCHECK(ncclRecv(
          recvSeg + directOffset * elementSize,
          directSize,
          datatype,
          partner,
          comm,
          stream));
    } else {
      if (k < T) {
        for (int a = 0; a < cfg.nActiveRanks; a++) {
          NCCLCHECK(ncclRecv(
              hbuff +
                  (static_cast<size_t>(a) * 2 + static_cast<size_t>(k % 2)) *
                      u * elementSize,
              u,
              datatype,
              cfg.activeRanks[a],
              comm,
              stream));
        }
      }
      if (k > 0) {
        for (int a = 0; a < cfg.nActiveRanks; a++) {
          NCCLCHECK(ncclSend(
              hbuff +
                  (static_cast<size_t>(a) * 2 +
                   static_cast<size_t>((k - 1) % 2)) *
                      u * elementSize,
              u,
              datatype,
              cfg.activeRanks[1 - a],
              comm,
              stream));
        }
      }
    }
    NCCLCHECK(ncclGroupEnd());
  }

  return ncclSuccess;
}

/**
 * A=4 no-pack XOR/Latin relay path.
 *
 * Each off-diagonal segment stays in place and is split into contiguous
 * directA, relay, and directB regions. The relay region takes two hops through
 * one of the four helpers; both direct regions take one hop to the destination.
 * The diagonal remains a local copy. OUT-OF-PLACE ONLY.
 */
static ncclResult_t shardedRelayAllToAllA4XorRelay(
    const void* const* sendBuffs,
    void* const* recvBuffs,
    const size_t* segmentCounts,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream,
    const ShardedRelayRankConfig* configs,
    int myActiveGroup,
    int nGroups,
    size_t elementSize) {
  size_t directACounts[SHARDED_RELAY_MAX_GROUPS];
  size_t relayCounts[SHARDED_RELAY_MAX_GROUPS];
  size_t directBCounts[SHARDED_RELAY_MAX_GROUPS];
  A4RelayTask tasks[SHARDED_RELAY_MAX_GROUPS][A4_RELAY_TASKS];

  for (int g = 0; g < nGroups; g++) {
    relayCounts[g] = rcclx::relay::allToAllA4RelayCount(segmentCounts[g]);
    directACounts[g] = relayCounts[g];
    directBCounts[g] = segmentCounts[g] - directACounts[g] - relayCounts[g];
    if (!buildA4RelayTasks(configs[g], tasks[g])) {
      return ncclInvalidArgument;
    }
  }

  // Keep the existing out-of-place diagonal copy unchanged.
  if (myActiveGroup >= 0) {
    const ShardedRelayRankConfig& cfg = configs[myActiveGroup];
    const size_t sc = segmentCounts[myActiveGroup];
    const size_t diagOffset = static_cast<size_t>(cfg.myActiveIndex) * sc;
    cudaMemcpyAsync(
        static_cast<char*>(recvBuffs[myActiveGroup]) + diagOffset * elementSize,
        static_cast<const char*>(sendBuffs[myActiveGroup]) +
            diagOffset * elementSize,
        sc * elementSize,
        cudaMemcpyDeviceToDevice,
        stream);
  }

  // Helper staging is kernel-owned scratch: callers pass a placeholder buffer
  // for groups where they are a helper. Each helper group holds its compact
  // relay slots (bounded by the segment count).
  void* helperScratch[SHARDED_RELAY_MAX_GROUPS] = {nullptr};
  for (int g = 0; g < nGroups; g++) {
    const ShardedRelayRankConfig& cfg = configs[g];
    if (!cfg.isActiveRank && relayCounts[g] > 0) {
      size_t needBytes = segmentCounts[g] * elementSize;
      helperScratch[g] = ScratchBufferCache::getInstance().get(
          kHelperScratchKeyBase + g, needBytes, stream);
      if (helperScratch[g] == nullptr) {
        return ncclInternalError;
      }
    }
  }

  // Phase 1 sends the first direct third to its final destination while the
  // middle third goes to its assigned helper's compact three-slot scratch.
  NCCLCHECK(ncclGroupStart());
  for (int g = 0; g < nGroups; g++) {
    const ShardedRelayRankConfig& cfg = configs[g];
    const size_t sc = segmentCounts[g];
    const size_t directA = directACounts[g];
    const size_t relay = relayCounts[g];
    for (const A4RelayTask& task : tasks[g]) {
      if (cfg.isActiveRank) {
        const int myIndex = cfg.myActiveIndex;
        if (task.sourceIndex == myIndex) {
          const char* sendSegment = static_cast<const char*>(sendBuffs[g]) +
              static_cast<size_t>(task.destIndex) * sc * elementSize;
          if (directA > 0) {
            NCCLCHECK(ncclSend(
                sendSegment,
                directA,
                datatype,
                cfg.activeRanks[task.destIndex],
                comm,
                stream));
          }
          if (relay > 0) {
            NCCLCHECK(ncclSend(
                sendSegment + directA * elementSize,
                relay,
                datatype,
                cfg.helperRanks[task.helperIndex],
                comm,
                stream));
          }
        }
        if (task.destIndex == myIndex) {
          char* recvSegment = static_cast<char*>(recvBuffs[g]) +
              static_cast<size_t>(task.sourceIndex) * sc * elementSize;
          if (directA > 0) {
            NCCLCHECK(ncclRecv(
                recvSegment,
                directA,
                datatype,
                cfg.activeRanks[task.sourceIndex],
                comm,
                stream));
          }
        }
      } else if (relay > 0 && task.helperIndex == cfg.myHelperIndex) {
        char* helperSlot = static_cast<char*>(helperScratch[g]) +
            static_cast<size_t>(task.helperSlot) * relay * elementSize;
        NCCLCHECK(ncclRecv(
            helperSlot,
            relay,
            datatype,
            cfg.activeRanks[task.sourceIndex],
            comm,
            stream));
      }
    }
  }
  NCCLCHECK(ncclGroupEnd());

  // Phase 2 sends the tail directly and forwards each compact helper slot into
  // recv[source * sc + directA]. Every byte is covered by the adjacent
  // [directA, relay, directB] regions; alignment loss and division tails are in
  // directB.
  NCCLCHECK(ncclGroupStart());
  for (int g = 0; g < nGroups; g++) {
    const ShardedRelayRankConfig& cfg = configs[g];
    const size_t sc = segmentCounts[g];
    const size_t directA = directACounts[g];
    const size_t relay = relayCounts[g];
    const size_t directB = directBCounts[g];
    for (const A4RelayTask& task : tasks[g]) {
      if (cfg.isActiveRank) {
        const int myIndex = cfg.myActiveIndex;
        if (task.sourceIndex == myIndex) {
          const char* sendSegment = static_cast<const char*>(sendBuffs[g]) +
              static_cast<size_t>(task.destIndex) * sc * elementSize;
          NCCLCHECK(ncclSend(
              sendSegment + (directA + relay) * elementSize,
              directB,
              datatype,
              cfg.activeRanks[task.destIndex],
              comm,
              stream));
        }
        if (task.destIndex == myIndex) {
          char* recvSegment = static_cast<char*>(recvBuffs[g]) +
              static_cast<size_t>(task.sourceIndex) * sc * elementSize;
          NCCLCHECK(ncclRecv(
              recvSegment + (directA + relay) * elementSize,
              directB,
              datatype,
              cfg.activeRanks[task.sourceIndex],
              comm,
              stream));
          if (relay > 0) {
            NCCLCHECK(ncclRecv(
                recvSegment + directA * elementSize,
                relay,
                datatype,
                cfg.helperRanks[task.helperIndex],
                comm,
                stream));
          }
        }
      } else if (relay > 0 && task.helperIndex == cfg.myHelperIndex) {
        const char* helperSlot = static_cast<const char*>(helperScratch[g]) +
            static_cast<size_t>(task.helperSlot) * relay * elementSize;
        NCCLCHECK(ncclSend(
            helperSlot,
            relay,
            datatype,
            cfg.activeRanks[task.destIndex],
            comm,
            stream));
      }
    }
  }
  NCCLCHECK(ncclGroupEnd());

  return ncclSuccess;
}

/**
 * Software-pipelined single-group A=4 XOR/Latin relay all-to-all.
 *
 * Same helper assignment as shardedRelayAllToAllA4XorRelay -- each (source,
 * dest) pair's relay region takes two hops through the one helper that the
 * Latin permutation gives it -- but for nGroups == 1, where the active ranks
 * and the helpers are disjoint, the relay is tiled so the scatter and the
 * forward share a group and each cross link runs duplex. See
 * relayPipelineTiles().
 *
 * The unpipelined schedule pays one relay unit on the cross links plus one
 * direct unit on the intra links in each of two groups, hence 2*sc/3. Merging
 * makes each of the T+1 groups carry one relay unit UP and one DOWN on every
 * cross link, matched by one direct unit on the intra links, so with u =
 * align(sc/(2*T + 1)) the cost is (T+1)*u: 2*sc/3 at T = 1 falling towards
 * sc/2.
 *
 * Segment layout for dest j:
 *   [0, T*u)      relay region, tile t at t*u, two hops via helper h(j)
 *   [T*u, sc)     direct region as T+1 chunks of u, the last absorbing the
 *                 /(2*T + 1) remainder and the alignment loss
 *
 * Each helper owns 3 tasks with three DISTINCT sources and three DISTINCT
 * destinations, so per group every rank posts exactly one send and one recv per
 * link per direction -- 3 relay plus 3 direct each way for an active rank, 3
 * each way for a helper.
 *
 * OUT-OF-PLACE ONLY, like every other all-to-all route.
 */
static ncclResult_t shardedRelayAllToAllA4XorRelayPipelined(
    const void* const* sendBuffs,
    void* const* recvBuffs,
    const size_t* segmentCounts,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream,
    const ShardedRelayRankConfig* configs,
    int myActiveGroup,
    int nTiles,
    size_t elementSize) {
  const ShardedRelayRankConfig& cfg = configs[0];
  const size_t sc = segmentCounts[0];
  const int T = nTiles;
  const size_t u =
      ((sc / (static_cast<size_t>(2) * T + 1)) / CHUNK_ALIGN_ELEMENTS) *
      CHUNK_ALIGN_ELEMENTS;
  if (u == 0) {
    return ncclInvalidArgument;
  }
  const size_t relayTotal = static_cast<size_t>(T) * u;
  const size_t lastDirect = sc - relayTotal - relayTotal;

  A4RelayTask tasks[A4_RELAY_TASKS];
  if (!buildA4RelayTasks(cfg, tasks)) {
    return ncclInvalidArgument;
  }

  // Diagonal stays a local copy.
  if (myActiveGroup == 0) {
    const size_t diagOffset = static_cast<size_t>(cfg.myActiveIndex) * sc;
    cudaMemcpyAsync(
        static_cast<char*>(recvBuffs[0]) + diagOffset * elementSize,
        static_cast<const char*>(sendBuffs[0]) + diagOffset * elementSize,
        sc * elementSize,
        cudaMemcpyDeviceToDevice,
        stream);
  }

  // Helper staging: two ping-pong units per task.
  char* hbuff = nullptr;
  if (!cfg.isActiveRank) {
    hbuff = static_cast<char*>(ScratchBufferCache::getInstance().get(
        kHelperScratchKeyBase,
        static_cast<size_t>(A4_RELAY_TASKS_PER_HELPER) * 2 * u * elementSize,
        stream));
    if (hbuff == nullptr) {
      return ncclInternalError;
    }
  }

  for (int k = 0; k <= T; k++) {
    NCCLCHECK(ncclGroupStart());
    const int m = cfg.myActiveIndex;
    const size_t directOffset = relayTotal + static_cast<size_t>(k) * u;
    const size_t directSize = (k < T) ? u : lastDirect;

    for (const A4RelayTask& task : tasks) {
      if (cfg.isActiveRank) {
        if (task.sourceIndex == m) {
          const char* sendSegment = static_cast<const char*>(sendBuffs[0]) +
              static_cast<size_t>(task.destIndex) * sc * elementSize;
          if (k < T) {
            NCCLCHECK(ncclSend(
                sendSegment + static_cast<size_t>(k) * u * elementSize,
                u,
                datatype,
                cfg.helperRanks[task.helperIndex],
                comm,
                stream));
          }
          NCCLCHECK(ncclSend(
              sendSegment + directOffset * elementSize,
              directSize,
              datatype,
              cfg.activeRanks[task.destIndex],
              comm,
              stream));
        }
        if (task.destIndex == m) {
          char* recvSegment = static_cast<char*>(recvBuffs[0]) +
              static_cast<size_t>(task.sourceIndex) * sc * elementSize;
          if (k > 0) {
            NCCLCHECK(ncclRecv(
                recvSegment + static_cast<size_t>(k - 1) * u * elementSize,
                u,
                datatype,
                cfg.helperRanks[task.helperIndex],
                comm,
                stream));
          }
          NCCLCHECK(ncclRecv(
              recvSegment + directOffset * elementSize,
              directSize,
              datatype,
              cfg.activeRanks[task.sourceIndex],
              comm,
              stream));
        }
      } else if (task.helperIndex == cfg.myHelperIndex) {
        char* slotBase =
            hbuff + static_cast<size_t>(task.helperSlot) * 2 * u * elementSize;
        if (k < T) {
          NCCLCHECK(ncclRecv(
              slotBase + static_cast<size_t>(k % 2) * u * elementSize,
              u,
              datatype,
              cfg.activeRanks[task.sourceIndex],
              comm,
              stream));
        }
        if (k > 0) {
          NCCLCHECK(ncclSend(
              slotBase + static_cast<size_t>((k - 1) % 2) * u * elementSize,
              u,
              datatype,
              cfg.activeRanks[task.destIndex],
              comm,
              stream));
        }
      }
    }
    NCCLCHECK(ncclGroupEnd());
  }

  return ncclSuccess;
}

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

  // The diagonal (recvSeg[m] = sendSeg[m]) rides along in the comm group below
  // as a P2P pair whose peer is the issuing rank: that is serviced as a local
  // copy inside the same kernel, so it costs no transfer and, unlike a separate
  // cudaMemcpyAsync, no second launch -- worth ~5 us, which at these sizes is
  // 10% of the whole call.
  //
  // Restricted to nGroups == 1. The same fold in the all-gather's fused case
  // made a fused routing-threshold test fail intermittently (once in four
  // runs); there all eight ranks issue a self pair alongside a real one in the
  // same group rather than just the active ones. Every sub-1x cell this targets
  // is single-group, so the fused route keeps the separate copy. Do not lift
  // this gate without running the fused suites repeatedly.
  const bool foldDiagonalIntoGroup = (nGroups == 1);

  if (!foldDiagonalIntoGroup && myActiveGroup >= 0 &&
      segmentCounts[myActiveGroup] > 0) {
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
      if (j == m && !foldDiagonalIntoGroup)
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
      if (s == m && !foldDiagonalIntoGroup)
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
 * nActiveRanksPerGroup must be a power of two (2 or 4). A==2 selects its direct
 * or dedicated 6-helper relay schedule by size. A==4 uses the deterministic
 * XOR/Latin helper schedule from its lower size bound up, and exact direct
 * below it.
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
  // Size-adaptive routing. The 2Active relay (scatter/forward/direct = 3 group
  // boundaries + a helper HBM round trip) wins on bandwidth at large sizes, but
  // at small sizes it is latency-bound; the A-generic Flat path does a
  // single-group pure-direct exchange (the active ranks swap their off-diagonal
  // segments directly, helpers idle) with minimal latency. The size -> route
  // mapping, including the A==4 XOR-relay window, lives in
  // selectAllToAllRoute() so the tests assert the same definition the
  // implementation dispatches on.
  const rcclx::relay::AllToAllRoute route = rcclx::relay::selectAllToAllRoute(
      nActiveRanksPerGroup, numHelpers, nGroups, segmentCounts, elementSize);
  if (route == rcclx::relay::AllToAllRoute::A2Relay) {
    // A single-group call has the helpers to itself, so the scatter and the
    // forward run on opposite directions of each cross link and can be
    // software-pipelined into one duplex stream; relayPipelineTiles() returns
    // 1 whenever that does not apply.
    const int nTiles = rcclx::relay::relayPipelineTiles(
        nGroups,
        rcclx::relay::relayShapeA2(numHelpers),
        rcclx::relay::relayMaxCount(segmentCounts, nGroups),
        elementSize);
    r = (nTiles > 1) ? shardedRelayAllToAll2ActivePipelined(
                           sendBuffs2,
                           recvBuffs2,
                           segmentCounts,
                           datatype,
                           comm,
                           stream,
                           configs,
                           myActiveGroup,
                           numHelpers,
                           nTiles,
                           elementSize)
                     : shardedRelayAllToAll2Active(
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
  } else if (route == rcclx::relay::AllToAllRoute::A4XorRelay) {
    // A single-group call has the helpers to itself, so the relay's two hops
    // run on opposite directions of each cross link and can be
    // software-pipelined into one duplex stream.
    const int nTiles = rcclx::relay::relayPipelineTiles(
        nGroups,
        rcclx::relay::kRelayShapeA4AllToAll,
        rcclx::relay::relayMaxCount(segmentCounts, nGroups),
        elementSize);
    r = (nTiles > 1) ? shardedRelayAllToAllA4XorRelayPipelined(
                           sendBuffs2,
                           recvBuffs2,
                           segmentCounts,
                           datatype,
                           comm,
                           stream,
                           configs,
                           myActiveGroup,
                           nTiles,
                           elementSize)
                     : shardedRelayAllToAllA4XorRelay(
                           sendBuffs2,
                           recvBuffs2,
                           segmentCounts,
                           datatype,
                           comm,
                           stream,
                           configs,
                           myActiveGroup,
                           nGroups,
                           elementSize);
  } else {
    // A==4 outside the routed window, or small A==2: exact direct all-to-all.
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
