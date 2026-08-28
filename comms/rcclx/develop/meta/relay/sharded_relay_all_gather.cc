/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "sharded_relay_all_gather.h"
#include "comm.h"
#include "sharded_relay_graph_scratch.h"
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
// in sharded_relay_allreduce.cc (also mirrored in the reduce-scatter and
// all-to-all relays). It is file-local there, so it cannot be linked across
// translation units; this TU re-declares its own copy in an anonymous namespace
// to keep it internal and ODR-safe. All-gather performs no reduction, so NO
// reduction kernels are used, and the relay paths land every byte directly in
// recvBuff / the helper passthrough buffers (no working-buffer scratch).
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

    // A capturing stream must not use this cache. hipMallocAsync would record
    // an allocation node whose address is only valid while the graph runs, and
    // a later growth would hipFreeAsync a pointer this graph has already baked
    // in. Captures get a graph-scoped buffer instead; see
    // sharded_relay_graph_scratch.h.
    //
    // Ahead of the lock on purpose: this is a HIP runtime call, and there is no
    // reason to hold a process-wide mutex across it while relay collectives run
    // concurrently on other streams. Measured either way it is in the noise, so
    // this is hygiene rather than a fix for anything.
    struct ncclCudaGraph graph;
    if (ncclCudaGetCapturingGraph(&graph, stream) != ncclSuccess) {
      return nullptr;
    }
    if (ncclCudaGraphValid(graph)) {
      return rcclx::relay::graphScratchGet(
          this, key, requiredBytes, stream, graph);
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

  // Helper staging is kernel-owned scratch: callers pass a placeholder buffer
  // for groups where they are a helper. Each helper group holds nActiveRanks
  // chunks (one per active source) to receive and forward.
  void* helperScratch[SHARDED_RELAY_MAX_GROUPS] = {nullptr};
  for (int g = 0; g < nGroups; g++) {
    const ShardedRelayRankConfig& cfg = configs[g];
    if (!cfg.isActiveRank && sendCounts[g] > 0 && chunkSizes[g] > 0) {
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
 * Software-pipelined single-group 2-active sharded relay all-gather.
 *
 * Same logical collective and the same helper roles as
 * shardedRelayAllGather2Active, but for nGroups == 1 -- where the active ranks
 * and the helpers are disjoint sets -- the relay is tiled and pipelined so both
 * directions of every cross link stay busy. See relayPipelineTiles() for why
 * the two-group schedule cannot do that and what it costs.
 *
 * With T tiles and unit u = align(sendCount / ((H+1)*T + 1)):
 *   sendBuff [0, H*T*u)   offload region; helper h owns [h*T*u, (h+1)*T*u),
 *                         its tile t at h*T*u + t*u
 *   sendBuff [H*T*u, sc)  direct region as T+1 chunks of u, the last absorbing
 *                         the /((H+1)*T + 1) remainder and the alignment loss
 *
 * Group k, for k in [0, T]:
 *   active   scatter tile k (k < T) to every helper, receive the partner's
 *            tile k-1 (k > 0) from every helper straight into the gather slot,
 *            and exchange direct chunk k over the active<->active link.
 *   helper h receive tile k of each active's chunk into ping-pong buffer k%2,
 *            forward buffer (k-1)%2 to the OTHER active.
 *
 * Every rank therefore posts exactly one send and one recv per link per group,
 * which is also the best case for p2p channel assignment. The helper's staging
 * is two units per active rather than the whole chunk, so a forwarded tile is
 * still cache-resident when it is read back.
 *
 * Both in-place and out-of-place are supported: the gather slot never overlaps
 * the send source.
 */
static ncclResult_t shardedRelayAllGather2ActivePipelined(
    const void* const* sendBuffs,
    void* const* recvBuffs,
    const size_t* sendCounts,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream,
    const ShardedRelayRankConfig* configs,
    int myActiveGroup,
    int numHelpers,
    int nTiles,
    size_t elementSize) {
  const ShardedRelayRankConfig& cfg = configs[0];
  const size_t sc = sendCounts[0];
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

  // Diagonal: recvBuff[m*sc] = sendBuff, a no-op when in-place.
  size_t gatherSlot = 0;
  if (myActiveGroup == 0) {
    const size_t diagSlot = static_cast<size_t>(cfg.myActiveIndex) * sc;
    gatherSlot = static_cast<size_t>(1 - cfg.myActiveIndex) * sc;
    char* diag = static_cast<char*>(recvBuffs[0]) + diagSlot * elementSize;
    if (static_cast<const void*>(sendBuffs[0]) !=
        static_cast<const void*>(diag)) {
      cudaMemcpyAsync(
          diag,
          sendBuffs[0],
          sc * elementSize,
          cudaMemcpyDeviceToDevice,
          stream);
    }
  }

  // Helper staging: two ping-pong units per active source. Tiny by design, so a
  // tile is forwarded out of cache rather than re-read from HBM.
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
      const char* sendbuff = static_cast<const char*>(sendBuffs[0]);
      char* recvbuff = static_cast<char*>(recvBuffs[0]);
      const int partner = cfg.activeRanks[1 - cfg.myActiveIndex];
      const size_t directOffset = directBase + static_cast<size_t>(k) * u;
      const size_t directSize = (k < T) ? u : lastDirect;

      if (k < T) {
        for (int h = 0; h < H; h++) {
          NCCLCHECK(ncclSend(
              sendbuff +
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
          sendbuff + directOffset * elementSize,
          directSize,
          datatype,
          partner,
          comm,
          stream));
      if (k > 0) {
        for (int h = 0; h < H; h++) {
          NCCLCHECK(ncclRecv(
              recvbuff +
                  (gatherSlot + static_cast<size_t>(h) * tileStride +
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
          recvbuff + (gatherSlot + directOffset) * elementSize,
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

  // An independent call has the cross links to itself, so the offload pays off
  // earlier than in a fused call where every group is driving them at once. The
  // size -> route mapping lives in selectAllGatherRoute() so the tests assert
  // the same definition this dispatch uses; it also encodes why A == 2 never
  // takes the offload (it only reaches this function in the small-message
  // regime, where the dedicated 2-active relay has already been ruled out, so
  // offloading would just re-add the hop it was routed here to avoid).
  const bool useOffload = rcclx::relay::selectAllGatherRoute(
                              A, H, nGroups, sendCounts, elementSize) ==
      rcclx::relay::AllGatherRoute::FlatOffload;

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
  //
  // With the offload disabled the whole shard goes out in one group, so the
  // diagonal can ride along in it as a P2P pair whose peer is the issuing rank:
  // that is serviced as a local copy inside the same kernel, costing no
  // transfer and, unlike this cudaMemcpyAsync, no second launch. With the
  // offload enabled the direct region is only d1 of the sc elements, so the
  // diagonal does not correspond to any single operation in the group and has
  // to stay a separate copy.
  //
  // Restricted to nGroups == 1. Folding it in the FUSED case made
  // Correctness_4Groups_A2_FusedRoutingThresholds fail intermittently (once in
  // four runs) with mismatched slots. There, all eight ranks issue a self pair
  // alongside a real one in the same group, rather than just the two active
  // ranks; whether that is an RCCL self-P2P ordering issue or something else
  // was not chased, because every sub-1x cell this targets is single-group and
  // the fused route has its own tuned reference. Do not lift this gate without
  // running the fused suite repeatedly -- a single green run is not evidence.
  const bool foldDiagonalIntoGroup = !useOffload && !isInPlace && nGroups == 1;
  if (myActiveGroup >= 0 && sendCounts[myActiveGroup] > 0 && !isInPlace &&
      !foldDiagonalIntoGroup) {
    cudaMemcpyAsync(
        static_cast<char*>(recvBuffs[myActiveGroup]) +
            myDiagOffset * elementSize,
        sendBuffs[myActiveGroup],
        sendCounts[myActiveGroup] * elementSize,
        cudaMemcpyDeviceToDevice,
        stream);
  }

  // Helper staging is kernel-owned scratch: callers pass a placeholder buffer
  // for groups where they are a helper. Each helper group holds A slices of cs
  // (one per active source, A*cs <= sc).
  void* helperScratch[SHARDED_RELAY_MAX_GROUPS] = {nullptr};
  for (int g = 0; g < nGroups; g++) {
    const ShardedRelayRankConfig& cfg = configs[g];
    if (!cfg.isActiveRank && sendCounts[g] > 0 && csArr[g] > 0) {
      size_t needBytes =
          static_cast<size_t>(cfg.nActiveRanks) * csArr[g] * elementSize;
      helperScratch[g] = ScratchBufferCache::getInstance().get(
          kHelperScratchKeyBase + g, needBytes, stream);
      if (helperScratch[g] == nullptr) {
        return ncclInternalError;
      }
    }
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
          if (d == m && !foldDiagonalIntoGroup)
            continue;
          NCCLCHECK(ncclSend(
              sendbuff, d1, datatype, cfg.activeRanks[d], comm, stream));
        }
        for (int s = 0; s < A; s++) {
          if (s == m && !foldDiagonalIntoGroup)
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
      char* hbuff = static_cast<char*>(helperScratch[g]);
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
      const char* hbuff = static_cast<const char*>(helperScratch[g]);
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
 * Software-pipelined single-group A>2 flat all-gather.
 *
 * Same scatter/forward relay as shardedRelayAllGatherFlat, but for nGroups == 1
 * -- where the active ranks and the helpers are disjoint -- the offload is
 * tiled so the scatter and the forward share a group and each cross link runs
 * duplex. See relayPipelineTiles().
 *
 * This schedule is ASYMMETRIC, unlike the 2-active one: a helper fans each
 * source's slice out to the A-1 other destinations, so a cross link carries one
 * unit UP and A-1 units DOWN. Merging is therefore bounded by the down
 * direction and the up direction stays underused, which is why the payoff is
 * smaller here. With v = align(sc / ((H + A - 1)*T + 1)) each group carries
 * (A-1)*v on the busiest link and the direct region is split to match, so the
 * cost is ((A-1)*T + 1)*v: at A = H = 4 that is sc/2 at T = 1 (identical to the
 * two-group schedule) falling towards 3*sc/7. That floor is also the hard one
 * -- an active rank has to take in A-1 = 3 whole buffers across its 7 links.
 *
 * The divisor is derived from A and H rather than fixed at the A = H = 4 value
 * of 7 because the layout below consumes H*T + 1 + (T-1)*(A-1) units before the
 * final direct chunk; any other geometry that reaches this path (A = 4 with
 * H = 5..8 on a 9-to-12-rank comm, or A = 8 with H = 8 on a 16-rank one) would
 * otherwise push directOffset(T) past sc and underflow directSize(T).
 *
 * Buffer layout of sendBuff [0, sc):
 *   [0, H*T*v)   offload region; helper h owns [h*T*v, (h+1)*T*v), tile t at
 *                h*T*v + t*v
 *   [H*T*v, sc)  direct region, one chunk per group sized to that group's cross
 *                load: v for group 0 (which has no forward to receive yet) and
 *                (A-1)*v for the rest, the last absorbing the remainder
 *
 * Both in-place and out-of-place are supported; the gather slots never overlap
 * the send source.
 */
static ncclResult_t shardedRelayAllGatherFlatPipelined(
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
    int nTiles,
    size_t elementSize) {
  const ShardedRelayRankConfig& cfg = configs[0];
  const size_t sc = sendCounts[0];
  const int A = nActiveRanksPerGroup;
  const int H = numHelpers;
  const int T = nTiles;
  const size_t v = ((sc /
                     (static_cast<size_t>(
                          rcclx::relay::relayShapeFanout(A, H).totalPerTile) *
                          T +
                      1)) /
                    CHUNK_ALIGN_ELEMENTS) *
      CHUNK_ALIGN_ELEMENTS;
  if (v == 0) {
    return ncclInvalidArgument;
  }
  const size_t tileStride = static_cast<size_t>(T) * v;
  const size_t directBase = static_cast<size_t>(H) * tileStride;
  // Group 0 carries one unit of direct traffic, every later group carries A-1.
  const size_t heavy = static_cast<size_t>(A - 1) * v;
  auto directOffset = [&](int k) -> size_t {
    return directBase + (k == 0 ? 0 : v + static_cast<size_t>(k - 1) * heavy);
  };
  auto directSize = [&](int k) -> size_t {
    if (k == 0) {
      return v;
    }
    return (k < T) ? heavy : (sc - directOffset(T));
  };

  // Diagonal: recvBuff[m*sc] = sendBuff, a no-op when in-place.
  if (myActiveGroup == 0) {
    char* diag = static_cast<char*>(recvBuffs[0]) +
        static_cast<size_t>(cfg.myActiveIndex) * sc * elementSize;
    if (static_cast<const void*>(sendBuffs[0]) !=
        static_cast<const void*>(diag)) {
      cudaMemcpyAsync(
          diag,
          sendBuffs[0],
          sc * elementSize,
          cudaMemcpyDeviceToDevice,
          stream);
    }
  }

  // Helper staging: two ping-pong tiles per active source.
  char* hbuff = nullptr;
  if (!cfg.isActiveRank) {
    hbuff = static_cast<char*>(ScratchBufferCache::getInstance().get(
        kHelperScratchKeyBase,
        static_cast<size_t>(A) * 2 * v * elementSize,
        stream));
    if (hbuff == nullptr) {
      return ncclInternalError;
    }
  }
  auto helperSlot = [&](int s, int k) -> char* {
    return hbuff +
        (static_cast<size_t>(s) * 2 + static_cast<size_t>(k % 2)) * v *
        elementSize;
  };

  for (int k = 0; k <= T; k++) {
    NCCLCHECK(ncclGroupStart());
    if (cfg.isActiveRank) {
      const char* sendbuff = static_cast<const char*>(sendBuffs[0]);
      char* recvbuff = static_cast<char*>(recvBuffs[0]);
      const int m = cfg.myActiveIndex;
      const size_t dOff = directOffset(k);
      const size_t dSz = directSize(k);

      // Direct exchange, issued as v-sized pieces rather than one (A-1)*v op,
      // so that every operation in the group is the same size and all seven
      // links get a comparable channel budget. See
      // kRelayUniformDirectOpMinBytes for why this is size-gated here but
      // unconditional in the reduce-scatter mirror.
      const size_t dPiece = v;
      const int dN =
          (sc * elementSize >= rcclx::relay::kRelayUniformDirectOpMinBytes &&
           dSz >= dPiece)
          ? static_cast<int>(dSz / dPiece)
          : 1;
      auto dPieceOff = [&](int i) -> size_t {
        return dOff + static_cast<size_t>(i) * dPiece;
      };
      auto dPieceSz = [&](int i) -> size_t {
        return (i < dN - 1) ? dPiece
                            : (dSz - static_cast<size_t>(dN - 1) * dPiece);
      };
      for (int d = 0; d < A; d++) {
        if (d == m) {
          continue;
        }
        for (int i = 0; i < dN; i++) {
          NCCLCHECK(ncclSend(
              sendbuff + dPieceOff(i) * elementSize,
              dPieceSz(i),
              datatype,
              cfg.activeRanks[d],
              comm,
              stream));
        }
      }
      if (k < T) {
        for (int h = 0; h < H; h++) {
          NCCLCHECK(ncclSend(
              sendbuff +
                  (static_cast<size_t>(h) * tileStride +
                   static_cast<size_t>(k) * v) *
                      elementSize,
              v,
              datatype,
              cfg.helperRanks[h],
              comm,
              stream));
        }
      }
      for (int s = 0; s < A; s++) {
        if (s == m) {
          continue;
        }
        for (int i = 0; i < dN; i++) {
          NCCLCHECK(ncclRecv(
              recvbuff +
                  (static_cast<size_t>(s) * sc + dPieceOff(i)) * elementSize,
              dPieceSz(i),
              datatype,
              cfg.activeRanks[s],
              comm,
              stream));
        }
      }
      if (k > 0) {
        // Helper h delivers tile k-1 of every other source's offload chunk.
        for (int h = 0; h < H; h++) {
          for (int s = 0; s < A; s++) {
            if (s == m) {
              continue;
            }
            NCCLCHECK(ncclRecv(
                recvbuff +
                    (static_cast<size_t>(s) * sc +
                     static_cast<size_t>(h) * tileStride +
                     static_cast<size_t>(k - 1) * v) *
                        elementSize,
                v,
                datatype,
                cfg.helperRanks[h],
                comm,
                stream));
          }
        }
      }
    } else {
      if (k < T) {
        for (int s = 0; s < A; s++) {
          NCCLCHECK(ncclRecv(
              helperSlot(s, k), v, datatype, cfg.activeRanks[s], comm, stream));
        }
      }
      if (k > 0) {
        // Per-peer send order matches each destination's receive order above.
        for (int s = 0; s < A; s++) {
          for (int d = 0; d < A; d++) {
            if (d == s) {
              continue;
            }
            NCCLCHECK(ncclSend(
                helperSlot(s, k - 1),
                v,
                datatype,
                cfg.activeRanks[d],
                comm,
                stream));
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
  // shard exchange with minimal latency. The size -> route mapping lives in
  // selectAllGatherRoute() so the tests assert the same definition this
  // dispatch uses.
  const rcclx::relay::AllGatherRoute route = rcclx::relay::selectAllGatherRoute(
      nActiveRanksPerGroup, numHelpers, nGroups, sendCounts, elementSize);
  if (route == rcclx::relay::AllGatherRoute::A2Relay) {
    // A single-group call has the helpers to itself, so the scatter and the
    // forward run on opposite directions of each cross link and can be
    // software-pipelined into one duplex stream; relayPipelineTiles() returns
    // 1 whenever that does not apply.
    const int nTiles = rcclx::relay::relayPipelineTiles(
        nGroups,
        rcclx::relay::relayShapeA2(numHelpers),
        rcclx::relay::relayMaxCount(sendCounts, nGroups),
        elementSize);
    r = (nTiles > 1) ? shardedRelayAllGather2ActivePipelined(
                           sendBuffs2,
                           recvBuffs2,
                           sendCounts,
                           datatype,
                           comm,
                           stream,
                           configs,
                           myActiveGroup,
                           numHelpers,
                           nTiles,
                           elementSize)
                     : shardedRelayAllGather2Active(
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
    // with a size-adaptive pure-direct small-size mode. A single-group call
    // with the offload enabled can additionally be software-pipelined so the
    // scatter and the forward share each cross link's two directions.
    const bool offload = route == rcclx::relay::AllGatherRoute::FlatOffload;
    const int nTiles = offload
        ? rcclx::relay::relayPipelineTiles(
              nGroups,
              rcclx::relay::relayShapeFanout(nActiveRanksPerGroup, numHelpers),
              rcclx::relay::relayMaxCount(sendCounts, nGroups),
              elementSize)
        : 1;
    if (nTiles > 1) {
      r = shardedRelayAllGatherFlatPipelined(
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
          nTiles,
          elementSize);
    } else {
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
  }

  return r;
}
