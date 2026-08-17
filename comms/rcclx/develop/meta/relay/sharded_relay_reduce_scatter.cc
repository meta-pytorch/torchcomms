/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "sharded_relay_reduce_scatter.h"
#include "comm.h"
#include "sharded_relay_allreduce_kernels.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <mutex>
#include <unordered_map>

// GPU memory access alignment in elements for chunk size rounding.
// Distinct from the CPU CACHE_LINE_SIZE (64 bytes) defined in comm.h
// which is used for struct padding.
static constexpr size_t CHUNK_ALIGN_ELEMENTS = 128;

// Infra below (ScratchBufferCache, rank-config builder, DISPATCH macros) is a
// deliberate copy of the file-local helpers in sharded_relay_allreduce.cc.
// They are file-local (static / anonymous namespace) there, so they cannot be
// linked across translation units; the reduce-scatter TU re-declares its own
// copies in an anonymous namespace to keep them internal and ODR-safe. The
// GPU kernels themselves are NOT duplicated — they are reused via
// sharded_relay_allreduce_kernels.h (dtype-generic, collective-agnostic).
namespace {

/**
 * Scratch Buffer Cache Singleton
 *
 * Amortizes cudaMalloc/cudaFree costs by caching and reusing scratch buffers.
 * Thread-safe with per-device buffer management. See the allreduce copy for a
 * full description; this is an independent cache scoped to reduce-scatter.
 */
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

    // Create composite key from device and user key.
    int64_t compositeKey = static_cast<int64_t>(device) * 4096 + key;
    auto& entry = buffers_[compositeKey];

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

  void clear(cudaStream_t stream = nullptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& pair : buffers_) {
      if (pair.second.buffer != nullptr) {
        cudaFreeAsync(pair.second.buffer, stream);
        pair.second.buffer = nullptr;
        pair.second.size = 0;
      }
    }
    buffers_.clear();
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
  std::unordered_map<int64_t, BufferEntry> buffers_; // compositeKey -> buffer
};

// Maximum number of helper ranks supported per group.
constexpr int SHARDED_RELAY_MAX_HELPERS = 8;

// Maximum number of active ranks per group. The recursive-halving relay
// schedule (round-r partner = myActiveIndex XOR round) requires nActiveRanks to
// be a power of two; supported values are 2 and 4 (on an 8-GPU node this leaves
// 6 or 4 helpers respectively).
constexpr int SHARDED_RELAY_MAX_ACTIVE = 8;

// Returns true if v is a power of two (v >= 1).
inline bool isPowerOfTwo(int v) {
  return v > 0 && (v & (v - 1)) == 0;
}

// Reverse the low `bits` bits of x. The recursive-halving reduce-scatter leaves
// active rank mi owning the segment at bit-reversed position bitReverse(mi);
// the reduce-scatter gather places block[j] at bit-reversed position so owner j
// ends up holding the reduced block[j].
inline int bitReverse(int x, int bits) {
  int r = 0;
  for (int b = 0; b < bits; b++) {
    r = (r << 1) | ((x >> b) & 1);
  }
  return r;
}

/**
 * Rank Configuration for Sharded Relay Reduce-Scatter
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
 * the XOR round schedule of the A>2 recursive path depends on it.
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

} // namespace

// Host-side dispatch macros for the reused generic kernels. These mirror the
// file-local DISPATCH_* macros in sharded_relay_allreduce.cc (the kernels they
// launch are declared in sharded_relay_allreduce_kernels.h).

#define LAUNCH_INCREMENTAL_ADD_KERNEL(TYPE, output, input, count, stream) \
  launchIncrementalAddKernel<TYPE>(output, input, count, stream)

#define DISPATCH_INCREMENTAL_ADD(datatype, output, input, count, stream)       \
  do {                                                                         \
    switch (datatype) {                                                        \
      case ncclInt8:                                                           \
        LAUNCH_INCREMENTAL_ADD_KERNEL(int8_t, output, input, count, stream);   \
        break;                                                                 \
      case ncclUint8:                                                          \
        LAUNCH_INCREMENTAL_ADD_KERNEL(uint8_t, output, input, count, stream);  \
        break;                                                                 \
      case ncclInt32:                                                          \
        LAUNCH_INCREMENTAL_ADD_KERNEL(int32_t, output, input, count, stream);  \
        break;                                                                 \
      case ncclUint32:                                                         \
        LAUNCH_INCREMENTAL_ADD_KERNEL(uint32_t, output, input, count, stream); \
        break;                                                                 \
      case ncclInt64:                                                          \
        LAUNCH_INCREMENTAL_ADD_KERNEL(int64_t, output, input, count, stream);  \
        break;                                                                 \
      case ncclUint64:                                                         \
        LAUNCH_INCREMENTAL_ADD_KERNEL(uint64_t, output, input, count, stream); \
        break;                                                                 \
      case ncclFloat16:                                                        \
        LAUNCH_INCREMENTAL_ADD_KERNEL(__half, output, input, count, stream);   \
        break;                                                                 \
      case ncclFloat:                                                          \
        LAUNCH_INCREMENTAL_ADD_KERNEL(float, output, input, count, stream);    \
        break;                                                                 \
      case ncclDouble:                                                         \
        LAUNCH_INCREMENTAL_ADD_KERNEL(double, output, input, count, stream);   \
        break;                                                                 \
      case ncclBfloat16:                                                       \
        LAUNCH_INCREMENTAL_ADD_KERNEL(                                         \
            __nv_bfloat16, output, input, count, stream);                      \
        break;                                                                 \
      default:                                                                 \
        break;                                                                 \
    }                                                                          \
  } while (0)

#define LAUNCH_SCALE_KERNEL(TYPE, data, count, divisor, stream) \
  launchScaleKernel<TYPE>(data, count, divisor, stream)

#define DISPATCH_SCALE(datatype, data, count, divisor, stream)            \
  do {                                                                    \
    switch (datatype) {                                                   \
      case ncclInt8:                                                      \
        LAUNCH_SCALE_KERNEL(int8_t, data, count, divisor, stream);        \
        break;                                                            \
      case ncclUint8:                                                     \
        LAUNCH_SCALE_KERNEL(uint8_t, data, count, divisor, stream);       \
        break;                                                            \
      case ncclInt32:                                                     \
        LAUNCH_SCALE_KERNEL(int32_t, data, count, divisor, stream);       \
        break;                                                            \
      case ncclUint32:                                                    \
        LAUNCH_SCALE_KERNEL(uint32_t, data, count, divisor, stream);      \
        break;                                                            \
      case ncclInt64:                                                     \
        LAUNCH_SCALE_KERNEL(int64_t, data, count, divisor, stream);       \
        break;                                                            \
      case ncclUint64:                                                    \
        LAUNCH_SCALE_KERNEL(uint64_t, data, count, divisor, stream);      \
        break;                                                            \
      case ncclFloat16:                                                   \
        LAUNCH_SCALE_KERNEL(__half, data, count, divisor, stream);        \
        break;                                                            \
      case ncclFloat:                                                     \
        LAUNCH_SCALE_KERNEL(float, data, count, divisor, stream);         \
        break;                                                            \
      case ncclDouble:                                                    \
        LAUNCH_SCALE_KERNEL(double, data, count, divisor, stream);        \
        break;                                                            \
      case ncclBfloat16:                                                  \
        LAUNCH_SCALE_KERNEL(__nv_bfloat16, data, count, divisor, stream); \
        break;                                                            \
      default:                                                            \
        break;                                                            \
    }                                                                     \
  } while (0)

#define LAUNCH_INCREMENTAL_ADD_AND_SCALE_KERNEL( \
    TYPE, output, input, count, divisor, stream) \
  launchIncrementalAddAndScaleKernel<TYPE>(      \
      output, input, count, divisor, stream)

#define DISPATCH_INCREMENTAL_ADD_AND_SCALE(                        \
    datatype, output, input, count, divisor, stream)               \
  do {                                                             \
    switch (datatype) {                                            \
      case ncclInt8:                                               \
        LAUNCH_INCREMENTAL_ADD_AND_SCALE_KERNEL(                   \
            int8_t, output, input, count, divisor, stream);        \
        break;                                                     \
      case ncclUint8:                                              \
        LAUNCH_INCREMENTAL_ADD_AND_SCALE_KERNEL(                   \
            uint8_t, output, input, count, divisor, stream);       \
        break;                                                     \
      case ncclInt32:                                              \
        LAUNCH_INCREMENTAL_ADD_AND_SCALE_KERNEL(                   \
            int32_t, output, input, count, divisor, stream);       \
        break;                                                     \
      case ncclUint32:                                             \
        LAUNCH_INCREMENTAL_ADD_AND_SCALE_KERNEL(                   \
            uint32_t, output, input, count, divisor, stream);      \
        break;                                                     \
      case ncclInt64:                                              \
        LAUNCH_INCREMENTAL_ADD_AND_SCALE_KERNEL(                   \
            int64_t, output, input, count, divisor, stream);       \
        break;                                                     \
      case ncclUint64:                                             \
        LAUNCH_INCREMENTAL_ADD_AND_SCALE_KERNEL(                   \
            uint64_t, output, input, count, divisor, stream);      \
        break;                                                     \
      case ncclFloat16:                                            \
        LAUNCH_INCREMENTAL_ADD_AND_SCALE_KERNEL(                   \
            __half, output, input, count, divisor, stream);        \
        break;                                                     \
      case ncclFloat:                                              \
        LAUNCH_INCREMENTAL_ADD_AND_SCALE_KERNEL(                   \
            float, output, input, count, divisor, stream);         \
        break;                                                     \
      case ncclDouble:                                             \
        LAUNCH_INCREMENTAL_ADD_AND_SCALE_KERNEL(                   \
            double, output, input, count, divisor, stream);        \
        break;                                                     \
      case ncclBfloat16:                                           \
        LAUNCH_INCREMENTAL_ADD_AND_SCALE_KERNEL(                   \
            __nv_bfloat16, output, input, count, divisor, stream); \
        break;                                                     \
      default:                                                     \
        break;                                                     \
    }                                                              \
  } while (0)

#define LAUNCH_FUSED_REDUCE_KERNEL(                       \
    TYPE, output, inputA, inputB, count, divisor, stream) \
  launchFusedReduceKernel<TYPE>(output, inputA, inputB, count, divisor, stream)

#define DISPATCH_FUSED_REDUCE(                                              \
    datatype, output, inputA, inputB, count, divisor, stream)               \
  do {                                                                      \
    switch (datatype) {                                                     \
      case ncclInt8:                                                        \
        LAUNCH_FUSED_REDUCE_KERNEL(                                         \
            int8_t, output, inputA, inputB, count, divisor, stream);        \
        break;                                                              \
      case ncclUint8:                                                       \
        LAUNCH_FUSED_REDUCE_KERNEL(                                         \
            uint8_t, output, inputA, inputB, count, divisor, stream);       \
        break;                                                              \
      case ncclInt32:                                                       \
        LAUNCH_FUSED_REDUCE_KERNEL(                                         \
            int32_t, output, inputA, inputB, count, divisor, stream);       \
        break;                                                              \
      case ncclUint32:                                                      \
        LAUNCH_FUSED_REDUCE_KERNEL(                                         \
            uint32_t, output, inputA, inputB, count, divisor, stream);      \
        break;                                                              \
      case ncclInt64:                                                       \
        LAUNCH_FUSED_REDUCE_KERNEL(                                         \
            int64_t, output, inputA, inputB, count, divisor, stream);       \
        break;                                                              \
      case ncclUint64:                                                      \
        LAUNCH_FUSED_REDUCE_KERNEL(                                         \
            uint64_t, output, inputA, inputB, count, divisor, stream);      \
        break;                                                              \
      case ncclFloat16:                                                     \
        LAUNCH_FUSED_REDUCE_KERNEL(                                         \
            __half, output, inputA, inputB, count, divisor, stream);        \
        break;                                                              \
      case ncclFloat:                                                       \
        LAUNCH_FUSED_REDUCE_KERNEL(                                         \
            float, output, inputA, inputB, count, divisor, stream);         \
        break;                                                              \
      case ncclDouble:                                                      \
        LAUNCH_FUSED_REDUCE_KERNEL(                                         \
            double, output, inputA, inputB, count, divisor, stream);        \
        break;                                                              \
      case ncclBfloat16:                                                    \
        LAUNCH_FUSED_REDUCE_KERNEL(                                         \
            __nv_bfloat16, output, inputA, inputB, count, divisor, stream); \
        break;                                                              \
      default:                                                              \
        break;                                                              \
    }                                                                       \
  } while (0)

#define LAUNCH_MULTI_REDUCE_KERNEL(                           \
    TYPE, dst, contribs, numContribs, count, divisor, stream) \
  launchMultiReduceKernel<TYPE>(                              \
      dst, contribs, numContribs, count, divisor, stream)

// Fused multi-input reduce: dst = (dst + sum of `numContribs` contiguous
// contribution blocks) [/ divisor], in one launch. Replaces a loop of
// per-contribution incremental adds plus a trailing scale.
#define DISPATCH_MULTI_REDUCE(                                             \
    datatype, dst, contribs, numContribs, count, divisor, stream)          \
  do {                                                                     \
    switch (datatype) {                                                    \
      case ncclInt8:                                                       \
        LAUNCH_MULTI_REDUCE_KERNEL(                                        \
            int8_t, dst, contribs, numContribs, count, divisor, stream);   \
        break;                                                             \
      case ncclUint8:                                                      \
        LAUNCH_MULTI_REDUCE_KERNEL(                                        \
            uint8_t, dst, contribs, numContribs, count, divisor, stream);  \
        break;                                                             \
      case ncclInt32:                                                      \
        LAUNCH_MULTI_REDUCE_KERNEL(                                        \
            int32_t, dst, contribs, numContribs, count, divisor, stream);  \
        break;                                                             \
      case ncclUint32:                                                     \
        LAUNCH_MULTI_REDUCE_KERNEL(                                        \
            uint32_t, dst, contribs, numContribs, count, divisor, stream); \
        break;                                                             \
      case ncclInt64:                                                      \
        LAUNCH_MULTI_REDUCE_KERNEL(                                        \
            int64_t, dst, contribs, numContribs, count, divisor, stream);  \
        break;                                                             \
      case ncclUint64:                                                     \
        LAUNCH_MULTI_REDUCE_KERNEL(                                        \
            uint64_t, dst, contribs, numContribs, count, divisor, stream); \
        break;                                                             \
      case ncclFloat16:                                                    \
        LAUNCH_MULTI_REDUCE_KERNEL(                                        \
            __half, dst, contribs, numContribs, count, divisor, stream);   \
        break;                                                             \
      case ncclFloat:                                                      \
        LAUNCH_MULTI_REDUCE_KERNEL(                                        \
            float, dst, contribs, numContribs, count, divisor, stream);    \
        break;                                                             \
      case ncclDouble:                                                     \
        LAUNCH_MULTI_REDUCE_KERNEL(                                        \
            double, dst, contribs, numContribs, count, divisor, stream);   \
        break;                                                             \
      case ncclBfloat16:                                                   \
        LAUNCH_MULTI_REDUCE_KERNEL(                                        \
            __nv_bfloat16,                                                 \
            dst,                                                           \
            contribs,                                                      \
            numContribs,                                                   \
            count,                                                         \
            divisor,                                                       \
            stream);                                                       \
        break;                                                             \
      default:                                                             \
        break;                                                             \
    }                                                                      \
  } while (0)

namespace {

// The DISPATCH_* macros above instantiate reduce kernels for exactly these
// types and fall through silently (default: break) for anything else, so an
// unsupported datatype would return ncclSuccess having never reduced anything.
//
// This is deliberately a supported-set test rather than the
// `datatype < 0 || datatype >= ncclNumTypes` range test used by upstream
// ArgsCheck: ncclFloat8e4m3 and ncclFloat8e5m2 are valid NCCL types that
// ncclTypeSize() sizes at 1 byte, so they pass a range test but have no reduce
// kernel here. Keep this list in sync with the DISPATCH_* macros above.
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
 * Two-active sharded relay reduce-scatter (original, performant path).
 *
 * Each group has exactly 2 active ranks; the logical collective is a 2-rank
 * reduce-scatter between them, accelerated by passthrough helpers that relay
 * sharded chunks of a single block (recvCounts[g] elements). This is the
 * production BM-FM path and is byte-for-byte unchanged from the original
 * implementation; the A>2 path lives in shardedRelayReduceScatterRecursive.
 *
 * Per active rank (index myActiveIndex), with recvcount = recvCounts[g]:
 *   - sendBuff holds 2 × recvcount elements; block[i] = sendBuff[i*recvcount].
 *   - ownBlockOffset  = myActiveIndex    × recvcount (local contribution)
 *   - sendBlockOffset = otherActiveIndex × recvcount (shipped to other rank)
 *   - recvBuff[0..recvcount) = block[myActiveIndex](self) +
 *                              block[myActiveIndex](other).
 *
 * The relay relays the sendBlockOffset block chunk-by-chunk; the output block
 * (recvBuff) is seeded with the ownBlockOffset contribution then accumulates
 * the relayed/direct-exchanged chunks from the other active rank.
 *
 * In-place is detected when recvBuff == sendBuff + ownBlockOffset (the NCCL
 * reduce-scatter in-place convention). In that case recvBuff already holds the
 * local contribution (no seeding copy) and the direct chunk is reduced via a
 * scratch buffer to avoid overwriting the local data before it is read.
 */
static ncclResult_t shardedRelayReduceScatter2Active(
    const void* const* sendBuffs,
    void* const* recvBuffs,
    const size_t* recvCounts,
    ncclDataType_t datatype,
    int reductionDivisor,
    ncclComm_t comm,
    cudaStream_t stream,
    const ShardedRelayRankConfig* configs,
    int myActiveGroup,
    int numHelpers,
    int nGroups,
    size_t elementSize) {
  int numChunks = numHelpers + 1;

  // =========================================================================
  // CALCULATE PER-GROUP CHUNK SIZES (from the OUTPUT recvCount, i.e. one block)
  // =========================================================================
  size_t chunkSizes[SHARDED_RELAY_MAX_GROUPS];
  size_t lastChunkSizes[SHARDED_RELAY_MAX_GROUPS];
  size_t directChunkOffsets[SHARDED_RELAY_MAX_GROUPS];
  size_t directChunkSizes[SHARDED_RELAY_MAX_GROUPS];

  for (int g = 0; g < nGroups; g++) {
    size_t count = recvCounts[g];

    // Skip groups with recvCount == 0; the per-phase loops below already check
    // recvCounts[g] == 0 and bypass NCCL ops for those groups.
    if (count == 0) {
      chunkSizes[g] = 0;
      lastChunkSizes[g] = 0;
      directChunkOffsets[g] = 0;
      directChunkSizes[g] = 0;
      continue;
    }

    // Calculate chunk size (aligned to CHUNK_ALIGN_ELEMENTS). When the
    // per-chunk size rounded down to CHUNK_ALIGN_ELEMENTS is zero, the block
    // is too small to scatter and the caller should fall back to a regular
    // reduce-scatter.
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

    // Direct exchange chunk info (within the single output block)
    int directChunkIndex = numHelpers;
    directChunkOffsets[g] = static_cast<size_t>(directChunkIndex) * chunkSize;
    directChunkSizes[g] = lastChunkSize;
  }

  // =========================================================================
  // SCRATCH BUFFER ALLOCATION
  // =========================================================================
  // Relay scratch: numHelpers × chunkSize for batched passthrough recv.
  // Direct-exchange scratch: (nActiveRanks-1) × directChunkSize (in-place).
  void* relayScratch = nullptr;
  void* directScratch = nullptr;

  // For an active rank, compute its block offsets up-front.
  size_t ownBlockOffset = 0;
  size_t sendBlockOffset = 0;
  bool isInPlace = false;
  if (myActiveGroup >= 0 && recvCounts[myActiveGroup] > 0) {
    const ShardedRelayRankConfig& cfg = configs[myActiveGroup];
    size_t recvcount = recvCounts[myActiveGroup];
    int otherActiveIndex = 1 - cfg.myActiveIndex;
    ownBlockOffset = static_cast<size_t>(cfg.myActiveIndex) * recvcount;
    sendBlockOffset = static_cast<size_t>(otherActiveIndex) * recvcount;

    // In-place when recvBuff aliases the local contribution block of sendBuff.
    const char* sendBlockStart =
        static_cast<const char*>(sendBuffs[myActiveGroup]) +
        ownBlockOffset * elementSize;
    isInPlace =
        (static_cast<const void*>(recvBuffs[myActiveGroup]) ==
         static_cast<const void*>(sendBlockStart));

    // Relay scratch sized to numHelpers × chunkSize.
    size_t relayScratchBytes = static_cast<size_t>(numHelpers) *
        chunkSizes[myActiveGroup] * elementSize;
    relayScratch = ScratchBufferCache::getInstance().get(
        SHARDED_RELAY_MAX_GROUPS, relayScratchBytes, stream);
    if (relayScratch == nullptr) {
      return ncclInternalError;
    }

    // Direct-exchange scratch (in-place only)
    if (isInPlace) {
      int nOtherActives = cfg.nActiveRanks - 1;
      size_t directScratchBytes = static_cast<size_t>(nOtherActives) *
          directChunkSizes[myActiveGroup] * elementSize;
      directScratch = ScratchBufferCache::getInstance().get(
          myActiveGroup, directScratchBytes, stream);
      if (directScratch == nullptr) {
        return ncclInternalError;
      }
    }
  }

  // =========================================================================
  // PHASE 1 (active→helpers): Both active ranks scatter their sendBlock chunks
  // =========================================================================
  // Helpers receive from each active rank into offset-based slots:
  //   slot a at offset (a × chunkSize) holds data from active rank a.
  // Active ranks send chunks of the sendBlockOffset block (block destined for
  // the OTHER active rank).
  NCCLCHECK(ncclGroupStart());

  for (int g = 0; g < nGroups; g++) {
    if (recvCounts[g] == 0)
      continue;
    const ShardedRelayRankConfig& cfg = configs[g];
    const void* sendbuff = sendBuffs[g];
    void* recvbuff = recvBuffs[g];
    size_t chunkSize = chunkSizes[g];

    if (cfg.isActiveRank) {
      // Active rank: send chunk h of my sendBlock to helper h
      for (int h = 0; h < cfg.numHelpers; h++) {
        int helperRank = cfg.helperRanks[h];
        size_t chunkOffset =
            sendBlockOffset + static_cast<size_t>(h) * chunkSize;

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

  // For out-of-place active groups: seed recvbuff's relay region with the
  // local contribution (the ownBlockOffset block of sendbuff) so the
  // incremental add in Phase 3 works uniformly. In-place already has it.
  if (myActiveGroup >= 0 && recvCounts[myActiveGroup] > 0 && !isInPlace) {
    const void* sendbuff = sendBuffs[myActiveGroup];
    void* recvbuff = recvBuffs[myActiveGroup];
    size_t relayBytes = static_cast<size_t>(numHelpers) *
        chunkSizes[myActiveGroup] * elementSize;
    cudaMemcpyAsync(
        recvbuff,
        static_cast<const char*>(sendbuff) + ownBlockOffset * elementSize,
        relayBytes,
        cudaMemcpyDeviceToDevice,
        stream);
  }

  // =========================================================================
  // PHASE 2 (helpers→active, batched): Passthrough forward
  // =========================================================================
  // Each helper sends slot 0 (a0's data) → a1 and slot 1 (a1's data) → a0.
  // Active rank receives all numHelpers chunks into relay scratch at offset
  // h × chunkSize per helper h.
  NCCLCHECK(ncclGroupStart());

  for (int g = 0; g < nGroups; g++) {
    if (recvCounts[g] == 0)
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
      // Active rank: receive ALL forwarded data from each helper into the
      // relay scratch at offset h × chunkSize.
      for (int h = 0; h < cfg.numHelpers; h++) {
        int helperRank = cfg.helperRanks[h];
        size_t scratchOffset = static_cast<size_t>(h) * chunkSize;
        NCCLCHECK(ncclRecv(
            static_cast<char*>(relayScratch) + scratchOffset * elementSize,
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
  // PHASE 3 (active reduce): Fused add + scale on the relay region
  // =========================================================================
  // recvbuff[i] = (recvbuff[i] + relayScratch[i]) / divisor over the relay
  // region (numHelpers × chunkSize). recvbuff was seeded with the local
  // ownBlock contribution above.
  for (int g = 0; g < nGroups; g++) {
    if (recvCounts[g] == 0)
      continue;
    const ShardedRelayRankConfig& cfg = configs[g];
    if (cfg.isActiveRank) {
      void* recvbuff = recvBuffs[g];
      size_t chunkSize = chunkSizes[g];
      size_t relayTotal = static_cast<size_t>(numHelpers) * chunkSize;

      if (reductionDivisor > 1) {
        DISPATCH_INCREMENTAL_ADD_AND_SCALE(
            datatype,
            recvbuff,
            relayScratch,
            relayTotal,
            reductionDivisor,
            stream);
      } else {
        DISPATCH_INCREMENTAL_ADD(
            datatype, recvbuff, relayScratch, relayTotal, stream);
      }
    }
  }

  // =========================================================================
  // PHASE 4 (active↔active): Direct exchange of the last chunk
  // =========================================================================
  // Active ranks exchange the direct chunk of their sendBlock; it lands in the
  // output block's direct-chunk slot (offset directChunkOffset in recvbuff).
  NCCLCHECK(ncclGroupStart());

  for (int g = 0; g < nGroups; g++) {
    if (recvCounts[g] == 0)
      continue;
    const ShardedRelayRankConfig& cfg = configs[g];

    if (cfg.isActiveRank) {
      const void* sendbuff = sendBuffs[g];
      void* recvbuff = recvBuffs[g];
      size_t directChunkOffset = directChunkOffsets[g];
      size_t directChunkSize = directChunkSizes[g];
      // Source in sendbuff is within the sendBlock (shipped to other rank).
      size_t sendDirectOffset = sendBlockOffset + directChunkOffset;

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

      // Receive direct chunks from all other active ranks
      int scratchIdx = 0;
      for (int a = 0; a < cfg.nActiveRanks; a++) {
        if (a == cfg.myActiveIndex)
          continue;
        int otherActiveRank = cfg.activeRanks[a];

        if (isInPlace) {
          size_t scratchOffset =
              static_cast<size_t>(scratchIdx) * directChunkSize;
          NCCLCHECK(ncclRecv(
              static_cast<char*>(directScratch) + scratchOffset * elementSize,
              directChunkSize,
              datatype,
              otherActiveRank,
              comm,
              stream));
        } else {
          NCCLCHECK(ncclRecv(
              static_cast<char*>(recvbuff) + directChunkOffset * elementSize,
              directChunkSize,
              datatype,
              otherActiveRank,
              comm,
              stream));
        }
        scratchIdx++;
      }
    }
  }

  NCCLCHECK(ncclGroupEnd());

  // =========================================================================
  // PHASE 5 (active reduce): Final reduction on the direct-exchange chunk
  // =========================================================================
  for (int g = 0; g < nGroups; g++) {
    if (recvCounts[g] == 0)
      continue;
    const ShardedRelayRankConfig& cfg = configs[g];

    if (cfg.isActiveRank) {
      const void* sendbuff = sendBuffs[g];
      void* recvbuff = recvBuffs[g];
      size_t directChunkOffset = directChunkOffsets[g];
      size_t directChunkSize = directChunkSizes[g];

      void* directChunkDst =
          static_cast<char*>(recvbuff) + directChunkOffset * elementSize;

      if (isInPlace) {
        // recvbuff direct slot already holds the local contribution.
        int scratchIdx2 = 0;
        for (int a = 0; a < cfg.nActiveRanks; a++) {
          if (a == cfg.myActiveIndex)
            continue;
          size_t scratchOffset =
              static_cast<size_t>(scratchIdx2) * directChunkSize;
          const void* received = static_cast<const char*>(directScratch) +
              scratchOffset * elementSize;
          DISPATCH_INCREMENTAL_ADD(
              datatype, directChunkDst, received, directChunkSize, stream);
          scratchIdx2++;
        }

        if (reductionDivisor > 1) {
          DISPATCH_SCALE(
              datatype,
              directChunkDst,
              directChunkSize,
              reductionDivisor,
              stream);
        }
      } else {
        // Out-of-place: local contribution is in sendbuff's ownBlock; the
        // received chunk is already in recvbuff's direct slot.
        const void* localContribution = static_cast<const char*>(sendbuff) +
            (ownBlockOffset + directChunkOffset) * elementSize;
        const void* receivedContribution = directChunkDst;

        DISPATCH_FUSED_REDUCE(
            datatype,
            directChunkDst,
            localContribution,
            receivedContribution,
            directChunkSize,
            reductionDivisor,
            stream);
      }
    }
  }

  return ncclSuccess;
}

/**
 * Reduce-scatter for > 2 active ranks (bandwidth-optimal recursive path).
 *
 * A recursive-halving relay over the cross links (active<->helper, 2-hop) woven
 * with a direct all-to-all over the intra links (active<->active, 1-hop), split
 * DIRECT_NUM=60% / 40% to equalize per-link load. There is NO all-gather phase
 * (reduce-scatter keeps only the owner's reduced shard).
 *
 * The active rank's sendBuff holds A blocks of recvCount (block[j] is this
 * rank's contribution to owner j). It is gathered into a working buffer W of
 * size A*recvCount laid out as [R region (pR) | D region (pD)], where every
 * block contributes an R part (pR/A elements, its first elements) and a D part
 * (pD/A elements, the rest):
 *   - block[j]_R is placed at R-part bitReverse(j): the recursive-halving
 *     reduce-scatter leaves owner mi holding R-part bitReverse(mi), so owner j
 *     ends up with the reduced block[j]_R.
 *   - block[j]_D is placed at D-part j: the direct all-to-all reduce-scatter
 *     keeps index j with owner j.
 * After the halving relay + direct all-to-all reduce, owner mi holds the
 * reduced block[mi] in two pieces in W (R at the final segOff, D at
 * pR + mi*(pD/A)); both are copied into recvBuff (recvCount = pR/A + pD/A).
 *
 * In-place (recvBuff == sendBuff + myActiveIndex*recvCount) and out-of-place
 * are both supported transparently: W is a separate scratch, so sendBuff is
 * fully read into W before recvBuff is written, and the only difference is the
 * (identical) final copy destination.
 *
 * Requires a power-of-two active count (count = A*recvCount is divisible by A
 * by construction, so the R/D split and per-block parts are exact).
 */
static ncclResult_t shardedRelayReduceScatterRecursive(
    const void* const* sendBuffs,
    void* const* recvBuffs,
    const size_t* recvCounts,
    ncclDataType_t datatype,
    int reductionDivisor,
    ncclComm_t comm,
    cudaStream_t stream,
    const ShardedRelayRankConfig* configs,
    int myActiveGroup,
    int numHelpers,
    int nActiveRanksPerGroup,
    int nGroups,
    size_t elementSize) {
  const int A = nActiveRanksPerGroup;
  const int numChunks = numHelpers + 1;
  int logA = 0;
  while ((1 << logA) < A) {
    logA++;
  }
  const int nRsGroups = 2 * logA;

  // Direct (intra) fraction = 60%. Balances per-link load: each intra link
  // carries 0.5*D (1-hop), each cross link 0.75*R (2-hop); equal at
  // D:R=0.6:0.4.
  const size_t DIRECT_NUM = 60, DIRECT_DEN = 100;

  // Per-group working-buffer total count = A * recvCount (the full input), and
  // its R/D split. count is divisible by A by construction.
  size_t countArr[SHARDED_RELAY_MAX_GROUPS];
  size_t pRArr[SHARDED_RELAY_MAX_GROUPS];
  size_t pDArr[SHARDED_RELAY_MAX_GROUPS];
  for (int g = 0; g < nGroups; g++) {
    size_t recvcount = recvCounts[g];
    if (recvcount == 0) {
      countArr[g] = 0;
      pRArr[g] = 0;
      pDArr[g] = 0;
      continue;
    }
    size_t count = static_cast<size_t>(A) * recvcount;
    countArr[g] = count;
    size_t alignA = static_cast<size_t>(A) * CHUNK_ALIGN_ELEMENTS;
    size_t pD = (count * DIRECT_NUM) / DIRECT_DEN;
    pD = (pD / alignA) * alignA;
    if (pD > count) {
      pD = (count / alignA) * alignA;
    }
    pDArr[g] = pD;
    pRArr[g] = count - pD;
  }

  // =========================================================================
  // SCRATCH BUFFER ALLOCATION
  // =========================================================================
  // workScratch (W): A*recvCount working buffer (active rank only).
  // rsScratch: relay reduce-scatter receive buffer (max half pR/2).
  // directScratch: direct all-to-all reduce-scatter buffer ((A-1) shards).
  void* workScratch = nullptr;
  void* rsScratch = nullptr;
  void* directScratch = nullptr;
  if (myActiveGroup >= 0 && recvCounts[myActiveGroup] > 0) {
    size_t count = countArr[myActiveGroup];
    size_t pR = pRArr[myActiveGroup];
    size_t pD = pDArr[myActiveGroup];

    size_t workBytes = count * elementSize;
    workScratch = ScratchBufferCache::getInstance().get(
        SHARDED_RELAY_MAX_GROUPS, workBytes, stream);
    if (workScratch == nullptr) {
      return ncclInternalError;
    }
    if (pR > 0) {
      size_t maxHalfBytes = (pR / 2 + CHUNK_ALIGN_ELEMENTS) * elementSize;
      rsScratch = ScratchBufferCache::getInstance().get(
          SHARDED_RELAY_MAX_GROUPS + 1, maxHalfBytes, stream);
      if (rsScratch == nullptr) {
        return ncclInternalError;
      }
    }
    if (pD > 0) {
      size_t directBytes = static_cast<size_t>(A - 1) * (pD / A) * elementSize;
      directScratch = ScratchBufferCache::getInstance().get(
          SHARDED_RELAY_MAX_GROUPS + 2, directBytes, stream);
      if (directScratch == nullptr) {
        return ncclInternalError;
      }
    }
  }

  // =========================================================================
  // GATHER: scatter the A input blocks into W with the R/D split + bit-reversed
  // R placement (active rank only). Reads directly from the caller's contiguous
  // input buffer (block j spans logical [j*recvcount, (j+1)*recvcount)).
  // =========================================================================
  if (myActiveGroup >= 0 && recvCounts[myActiveGroup] > 0) {
    char* W = static_cast<char*>(workScratch);
    const char* inPtr = static_cast<const char*>(sendBuffs[myActiveGroup]);
    size_t recvcount = recvCounts[myActiveGroup];
    size_t pR = pRArr[myActiveGroup];
    size_t rPart = pR / A; // per-block R part
    size_t dPart = pDArr[myActiveGroup] / A; // per-block D part
    for (int j = 0; j < A; j++) {
      size_t blkOff = static_cast<size_t>(j) * recvcount;
      if (rPart > 0) {
        int p = bitReverse(j, logA);
        cudaMemcpyAsync(
            W + static_cast<size_t>(p) * rPart * elementSize,
            inPtr + blkOff * elementSize,
            rPart * elementSize,
            cudaMemcpyDeviceToDevice,
            stream);
      }
      if (dPart > 0) {
        cudaMemcpyAsync(
            W + (pR + static_cast<size_t>(j) * dPart) * elementSize,
            inPtr + (blkOff + rPart) * elementSize,
            dPart * elementSize,
            cudaMemcpyDeviceToDevice,
            stream);
      }
    }
  }

  // Direct shard geometry for this rank's active group.
  size_t myShardLen = 0, dBase = 0;
  int myIdx = -1;
  if (myActiveGroup >= 0 && recvCounts[myActiveGroup] > 0 &&
      pDArr[myActiveGroup] > 0) {
    myShardLen = pDArr[myActiveGroup] / A;
    dBase = pRArr[myActiveGroup];
    myIdx = configs[myActiveGroup].myActiveIndex;
  }

  // Relay segment within the R region [0, pR).
  size_t segOff = 0;
  size_t segLen = (myActiveGroup >= 0) ? pRArr[myActiveGroup] : 0;

  // ===== Reduce-scatter: relay halving (R) + direct all-to-all (D) =====
  int giRs = 0;
  for (int k = 0; k < logA; k++) {
    int mask = 1 << k;
    bool lastRsStep = (k == logA - 1);

    size_t halfLen = 0, sendOff = 0, keepOff = 0;
    int partner = -1;
    if (myActiveGroup >= 0 && pRArr[myActiveGroup] > 0) {
      const ShardedRelayRankConfig& cfg = configs[myActiveGroup];
      halfLen = segLen / 2;
      int mi = cfg.myActiveIndex;
      if (mi & mask) {
        keepOff = segOff + halfLen;
        sendOff = segOff;
      } else {
        keepOff = segOff;
        sendOff = segOff + halfLen;
      }
      partner = cfg.activeRanks[mi ^ mask];
    }

    for (int phase = 0; phase < 2; phase++) { // 0 = scatter, 1 = forward
      int gi = giRs + phase;
      size_t dSliceLen = 0, dSliceOff = 0;
      if (myShardLen > 0) {
        size_t base = myShardLen / nRsGroups;
        dSliceOff = static_cast<size_t>(gi) * base;
        dSliceLen = (gi == nRsGroups - 1) ? (myShardLen - dSliceOff) : base;
      }

      NCCLCHECK(ncclGroupStart());
      for (int g = 0; g < nGroups; g++) {
        if (recvCounts[g] == 0)
          continue;
        const ShardedRelayRankConfig& cfg = configs[g];
        void* workbuff = (g == myActiveGroup) ? workScratch : recvBuffs[g];
        size_t pRg = pRArr[g];

        // ---- RELAY (cross links), R region ----
        if (pRg > 0) {
          size_t xg = pRg >> (k + 1);
          size_t csz = (xg / numChunks);
          csz = (csz / CHUNK_ALIGN_ELEMENTS) * CHUNK_ALIGN_ELEMENTS;
          if (cfg.isActiveRank) {
            if (csz == 0) {
              // No helpers: the whole half is a direct partner swap. Send AND
              // recv MUST be in the same ncclGroup (phase 0) — NCCL matches P2P
              // within a group, so splitting send/recv across phases deadlocks.
              if (phase == 0) {
                NCCLCHECK(ncclSend(
                    static_cast<const char*>(workbuff) + sendOff * elementSize,
                    xg,
                    datatype,
                    partner,
                    comm,
                    stream));
                NCCLCHECK(ncclRecv(
                    static_cast<char*>(rsScratch),
                    xg,
                    datatype,
                    partner,
                    comm,
                    stream));
              }
            } else {
              size_t directSz = xg - static_cast<size_t>(numHelpers) * csz;
              if (phase == 0) {
                for (int h = 0; h < cfg.numHelpers; h++) {
                  NCCLCHECK(ncclSend(
                      static_cast<const char*>(workbuff) +
                          (sendOff + (size_t)h * csz) * elementSize,
                      csz,
                      datatype,
                      cfg.helperRanks[h],
                      comm,
                      stream));
                }
                NCCLCHECK(ncclSend(
                    static_cast<const char*>(workbuff) +
                        (sendOff + (size_t)numHelpers * csz) * elementSize,
                    directSz,
                    datatype,
                    partner,
                    comm,
                    stream));
                NCCLCHECK(ncclRecv(
                    static_cast<char*>(rsScratch) +
                        (size_t)numHelpers * csz * elementSize,
                    directSz,
                    datatype,
                    partner,
                    comm,
                    stream));
              } else {
                for (int h = 0; h < cfg.numHelpers; h++) {
                  NCCLCHECK(ncclRecv(
                      static_cast<char*>(rsScratch) +
                          (size_t)h * csz * elementSize,
                      csz,
                      datatype,
                      cfg.helperRanks[h],
                      comm,
                      stream));
                }
              }
            }
          } else {
            size_t cszH = (xg / numChunks);
            cszH = (cszH / CHUNK_ALIGN_ELEMENTS) * CHUNK_ALIGN_ELEMENTS;
            if (cszH != 0) {
              if (phase == 0) {
                for (int a = 0; a < cfg.nActiveRanks; a++) {
                  NCCLCHECK(ncclRecv(
                      static_cast<char*>(workbuff) +
                          (size_t)a * cszH * elementSize,
                      cszH,
                      datatype,
                      cfg.activeRanks[a],
                      comm,
                      stream));
                }
              } else {
                for (int a = 0; a < cfg.nActiveRanks; a++) {
                  int target = cfg.activeRanks[a ^ mask];
                  NCCLCHECK(ncclSend(
                      static_cast<const char*>(workbuff) +
                          (size_t)a * cszH * elementSize,
                      cszH,
                      datatype,
                      target,
                      comm,
                      stream));
                }
              }
            }
          }
        }

        // ---- DIRECT all-to-all RS (intra links), D region ----
        if (g == myActiveGroup && myShardLen > 0 && dSliceLen > 0) {
          int peerIdx = 0;
          for (int j = 0; j < A; j++) {
            if (j == myIdx) {
              continue;
            }
            NCCLCHECK(ncclSend(
                static_cast<const char*>(workbuff) +
                    (dBase + (size_t)j * myShardLen + dSliceOff) * elementSize,
                dSliceLen,
                datatype,
                cfg.activeRanks[j],
                comm,
                stream));
            NCCLCHECK(ncclRecv(
                static_cast<char*>(directScratch) +
                    ((size_t)peerIdx * myShardLen + dSliceOff) * elementSize,
                dSliceLen,
                datatype,
                cfg.activeRanks[j],
                comm,
                stream));
            peerIdx++;
          }
        }
      }
      NCCLCHECK(ncclGroupEnd());

      if (phase == 1 && myActiveGroup >= 0 && pRArr[myActiveGroup] > 0) {
        void* keepDst = static_cast<char*>(workScratch) + keepOff * elementSize;
        if (lastRsStep && reductionDivisor > 1) {
          DISPATCH_INCREMENTAL_ADD_AND_SCALE(
              datatype, keepDst, rsScratch, halfLen, reductionDivisor, stream);
        } else {
          DISPATCH_INCREMENTAL_ADD(
              datatype, keepDst, rsScratch, halfLen, stream);
        }
        segOff = keepOff;
        segLen = halfLen;
      }
    }
    giRs += 2;
  }

  // Direct RS reduce: fold the A-1 contributions into the owned shard, scale.
  // Fused single-pass multi-input reduce (read owned + all A-1 contribs once,
  // write owned once) instead of A-1 separate read-modify-write adds + scale.
  if (myShardLen > 0) {
    void* ownedDst = static_cast<char*>(workScratch) +
        (dBase + (size_t)myIdx * myShardLen) * elementSize;
    DISPATCH_MULTI_REDUCE(
        datatype,
        ownedDst,
        directScratch,
        A - 1,
        myShardLen,
        reductionDivisor,
        stream);
  }

  // =========================================================================
  // SCATTER OUT: copy owner mi's reduced block from W into the caller's output
  // buffer (recvCount = rPart + dPart elements). R result at logical
  // [0, rPart); D result at [rPart, rPart+dPart).
  // =========================================================================
  if (myActiveGroup >= 0 && recvCounts[myActiveGroup] > 0) {
    const char* W = static_cast<const char*>(workScratch);
    char* outPtr = static_cast<char*>(recvBuffs[myActiveGroup]);
    size_t pR = pRArr[myActiveGroup];
    size_t rPart = pR / A;
    size_t dPart = pDArr[myActiveGroup] / A;
    int mi = configs[myActiveGroup].myActiveIndex;
    if (rPart > 0) {
      // Owner mi's R result sits at the final segOff (== bitReverse(mi)*rPart).
      cudaMemcpyAsync(
          outPtr,
          W + segOff * elementSize,
          rPart * elementSize,
          cudaMemcpyDeviceToDevice,
          stream);
    }
    if (dPart > 0) {
      cudaMemcpyAsync(
          outPtr + rPart * elementSize,
          W + (pR + static_cast<size_t>(mi) * dPart) * elementSize,
          dPart * elementSize,
          cudaMemcpyDeviceToDevice,
          stream);
    }
  }

  return ncclSuccess;
}

/**
 * Fused Multi-Group Sharded Relay Reduce-Scatter.
 *
 * Executes multiple sharded relay reduce-scatters in one fused call,
 * phase-synced across all groups so XGMI links carry unidirectional traffic.
 * Helpers are pure passthrough; reductions happen on the active ranks. Each
 * rank is ACTIVE for exactly one group and a HELPER for the others.
 *
 * nActiveRanksPerGroup must be a power of two (2 or 4): A==2 uses the original
 * 2-active path; A>2 uses the bandwidth-optimal recursive path.
 */
HOT ncclResult_t ncclShardedRelayMultiGroupReduceScatterImpl(
    const void* const* sendBuffs,
    void* const* recvBuffs,
    const size_t* recvCounts,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    cudaStream_t stream,
    const int* const* allActiveRanks,
    int nActiveRanksPerGroup,
    int nGroups) {
  int nRanks, rank;
  NCCLCHECK(ncclCommCount(comm, &nRanks));
  NCCLCHECK(ncclCommUserRank(comm, &rank));

  // Validate every argument before touching recvCounts: the all-zero scan
  // below indexes recvCounts[0..nGroups), so a null pointer or an out-of-range
  // nGroups has to be rejected first. Bounds-checking nGroups up here also
  // means nGroups <= 0 reports ncclInvalidArgument rather than skipping the
  // scan entirely and returning ncclSuccess.
  if (nGroups < 1 || nGroups > SHARDED_RELAY_MAX_GROUPS) {
    return ncclInvalidArgument;
  }

  if (recvBuffs == nullptr || allActiveRanks == nullptr ||
      recvCounts == nullptr || sendBuffs == nullptr) {
    return ncclInvalidArgument;
  }

  // Require a power-of-two active-rank count (>= 2) for the XOR schedule.
  if (nActiveRanksPerGroup < 2 || !isPowerOfTwo(nActiveRanksPerGroup)) {
    return ncclInvalidArgument;
  }

  // Validate operation - only SUM and AVG are supported
  if (op != ncclSum && op != ncclAvg) {
    return ncclInvalidArgument;
  }

  if (!isSupportedRelayDataType(datatype)) {
    return ncclInvalidArgument;
  }

  // Check if all recvCounts are zero
  bool allZero = true;
  for (int g = 0; g < nGroups; g++) {
    if (recvCounts[g] != 0) {
      allZero = false;
      break;
    }
  }
  if (allZero) {
    return ncclSuccess;
  }

  size_t elementSize = ncclTypeSize(datatype);

  // Compute divisor for reduction: 1 for SUM, nActiveRanksPerGroup for AVG
  int reductionDivisor = (op == ncclAvg) ? nActiveRanksPerGroup : 1;

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

  if (nActiveRanksPerGroup == 2) {
    // 2-active path is unchanged internally; feed it per-group contiguous
    // buffers. Helper groups use their scratch (recvBuffs[g]); the active
    // group uses its caller input/output buffers directly.
    const void* sendBuffs2[SHARDED_RELAY_MAX_GROUPS];
    void* recvBuffs2[SHARDED_RELAY_MAX_GROUPS];
    for (int g = 0; g < nGroups; g++) {
      sendBuffs2[g] = sendBuffs[g];
      recvBuffs2[g] = recvBuffs[g];
    }
    return shardedRelayReduceScatter2Active(
        sendBuffs2,
        recvBuffs2,
        recvCounts,
        datatype,
        reductionDivisor,
        comm,
        stream,
        configs,
        myActiveGroup,
        numHelpers,
        nGroups,
        elementSize);
  }
  // A>2: bandwidth-optimal recursive-halving relay + woven direct all-to-all.
  return shardedRelayReduceScatterRecursive(
      sendBuffs,
      recvBuffs,
      recvCounts,
      datatype,
      reductionDivisor,
      comm,
      stream,
      configs,
      myActiveGroup,
      numHelpers,
      nActiveRanksPerGroup,
      nGroups,
      elementSize);
}
