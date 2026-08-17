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

/**
 * Rank Configuration for Sharded Relay Reduce-Scatter
 *
 * Holds parsed active and helper rank information for a single group.
 * NOTE: This implementation requires exactly 2 active ranks per group.
 */
struct ShardedRelayRankConfig {
  int activeRanks[2]; // Active rank IDs (exactly 2 required)
  int nActiveRanks; // Number of active ranks (must be 2)
  int helperRanks[SHARDED_RELAY_MAX_HELPERS]; // Helper rank IDs
  int numHelpers; // Number of helper ranks
  bool isActiveRank; // Is current rank active?
  int myActiveIndex; // Index in activeRanks array (-1 if helper)
  int myHelperIndex; // Index in helperRanks array (-1 if active)
};

/**
 * Build rank configuration from provided active ranks array.
 * NOTE: This implementation requires exactly 2 active ranks per group.
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

  // Validate input - require EXACTLY 2 active ranks
  if (activeRanksInput == nullptr || nActiveRanksInput != 2) {
    return false;
  }

  // Copy active ranks and validate
  for (int i = 0; i < 2; i++) {
    int rankId = activeRanksInput[i];
    if (rankId >= 0 && rankId < nRanks) {
      config.activeRanks[config.nActiveRanks++] = rankId;
    }
  }

  // Validate: need exactly 2 valid active ranks
  if (config.nActiveRanks != 2) {
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
 * Fused Multi-Group Sharded Relay Reduce-Scatter — Phase-Synchronized
 * Passthrough.
 *
 * Reduce-scatter analogue of ncclShardedRelayMultiGroupAllReduceImpl. Each
 * group has exactly 2 active ranks; the logical collective is a 2-rank
 * reduce-scatter between them, accelerated by passthrough helpers that relay
 * sharded chunks of a single block (recvCounts[g] elements).
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

  if (sendBuffs == nullptr || recvBuffs == nullptr ||
      allActiveRanks == nullptr || recvCounts == nullptr) {
    return ncclInvalidArgument;
  }

  // Require at least 2 active ranks per group
  if (nActiveRanksPerGroup < 2) {
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
