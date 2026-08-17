/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "sharded_relay_allreduce.h"
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

/**
 * Scratch Buffer Cache Singleton
 *
 * Amortizes cudaMalloc/cudaFree costs by caching and reusing scratch buffers.
 * Thread-safe with per-device buffer management.
 *
 * Key features:
 * - Multiple buffers per device (keyed for multi-group support)
 * - Automatically grows buffer if larger size needed
 * - Never shrinks (to avoid repeated alloc/free for varying sizes)
 * - Thread-safe access with mutex protection
 */
class ScratchBufferCache {
 public:
  static ScratchBufferCache& getInstance() {
    static ScratchBufferCache instance;
    return instance;
  }

  /**
   * Get a scratch buffer with a specific key (for multi-group support).
   * Each key maintains its own buffer, allowing multiple independent scratch
   * buffers per device.
   *
   * @param key Unique key to identify this scratch buffer (e.g., group index)
   * @param requiredBytes Minimum size in bytes needed
   * @param stream CUDA stream
   * @return Pointer to device memory of at least requiredBytes
   */
  void* get(int key, size_t requiredBytes, cudaStream_t stream) {
    if (requiredBytes == 0) {
      return nullptr;
    }

    int device;
    cudaGetDevice(&device);

    std::lock_guard<std::mutex> lock(mutex_);

    // Create composite key from device and user key.
    // Use 4096 as multiplier to avoid collisions and allow for future growth
    // (SHARDED_RELAY_MAX_GROUPS = 8, so keys are at most a few hundred).
    int64_t compositeKey = static_cast<int64_t>(device) * 4096 + key;
    auto& entry = buffers_[compositeKey];

    if (entry.buffer == nullptr || entry.size < requiredBytes) {
      if (entry.buffer != nullptr) {
        // Use async free to avoid blocking - memory will be freed after
        // all preceding operations on this stream complete
        cudaFreeAsync(entry.buffer, stream);
      }

      size_t allocSize = requiredBytes;
      if (allocSize >= 1024 * 1024) {
        // Round up to next 64MB boundary for larger buffers
        allocSize =
            ((requiredBytes + 64 * 1024 * 1024 - 1) / (64 * 1024 * 1024)) *
            (64 * 1024 * 1024);
      }

      // Use async malloc to avoid blocking - this is critical for avoiding
      // deadlocks when different ranks reach this point at different times
      // while others are waiting in NCCL collectives
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

  /**
   * Clear all cached buffers. Call during shutdown or when memory pressure is
   * high.
   */
  void clear(cudaStream_t stream = nullptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& pair : buffers_) {
      if (pair.second.buffer != nullptr) {
        // Use async free to match async allocation
        cudaFreeAsync(pair.second.buffer, stream);
        pair.second.buffer = nullptr;
        pair.second.size = 0;
      }
    }
    buffers_.clear();
  }

  // Prevent copying
  ScratchBufferCache(const ScratchBufferCache&) = delete;
  ScratchBufferCache& operator=(const ScratchBufferCache&) = delete;

 private:
  ScratchBufferCache() = default;

  ~ScratchBufferCache() {
    // Note: Don't call cudaFree in destructor during program exit,
    // as CUDA runtime may already be shut down.
    // Buffers will be automatically freed when the process exits.
  }

  struct BufferEntry {
    void* buffer = nullptr;
    size_t size = 0;
  };

  std::mutex mutex_;
  std::unordered_map<int64_t, BufferEntry> buffers_; // compositeKey -> buffer
};

/**
 * GPU kernel for incremental reduction: output[i] += input[i]
 * Used to add received chunks directly into the buffer.
 *
 * Kernel and launcher live in sharded_relay_allreduce_kernels.cu so the
 * `__global__` body is preserved when the surrounding TU is compiled with
 * `--offload-host-only` (see comments in def_build.bzl).
 */

#define LAUNCH_INCREMENTAL_ADD_KERNEL(TYPE, output, input, count, stream) \
  launchIncrementalAddKernel<TYPE>(output, input, count, stream)

/**
 * Helper macro to dispatch incremental add kernel by datatype
 */
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

/**
 * GPU kernel for scaling: output[i] = output[i] / divisor
 * Used to compute average after sum reduction (for ncclAvg operation).
 *
 * Kernel and launcher live in sharded_relay_allreduce_kernels.cu.
 */

#define LAUNCH_SCALE_KERNEL(TYPE, data, count, divisor, stream) \
  launchScaleKernel<TYPE>(data, count, divisor, stream)

/**
 * Helper macro to dispatch scale kernel by datatype
 */
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

/**
 * GPU kernel for fused incremental add + scale:
 *   output[i] = (output[i] + input[i]) / divisor
 *
 * Combines DISPATCH_INCREMENTAL_ADD + DISPATCH_SCALE into a single HBM pass
 * (read output, read input, write output once instead of twice).  Used by
 * the active rank to merge passthrough relay scratch into recvbuff while
 * applying the AVG divisor in one fused kernel.
 *
 * When divisor == 1, this collapses to a plain incremental add — but the
 * caller should prefer DISPATCH_INCREMENTAL_ADD in that case to avoid the
 * unnecessary divide.
 *
 * Kernel and launcher live in sharded_relay_allreduce_kernels.cu.
 */

#define LAUNCH_INCREMENTAL_ADD_AND_SCALE_KERNEL( \
    TYPE, output, input, count, divisor, stream) \
  launchIncrementalAddAndScaleKernel<TYPE>(      \
      output, input, count, divisor, stream)

/**
 * Helper macro to dispatch fused incremental-add + scale kernel by datatype
 */
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

/**
 * GPU kernel for fused reduction: output[i] = (a[i] + b[i]) / divisor
 * When divisor == 1, this is a simple sum: output[i] = a[i] + b[i]
 * When divisor == 2, this computes the average: output[i] = (a[i] + b[i]) / 2
 *
 * Used by helper ranks to combine data from both active ranks and compute
 * sum or average in a single kernel launch (avoiding separate add + scale).
 *
 * Kernel and launcher live in sharded_relay_allreduce_kernels.cu.
 */

#define LAUNCH_FUSED_REDUCE_KERNEL(                       \
    TYPE, output, inputA, inputB, count, divisor, stream) \
  launchFusedReduceKernel<TYPE>(output, inputA, inputB, count, divisor, stream)

/**
 * Helper macro to dispatch fused reduce kernel by datatype
 * divisor == 1 for SUM, divisor == 2 for AVG (with 2 active ranks)
 */
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
// kernel here. It also keeps ncclTypeSize()'s int -1 for an unknown type out of
// the `size_t elementSize` below, where it would become SIZE_MAX rather than 0.
// Keep this list in sync with the DISPATCH_* macros above.
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

// Maximum number of helper ranks supported per group.
static constexpr int SHARDED_RELAY_MAX_HELPERS = 8;

// Maximum number of active ranks per group. The hypercube exchange schedule
// (round-r partner = myActiveIndex XOR round) requires nActiveRanks to be a
// power of two; supported values are 2 and 4 (on an 8-GPU node this leaves 6
// or 4 helpers respectively).
static constexpr int SHARDED_RELAY_MAX_ACTIVE = 8;

// Returns true if v is a power of two (v >= 1).
static inline bool isPowerOfTwo(int v) {
  return v > 0 && (v & (v - 1)) == 0;
}

/**
 * Rank Configuration for Sharded Relay AllReduce
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
 * NOTE: This implementation requires exactly 2 active ranks per group.
 *
 * @param nRanks Total number of ranks in the communicator
 * @param rank Current rank ID
 * @param activeRanksInput Array of active rank IDs from caller (must have
 * exactly 2)
 * @param nActiveRanksInput Number of active ranks (must be exactly 2)
 * @param config Output configuration struct
 * @return true if configuration is valid, false otherwise
 */
static bool buildShardedRelayRankConfig(
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
  // Bounded by SHARDED_RELAY_MAX_HELPERS to prevent stack buffer overflow
  // for communicators with more than (SHARDED_RELAY_MAX_HELPERS + 2) ranks.
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

/**
 * Two-active sharded relay allreduce (original, performant path).
 *
 * Helpers forward slot a -> activeRanks[1-a]; the active rank batches all
 * numHelpers forwarded chunks into relay scratch, fuses add+scale into
 * recvBuff, then directly exchanges the final chunk between the two active
 * ranks.
 */
static ncclResult_t shardedRelayAllReduce2Active(
    const void* const* sendBuffs,
    void* const* recvBuffs,
    const size_t* counts,
    ncclDataType_t datatype,
    int reductionDivisor,
    ncclComm_t comm,
    cudaStream_t stream,
    const ShardedRelayRankConfig* configs,
    int myActiveGroup,
    int numHelpers,
    int nGroups,
    size_t elementSize) {
  int numChunks = numHelpers + 1; // 7 for 8 ranks with 2 active per group

  // ==========================================================================
  // SIZE-ADAPTIVE PURE-DIRECT FAST PATH (A==2)
  // ==========================================================================
  // At small sizes the 3-group helper relay (scatter/forward/direct + a helper
  // HBM round trip) is dominated by launch+handshake latency. Instead do a
  // classic 2-rank RS+AG allreduce directly between the two active ranks (two
  // groups, helpers idle): swap owned halves and reduce, then swap the reduced
  // halves back. maxBytes (count*elemSize) equals the bench per-rank input
  // label.
  size_t maxCount2 = 0;
  for (int g = 0; g < nGroups; g++) {
    if (counts[g] > maxCount2) {
      maxCount2 = counts[g];
    }
  }
  const size_t maxBytes2 = maxCount2 * elementSize;
  // A=2 relay wins big at large (one group across all helpers -> ~1.7x fused);
  // pure-direct only helps small. Crossover (measured MI350X, bf16, 8 GPUs):
  // fused relay overtakes at ~4.5 MB, independent at ~9 MB. Cross over below.
  const size_t kA2PureDirectMaxBytes = (nGroups > 1)
      ? (static_cast<size_t>(2) << 20) // fused: < 2 MB
      : (static_cast<size_t>(6) << 20); // independent: < 6 MB
  if (maxBytes2 < kA2PureDirectMaxBytes) {
    void* pdScratch = nullptr;
    size_t pdCount = 0, pdOwnOff = 0, pdOwnLen = 0, pdOthOff = 0, pdOthLen = 0;
    int pdPartner = -1;
    if (myActiveGroup >= 0 && counts[myActiveGroup] > 0) {
      const ShardedRelayRankConfig& cfg = configs[myActiveGroup];
      pdCount = counts[myActiveGroup];
      size_t h0 = pdCount / 2;
      size_t h1 = pdCount - h0;
      int mi = cfg.myActiveIndex;
      pdOwnOff = (mi == 0) ? 0 : h0;
      pdOwnLen = (mi == 0) ? h0 : h1;
      pdOthOff = (mi == 0) ? h0 : 0;
      pdOthLen = (mi == 0) ? h1 : h0;
      pdPartner = cfg.activeRanks[1 - mi];
      pdScratch = ScratchBufferCache::getInstance().get(
          SHARDED_RELAY_MAX_GROUPS, (pdOwnLen + 1) * elementSize, stream);
      if (pdScratch == nullptr) {
        return ncclInternalError;
      }
      // Seed recvBuff with local contribution (out-of-place); in-place already
      // holds it.
      if (sendBuffs[myActiveGroup] != recvBuffs[myActiveGroup]) {
        cudaMemcpyAsync(
            recvBuffs[myActiveGroup],
            sendBuffs[myActiveGroup],
            pdCount * elementSize,
            cudaMemcpyDeviceToDevice,
            stream);
      }
    }

    // Group 1 (reduce-scatter swap): send my partner's owned half; receive my
    // owned half's other contribution into scratch.
    NCCLCHECK(ncclGroupStart());
    for (int g = 0; g < nGroups; g++) {
      if (counts[g] == 0) {
        continue;
      }
      const ShardedRelayRankConfig& cfg = configs[g];
      if (!cfg.isActiveRank) {
        continue; // helpers idle
      }
      char* recvbuff = static_cast<char*>(recvBuffs[g]);
      if (pdOthLen > 0) {
        NCCLCHECK(ncclSend(
            recvbuff + pdOthOff * elementSize,
            pdOthLen,
            datatype,
            pdPartner,
            comm,
            stream));
      }
      if (pdOwnLen > 0) {
        NCCLCHECK(ncclRecv(
            static_cast<char*>(pdScratch),
            pdOwnLen,
            datatype,
            pdPartner,
            comm,
            stream));
      }
    }
    NCCLCHECK(ncclGroupEnd());

    // Reduce my owned half (local + received), scale for AVG.
    if (myActiveGroup >= 0 && pdOwnLen > 0) {
      char* ownDst =
          static_cast<char*>(recvBuffs[myActiveGroup]) + pdOwnOff * elementSize;
      if (reductionDivisor > 1) {
        DISPATCH_INCREMENTAL_ADD_AND_SCALE(
            datatype, ownDst, pdScratch, pdOwnLen, reductionDivisor, stream);
      } else {
        DISPATCH_INCREMENTAL_ADD(datatype, ownDst, pdScratch, pdOwnLen, stream);
      }
    }

    // Group 2 (all-gather swap): exchange the reduced owned halves.
    NCCLCHECK(ncclGroupStart());
    for (int g = 0; g < nGroups; g++) {
      if (counts[g] == 0) {
        continue;
      }
      const ShardedRelayRankConfig& cfg = configs[g];
      if (!cfg.isActiveRank) {
        continue;
      }
      char* recvbuff = static_cast<char*>(recvBuffs[g]);
      if (pdOwnLen > 0) {
        NCCLCHECK(ncclSend(
            recvbuff + pdOwnOff * elementSize,
            pdOwnLen,
            datatype,
            pdPartner,
            comm,
            stream));
      }
      if (pdOthLen > 0) {
        NCCLCHECK(ncclRecv(
            recvbuff + pdOthOff * elementSize,
            pdOthLen,
            datatype,
            pdPartner,
            comm,
            stream));
      }
    }
    NCCLCHECK(ncclGroupEnd());
    return ncclSuccess;
  }

  // =========================================================================
  // CALCULATE PER-GROUP CHUNK SIZES
  // =========================================================================
  // Each group may have a different count, so we compute chunk sizes per group
  size_t chunkSizes[SHARDED_RELAY_MAX_GROUPS];
  size_t lastChunkSizes[SHARDED_RELAY_MAX_GROUPS];
  size_t directChunkOffsets[SHARDED_RELAY_MAX_GROUPS];
  size_t directChunkSizes[SHARDED_RELAY_MAX_GROUPS];

  for (int g = 0; g < nGroups; g++) {
    size_t count = counts[g];

    // Skip groups with count == 0; the per-phase loops below already check
    // counts[g] == 0 and bypass NCCL ops for those groups, so chunkSizes
    // entries for zero-count groups are never read.
    if (count == 0) {
      chunkSizes[g] = 0;
      lastChunkSizes[g] = 0;
      directChunkOffsets[g] = 0;
      directChunkSizes[g] = 0;
      continue;
    }

    // Calculate chunk size (aligned to cache line).
    // The algorithm scatters numChunks chunks of size chunkSize over the
    // input buffer, which requires count >= numChunks * chunkSize. When the
    // per-chunk size rounded down to CHUNK_ALIGN_ELEMENTS is zero, the
    // buffer is too small to scatter and the algorithm cannot proceed
    // safely; the caller should fall back to a regular allreduce.
    size_t chunkSize = count / numChunks;
    chunkSize = (chunkSize / CHUNK_ALIGN_ELEMENTS) * CHUNK_ALIGN_ELEMENTS;
    if (chunkSize == 0) {
      return ncclInvalidArgument;
    }
    chunkSizes[g] = chunkSize;

    // Calculate the size of the last chunk
    size_t totalChunkedElements = static_cast<size_t>(numChunks) * chunkSize;
    size_t lastChunkSize = chunkSize;
    if (totalChunkedElements < count) {
      lastChunkSize = chunkSize + (count - totalChunkedElements);
    }
    lastChunkSizes[g] = lastChunkSize;

    // Direct exchange chunk info
    int directChunkIndex = numHelpers;
    directChunkOffsets[g] = static_cast<size_t>(directChunkIndex) * chunkSize;
    directChunkSizes[g] = lastChunkSize;
  }

  // =========================================================================
  // SCRATCH BUFFER ALLOCATION
  // =========================================================================
  // Relay scratch: numHelpers × chunkSize for batched passthrough recv.
  //   Sized to receive ALL forwarded chunks from helpers in a single
  //   ncclGroupStart/End — matches original phase-sync structure.
  // Direct-exchange scratch: (nActiveRanks-1) × directChunkSize (in-place)
  void* relayScratch = nullptr;
  void* directScratch = nullptr;

  if (myActiveGroup >= 0 && counts[myActiveGroup] > 0) {
    // Relay scratch sized to numHelpers × chunkSize so that the active rank
    // can receive ALL forwarded chunks in one batched phase.
    size_t relayScratchBytes = static_cast<size_t>(numHelpers) *
        chunkSizes[myActiveGroup] * elementSize;
    relayScratch = ScratchBufferCache::getInstance().get(
        SHARDED_RELAY_MAX_GROUPS, relayScratchBytes, stream);
    if (relayScratch == nullptr) {
      return ncclInternalError;
    }

    // Direct-exchange scratch (in-place only)
    bool isInPlace = (sendBuffs[myActiveGroup] == recvBuffs[myActiveGroup]);
    if (isInPlace) {
      int nOtherActives = configs[myActiveGroup].nActiveRanks - 1;
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
  // PHASE 1 (active→helpers): Both active ranks scatter chunks to helpers
  // =========================================================================
  // Helpers receive from each active rank into offset-based slots:
  //   slot a at offset (a × chunkSize) holds data from active rank a
  NCCLCHECK(ncclGroupStart());

  for (int g = 0; g < nGroups; g++) {
    if (counts[g] == 0)
      continue;
    const ShardedRelayRankConfig& cfg = configs[g];
    const void* sendbuff = sendBuffs[g];
    void* recvbuff = recvBuffs[g];
    size_t chunkSize = chunkSizes[g];

    if (cfg.isActiveRank) {
      // Active rank: send my chunk h to helper h
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

  // For out-of-place active groups: copy sendbuff relay region to recvbuff
  // so that the incremental add in Phase 3 works uniformly.
  if (myActiveGroup >= 0 && counts[myActiveGroup] > 0) {
    const void* sendbuff = sendBuffs[myActiveGroup];
    void* recvbuff = recvBuffs[myActiveGroup];
    if (sendbuff != recvbuff) {
      size_t relayBytes = static_cast<size_t>(numHelpers) *
          chunkSizes[myActiveGroup] * elementSize;
      cudaMemcpyAsync(
          recvbuff, sendbuff, relayBytes, cudaMemcpyDeviceToDevice, stream);
    }
  }

  // =========================================================================
  // PHASE 2 (helpers→active, batched): Passthrough forward
  // =========================================================================
  // ALL helpers forward simultaneously in ONE ncclGroupStart/End.
  // Each helper sends slot 0 (a0's data) → a1 and slot 1 (a1's data) → a0.
  // Active rank receives all numHelpers chunks into relay scratch
  // (numHelpers × chunkSize), at offset h × chunkSize per helper h.
  NCCLCHECK(ncclGroupStart());

  for (int g = 0; g < nGroups; g++) {
    if (counts[g] == 0)
      continue;
    const ShardedRelayRankConfig& cfg = configs[g];
    void* recvbuff = recvBuffs[g];
    size_t chunkSize = chunkSizes[g];

    if (!cfg.isActiveRank) {
      // I am a helper for group g: forward each slot to the OTHER active.
      // slot a → activeRanks[1-a] (swap for 2 active ranks)
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
  // Single-pass fused kernel: recvbuff[i] = (recvbuff[i] + relayScratch[i]) /
  // divisor.  Halves HBM traffic vs separate ADD + SCALE passes.
  for (int g = 0; g < nGroups; g++) {
    if (counts[g] == 0)
      continue;
    const ShardedRelayRankConfig& cfg = configs[g];
    if (cfg.isActiveRank) {
      void* recvbuff = recvBuffs[g];
      size_t chunkSize = chunkSizes[g];
      size_t relayTotal = static_cast<size_t>(numHelpers) * chunkSize;

      if (reductionDivisor > 1) {
        // Fused: add + AVG-scale in one HBM pass.
        DISPATCH_INCREMENTAL_ADD_AND_SCALE(
            datatype,
            recvbuff,
            relayScratch,
            relayTotal,
            reductionDivisor,
            stream);
      } else {
        // SUM only: plain incremental add.
        DISPATCH_INCREMENTAL_ADD(
            datatype, recvbuff, relayScratch, relayTotal, stream);
      }
    }
  }

  // =========================================================================
  // PHASE 4 (active↔active): Direct exchange between active ranks
  // =========================================================================
  NCCLCHECK(ncclGroupStart());

  for (int g = 0; g < nGroups; g++) {
    if (counts[g] == 0)
      continue;
    const ShardedRelayRankConfig& cfg = configs[g];

    if (cfg.isActiveRank) {
      const void* sendbuff = sendBuffs[g];
      void* recvbuff = recvBuffs[g];
      bool isInPlace = (sendbuff == recvbuff);
      size_t directChunkOffset = directChunkOffsets[g];
      size_t directChunkSize = directChunkSizes[g];

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
    if (counts[g] == 0)
      continue;
    const ShardedRelayRankConfig& cfg = configs[g];

    if (cfg.isActiveRank) {
      const void* sendbuff = sendBuffs[g];
      void* recvbuff = recvBuffs[g];
      bool isInPlace = (sendbuff == recvbuff);
      size_t directChunkOffset = directChunkOffsets[g];
      size_t directChunkSize = directChunkSizes[g];

      void* directChunkDst =
          static_cast<char*>(recvbuff) + directChunkOffset * elementSize;

      if (isInPlace) {
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
        const void* localContribution = static_cast<const char*>(sendbuff) +
            directChunkOffset * elementSize;
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
 * AllReduce for > 2 active ranks (flat helper-reduce-AND-broadcast). Combines
 * the two patterns that individually beat NCCL at 4-active: the
 * reduce-AT-helper of the flat reduce-scatter and the broadcast-AT-helper of
 * the flat all-gather.
 *
 * Each rank's `count` is split into a DIRECT region pD (allreduced among the A
 * active ranks over the 1-hop intra links via a direct all-to-all
 * reduce-scatter + all-gather) and an OFFLOAD region pO (allreduced 2-hop
 * through the otherwise-idle helpers: every helper SUMS all A active ranks'
 * chunk and BROADCASTS the result back to all A). Both regions run concurrently
 * in two ncclGroups -- G1 scatter and G2 gather/broadcast -- with the
 * reductions in between. With f = pO/count = 0.5 each XGMI link carries
 * ~0.25*count (intra: 0.5*(1-f)*count for the direct RS+AG; cross:
 * 2*f*count/H for the offload scatter+broadcast), vs ~0.5*count/link for a
 * 4-rank NCCL allreduce that uses only the intra links. Replaces the
 * recursive-halving path.
 *
 * Scratch: dScratch holds the A-1 received direct shards. The helper's scratch
 * is recvBuffs[g] = rawBuf [0, A*oChunk) (one chunk per active source); the
 * fused helper reduce sums all A chunks in place into rawBuf[0], which is then
 * broadcast back to the active ranks. recvBuff is operated on in
 * place after being seeded from sendBuff (out-of-place) or aliasing it
 * (in-place). Requires count % A == 0 and assumes A == numHelpers (the 8-GPU
 * 4-active topology); the AVG divisor is applied once over the whole count.
 */
static ncclResult_t shardedRelayAllReduceFlat(
    const void* const* sendBuffs,
    void* const* recvBuffs,
    const size_t* counts,
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
  const int H = numHelpers;

  // Offload (helper reduce+broadcast) share. Empirically ~0.78 is optimal at
  // large sizes on MI350X (the offload path through otherwise-idle cross links
  // is more efficient per byte than direct intra RS+AG). But at small sizes the
  // 2-hop offload only adds helper-hop latency, so disable it there and run a
  // pure-direct RS+AG among the A active ranks (helpers idle). maxBytes here
  // (count*elemSize) equals the bench per-rank input label; crossover set
  // below.
  size_t maxCount = 0;
  for (int g = 0; g < nGroups; g++) {
    if (counts[g] > maxCount) {
      maxCount = counts[g];
    }
  }
  const size_t maxBytes = maxCount * elementSize;
  // Crossover is scenario-dependent (measured MI350X, A=4, bf16, 8 GPUs): fused
  // offload overtakes pure-direct at ~4.5 MB (pure-direct wins the small end,
  // e.g. 576 KB 1.08x->1.23x), independent at ~27 MB (0.62->0.74 @4KB,
  // 0.69->0.86 @4.5MB). Independent has no cross-group contention so direct
  // holds on longer. A=4 fused was already >=1.02x, so keep the threshold low
  // there.
  const size_t kFlatPureDirectMaxBytes = (nGroups > 1)
      ? 0 // fused A=4 already >=1.02x with offload; pure-direct doesn't help
      : (static_cast<size_t>(12) << 20); // independent: < 12 MB
  const size_t kOffPermille = (maxBytes < kFlatPureDirectMaxBytes) ? 0 : 780;

  for (int g = 0; g < nGroups; g++) {
    if (counts[g] != 0 && (counts[g] % static_cast<size_t>(A)) != 0) {
      return ncclInvalidArgument;
    }
  }

  // Per-group geometry: pO (offload) aligned to H*CHUNK_ALIGN; pD = count - pO
  // is then divisible by A (== H). dShard = pD/A, oChunk = pO/H.
  size_t pDArr[SHARDED_RELAY_MAX_GROUPS];
  size_t pOArr[SHARDED_RELAY_MAX_GROUPS];
  for (int g = 0; g < nGroups; g++) {
    size_t count = counts[g];
    if (count == 0) {
      pDArr[g] = 0;
      pOArr[g] = 0;
      continue;
    }
    size_t alignH = static_cast<size_t>(H) * CHUNK_ALIGN_ELEMENTS;
    size_t pO = (count * kOffPermille) / 1000;
    pO = (pO / alignH) * alignH;
    if (pO > count) {
      pO = (count / alignH) * alignH;
    }
    pOArr[g] = pO;
    pDArr[g] = count - pO;
  }

  // Out-of-place: seed recvBuff = sendBuff once, then operate in place.
  if (myActiveGroup >= 0 && counts[myActiveGroup] > 0 &&
      sendBuffs[myActiveGroup] != recvBuffs[myActiveGroup]) {
    cudaMemcpyAsync(
        recvBuffs[myActiveGroup],
        sendBuffs[myActiveGroup],
        counts[myActiveGroup] * elementSize,
        cudaMemcpyDeviceToDevice,
        stream);
  }

  // Direct reduce-scatter scratch: the A-1 received direct shards.
  void* dScratch = nullptr;
  if (myActiveGroup >= 0 && counts[myActiveGroup] > 0 &&
      pDArr[myActiveGroup] > 0) {
    size_t dShard = pDArr[myActiveGroup] / A;
    size_t dBytes = static_cast<size_t>(A - 1) * dShard * elementSize;
    dScratch = ScratchBufferCache::getInstance().get(
        SHARDED_RELAY_MAX_GROUPS, dBytes, stream);
    if (dScratch == nullptr) {
      return ncclInternalError;
    }
  }

  // ===== Group 1: direct RS (active<->active) + offload scatter
  // (active->helper). ========================================================
  NCCLCHECK(ncclGroupStart());
  for (int g = 0; g < nGroups; g++) {
    if (counts[g] == 0)
      continue;
    const ShardedRelayRankConfig& cfg = configs[g];
    size_t pD = pDArr[g];
    size_t pO = pOArr[g];
    size_t dShard = (pD > 0) ? pD / A : 0;
    size_t oChunk = (pO > 0) ? pO / H : 0;
    if (cfg.isActiveRank) {
      char* recvbuff = static_cast<char*>(recvBuffs[g]);
      int m = cfg.myActiveIndex;
      // Direct RS: send my shard k to owner k; recv source s's shard m into its
      // dScratch slot p.
      if (dShard > 0) {
        for (int k = 0; k < A; k++) {
          if (k == m)
            continue;
          NCCLCHECK(ncclSend(
              recvbuff + static_cast<size_t>(k) * dShard * elementSize,
              dShard,
              datatype,
              cfg.activeRanks[k],
              comm,
              stream));
        }
        for (int s = 0; s < A; s++) {
          if (s == m)
            continue;
          int p = (s < m) ? s : s - 1;
          NCCLCHECK(ncclRecv(
              static_cast<char*>(dScratch) +
                  static_cast<size_t>(p) * dShard * elementSize,
              dShard,
              datatype,
              cfg.activeRanks[s],
              comm,
              stream));
        }
      }
      // Offload scatter: send my offload chunk h to helper h.
      if (oChunk > 0) {
        for (int h = 0; h < cfg.numHelpers; h++) {
          NCCLCHECK(ncclSend(
              recvbuff + (pD + static_cast<size_t>(h) * oChunk) * elementSize,
              oChunk,
              datatype,
              cfg.helperRanks[h],
              comm,
              stream));
        }
      }
    } else if (oChunk > 0) {
      // Helper: recv this helper's offload chunk from each active a into
      // rawBuf[a].
      char* rawBuf = static_cast<char*>(recvBuffs[g]);
      for (int a = 0; a < cfg.nActiveRanks; a++) {
        NCCLCHECK(ncclRecv(
            rawBuf + static_cast<size_t>(a) * oChunk * elementSize,
            oChunk,
            datatype,
            cfg.activeRanks[a],
            comm,
            stream));
      }
    }
  }
  NCCLCHECK(ncclGroupEnd());

  // ===== Reduce: direct (active sums its shard) + offload (helper sums all A).
  for (int g = 0; g < nGroups; g++) {
    if (counts[g] == 0)
      continue;
    const ShardedRelayRankConfig& cfg = configs[g];
    size_t pD = pDArr[g];
    size_t pO = pOArr[g];
    size_t dShard = (pD > 0) ? pD / A : 0;
    size_t oChunk = (pO > 0) ? pO / H : 0;
    if (cfg.isActiveRank) {
      if (dShard > 0 && g == myActiveGroup) {
        char* recvbuff = static_cast<char*>(recvBuffs[g]);
        int m = cfg.myActiveIndex;
        char* dst = recvbuff + static_cast<size_t>(m) * dShard * elementSize;
        // Fused single-pass reduce: dst = (dst + A-1 direct contribs) / divisor
        // (AVG folded in), before the all-gather broadcasts the owned shard.
        DISPATCH_MULTI_REDUCE(
            datatype, dst, dScratch, A - 1, dShard, reductionDivisor, stream);
      }
    } else if (oChunk > 0) {
      // Helper: fused single-pass reduce of the A received chunks into
      // rawBuf[0] in-place (rawBuf[0] = (sum_a rawBuf[a]) / divisor), one
      // launch instead of memcpy + A-1 adds + scale. The broadcast below reads
      // rawBuf[0].
      char* rawBuf = static_cast<char*>(recvBuffs[g]);
      DISPATCH_MULTI_REDUCE(
          datatype,
          rawBuf,
          rawBuf + oChunk * elementSize,
          cfg.nActiveRanks - 1,
          oChunk,
          reductionDivisor,
          stream);
    }
  }

  // ===== Group 2: direct AG (active<->active) + offload broadcast
  // (helper->active). ========================================================
  NCCLCHECK(ncclGroupStart());
  for (int g = 0; g < nGroups; g++) {
    if (counts[g] == 0)
      continue;
    const ShardedRelayRankConfig& cfg = configs[g];
    size_t pD = pDArr[g];
    size_t pO = pOArr[g];
    size_t dShard = (pD > 0) ? pD / A : 0;
    size_t oChunk = (pO > 0) ? pO / H : 0;
    if (cfg.isActiveRank) {
      char* recvbuff = static_cast<char*>(recvBuffs[g]);
      int m = cfg.myActiveIndex;
      // Direct AG: send my reduced shard m to every other active; recv their
      // reduced shard k into recvBuff[k].
      if (dShard > 0) {
        for (int k = 0; k < A; k++) {
          if (k == m)
            continue;
          NCCLCHECK(ncclSend(
              recvbuff + static_cast<size_t>(m) * dShard * elementSize,
              dShard,
              datatype,
              cfg.activeRanks[k],
              comm,
              stream));
          NCCLCHECK(ncclRecv(
              recvbuff + static_cast<size_t>(k) * dShard * elementSize,
              dShard,
              datatype,
              cfg.activeRanks[k],
              comm,
              stream));
        }
      }
      // Offload broadcast: recv reduced chunk h from helper h into recvBuff.
      if (oChunk > 0) {
        for (int h = 0; h < cfg.numHelpers; h++) {
          NCCLCHECK(ncclRecv(
              recvbuff + (pD + static_cast<size_t>(h) * oChunk) * elementSize,
              oChunk,
              datatype,
              cfg.helperRanks[h],
              comm,
              stream));
        }
      }
    } else if (oChunk > 0) {
      // Helper: broadcast the reduced chunk (now in rawBuf[0]) to all A active
      // ranks.
      char* rawBuf = static_cast<char*>(recvBuffs[g]);
      for (int a = 0; a < cfg.nActiveRanks; a++) {
        NCCLCHECK(ncclSend(
            rawBuf, oChunk, datatype, cfg.activeRanks[a], comm, stream));
      }
    }
  }
  NCCLCHECK(ncclGroupEnd());

  // The AVG divisor was already folded into the per-shard direct reduce and the
  // per-chunk helper reduce above, so no separate whole-count scale is needed.
  return ncclSuccess;
}

/**
 * Fused Multi-Group Sharded Relay AllReduce.
 *
 * Executes multiple sharded relay allreduces in one fused call, phase-synced
 * across all groups so XGMI links carry unidirectional traffic. Helpers are
 * pure passthrough; reductions happen on the active ranks. Each rank is ACTIVE
 * for exactly one group and a HELPER for the others.
 *
 * nActiveRanksPerGroup must be a power of two (2 or 4): A==2 uses the original
 * 2-active path; A>2 uses the flat helper-reduce-and-broadcast path.
 */
HOT ncclResult_t ncclShardedRelayMultiGroupAllReduceImpl(
    const void* const* sendBuffs,
    void* const* recvBuffs,
    const size_t* counts,
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

  // Validate every argument before touching counts: the all-zero scan below
  // indexes counts[0..nGroups), so a null pointer or an out-of-range nGroups
  // has to be rejected first. Bounds-checking nGroups up here also means
  // nGroups <= 0 reports ncclInvalidArgument rather than skipping the scan
  // entirely and returning ncclSuccess.
  if (nGroups < 1 || nGroups > SHARDED_RELAY_MAX_GROUPS) {
    return ncclInvalidArgument;
  }
  if (recvBuffs == nullptr || allActiveRanks == nullptr || counts == nullptr ||
      sendBuffs == nullptr) {
    return ncclInvalidArgument;
  }

  // Require a power-of-two active-rank count (>= 2) for the XOR schedule.
  if (nActiveRanksPerGroup < 2 || !isPowerOfTwo(nActiveRanksPerGroup)) {
    return ncclInvalidArgument;
  }

  if (op != ncclSum && op != ncclAvg) {
    return ncclInvalidArgument;
  }

  if (!isSupportedRelayDataType(datatype)) {
    return ncclInvalidArgument;
  }

  bool allZero = true;
  for (int g = 0; g < nGroups; g++) {
    if (counts[g] != 0) {
      allZero = false;
      break;
    }
  }
  if (allZero) {
    return ncclSuccess;
  }

  size_t elementSize = ncclTypeSize(datatype);
  int reductionDivisor = (op == ncclAvg) ? nActiveRanksPerGroup : 1;

  ShardedRelayRankConfig configs[SHARDED_RELAY_MAX_GROUPS];
  int myActiveGroup = -1;
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
  int numHelpers = configs[0].numHelpers;

  // Build per-group buffer arrays for the unchanged kernels. Helper groups use
  // their scratch (recvBuffs[g]); the active group uses the caller's contiguous
  // input/output buffers directly. Allreduce may be in-place (sendBuffs[g]
  // aliases recvBuffs[g]) or out-of-place, keyed off
  // sendBuffs[g]==recvBuffs[g].
  const void* sendBuffs2[SHARDED_RELAY_MAX_GROUPS];
  void* recvBuffs2[SHARDED_RELAY_MAX_GROUPS];
  for (int g = 0; g < nGroups; g++) {
    sendBuffs2[g] = sendBuffs[g];
    recvBuffs2[g] = recvBuffs[g];
  }

  ncclResult_t r;
  if (nActiveRanksPerGroup == 2) {
    r = shardedRelayAllReduce2Active(
        sendBuffs2,
        recvBuffs2,
        counts,
        datatype,
        reductionDivisor,
        comm,
        stream,
        configs,
        myActiveGroup,
        numHelpers,
        nGroups,
        elementSize);
  } else {
    // A>2: flat helper-reduce-and-broadcast (direct RS+AG over intra woven with
    // offload reduce+broadcast through the helpers).
    r = shardedRelayAllReduceFlat(
        sendBuffs2,
        recvBuffs2,
        counts,
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

  return r;
}
