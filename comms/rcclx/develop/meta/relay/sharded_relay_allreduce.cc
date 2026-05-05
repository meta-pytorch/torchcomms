/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "sharded_relay_allreduce.h"
#include "comm.h"

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
 */
template <typename T>
__global__ void incrementalAddKernel(T* output, const T* input, size_t count) {
  size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
  size_t totalThreads = blockDim.x * gridDim.x;

  for (size_t elemIdx = threadId; elemIdx < count; elemIdx += totalThreads) {
    output[elemIdx] += input[elemIdx];
  }
}

template <typename T>
void launchIncrementalAddKernel(
    void* output,
    const void* input,
    size_t count,
    cudaStream_t stream) {
  const int blockSize = 256;
  int gridSize = (count + blockSize - 1) / blockSize;
  incrementalAddKernel<T><<<gridSize, blockSize, 0, stream>>>(
      static_cast<T*>(output), static_cast<const T*>(input), count);
}

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
 */
template <typename T>
__global__ void scaleKernel(T* data, size_t count, int divisor) {
  size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
  size_t totalThreads = blockDim.x * gridDim.x;

  for (size_t elemIdx = threadId; elemIdx < count; elemIdx += totalThreads) {
    data[elemIdx] = data[elemIdx] / static_cast<T>(divisor);
  }
}

template <typename T>
void launchScaleKernel(
    void* data,
    size_t count,
    int divisor,
    cudaStream_t stream) {
  const int blockSize = 256;
  int gridSize = (count + blockSize - 1) / blockSize;
  scaleKernel<T><<<gridSize, blockSize, 0, stream>>>(
      static_cast<T*>(data), count, divisor);
}

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
 */
template <typename T>
__global__ void incrementalAddAndScaleKernel(
    T* output,
    const T* input,
    size_t count,
    int divisor) {
  size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
  size_t totalThreads = blockDim.x * gridDim.x;

  for (size_t elemIdx = threadId; elemIdx < count; elemIdx += totalThreads) {
    T sum = output[elemIdx] + input[elemIdx];
    if (divisor > 1) {
      output[elemIdx] = sum / static_cast<T>(divisor);
    } else {
      output[elemIdx] = sum;
    }
  }
}

template <typename T>
void launchIncrementalAddAndScaleKernel(
    void* output,
    const void* input,
    size_t count,
    int divisor,
    cudaStream_t stream) {
  const int blockSize = 256;
  int gridSize = (count + blockSize - 1) / blockSize;
  incrementalAddAndScaleKernel<T><<<gridSize, blockSize, 0, stream>>>(
      static_cast<T*>(output), static_cast<const T*>(input), count, divisor);
}

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
 */
template <typename T>
__global__ void fusedReduceKernel(
    T* output,
    const T* inputA,
    const T* inputB,
    size_t count,
    int divisor) {
  size_t threadId = blockIdx.x * blockDim.x + threadIdx.x;
  size_t totalThreads = blockDim.x * gridDim.x;

  for (size_t elemIdx = threadId; elemIdx < count; elemIdx += totalThreads) {
    T sum = inputA[elemIdx] + inputB[elemIdx];
    if (divisor > 1) {
      output[elemIdx] = sum / static_cast<T>(divisor);
    } else {
      output[elemIdx] = sum;
    }
  }
}

template <typename T>
void launchFusedReduceKernel(
    void* output,
    const void* inputA,
    const void* inputB,
    size_t count,
    int divisor,
    cudaStream_t stream) {
  const int blockSize = 256;
  int gridSize = (count + blockSize - 1) / blockSize;
  fusedReduceKernel<T><<<gridSize, blockSize, 0, stream>>>(
      static_cast<T*>(output),
      static_cast<const T*>(inputA),
      static_cast<const T*>(inputB),
      count,
      divisor);
}

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

// Maximum number of helper ranks supported per group.
// With exactly 2 active ranks, this implies a maximum communicator size of
// SHARDED_RELAY_MAX_HELPERS + 2 ranks for this algorithm.
static constexpr int SHARDED_RELAY_MAX_HELPERS = 8;

/**
 * Rank Configuration for Sharded Relay AllReduce
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
 * Fused Multi-Group Sharded Relay AllReduce — Phase-Synchronized Passthrough
 *
 * This implementation executes multiple sharded relay allreduces in a single
 * fused call, coordinating phases across ALL groups to prevent XGMI link
 * contention.  Helpers act as pure passthrough (no local compute); all
 * reductions are performed on the active ranks.
 *
 * Phase-Synchronized Execution with Passthrough Helpers:
 * ======================================================
 *
 *   PHASE 1 (active→helpers): ALL groups scatter simultaneously.
 *            Both active ranks send their chunks to helpers.
 *            Helpers receive into a two-slot buffer:
 *              slot 0 = data from active rank a0
 *              slot 1 = data from active rank a1
 *            All XGMI links carry unidirectional traffic: active→helper
 *
 *   PHASE 2 (helpers→active, batched): ALL helpers forward simultaneously
 *            in ONE ncclGroupStart/End.  Each helper sends slot 0 (a0's data)
 *            to a1 and slot 1 (a1's data) to a0.  Active rank receives all
 *            numHelpers chunks into relay scratch (numHelpers × chunkSize).
 *
 *   PHASE 3 (active reduce): Active ranks add the relay scratch to recvBuff
 *            using a single fused add+scale kernel
 *            (recvBuff[i] = (recvBuff[i] + scratch[i]) / nActiveRanks)
 *            over the whole relay region (numHelpers × chunkSize) — halves
 *            HBM traffic vs separate add + scale passes.
 *
 *   PHASE 4 (active↔active): Direct exchange of the last chunk between
 *            the two active ranks (same as the original algorithm).
 *
 *   PHASE 5 (active reduce): Final reduction on the direct-exchange chunk.
 *            Compute-only, no XGMI traffic.
 *
 * Memory Model:
 * =============
 * Each rank is ACTIVE for exactly ONE group (has real tensor data).
 * For other groups, the rank is a HELPER (uses provided two-slot scratch).
 *
 * For ACTIVE ranks:
 *   - sendBuff and recvBuff can be the same (in-place) or different
 *   - Relay scratch: numHelpers × chunkSize (from ScratchBufferCache, batched)
 *   - Direct-exchange scratch: (nActiveRanks-1) × directChunkSize (in-place)
 *
 * For HELPER ranks:
 *   - Buffer must hold nActiveRanks × chunkSize elements (two slots)
 *   - Slot a at offset a × chunkSize holds data from active rank a
 *   - Each helper group MUST have its own buffer (no aliasing across groups)
 *     because all groups are processed simultaneously under phase-sync
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

  // Require at least 2 active ranks per group
  if (nActiveRanksPerGroup < 2) {
    return ncclInvalidArgument;
  }

  // Check if all counts are zero
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

  // Validate operation - only SUM and AVG are supported
  if (op != ncclSum && op != ncclAvg) {
    return ncclInvalidArgument;
  }

  if (nGroups < 1 || nGroups > SHARDED_RELAY_MAX_GROUPS) {
    return ncclInvalidArgument;
  }

  if (sendBuffs == nullptr || recvBuffs == nullptr ||
      allActiveRanks == nullptr || counts == nullptr) {
    return ncclInvalidArgument;
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
  int numChunks = numHelpers + 1; // 7 for 8 ranks with 2 active per group

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
