/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "low_precision_buffer_pool.h"
#include <hip/hip_runtime.h>
#include <algorithm>
#include "comm.h"
#include "low_precision_common.h"

// Buffer pool size constants optimized for MI300X single node (8 GPUs)
#define MIN_BUFFER_POOL_ELEMENTS \
  128ULL * 1024ULL * 1024ULL // Minimum 128M elements (~2.3GB for 8 ranks)
#define MI300X_OPTIMAL_PAGE_SIZE \
  4096UL // 4KB page alignment for optimal HBM3 access
#define MI300X_CACHE_LINE_SIZE 128UL // 128B cache line alignment

/**
 * Initializes a low precision buffer pool with the specified capacity.
 * Allocates GPU memory buffers for FP8 and float operations.
 */
ncclResult_t ncclLowPrecisionBufferPoolInit(
    struct ncclLowPrecisionBufferPool* pool,
    size_t maxElements,
    int maxRanks) {
  if (pool->initialized) {
    return ncclSuccess; // Already initialized
  }

  // Calculate maximum buffer size for worst case across all collective types
  // Use MI300X-optimized alignment for better memory efficiency and cache
  // performance
  size_t floatAlignSize =
      MI300X_CACHE_LINE_SIZE; // 128B alignment for optimal HBM3 access

  // Primary FP8 buffer: for quantized input data
  // Memory access pattern: Sequential writes during quantization, random reads
  // during exchange
  size_t fp8Phase1Size = maxElements * sizeof(rccl_float8);

  // Secondary FP8 buffer: for all-to-all exchanges (worst case: maxElements *
  // maxRanks) Memory access pattern: Interleaved writes from different ranks,
  // sequential reads for reduction
  size_t fp8Phase2Size = maxElements * maxRanks * sizeof(rccl_float8);

  // AllGather FP8 buffer: for final gathering (same size as primary for most
  // cases) Memory access pattern: Sequential writes from reduction, sequential
  // reads for dequantization
  size_t fp8AllGatherSize = maxElements * sizeof(rccl_float8);

  // Float reduction buffer: for intermediate reduction results
  // Memory access pattern: Sequential writes from reduction, sequential reads
  // for requantization
  size_t floatReductionSize = maxElements * sizeof(float);

  // Float output buffer: for final output (worst case: maxElements for most
  // collectives) Memory access pattern: Sequential writes during
  // dequantization, sequential reads by user
  size_t floatOutputSize = maxElements * sizeof(float);

  size_t totalBackingBufferSize = 0;

  // FP8 Phase 1 buffer - align to cache line for optimal HBM3 bandwidth
  pool->offsets.fp8Phase1Offset = totalBackingBufferSize;
  totalBackingBufferSize += fp8Phase1Size;
  totalBackingBufferSize =
      (totalBackingBufferSize + floatAlignSize - 1) & ~(floatAlignSize - 1);

  // FP8 Phase 2 buffer - align to cache line for optimal interleaved access
  // patterns
  pool->offsets.fp8Phase2Offset = totalBackingBufferSize;
  totalBackingBufferSize += fp8Phase2Size;
  totalBackingBufferSize =
      (totalBackingBufferSize + floatAlignSize - 1) & ~(floatAlignSize - 1);

  // FP8 AllGather buffer - align to cache line for optimal sequential access
  pool->offsets.fp8AllGatherOffset = totalBackingBufferSize;
  totalBackingBufferSize += fp8AllGatherSize;
  totalBackingBufferSize =
      (totalBackingBufferSize + floatAlignSize - 1) & ~(floatAlignSize - 1);

  // Float reduction buffer - align to cache line for optimal float
  // vectorization
  pool->offsets.floatReductionOffset = totalBackingBufferSize;
  totalBackingBufferSize += floatReductionSize;
  totalBackingBufferSize =
      (totalBackingBufferSize + floatAlignSize - 1) & ~(floatAlignSize - 1);

  // Float output buffer - align to cache line for optimal final output
  pool->offsets.floatOutputOffset = totalBackingBufferSize;
  totalBackingBufferSize += floatOutputSize;

  // Align total buffer size to MI300X page boundaries for optimal HBM3 access
  totalBackingBufferSize =
      (totalBackingBufferSize + MI300X_OPTIMAL_PAGE_SIZE - 1) &
      ~(MI300X_OPTIMAL_PAGE_SIZE - 1);

  // Allocate with MI300X-optimized flags for cache-resident buffer pool (512MB)
  // Use uncached memory for better multi-kernel streaming performance
  CUDACHECK(hipExtMallocWithFlags(
      &pool->backingBuffer, totalBackingBufferSize, hipDeviceMallocUncached));

  // Pre-touch memory to ensure it's resident in HBM3 and establish optimal page
  // mappings
  CUDACHECK(hipMemset(pool->backingBuffer, 0, totalBackingBufferSize));
  CUDACHECK(hipDeviceSynchronize());

  pool->maxBufferSize = totalBackingBufferSize;
  pool->currentSize = 0;
  pool->initialized = true;

  INFO(
      NCCL_COLL,
      "MI300X-optimized buffer pool allocated: %.1fMB total (uncached HBM3, 4KB page-aligned)",
      totalBackingBufferSize / (1024.0 * 1024.0));

  return ncclSuccess;
}

HOT ncclResult_t ncclLowPrecisionBufferPoolGetBuffers(
    struct ncclLowPrecisionBufferPool* pool,
    size_t count,
    int nRanks,
    rccl_float8** fp8Phase1Buffer,
    rccl_float8** fp8Phase2Buffer,
    rccl_float8** fp8AllGatherBuffer,
    float** floatReductionBuffer,
    float** floatOutputBuffer) {
  if (!pool->initialized) {
    return ncclInvalidUsage;
  }

  char* basePtr = static_cast<char*>(pool->backingBuffer);

  if (fp8Phase1Buffer) {
    *fp8Phase1Buffer =
        reinterpret_cast<rccl_float8*>(basePtr + pool->offsets.fp8Phase1Offset);
  }
  if (fp8Phase2Buffer) {
    *fp8Phase2Buffer =
        reinterpret_cast<rccl_float8*>(basePtr + pool->offsets.fp8Phase2Offset);
  }
  if (fp8AllGatherBuffer) {
    *fp8AllGatherBuffer = reinterpret_cast<rccl_float8*>(
        basePtr + pool->offsets.fp8AllGatherOffset);
  }
  if (floatReductionBuffer) {
    *floatReductionBuffer =
        reinterpret_cast<float*>(basePtr + pool->offsets.floatReductionOffset);
  }
  if (floatOutputBuffer) {
    *floatOutputBuffer =
        reinterpret_cast<float*>(basePtr + pool->offsets.floatOutputOffset);
  }

  return ncclSuccess;
}

/**
 * Destroys the buffer pool and releases allocated GPU memory.
 */
ncclResult_t ncclLowPrecisionBufferPoolDestroy(
    struct ncclLowPrecisionBufferPool* pool) {
  if (pool->initialized && pool->backingBuffer) {
    CUDACHECK(hipFree(pool->backingBuffer));
    pool->backingBuffer = nullptr;
    pool->initialized = false;
  }
  return ncclSuccess;
}

/**
 * Ensures the communicator's buffer pool is large enough for the requested
 * operation. Expands the pool if necessary to accommodate the specified element
 * count.
 */
ncclResult_t
ncclEnsureLowPrecisionBufferPool(ncclComm_t comm, size_t count, int nRanks) {
  struct ncclLowPrecisionBufferPool* pool = &comm->lowPrecisionBufferPool;

  // Initialize pool if not already done (e.g., env var set after comm creation)
  if (!pool->initialized) {
    INFO(
        NCCL_COLL,
        "Lazy-initializing low precision buffer pool (env var set after comm creation)");
    NCCLCHECK(ncclInitLowPrecisionBufferPoolForComm(comm, nRanks));
  }

  // Calculate exact memory requirement using precise formula from buffer
  // layout: Total = maxElements × (fp8Phase1 + fp8Phase2 + fp8AllGather +
  // floatReduction + floatOutput) Total = maxElements × (1 + nRanks + 1 + 4 +
  // 4) = maxElements × (10 + nRanks) bytes
  size_t bytesPerElement = 10 + nRanks; // Exact calculation, not estimate
  size_t requiredSize = count * bytesPerElement;

  if (requiredSize > pool->maxBufferSize) {
    // Back-calculate required maxElements from the exact memory formula
    size_t currentMaxElements = pool->maxBufferSize / bytesPerElement;

    // Dynamically grow the buffer pool if needed
    // Take the maximum of doubling current capacity or satisfying the required
    // size This ensures we never shrink and grow efficiently
    size_t targetBufferSize = (requiredSize > pool->maxBufferSize * 2)
        ? requiredSize
        : (pool->maxBufferSize * 2);
    size_t calculatedElements = targetBufferSize / bytesPerElement;
    size_t newMaxElements = (calculatedElements > MIN_BUFFER_POOL_ELEMENTS)
        ? calculatedElements
        : MIN_BUFFER_POOL_ELEMENTS;

    INFO(
        NCCL_COLL,
        "Growing low precision buffer pool from %zu to %zu elements (~%.1fMB total)",
        currentMaxElements,
        newMaxElements,
        (newMaxElements * bytesPerElement) / (1024.0 * 1024.0));

    // Destroy old buffer and reinitialize with larger size
    NCCLCHECK(ncclLowPrecisionBufferPoolDestroy(pool));
    NCCLCHECK(ncclLowPrecisionBufferPoolInit(pool, newMaxElements, nRanks));
  }

  return ncclSuccess;
}

/**
 * Initializes buffer pool during communicator creation with pre-calculated
 * sizes for 2G element support across all low precision collectives (8 GPU
 * single node).
 */
ncclResult_t ncclInitLowPrecisionBufferPoolForComm(
    ncclComm_t comm,
    int nRanks) {
  struct ncclLowPrecisionBufferPool* pool = &comm->lowPrecisionBufferPool;

  if (pool->initialized) {
    return ncclSuccess; // Already initialized
  }

  // Optimized initial allocation to handle 512MB float messages (128M elements)
  // Can grow dynamically if larger messages are encountered
  constexpr size_t INITIAL_ELEMENTS =
      static_cast<size_t>(MIN_BUFFER_POOL_ELEMENTS);
  constexpr int MAX_RANKS = 8; // Single node with 8 GPUs

  size_t maxElements = INITIAL_ELEMENTS;
  int maxRanks = (nRanks > MAX_RANKS) ? nRanks : MAX_RANKS;

  NCCLCHECK(ncclLowPrecisionBufferPoolInit(pool, maxElements, maxRanks));

  // Use the same precise calculation as the growth logic for consistency
  size_t bytesPerElement =
      10 + maxRanks; // fp8Phase1(1) + fp8Phase2(maxRanks) + fp8AllGather(1) +
                     // floatReduction(4) + floatOutput(4)

  INFO(
      NCCL_COLL,
      "Pre-initialized low precision buffer pool (optimized): maxElements=%zu (~%.1fMB total), maxRanks=%d",
      maxElements,
      (maxElements * bytesPerElement) / (1024.0 * 1024.0),
      maxRanks);

  return ncclSuccess;
}

HOT ncclResult_t ncclCalculateLowPrecisionKernelConfig(
    size_t totalElements,
    size_t chunkElements,
    struct ncclLowPrecisionKernelConfig* config) {
  config->blockSize = DEFAULT_BLOCK_SIZE;
  config->maxBlocks = DEFAULT_MAX_BLOCKS;
  config->fullGridSize =
      (config->maxBlocks <
       (int)((totalElements + config->blockSize - 1) / config->blockSize))
      ? config->maxBlocks
      : (int)((totalElements + config->blockSize - 1) / config->blockSize);
  config->chunkGridSize =
      (config->maxBlocks <
       (int)((chunkElements + config->blockSize - 1) / config->blockSize))
      ? config->maxBlocks
      : (int)((chunkElements + config->blockSize - 1) / config->blockSize);

  return ncclSuccess;
}
