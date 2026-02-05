/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdlib>
#include <cstring>
#include "low_precision_kernels.h"
#include "low_precision_utility.h"

// Helper function to check if low precision FP8 E4M3 is enabled
inline bool isLowPrecisionFp8E4M3Enabled() {
  const char* lowPrecisionEnable = getenv("RCCL_LOW_PRECISION_ENABLE");
  return (lowPrecisionEnable && strcmp(lowPrecisionEnable, "1") == 0);
}

/**
 * Unified buffer pool for all low precision collectives.
 * Manages pre-allocated GPU memory buffers to avoid per-operation allocations.
 */
struct ncclLowPrecisionBufferPool {
  void* backingBuffer;
  size_t maxBufferSize;
  size_t currentSize;
  bool initialized;

  /**
   * Pre-calculated offsets for different buffer types used across collectives.
   * Enables efficient buffer layout with proper alignment for GPU access
   * patterns.
   */
  struct BufferOffsets {
    size_t fp8Phase1Offset; // Primary FP8 buffer (input/intermediate)
    size_t fp8Phase2Offset; // Secondary FP8 buffer (all-to-all/exchange)
    size_t fp8AllGatherOffset; // AllGather result buffer
    size_t floatReductionOffset; // Float reduction buffer
    size_t floatOutputOffset; // Final float output buffer
  } offsets;
};

/**
 * Initialize buffer pool with maximum expected size for all collective types.
 */
ncclResult_t ncclLowPrecisionBufferPoolInit(
    struct ncclLowPrecisionBufferPool* pool,
    size_t maxElements,
    int maxRanks);

/**
 * Get buffer pointers for a specific operation.
 * Different collectives can request different combinations of buffers.
 */
ncclResult_t ncclLowPrecisionBufferPoolGetBuffers(
    struct ncclLowPrecisionBufferPool* pool,
    size_t count,
    int nRanks,
    rccl_float8** fp8Phase1Buffer,
    rccl_float8** fp8Phase2Buffer,
    rccl_float8** fp8AllGatherBuffer,
    float** floatReductionBuffer,
    float** floatOutputBuffer);

/**
 * Clean up buffer pool.
 */
ncclResult_t ncclLowPrecisionBufferPoolDestroy(
    struct ncclLowPrecisionBufferPool* pool);

/**
 * Common utility function to ensure buffer pool is initialized.
 * Automatically determines appropriate size based on collective type and
 * parameters.
 */
ncclResult_t
ncclEnsureLowPrecisionBufferPool(ncclComm_t comm, size_t count, int nRanks);

/**
 * Common kernel launch parameter calculation structure.
 * Optimizes GPU kernel execution based on problem size and hardware
 * characteristics.
 */
struct ncclLowPrecisionKernelConfig {
  int blockSize;
  int maxBlocks;
  int fullGridSize;
  int chunkGridSize;
};

/**
 * Calculates optimal kernel launch configuration for low precision operations.
 */
ncclResult_t ncclCalculateLowPrecisionKernelConfig(
    size_t totalElements,
    size_t chunkElements,
    struct ncclLowPrecisionKernelConfig* config);

/**
 * Low precision allreduce operation using FP8 quantization for bandwidth
 * efficiency.
 */
HOT ncclResult_t ncclLowPrecisionAllReduce(
    const void* RESTRICT sendbuff,
    void* RESTRICT recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    cudaStream_t stream);

/**
 * Low precision allgather operation by quantizing input to FP8 and exchanging
 * between ranks.
 */
HOT ncclResult_t ncclLowPrecisionAllGather(
    const void* RESTRICT sendbuff,
    void* RESTRICT recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream);

/**
 * Low precision all-to-all communication using FP8 quantization for bandwidth
 * efficiency.
 */
HOT ncclResult_t ncclLowPrecisionAllToAll(
    const void* RESTRICT sendbuff,
    void* RESTRICT recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream);

/**
 * Low precision reduce-scatter operation combining reduction with FP8
 * quantization.
 */
HOT ncclResult_t ncclLowPrecisionReduceScatter(
    const void* RESTRICT sendbuff,
    void* RESTRICT recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    cudaStream_t stream);
