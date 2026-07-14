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

// Returns true if the named environment variable is set to "1".
inline bool isLowPrecisionEnvFlagSet(const char* envName) {
  const char* value = getenv(envName);
  return (value && strcmp(value, "1") == 0);
}

// Global switch: enables FP8 E4M3 low precision for ALL collectives.
inline bool isLowPrecisionFp8E4M3Enabled() {
  return isLowPrecisionEnvFlagSet("RCCL_LOW_PRECISION_ENABLE");
}

// Forces the per-comm low precision buffer pool to be pre-allocated at comm
// init regardless of whether any LP collective is currently enabled. The pool
// must exist before any LP collective runs (e.g. it cannot be allocated during
// CUDA graph capture), so setting this lets the per-collective LP switches be
// toggled dynamically at runtime without restarting the job.
inline bool isLowPrecisionInitEnabled() {
  return isLowPrecisionEnvFlagSet("RCCL_LOW_PRECISION_ENABLE_INIT");
}

// Per-collective switches. A collective uses low precision if the global
// switch (RCCL_LOW_PRECISION_ENABLE) OR its dedicated switch is set.
inline bool isLowPrecisionFp8E4M3AllGatherEnabled() {
  return isLowPrecisionFp8E4M3Enabled() ||
      isLowPrecisionEnvFlagSet("RCCL_LOW_PRECISION_ENABLE_ALLGATHER");
}

inline bool isLowPrecisionFp8E4M3AllReduceEnabled() {
  return isLowPrecisionFp8E4M3Enabled() ||
      isLowPrecisionEnvFlagSet("RCCL_LOW_PRECISION_ENABLE_ALLREDUCE");
}

inline bool isLowPrecisionFp8E4M3ReduceScatterEnabled() {
  return isLowPrecisionFp8E4M3Enabled() ||
      isLowPrecisionEnvFlagSet("RCCL_LOW_PRECISION_ENABLE_REDUCESCATTER");
}

inline bool isLowPrecisionFp8E4M3AllToAllEnabled() {
  return isLowPrecisionFp8E4M3Enabled() ||
      isLowPrecisionEnvFlagSet("RCCL_LOW_PRECISION_ENABLE_ALLTOALL");
}

// True if low precision is enabled for the whole comm (global switch) or for
// any individual collective. Used to decide whether the per-comm low precision
// buffer pool must be pre-allocated at init time (required for CUDA graph
// compatibility, since the pool cannot be allocated during graph capture).
inline bool isAnyLowPrecisionFp8E4M3Enabled() {
  return isLowPrecisionFp8E4M3AllGatherEnabled() ||
      isLowPrecisionFp8E4M3AllReduceEnabled() ||
      isLowPrecisionFp8E4M3ReduceScatterEnabled() ||
      isLowPrecisionFp8E4M3AllToAllEnabled();
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
