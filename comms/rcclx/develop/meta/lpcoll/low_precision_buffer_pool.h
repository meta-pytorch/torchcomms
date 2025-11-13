/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "low_precision_common.h"

/**
 * Initializes a low precision buffer pool with the specified capacity.
 * Allocates GPU memory buffers for FP8 and float operations.
 */
HOT ncclResult_t ncclLowPrecisionBufferPoolInit(
    struct ncclLowPrecisionBufferPool* pool,
    size_t maxElements,
    int maxRanks);

/**
 * Retrieves pre-allocated buffers from the buffer pool for low precision
 * operations. Returns pointers to phase buffers, all-gather buffers, and
 * reduction buffers.
 */
HOT ncclResult_t ncclLowPrecisionBufferPoolGetBuffers(
    struct ncclLowPrecisionBufferPool* pool,
    size_t count,
    int nRanks,
    rccl_float8** fp8Phase1Buffer,
    rccl_float8** fp8Phase2Buffer,
    rccl_float8** fp8AllGatherBuffer,
    float** floatReductionBuffer,
    float** floatOutputBuffer);

/**
 * Destroys the buffer pool and releases allocated GPU memory.
 */
ncclResult_t ncclLowPrecisionBufferPoolDestroy(
    struct ncclLowPrecisionBufferPool* pool);

/**
 * Ensures the communicator's buffer pool is large enough for the requested
 * operation. Expands the pool if necessary to accommodate the specified element
 * count.
 */
HOT ncclResult_t
ncclEnsureLowPrecisionBufferPool(ncclComm_t comm, size_t count, int nRanks);

/**
 * Initializes buffer pool during communicator creation with pre-calculated
 * sizes for 2G element support across all low precision collectives (8 GPU
 * single node).
 */
ncclResult_t ncclInitLowPrecisionBufferPoolForComm(ncclComm_t comm, int nRanks);

/**
 * Calculates optimal kernel launch configuration for low precision operations.
 * Determines grid size and block size based on total elements and chunk size.
 */
HOT ncclResult_t ncclCalculateLowPrecisionKernelConfig(
    size_t totalElements,
    size_t chunkElements,
    struct ncclLowPrecisionKernelConfig* config);
