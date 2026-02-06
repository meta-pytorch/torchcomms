/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "low_precision_buffer_pool.h"

/**
 * Performs low precision reduce-scatter operation using FP8 quantization.
 * Combines input data with reduction operation and distributes unique chunks to
 * each rank.
 */
HOT ncclResult_t ncclLowPrecisionReduceScatter(
    const void* RESTRICT sendbuff,
    void* RESTRICT recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    cudaStream_t stream);
