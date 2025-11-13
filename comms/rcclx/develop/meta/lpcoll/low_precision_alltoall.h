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
 * Performs low precision all-to-all communication using FP8 quantization.
 * Each rank sends unique data to every other rank with bandwidth efficiency.
 */
HOT ncclResult_t ncclLowPrecisionAllToAll(
    const void* RESTRICT sendbuff,
    void* RESTRICT recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream);
