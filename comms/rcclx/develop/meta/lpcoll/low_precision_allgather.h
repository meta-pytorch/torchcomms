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
 * Performs low precision allgather operation by quantizing input data to FP8
 * format, exchanging quantized chunks between ranks, and dequantizing back to
 * original format.
 */
HOT ncclResult_t ncclLowPrecisionAllGather(
    const void* RESTRICT sendbuff,
    void* RESTRICT recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream);
