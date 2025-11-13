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
 * Performs low precision allreduce operation using FP8 quantization for
 * bandwidth efficiency. Implements scatter-reduce-allgather pattern with local
 * reduction between scatter and allgather phases.
 */
HOT ncclResult_t ncclLowPrecisionAllReduce(
    const void* RESTRICT sendbuff,
    void* RESTRICT recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    cudaStream_t stream);
