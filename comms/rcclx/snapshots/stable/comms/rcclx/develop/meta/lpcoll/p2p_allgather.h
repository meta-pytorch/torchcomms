/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <hip/hip_runtime.h>
#include "comm.h"
#include "nccl.h"

/**
 * P2P AllGather collective using parallel point-to-point communication.
 * Optimized for large messages (>16MB) with better performance than traditional
 * ring algorithms.
 */
ncclResult_t ncclP2PAllGather(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream);
