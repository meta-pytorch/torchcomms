// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <stdio.h>
#include <cstddef>
#include "nccl.h"

template <typename T>
extern __global__ void ncclKernel_AllReduceSparseBlock_Unpack(
    T* unpackBuf,
    const T* packBuf,
    const size_t blockCount,
    const int64_t* unpackIndices,
    const size_t blockLength);
