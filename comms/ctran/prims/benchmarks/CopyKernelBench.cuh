// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>
#include <cstddef>

#include "comms/ctran/prims/CopyUtils.cuh"
#include "comms/ctran/prims/ThreadGroup.cuh"

namespace ctran::prims::benchmark {

using namespace ctran::prims;

__global__ void copyKernel(
    char* dst,
    const char* src,
    std::size_t nBytes,
    int nRuns,
    SyncScope groupScope);

// Dual-destination copy: load src once, store to both dst1 and dst2
__global__ void dualDstCopyKernel(
    char* dst1,
    char* dst2,
    const char* src,
    std::size_t nBytes,
    int nRuns,
    SyncScope groupScope);

// Two-pass copy in single kernel: read src twice, write dst1 then dst2
__global__ void twoPassCopyKernel(
    char* dst1,
    char* dst2,
    const char* src,
    std::size_t nBytes,
    int nRuns,
    SyncScope groupScope);

} // namespace ctran::prims::benchmark
