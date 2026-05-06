// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>
#include <cstddef>

#include "comms/pipes/CopyUtils.cuh"
#include "comms/pipes/ThreadGroup.cuh"

namespace comms::pipes::benchmark {

using namespace comms::pipes;

__global__ void copy_dual_kernel(
    char* dst1,
    char* dst2,
    const char* src,
    std::size_t nBytes,
    int nRuns);

__global__ void copy_two_sequential_kernel(
    char* dst1,
    char* dst2,
    const char* src,
    std::size_t nBytes,
    int nRuns);

} // namespace comms::pipes::benchmark
