// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>
#include <cstddef>

namespace comms::pipes::benchmark {

void launch_copy_dual(
    char* dst1,
    char* dst2,
    const char* src,
    std::size_t nBytes,
    int nRuns,
    int numBlocks,
    int numThreads,
    cudaStream_t stream);

void launch_copy_two_sequential(
    char* dst1,
    char* dst2,
    const char* src,
    std::size_t nBytes,
    int nRuns,
    int numBlocks,
    int numThreads,
    cudaStream_t stream);

} // namespace comms::pipes::benchmark
