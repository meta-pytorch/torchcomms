// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda_runtime.h>
#include <cstddef>

namespace comms::pipes::benchmark {

/**
 * Kernel that uses P2pSelfTransportDevice to copy data
 */
__global__ void selfTransportPutKernel(
    char* dst,
    const char* src,
    std::size_t nBytes,
    int nRuns);

} // namespace comms::pipes::benchmark
