// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda_runtime.h>
#include <cstddef>

namespace comms::pipes::benchmark {

/**
 * Kernel that uses P2pSelfTransportDevice to copy data (grid-collective)
 */
__global__ void selfTransportPutKernel(
    char* dst,
    const char* src,
    std::size_t nBytes,
    int nRuns);

/**
 * Kernel that uses P2pSelfTransportDevice to copy per-group tiles.
 * Each block copies one tile of tileSize bytes at group.group_id offset.
 */
__global__ void selfTransportPutTileKernel(
    char* dst,
    const char* src,
    std::size_t tileSize,
    int nRuns);

} // namespace comms::pipes::benchmark
