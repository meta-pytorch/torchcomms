// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>
#include <cstddef>

namespace comms::pipes::benchmark {

__global__ void
tile_copy_kernel(float* dst, const float* src, std::size_t nelems, int nruns);

__global__ void tile_reduce_sum_kernel(
    float* dst,
    const float* src_a,
    const float* src_b,
    std::size_t nelems,
    int nruns);

} // namespace comms::pipes::benchmark
