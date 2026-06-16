// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/prims/collectives/Ring.cuh"
#include "comms/prims/core/Timeout.cuh"

namespace comms::prims {

template <int NumRings, int kBlockSize>
__global__ __launch_bounds__(kBlockSize, 1) void ring_allgather_kernel(
    const __grid_constant__ RingAllgatherArgs<NumRings> args,
    Timeout timeout);

} // namespace comms::prims
