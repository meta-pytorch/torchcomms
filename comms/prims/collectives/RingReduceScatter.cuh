// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/prims/collectives/Ring.cuh"
#include "comms/prims/core/Timeout.cuh"

namespace comms::prims {

template <
    int NumRings,
    typename T,
    typename AccumOp,
    int kTileElems,
    int kBlockSize>
__global__ __launch_bounds__(kBlockSize, 1) void ring_reduce_scatter_kernel(
    const __grid_constant__ RingReduceScatterArgs<NumRings, T> args,
    Timeout timeout);

} // namespace comms::prims
