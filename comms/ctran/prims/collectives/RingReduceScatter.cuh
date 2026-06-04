// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/ctran/prims/DeviceMacros.cuh"
#include "comms/ctran/prims/Timeout.cuh"
#include "comms/ctran/prims/collectives/Ring.cuh"

namespace ctran::prims {

template <
    int NumRings,
    typename T,
    typename AccumOp,
    int kTileElems,
    int kBlockSize>
__global__ __launch_bounds__(kBlockSize, 1) void ring_reduce_scatter_kernel(
    const PIPES_GRID_CONSTANT RingReduceScatterArgs<NumRings, T> args,
    Timeout timeout);

} // namespace ctran::prims
