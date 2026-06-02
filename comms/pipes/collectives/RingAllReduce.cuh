// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/pipes/Timeout.cuh"
#include "comms/pipes/collectives/Ring.cuh"

namespace comms::pipes {

template <
    int NumRings,
    typename T,
    typename AccumOp,
    int kTileElems,
    int kBlockSize,
    bool EnableBidirAg = false>
__global__ __launch_bounds__(kBlockSize, 1) void ring_allreduce_kernel(
    const __grid_constant__ RingAllReduceArgs<NumRings, T> args,
    Timeout timeout);

} // namespace comms::pipes
