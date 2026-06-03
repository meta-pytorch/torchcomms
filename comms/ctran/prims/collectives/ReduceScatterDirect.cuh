// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/ctran/prims/DeviceMacros.cuh"
#include "comms/ctran/prims/Timeout.cuh"
#include "comms/ctran/prims/collectives/ReduceScatterDirectTypes.h"

namespace ctran::prims {

template <typename T, typename AccumOp, int kTileElems, int kBlockSize>
__global__
    __launch_bounds__(kBlockSize, 1) void direct_reduce_scatter_nvl_kernel(
        const PIPES_GRID_CONSTANT DirectReduceScatterNvlArgs<T> args,
        Timeout timeout);

} // namespace ctran::prims
