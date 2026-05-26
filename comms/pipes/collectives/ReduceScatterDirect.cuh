// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/pipes/Timeout.cuh"
#include "comms/pipes/collectives/ReduceScatterDirectTypes.h"

namespace comms::pipes {

template <typename T, typename AccumOp, int kTileElems, int kBlockSize>
__global__
    __launch_bounds__(kBlockSize, 1) void direct_reduce_scatter_nvl_kernel(
        const __grid_constant__ DirectReduceScatterNvlArgs<T> args,
        Timeout timeout);

__global__
    __launch_bounds__(512, 1) void hierarchical_reduce_scatter_fused_float_sum_kernel(
        const __grid_constant__ HierarchicalReduceScatterFusedArgs<float> args,
        Timeout timeout);

} // namespace comms::pipes
