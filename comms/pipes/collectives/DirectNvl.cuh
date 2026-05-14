// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Direct NVLink AllGather, ReduceScatter, and Hierarchical AllGather kernels.

#pragma once

#include "comms/pipes/Timeout.cuh"
#include "comms/pipes/collectives/DirectNvlTypes.h"

namespace comms::pipes {

template <int kBlockSize>
__global__ __launch_bounds__(kBlockSize, 1) void direct_allgather_nvl_kernel(
    const __grid_constant__ DirectAllgatherNvlArgs args,
    Timeout timeout);

template <typename T, typename AccumOp, int kTileElems, int kBlockSize>
__global__
    __launch_bounds__(kBlockSize, 1) void direct_reduce_scatter_nvl_kernel(
        const __grid_constant__ DirectReduceScatterNvlArgs<T> args,
        Timeout timeout);

template <int kBlockSize>
__global__
    __launch_bounds__(kBlockSize, 1) void hierarchical_allgather_fused_kernel(
        const __grid_constant__ HierarchicalAllgatherFusedArgs args,
        Timeout timeout);

} // namespace comms::pipes
