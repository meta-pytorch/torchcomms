// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Direct NVLink AllGather and Hierarchical AllGather kernels.

#pragma once

#include "comms/ctran/prims/DeviceMacros.cuh"
#include "comms/ctran/prims/Timeout.cuh"
#include "comms/ctran/prims/collectives/AllGatherDirectTypes.h"

namespace ctran::prims {

template <int kBlockSize>
__global__ __launch_bounds__(kBlockSize, 1) void direct_allgather_nvl_kernel(
    const PIPES_GRID_CONSTANT DirectAllgatherNvlArgs args,
    Timeout timeout);

template <int kBlockSize>
__global__
    __launch_bounds__(kBlockSize, 1) void hierarchical_allgather_fused_kernel(
        const PIPES_GRID_CONSTANT HierarchicalAllgatherFusedArgs args,
        Timeout timeout);

template <int kBlockSize>
__global__
    __launch_bounds__(kBlockSize, 1) void hierarchical_allgather_overlap_kernel(
        const PIPES_GRID_CONSTANT HierarchicalAllgatherOverlapArgs args,
        Timeout timeout);

} // namespace ctran::prims
