// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// Direct NVLink AllGather and Hierarchical AllGather kernels.

#pragma once

#include "comms/prims/collectives/AllGatherDirectTypes.h"
#include "comms/prims/core/Timeout.cuh"

namespace comms::prims {

template <int kBlockSize>
__global__ __launch_bounds__(kBlockSize, 1) void direct_allgather_nvl_kernel(
    const __grid_constant__ DirectAllgatherNvlArgs args,
    Timeout timeout);

template <int kBlockSize>
__global__
    __launch_bounds__(kBlockSize, 1) void hierarchical_allgather_fused_kernel(
        const __grid_constant__ HierarchicalAllgatherFusedArgs args,
        Timeout timeout);

template <int kBlockSize>
__global__
    __launch_bounds__(kBlockSize, 1) void hierarchical_allgather_overlap_kernel(
        const __grid_constant__ HierarchicalAllgatherOverlapArgs args,
        Timeout timeout);

} // namespace comms::prims
