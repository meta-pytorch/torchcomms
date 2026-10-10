// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/pipes/Timeout.cuh"
#include "comms/pipes/collectives/HierarchicalReduceScatterTypes.h"

namespace comms::pipes {

__global__
    __launch_bounds__(512, 1) void hierarchical_reduce_scatter_fused_float_sum_kernel(
        const __grid_constant__ HierarchicalReduceScatterFusedArgs<float> args,
        Timeout timeout);

} // namespace comms::pipes
