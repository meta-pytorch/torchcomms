// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/collectives/RingAllgather.cuh"

namespace comms::pipes {

template <int NumRings, int kBlockSize>
__global__ __launch_bounds__(kBlockSize, 1) void ring_allgather_kernel(
    const __grid_constant__ RingAllgatherArgs<NumRings> args,
    Timeout timeout) {
  ring_allgather_device<NumRings, kBlockSize>(args, timeout);
}

// Template instantiations
template __global__ void ring_allgather_kernel<1, 512>(
    const __grid_constant__ RingAllgatherArgs<1>,
    Timeout);
template __global__ void ring_allgather_kernel<2, 512>(
    const __grid_constant__ RingAllgatherArgs<2>,
    Timeout);
template __global__ void ring_allgather_kernel<4, 512>(
    const __grid_constant__ RingAllgatherArgs<4>,
    Timeout);

} // namespace comms::pipes
