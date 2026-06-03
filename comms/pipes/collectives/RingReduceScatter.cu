// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/collectives/RingReduceScatter.cuh"

namespace comms::pipes {

template <
    int NumRings,
    typename T,
    typename AccumOp,
    int kTileElems,
    int kBlockSize>
__global__ __launch_bounds__(kBlockSize, 1) void ring_reduce_scatter_kernel(
    const __grid_constant__ RingReduceScatterArgs<NumRings, T> args,
    Timeout timeout) {
  ring_reduce_scatter_device<NumRings, T, AccumOp, kTileElems, kBlockSize>(
      args, timeout);
}

// Template instantiations
template __global__ void
ring_reduce_scatter_kernel<1, float, SumOp, 16384, 512>(
    const __grid_constant__ RingReduceScatterArgs<1, float>,
    Timeout);
template __global__ void
ring_reduce_scatter_kernel<2, float, SumOp, 16384, 512>(
    const __grid_constant__ RingReduceScatterArgs<2, float>,
    Timeout);
template __global__ void
ring_reduce_scatter_kernel<4, float, SumOp, 16384, 512>(
    const __grid_constant__ RingReduceScatterArgs<4, float>,
    Timeout);

} // namespace comms::pipes
