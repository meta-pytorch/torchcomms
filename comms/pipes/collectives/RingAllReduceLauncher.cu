// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/collectives/RingAllReduceLauncher.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "comms/pipes/CopyOp.cuh"
#include "comms/pipes/TimeoutUtils.h"
#include "comms/pipes/collectives/RingAllReduce.cuh"

namespace comms::pipes {

namespace {

RingTopology to_ring_topology(
    const RingAllReduceLaunchParams::RingParams& src) {
  return RingTopology{
      .prev_rank = src.prev_rank,
      .next_rank = src.next_rank,
      .prev = src.prev,
      .next = src.next,
  };
}

void check_kernel_launch(const char* kernel_name) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(
        std::string(kernel_name) +
        " kernel launch failed: " + cudaGetErrorString(err));
  }
}

template <int NumRings, bool EnableBidirAg>
void launch_impl(const RingAllReduceLaunchParams& params, Timeout timeout) {
  const std::size_t chunk_elements = params.count / params.num_ranks;

  RingAllReduceArgs<NumRings, float> args{};
  args.my_rank = params.my_rank;
  args.num_ranks = params.num_ranks;
  args.chunk_elements = chunk_elements;
  args.signaling_data_size = params.signaling_data_size;
  args.ib_window_bytes = params.ib_window_bytes;
  args.skip_reduction = params.skip_reduction;
  args.input = params.input;
  args.output = params.output;

  for (int r = 0; r < NumRings; r++) {
    args.rings[r] = to_ring_topology(params.rings[r]);
  }

  ring_allreduce_kernel<NumRings, float, SumOp, 16384, 512, EnableBidirAg>
      <<<params.num_blocks, 512, 0, params.stream>>>(args, timeout);
  check_kernel_launch("RingAllReduce");
}

} // namespace

void launch_ring_allreduce(const RingAllReduceLaunchParams& params) {
  if (params.num_ranks == 0) {
    throw std::runtime_error("num_ranks must be > 0");
  }
  if (params.count % params.num_ranks != 0) {
    throw std::runtime_error(
        "count (" + std::to_string(params.count) +
        ") must be divisible by num_ranks (" +
        std::to_string(params.num_ranks) + ")");
  }

  Timeout timeout;
  if (params.timeout_ms > 0) {
    int device = 0;
    cudaError_t cudaErr = cudaGetDevice(&device);
    if (cudaErr != cudaSuccess) {
      throw std::runtime_error(
          "Failed to get CUDA device: " +
          std::string(cudaGetErrorString(cudaErr)));
    }

    timeout = makeTimeout(params.timeout_ms, device);
  }

  const bool bidir = params.enable_bidir_ag && params.num_ranks > 2;

  switch (params.num_rings) {
    case 1:
      if (bidir)
        launch_impl<1, true>(params, timeout);
      else
        launch_impl<1, false>(params, timeout);
      break;
    case 2:
      if (bidir)
        launch_impl<2, true>(params, timeout);
      else
        launch_impl<2, false>(params, timeout);
      break;
    case 4:
      if (bidir)
        launch_impl<4, true>(params, timeout);
      else
        launch_impl<4, false>(params, timeout);
      break;
    default:
      throw std::runtime_error(
          "Unsupported num_rings=" + std::to_string(params.num_rings) +
          " (supported: 1, 2, 4)");
  }
}

} // namespace comms::pipes
