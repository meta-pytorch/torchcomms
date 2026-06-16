// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/collectives/RingReduceScatterLauncher.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "comms/prims/collectives/RingReduceScatter.cuh"
#include "comms/prims/core/CopyOp.cuh"
#include "comms/prims/core/TimeoutUtils.h"

namespace comms::prims {

namespace {

template <int NumRings>
void launch_impl(const RingReduceScatterLaunchParams& params, Timeout timeout) {
  RingReduceScatterArgs<NumRings, float> args{};
  args.my_rank = params.my_rank;
  args.num_ranks = params.num_ranks;
  args.chunk_elements = params.chunk_elements;
  args.signaling_data_size = params.signaling_data_size;
  args.input = params.input;
  args.output = params.output;

  for (int r = 0; r < NumRings; r++) {
    auto& src = params.rings[r];
    args.rings[r] = RingTopology{
        .prev_rank = src.prev_rank,
        .next_rank = src.next_rank,
        .prev = src.prev,
        .next = src.next,
    };
  }

  ring_reduce_scatter_kernel<NumRings, float, SumOp, 16384, 512>
      <<<params.num_blocks, 512>>>(args, timeout);
}

} // namespace

void launch_ring_reduce_scatter(const RingReduceScatterLaunchParams& params) {
  Timeout timeout;
  if (params.timeout_ms > 0) {
    int device = 0;
    cudaGetDevice(&device);
    timeout = makeTimeout(params.timeout_ms, device);
  }

  switch (params.num_rings) {
    case 1:
      launch_impl<1>(params, timeout);
      break;
    case 2:
      launch_impl<2>(params, timeout);
      break;
    case 4:
      launch_impl<4>(params, timeout);
      break;
    default:
      throw std::runtime_error(
          "Unsupported num_rings=" + std::to_string(params.num_rings) +
          " (supported: 1, 2, 4)");
  }
}

} // namespace comms::prims
