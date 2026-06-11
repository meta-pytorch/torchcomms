// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/collectives/RingAllgatherLauncher.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "comms/prims/collectives/RingAllgather.cuh"
#include "comms/prims/core/TimeoutUtils.h"

namespace comms::prims {

namespace {

template <int NumRings>
void launch_impl(const RingAllgatherLaunchParams& params, Timeout timeout) {
  RingAllgatherArgs<NumRings> args{};
  args.my_rank = params.my_rank;
  args.num_ranks = params.num_ranks;
  args.sendcount = params.sendcount;
  args.signaling_data_size = params.signaling_data_size;
  args.sendbuf = params.sendbuf;
  args.recvbuf = params.recvbuf;

  for (int r = 0; r < NumRings; r++) {
    auto& src = params.rings[r];
    args.rings[r] = RingTopology{
        .prev_rank = src.prev_rank,
        .next_rank = src.next_rank,
        .prev = src.prev,
        .next = src.next,
    };
  }

  ring_allgather_kernel<NumRings, 512>
      <<<params.num_blocks, 512>>>(args, timeout);
}

} // namespace

void launch_ring_allgather(const RingAllgatherLaunchParams& params) {
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
