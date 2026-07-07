// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/collectives/ReduceScatterDirectIbLauncher.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "comms/prims/collectives/ReduceScatterDirectIb.cuh"
#include "comms/prims/core/Checks.h"
#include "comms/prims/core/TimeoutUtils.h"

namespace comms::prims {

namespace {

void validate_direct_ib(const DirectReduceScatterIbLaunchParams& params) {
  if (params.num_ranks < 1 ||
      params.num_ranks > kDirectReduceScatterIbMaxRanks) {
    throw std::runtime_error(
        "Unsupported direct IB num_ranks=" + std::to_string(params.num_ranks) +
        " (supported: 1.." + std::to_string(kDirectReduceScatterIbMaxRanks) +
        ")");
  }
  if (params.num_blocks < 1) {
    throw std::runtime_error(
        "direct IB reduce-scatter requires num_blocks >= 1, got " +
        std::to_string(params.num_blocks));
  }
}

Timeout make_launch_timeout(float timeout_ms) {
  Timeout timeout;
  if (timeout_ms > 0) {
    int device = 0;
    PIPES_CUDA_CHECK(cudaGetDevice(&device));
    timeout = makeTimeout(timeout_ms, device);
  }
  return timeout;
}

DirectReduceScatterIbArgs<float> make_args(
    const DirectReduceScatterIbLaunchParams& params) {
  validate_direct_ib(params);

  DirectReduceScatterIbArgs<float> args{};
  args.my_rank = params.my_rank;
  args.num_ranks = params.num_ranks;
  args.chunk_elements = params.chunk_elements;
  args.signaling_data_size = params.signaling_data_size;
  args.input = params.input;
  args.output = params.output;
  args.in_place = params.in_place;

  for (int peer = 0; peer < params.num_ranks; ++peer) {
    if (peer == params.my_rank) {
      continue;
    }
    args.peers[peer] = params.peers[peer];
  }

  return args;
}

} // namespace

void launch_direct_reduce_scatter_ib(
    const DirectReduceScatterIbLaunchParams& params) {
  launch_direct_reduce_scatter_ib_impl(
      make_args(params),
      params.num_blocks,
      params.stream,
      make_launch_timeout(params.timeout_ms));
}

} // namespace comms::prims
