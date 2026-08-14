// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/collectives/ReduceScatterDirectIbV2Launcher.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "comms/prims/collectives/ReduceScatterDirectIbV2.cuh"
#include "comms/prims/core/Checks.h"
#include "comms/prims/core/TimeoutUtils.h"

namespace comms::prims {

namespace {

void validate(const DirectReduceScatterIbV2LaunchParams& params) {
  // num_ranks == 1 is rejected, not handled: the kernel has no single-rank
  // path, so it would return having written nothing to a caller expecting a
  // BF16 -> FP32 converted copy of its own shard. ctran's support check
  // already excludes it; this makes the contract explicit for direct callers.
  if (params.num_ranks < 2 ||
      params.num_ranks > kDirectReduceScatterIbV2MaxRanks) {
    throw std::runtime_error(
        "Unsupported direct IB v2 num_ranks=" +
        std::to_string(params.num_ranks) + " (supported: 2.." +
        std::to_string(kDirectReduceScatterIbV2MaxRanks) + ")");
  }
  if (params.num_blocks < 1) {
    throw std::runtime_error(
        "direct IB v2 requires num_blocks >= 1, got " +
        std::to_string(params.num_blocks));
  }
  // The NIC reads `input` directly, so an unregistered buffer has no lkey and
  // the put would reference memory the NIC cannot translate.
  if (params.input_reg.ptr == nullptr) {
    throw std::runtime_error("direct IB v2 requires a registered input buffer");
  }
  // Send offsets are computed relative to `input` and applied to `input_reg`,
  // so registering a different buffer would silently send the wrong bytes.
  if (params.input_reg.ptr != static_cast<const void*>(params.input)) {
    throw std::runtime_error(
        "direct IB v2 registered buffer must start at the input pointer");
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

} // namespace

void launch_direct_reduce_scatter_ib_v2_impl(
    const DirectReduceScatterIbV2LaunchParams& params) {
  validate(params);

  DirectReduceScatterIbV2Args args{};
  args.my_rank = params.my_rank;
  args.num_ranks = params.num_ranks;
  args.chunk_elements = params.chunk_elements;
  args.signaling_data_size = params.signaling_data_size;
  args.input = params.input;
  args.output = params.output;
  args.input_reg = params.input_reg;
  for (int peer = 0; peer < params.num_ranks; ++peer) {
    if (peer != params.my_rank) {
      args.peers[peer] = params.peers[peer];
    }
  }

  launch_direct_reduce_scatter_ib_v2(
      args,
      params.num_blocks,
      params.block_threads,
      params.stream,
      make_launch_timeout(params.timeout_ms));
}

} // namespace comms::prims
