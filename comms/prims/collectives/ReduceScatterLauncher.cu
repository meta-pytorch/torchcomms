// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/collectives/ReduceScatterLauncher.h"

#include <cuda_runtime.h>

#include <new>
#include <stdexcept>
#include <string>

#include "comms/prims/collectives/ReduceScatterDirect.cuh"
#include "comms/prims/core/Checks.h"
#include "comms/prims/core/CopyOp.cuh"
#include "comms/prims/core/TimeoutUtils.h"

namespace comms::prims {

namespace {

void validate_direct_nvl(int num_ranks) {
  if (num_ranks < 1 || num_ranks > kDirectNvlMaxRanks) {
    throw std::runtime_error(
        "Unsupported direct NVLink num_ranks=" + std::to_string(num_ranks) +
        " (supported: 1.." + std::to_string(kDirectNvlMaxRanks) + ")");
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

void launch_direct_reduce_scatter_nvl(
    const DirectReduceScatterNvlLaunchParams& params) {
  validate_direct_nvl(params.num_ranks);

  DirectReduceScatterNvlArgs<float> args{};
  args.my_rank = params.my_rank;
  args.num_ranks = params.num_ranks;
  args.chunk_elements = params.chunk_elements;
  args.signaling_data_size = params.signaling_data_size;
  args.input = params.input;
  args.output = params.output;

  for (int peer = 0; peer < params.num_ranks; ++peer) {
    if (peer == params.my_rank) {
      continue;
    }
    new (&args.peers[peer]) P2pNvlTransportDevice(params.peers[peer]);
  }

  direct_reduce_scatter_nvl_kernel<float, SumOp, 16384, 512>
      <<<params.num_blocks, 512, 0, params.stream>>>(
          args, make_launch_timeout(params.timeout_ms));
  PIPES_CUDA_CHECK(cudaGetLastError());
}

} // namespace comms::prims
