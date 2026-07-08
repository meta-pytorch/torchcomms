// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

#include "comms/pipes/P2pNvlTransportDevice.cuh"
#include "comms/pipes/collectives/HierarchicalReduceScatterTypes.h"

namespace comms::pipes {

struct HierarchicalReduceScatterLaunchParams {
  int num_ranks{0};
  int ib_rank{0};
  int ib_size{0};
  int nvl_rank{0};
  int nvl_size{0};
  std::size_t chunk_elements{0};
  std::size_t ib_signaling_data_size{0};
  std::size_t nvl_signaling_data_size{0};
  const float* input{nullptr};
  float* output{nullptr};
  float* workspace{nullptr};
  int num_blocks{16};
  float timeout_ms{0.0f};
  cudaStream_t stream{nullptr};
  HierarchicalReduceScatterIbgdaRing ib_ring{};
  P2pNvlTransportDevice nvl_peers[kDirectNvlMaxRanks]{};
};

void launch_hierarchical_reduce_scatter_fused(
    const HierarchicalReduceScatterLaunchParams& params);

} // namespace comms::pipes
