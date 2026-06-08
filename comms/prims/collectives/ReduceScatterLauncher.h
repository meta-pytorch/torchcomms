// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

#include "comms/prims/collectives/ReduceScatterDirectTypes.h"
#include "comms/prims/transport/nvl/P2pNvlTransportDevice.cuh"

namespace comms::prims {

struct DirectReduceScatterNvlLaunchParams {
  int my_rank{0};
  int num_ranks{0};
  std::size_t chunk_elements{0};
  std::size_t signaling_data_size{0};
  const float* input{nullptr};
  float* output{nullptr};
  int num_blocks{16};
  float timeout_ms{0.0f};
  cudaStream_t stream{nullptr};
  P2pNvlTransportDevice peers[kDirectNvlMaxRanks]{};
};

void launch_direct_reduce_scatter_nvl(
    const DirectReduceScatterNvlLaunchParams& params);

} // namespace comms::prims
