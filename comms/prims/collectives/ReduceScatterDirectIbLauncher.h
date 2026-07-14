// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

#include "comms/prims/transport/P2pIbTransportDeviceDecl.cuh"

namespace comms::prims {

inline constexpr int kDirectReduceScatterIbMaxRanks = 256;

struct DirectReduceScatterIbLaunchParams {
  int my_rank{0};
  int num_ranks{0};
  std::size_t chunk_elements{0};
  std::size_t signaling_data_size{0};
  const float* input{nullptr};
  float* output{nullptr};
  bool in_place{false};
  int num_blocks{16};
  float timeout_ms{0.0f};
  cudaStream_t stream{nullptr};
  P2pIbTransportDevice peers[kDirectReduceScatterIbMaxRanks]{};
};

void launch_direct_reduce_scatter_ib(
    const DirectReduceScatterIbLaunchParams& params);

} // namespace comms::prims
