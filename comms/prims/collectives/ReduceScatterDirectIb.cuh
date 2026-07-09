// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

#include "comms/prims/core/Timeout.cuh"
#include "comms/prims/transport/P2pIbTransportDeviceDecl.cuh"

namespace comms::prims {

inline constexpr int kDirectIbDeviceMaxRanks = 256;

template <typename T>
struct DirectReduceScatterIbArgs {
  int my_rank{0};
  int num_ranks{0};
  std::size_t chunk_elements{0};
  std::size_t signaling_data_size{0};
  P2pIbTransportDevice peers[kDirectIbDeviceMaxRanks]{};
  const T* input{nullptr};
  T* output{nullptr};
  bool in_place{false};
};

void launch_direct_reduce_scatter_ib_impl(
    const DirectReduceScatterIbArgs<float>& args,
    int num_blocks,
    cudaStream_t stream,
    Timeout timeout);

} // namespace comms::prims
