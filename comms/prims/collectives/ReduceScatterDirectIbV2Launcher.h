// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

#include "comms/prims/transport/P2pIbTransportDeviceDecl.cuh"
#include "comms/prims/transport/ibgda/IbgdaBuffer.h"

namespace comms::prims {

inline constexpr int kDirectReduceScatterIbV2MaxRanks = 256;

struct DirectReduceScatterIbV2LaunchParams {
  int my_rank{0};
  int num_ranks{0};
  std::size_t chunk_elements{0};
  std::size_t signaling_data_size{0};
  const __nv_bfloat16* input{nullptr};
  float* output{nullptr};
  IbgdaLocalBuffer input_reg{};
  int num_blocks{2};
  int block_threads{1024};
  float timeout_ms{0.0f};
  cudaStream_t stream{nullptr};
  P2pIbTransportDevice peers[kDirectReduceScatterIbV2MaxRanks]{};
};

void launch_direct_reduce_scatter_ib_v2_impl(
    const DirectReduceScatterIbV2LaunchParams& params);

} // namespace comms::prims
