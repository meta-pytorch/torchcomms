// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

#include "comms/prims/transport/P2pIbTransportDeviceDecl.cuh"

namespace comms::prims {

enum class RingReduceScatterDataType {
  kInt8,
  kUint8,
  kInt32,
  kUint32,
  kInt64,
  kUint64,
  kFloat32,
  kFloat64,
  kFloat16,
  kBfloat16,
};

enum class RingReduceScatterReduceOp {
  kSum,
  kMax,
  kMin,
};

struct RingReduceScatterLaunchParams {
  int my_rank{0};
  int num_ranks{0};
  std::size_t chunk_elements{0};
  std::size_t signaling_data_size{0};
  const void* input{nullptr};
  void* output{nullptr};
  RingReduceScatterDataType data_type{RingReduceScatterDataType::kFloat32};
  RingReduceScatterReduceOp reduce_op{RingReduceScatterReduceOp::kSum};
  int num_blocks{16};
  int num_rings{1};
  float timeout_ms{0.0f};
  cudaStream_t stream{nullptr};

  struct RingParams {
    int prev_rank{0};
    int next_rank{0};
    P2pIbTransportDevice prev{};
    P2pIbTransportDevice next{};
  };
  static constexpr int kMaxRings = 4;
  RingParams rings[kMaxRings]{};
};

void launch_ring_reduce_scatter(const RingReduceScatterLaunchParams& params);

} // namespace comms::prims
