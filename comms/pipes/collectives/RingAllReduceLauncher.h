// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>
#include <cstddef>

namespace comms::pipes {

class P2pIbgdaTransportDevice;

struct RingAllReduceLaunchParams {
  int my_rank{0};
  int num_ranks{0};
  std::size_t count{0};
  std::size_t signaling_data_size{0};
  const float* input{nullptr};
  float* output{nullptr};
  int num_blocks{16};
  int num_rings{1};
  float timeout_ms{0.0f};
  bool enable_bidir_ag{false};
  std::size_t ib_window_bytes{0};
  bool skip_reduction{false};
  cudaStream_t stream{nullptr};

  struct RingParams {
    int prev_rank{0};
    int next_rank{0};
    P2pIbgdaTransportDevice* prev{nullptr};
    P2pIbgdaTransportDevice* next{nullptr};
  };
  static constexpr int kMaxRings = 4;
  RingParams rings[kMaxRings]{};
};

// Launches a ring allreduce on the specified stream.
//
// Preconditions:
// - count must be divisible by num_ranks
// - transport devices must be bound and connected
void launch_ring_allreduce(const RingAllReduceLaunchParams& params);

} // namespace comms::pipes
