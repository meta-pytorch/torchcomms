// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>

#include "comms/prims/transport/P2pIbTransportDeviceDecl.cuh"

namespace comms::prims {

struct RingAllgatherLaunchParams {
  int my_rank{0};
  int num_ranks{0};
  std::size_t sendcount{0};
  std::size_t signaling_data_size{0};
  const char* sendbuf{nullptr};
  char* recvbuf{nullptr};
  int num_blocks{16};
  int num_rings{1};
  float timeout_ms{0.0f};

  struct RingParams {
    int prev_rank{0};
    int next_rank{0};
    P2pIbTransportDevice prev{};
    P2pIbTransportDevice next{};
  };
  static constexpr int kMaxRings = 4;
  RingParams rings[kMaxRings]{};
};

void launch_ring_allgather(const RingAllgatherLaunchParams& params);

} // namespace comms::prims
