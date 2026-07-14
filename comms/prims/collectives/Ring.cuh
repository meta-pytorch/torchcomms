// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>

#include "comms/prims/transport/P2pIbTransportDevice.cuh"

namespace comms::prims {

struct RingTopology {
  int prev_rank{0};
  int next_rank{0};
  P2pIbTransportDevice prev{};
  P2pIbTransportDevice next{};
};

template <int NumRings>
struct RingAllgatherArgs {
  int my_rank{0};
  int num_ranks{0};
  std::size_t sendcount{0};
  std::size_t signaling_data_size{0};
  RingTopology rings[NumRings]{};
  const char* sendbuf{nullptr};
  char* recvbuf{nullptr};
};

template <int NumRings, typename T>
struct RingReduceScatterArgs {
  int my_rank{0};
  int num_ranks{0};
  std::size_t chunk_elements{0};
  std::size_t signaling_data_size{0};
  RingTopology rings[NumRings]{};
  const T* input{nullptr};
  T* output{nullptr};
};

} // namespace comms::prims
