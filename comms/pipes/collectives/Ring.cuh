// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>

#include "comms/pipes/P2pIbgdaTransportDevice.cuh"

namespace comms::pipes {

struct RingTopology {
  int prev_rank{0};
  int next_rank{0};
  P2pIbgdaTransportDevice* prev{nullptr};
  P2pIbgdaTransportDevice* next{nullptr};
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

template <int NumRings, typename T>
struct RingAllReduceArgs {
  int my_rank{0};
  int num_ranks{0};
  std::size_t chunk_elements{0};
  std::size_t signaling_data_size{0};
  std::size_t ib_window_bytes{0};
  bool skip_reduction{false};
  RingTopology rings[NumRings]{};
  const T* input{nullptr};
  T* output{nullptr};
};

} // namespace comms::pipes
