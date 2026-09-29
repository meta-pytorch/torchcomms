// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>

#include "comms/prims/collectives/DirectCollectiveTypes.h"
#include "comms/prims/transport/nvl/P2pNvlTransportDevice.cuh"

namespace comms::prims {

template <typename T>
struct DirectReduceScatterNvlArgs {
  int my_rank{0};
  int num_ranks{0};
  std::size_t chunk_elements{0};
  std::size_t signaling_data_size{0};
  P2pNvlTransportDevice peers[kDirectNvlMaxRanks]{};
  const T* input{nullptr};
  T* output{nullptr};
};

} // namespace comms::prims
