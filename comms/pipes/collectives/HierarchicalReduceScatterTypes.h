// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>

#include "comms/pipes/P2pNvlTransportDevice.cuh"
#include "comms/pipes/collectives/DirectCollectiveTypes.h"

namespace comms::pipes {

class P2pIbgdaTransportDevice;

struct HierarchicalReduceScatterIbgdaRing {
  int prev_rank{0};
  int next_rank{0};
  P2pIbgdaTransportDevice* prev{nullptr};
  P2pIbgdaTransportDevice* next{nullptr};
};

template <typename T>
struct HierarchicalReduceScatterFusedArgs {
  int ib_rank{0};
  int ib_size{0};
  int nvl_rank{0};
  int nvl_size{0};
  std::size_t chunk_elements{0};
  std::size_t ib_signaling_data_size{0};
  std::size_t nvl_signaling_data_size{0};
  const T* input{nullptr};
  T* output{nullptr};
  T* workspace{nullptr};
  HierarchicalReduceScatterIbgdaRing ib_ring{};
  P2pNvlTransportDevice nvl_peers[kDirectNvlMaxRanks]{};
};

static_assert(
    sizeof(HierarchicalReduceScatterFusedArgs<float>) <= 32764,
    "Hierarchical reduce-scatter launch args exceed CUDA kernel parameter limit");

} // namespace comms::pipes
