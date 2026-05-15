// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>

#include "comms/pipes/P2pNvlTransportDevice.cuh"
#include "comms/pipes/collectives/DirectCollectiveTypes.h"

namespace comms::pipes {

class P2pIbgdaTransportDevice;

struct DirectAllgatherNvlArgs {
  int my_rank{0};
  int num_ranks{0};
  std::size_t sendcount{0};
  std::size_t signaling_data_size{0};
  P2pNvlTransportDevice peers[kDirectNvlMaxRanks]{};
  const char* sendbuf{nullptr};
  char* recvbuf{nullptr};
};

struct HierarchicalAllgatherIbgdaRing {
  int prev_rank{0};
  int next_rank{0};
  P2pIbgdaTransportDevice* prev{nullptr};
  P2pIbgdaTransportDevice* next{nullptr};
};

struct HierarchicalAllgatherFusedArgs {
  int ib_rank{0};
  int ib_size{0};
  int nvl_rank{0};
  int nvl_size{0};
  std::size_t sendcount{0};
  std::size_t ib_signaling_data_size{0};
  std::size_t nvl_signaling_data_size{0};
  const char* sendbuf{nullptr};
  char* recvbuf{nullptr};
  HierarchicalAllgatherIbgdaRing ib_ring{};
  P2pNvlTransportDevice nvl_peers[kDirectNvlMaxRanks]{};
};

} // namespace comms::pipes
