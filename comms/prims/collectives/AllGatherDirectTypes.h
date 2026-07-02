// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

#include "comms/prims/collectives/DirectCollectiveTypes.h"
#include "comms/prims/trace/PipesTraceTypes.h"
#include "comms/prims/transport/P2pIbTransportDeviceDecl.cuh"
#include "comms/prims/transport/nvl/P2pNvlTransportDevice.cuh"

namespace comms::prims {

struct DirectAllgatherNvlArgs {
  int my_rank{0};
  int num_ranks{0};
  std::size_t sendcount{0};
  std::size_t signaling_data_size{0};
  P2pNvlTransportDevice peers[kDirectNvlMaxRanks]{};
  const char* sendbuf{nullptr};
  char* recvbuf{nullptr};
};

struct HierarchicalAllgatherIbRing {
  int prev_rank{0};
  int next_rank{0};
  P2pIbTransportDevice prev{};
  P2pIbTransportDevice next{};
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
  HierarchicalAllgatherIbRing ib_ring{};
  P2pNvlTransportDevice nvl_peers[kDirectNvlMaxRanks]{};
};

struct HierarchicalAllgatherOverlapArgs {
  int num_ranks{0};
  int ib_rank{0};
  int ib_size{0};
  int nvl_rank{0};
  int nvl_size{0};
  int ib_num_blocks{0};
  int nvl_num_blocks{0};
  std::size_t sendcount{0};
  std::size_t ib_signaling_data_size{0};
  std::size_t nvl_signaling_data_size{0};
  std::size_t chunk_bytes{0};
  uint64_t ready_sequence{0};
  uint64_t* ready_counters{nullptr};
  const char* sendbuf{nullptr};
  char* recvbuf{nullptr};
  HierarchicalAllgatherIbRing ib_ring{};
  P2pNvlTransportDevice nvl_peers[kDirectNvlMaxRanks]{};
  PipesTraceHandle trace{};
};

} // namespace comms::prims
