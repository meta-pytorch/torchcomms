// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

#include "comms/prims/collectives/DirectCollectiveTypes.h"
#include "comms/prims/trace/PipesTraceTypes.h"
#include "comms/prims/transport/nvl/P2pNvlTransportDevice.cuh"

namespace comms::prims {

class P2pIbgdaTransportDevice;

// Max number of inter-node (IB) peers addressable by the direct/star small-
// message path. The testbed is 4 nodes; this is sized generously and the host
// falls back to the ring when nNodes exceeds it.
inline constexpr int kHierarchicalAgMaxNodes = 32;

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
  HierarchicalAllgatherIbgdaRing ib_ring{};
  P2pNvlTransportDevice nvl_peers[kDirectNvlMaxRanks]{};
  // Direct/star inter-node IB phase for small messages: when use_direct is set,
  // each rank sends its own slice directly to the same nvl_rank on every other
  // node (ib_peers[node]) and receives theirs, replacing the W-1 sequential
  // store-and-forward ring hops with a single parallel hop. ib_peers[ib_rank]
  // is unused (self).
  bool use_direct{false};
  P2pIbgdaTransportDevice* ib_peers[kHierarchicalAgMaxNodes]{};
  PipesTraceHandle trace{};
};

} // namespace comms::prims
