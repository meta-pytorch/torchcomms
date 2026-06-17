// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

#include "comms/prims/collectives/AllGatherDirectTypes.h"
#include "comms/prims/transport/nvl/P2pNvlTransportDevice.cuh"

namespace comms::prims {

struct DirectAllgatherNvlLaunchParams {
  int my_rank{0};
  int num_ranks{0};
  std::size_t sendcount{0};
  std::size_t signaling_data_size{0};
  const char* sendbuf{nullptr};
  char* recvbuf{nullptr};
  bool in_place{false};
  int num_blocks{16};
  float timeout_ms{0.0f};
  cudaStream_t stream{nullptr};
  P2pNvlTransportDevice peers[kDirectNvlMaxRanks]{};
};

void launch_direct_allgather_nvl(const DirectAllgatherNvlLaunchParams& params);

struct HierarchicalAllgatherLaunchParams {
  int num_ranks{0};
  int ib_rank{0};
  int ib_size{0};
  int nvl_rank{0};
  int nvl_size{0};
  std::size_t sendcount{0};
  std::size_t ib_signaling_data_size{0};
  std::size_t nvl_signaling_data_size{0};
  const char* sendbuf{nullptr};
  char* recvbuf{nullptr};
  bool in_place{false};
  int ib_num_blocks{16};
  float timeout_ms{0.0f};
  cudaStream_t stream{nullptr};
  HierarchicalAllgatherIbgdaRing ib_ring{};
  P2pNvlTransportDevice nvl_peers[kDirectNvlMaxRanks]{};
};

void launch_hierarchical_allgather_fused(
    const HierarchicalAllgatherLaunchParams& params);

struct HierarchicalAllgatherOverlapLaunchParams {
  int num_ranks{0};
  int ib_rank{0};
  int ib_size{0};
  int nvl_rank{0};
  int nvl_size{0};
  std::size_t sendcount{0};
  std::size_t ib_signaling_data_size{0};
  std::size_t nvl_signaling_data_size{0};
  std::size_t chunk_bytes{0};
  uint64_t ready_sequence{0};
  uint64_t* ready_counters{nullptr};
  const char* sendbuf{nullptr};
  char* recvbuf{nullptr};
  bool in_place{false};
  int ib_num_blocks{16};
  int nvl_num_blocks{16};
  float timeout_ms{0.0f};
  cudaStream_t stream{nullptr};
  HierarchicalAllgatherIbgdaRing ib_ring{};
  P2pNvlTransportDevice nvl_peers[kDirectNvlMaxRanks]{};
  bool use_direct{false};
  P2pIbgdaTransportDevice* ib_peers[kHierarchicalAgMaxNodes]{};
  PipesTraceHandle trace{};
};

void launch_hierarchical_allgather_overlap(
    const HierarchicalAllgatherOverlapLaunchParams& params);

} // namespace comms::prims
