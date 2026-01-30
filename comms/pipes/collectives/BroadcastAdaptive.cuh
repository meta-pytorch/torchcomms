// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>

#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/Transport.cuh"
#include "comms/pipes/collectives/BroadcastBinomialTree.cuh"
#include "comms/pipes/collectives/BroadcastFlat.cuh"
#include "comms/pipes/collectives/BroadcastRing.cuh"

namespace comms::pipes::collectives {
/**
 * Wrapper that selects between flat-tree and ring broadcast
 * based on message size and rank count.
 *
 * Selection criteria (updated from latest profiling on 8-rank NVLink):
 * - For small/medium messages (< 4MB): Use flat-tree (lower latency)
 * - For large messages (>= 4MB) with multiple ranks: Use ring
 *   (1.06x faster than flat-tree at 4MB, competitive at 8MB+)
 *
 * Updated benchmark results:
 * - 4MB: Ring 42.27 GB/s vs Flat 39.85 GB/s (Ring is 1.06x faster)
 * - 8MB: Ring 84.27 GB/s vs Flat 85.49 GB/s (essentially tied)
 * - 64MB: Ring 252.94 GB/s vs Flat 260.48 GB/s (essentially tied)
 *
 * The 4MB threshold was determined through updated benchmarking where ring
 * starts to outperform flat-tree.
 */
__device__ __forceinline__ void broadcast_adaptive(
    void* buff_d,
    int my_rank_id,
    int root_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    std::size_t nbytes) {
#ifdef __CUDA_ARCH__
  // Threshold for switching to ring algorithm (4MB)
  // Updated empirical profiling on 8-rank NVLink:
  // - < 4MB: Ring is 1.06x faster than flat (42.27 vs 39.85 GB/s)
  // - >= 4MB: Ring and flat are competitive (both ~85-260 GB/s)
  //
  // TODO: Make this configurable.
  constexpr std::size_t kRingThreshold = 4 * 1024 * 1024;

  const auto nranks = transports_per_rank.size();

  if (nbytes >= kRingThreshold && nranks > 2) {
    // Large messages (>= 4MB): use ring for best bandwidth
    // Ring achieves up to 77% of NCCL at 64MB (253-260 GB/s)
    broadcast_ring(
        buff_d, my_rank_id, root_rank_id, transports_per_rank, nbytes);
  } else {
    // Small/medium messages (< 4MB) or small rank count: use flat-tree
    // Flat-tree has lower latency overhead for smaller transfers
    broadcastFlat(
        buff_d, my_rank_id, root_rank_id, transports_per_rank, nbytes);
  }
#endif
}

} // namespace comms::pipes::collectives
