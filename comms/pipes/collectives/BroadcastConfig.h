// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>

#include "comms/pipes/MultiPeerNvlTransport.h"

namespace comms::pipes::collectives {

/**
 * Recommended launch parameters for broadcast collectives.
 */
struct BroadcastLaunchParams {
  MultiPeerNvlTransportConfig transportConfig;
  int numBlocks{0};
  int numThreads{0};
  bool spreadClusterLaunch{false};
};

/**
 * Returns recommended broadcast launch parameters for a given message size.
 *
 * Configs were validated on 8-rank H100 DGX with broadcast().
 * Callers can override for specific hardware or workload needs.
 *
 * Two-tier selection:
 * - nbytes >= 8MB: 64 blocks, 512 threads, 16KB chunks, 16MB buffer, depth 4,
 *   spread cluster launch. Achieved 0.98-1.01x NCCL across 64MB-1GB.
 *   Memory: 7 peers × 4 depth × 16MB = 448MB staging per rank.
 * - nbytes < 8MB: 16 blocks, 512 threads, 32KB chunks, 8MB buffer, depth 2,
 *   spread cluster launch. Ring topology (7 hops) cannot match NCCL's tree
 *   (3 hops) for latency-bound small messages; ~0.70x measured on 8-GPU DGX.
 *   16 blocks provides better per-hop parallelism than 4 blocks (0.70x vs
 * 0.45x).
 *
 * @param nbytes Message size in bytes.
 * @return BroadcastLaunchParams with recommended transport config and launch
 *         parameters.
 */
inline BroadcastLaunchParams recommended_broadcast_params(std::size_t nbytes) {
  constexpr std::size_t kLargeMessageThreshold = 8 * 1024 * 1024; // 8MB

  // Large messages (>= 8MB): 64 blocks, 16KB chunks, PD=4
  // Achieves ~0.98-1.01x NCCL for 64MB+ (measured).
  // Memory: 7 peers × 4 depth × 16MB = 448MB staging per rank.
  if (nbytes >= kLargeMessageThreshold) {
    return BroadcastLaunchParams{
        .transportConfig =
            MultiPeerNvlTransportConfig{
                .dataBufferSize = 16 * 1024 * 1024, // 16MB
                .chunkSize = 16 * 1024, // 16KB
                .pipelineDepth = 4,
            },
        .numBlocks = 64,
        .numThreads = 512,
        .spreadClusterLaunch = true,
    };
  }

  // Small messages (< 8MB): 16 blocks, 32KB chunks, PD=2
  // Ring topology (7 hops) cannot match NCCL's tree (3 hops) for latency-bound
  // small messages. Measured ~0.70x NCCL on 8-GPU DGX H100.
  // 16 blocks provides better per-hop parallelism than 4 blocks (0.70x vs
  // 0.45x).
  return BroadcastLaunchParams{
      .transportConfig =
          MultiPeerNvlTransportConfig{
              .dataBufferSize = 8 * 1024 * 1024, // 8MB
              .chunkSize = 32 * 1024, // 32KB
              .pipelineDepth = 2,
          },
      .numBlocks = 16,
      .numThreads = 512,
      .spreadClusterLaunch = true,
  };
}

} // namespace comms::pipes::collectives
