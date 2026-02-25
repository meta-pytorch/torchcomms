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
 * - nbytes >= 8MB: 64 blocks, 512 threads, 32KB chunks, 16MB buffer, depth 3,
 *   spread cluster launch. Achieved 0.98-1.00x NCCL across 64MB-1GB.
 *   Memory: 7 peers × 3 depth × 16MB = 336MB staging per rank.
 *   PD=3 captures 98-100% of PD=4 throughput at 25% less memory (336MB vs
 *   448MB per rank).
 *   Benchmark data: PD=2→3 gains 11 GB/s at 1GB; PD=3→4 gains only ~3 GB/s.
 *   512 chunks/step for 1024 warps (2 warps per 32KB chunk).
 * - nbytes < 8MB: 16 blocks, 512 threads, 16KB chunks, 4MB buffer, depth 2,
 *   spread cluster launch. 27% improvement over 32KB chunks at 4MB
 *   (0.89x vs 0.70x NCCL) by fixing warp utilization: 4MB/16KB = 256 chunks
 *   for 256 warps (100%) vs 4MB/32KB = 128 chunks for 256 warps (50%).
 *
 * Minimum chunks_per_step guidance:
 *
 * The absolute number of chunks per pipeline step (dataBufferSize / chunkSize)
 * directly determines the number of active warps and thus memory-level
 * parallelism (MLP). Benchmark data shows that the active warp count — not
 * the ratio to total warps — drives throughput:
 *
 *   chunks_per_step >= 256:  good          (0.84-1.01x NCCL)
 *   chunks_per_step =  128:  poor          (0.56x NCCL)
 *   chunks_per_step =   64:  marginal      (0.29x NCCL)
 *   chunks_per_step =   32:  catastrophic  (0.14x NCCL)
 *
 * Key evidence: 8-block (128 warps) and 16-block (256 warps) configs with
 * 512KB chunks both produce identical ~52 GB/s throughput, proving that only
 * the 32 active warps (= 32 chunks) matter, not the total warp count.
 * Doubling the buffer from 16MB to 32MB (32 → 64 chunks) doubles throughput
 * proportionally, confirming the linear MLP relationship.
 *
 * When overriding configs, ensure dataBufferSize / chunkSize >= 256 for
 * near-optimal performance. Below 256 chunks/step, throughput drops
 * sharply (128 chunks = 0.56x NCCL).
 *
 * @param nbytes Message size in bytes.
 * @return BroadcastLaunchParams with recommended transport config and launch
 *         parameters.
 */
inline BroadcastLaunchParams recommended_broadcast_params(std::size_t nbytes) {
  constexpr std::size_t kLargeMessageThreshold = 8 * 1024 * 1024; // 8MB

  // Large messages (>= 8MB): 64 blocks, 32KB chunks, PD=3
  // Matches NCCL within ~1-2% at 25% less memory than PD=4 (336MB vs 448MB).
  // 512 chunks/step for 1024 warps (2 warps per 32KB chunk).
  if (nbytes >= kLargeMessageThreshold) {
    return BroadcastLaunchParams{
        .transportConfig =
            MultiPeerNvlTransportConfig{
                .dataBufferSize = 16 * 1024 * 1024, // 16MB
                .chunkSize = 32 * 1024, // 32KB
                .pipelineDepth = 3,
            },
        .numBlocks = 64,
        .numThreads = 512,
        .spreadClusterLaunch = true,
    };
  }

  // Small messages (< 8MB): 16 blocks, 16KB chunks, PD=2
  // 27% improvement at 4MB (0.70x → 0.89x NCCL).
  // 256 chunks/step for 256 warps (1:1 ratio).
  return BroadcastLaunchParams{
      .transportConfig =
          MultiPeerNvlTransportConfig{
              .dataBufferSize = 4 * 1024 * 1024, // 4MB
              .chunkSize = 16 * 1024, // 16KB
              .pipelineDepth = 2,
          },
      .numBlocks = 16,
      .numThreads = 512,
      .spreadClusterLaunch = true,
  };
}

} // namespace comms::pipes::collectives
