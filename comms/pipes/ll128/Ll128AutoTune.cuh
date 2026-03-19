// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>

namespace comms::pipes {

/// Recommended launch configuration for LL128 kernels.
struct Ll128LaunchConfig {
  int numBlocks;
  int numThreads;
};

/**
 * Return the recommended (numBlocks, numThreads) for an LL128 kernel launch
 * given a unidirectional message of @p nbytes.
 *
 * The table is derived from empirical benchmarks on H100 NVLink P2P:
 *
 *   - LL128 scales dramatically with block count — far more than Simple
 *     or NCCL — because each warp independently processes 128-byte,
 *     cache-line-atomic packets with minimal contention.
 *   - 512 threads (16 warps/block) matches NCCL's LL128 configuration
 *     and outperforms 128 threads at every message size.
 *   - The sweet spot is roughly 2-4 packets per warp.  Beyond that,
 *     diminishing returns set in because packet processing saturates.
 *
 * @param nbytes  Message size in bytes (must be a multiple of 16 for LL128).
 * @return Recommended launch configuration.
 */
inline __host__ __device__ Ll128LaunchConfig ll128_auto_tune(size_t nbytes) {
  if (nbytes == 0) {
    return {0, 0}; // No kernel launch needed.
  }

  // All configs use 512 threads (16 warps/block).
  //
  // Target: ~2-4 packets/warp.
  //   packets = ceil(nbytes / 120)
  //   warps   = numBlocks * 16
  //
  // The table below is intentionally simple — a series of message-size
  // thresholds.  Sizes are chosen so that the recommended block count sits
  // just past the "knee" of the scaling curve (i.e., the point where adding
  // more blocks yields <5% additional bandwidth).

  constexpr int kThreads = 512;

  if (nbytes <= 2 * 1024) {
    // 64B-2KB: 1 block, 16 warps.
    // At 2KB, ~17 packets / 8 warps ≈ 2 pkts/warp — good utilization.
    return {1, kThreads};
  }
  if (nbytes <= 4 * 1024) {
    // 3-4KB: 2 blocks, 16 warps.
    return {2, kThreads};
  }
  if (nbytes <= 8 * 1024) {
    // 5-8KB: 4 blocks, 32 warps.
    // Benchmarks show 4 blocks at 8KB achieves ~2x NCCL; the previous
    // default (1 block/128t) was ~0.75x NCCL.
    return {4, kThreads};
  }
  if (nbytes <= 16 * 1024) {
    // 16KB: 8 blocks, 64 warps.
    return {8, kThreads};
  }
  if (nbytes <= 32 * 1024) {
    // 32KB: 16 blocks, 128 warps.  Benchmark: ~14 GB/s.
    return {16, kThreads};
  }
  if (nbytes <= 64 * 1024) {
    // 64KB: 32 blocks, 256 warps.
    return {32, kThreads};
  }
  if (nbytes <= 128 * 1024) {
    // 128KB: 64 blocks, 512 warps.  Benchmark: ~29 GB/s.
    return {64, kThreads};
  }
  if (nbytes <= 256 * 1024) {
    // 256KB: 128 blocks, 1024 warps.
    return {128, kThreads};
  }
  if (nbytes <= 512 * 1024) {
    // 512KB: 128-256 blocks.
    return {256, kThreads};
  }
  if (nbytes <= 1024 * 1024) {
    // 1MB: 512 blocks.  Benchmark: ~130-145 GB/s (varies across runs).
    return {512, kThreads};
  }
  // 2MB+: 1024 blocks.  Benchmark: 1024 beats 512 by 7% at 2MB uni,
  // 4.3% at 4MB bidir.
  return {1024, kThreads};
}

/**
 * Return the recommended (numBlocks, numThreads) for a bidirectional LL128
 * kernel launch.
 *
 * Bidirectional kernels partition warps between send and receive directions
 * (typically via partition_interleaved(2)), so each direction gets half the
 * warps.  To compensate, we roughly double the block count relative to
 * unidirectional recommendations.
 *
 * @param nbytes  Message size per direction in bytes.
 * @return Recommended launch configuration.
 */
inline __host__ __device__ Ll128LaunchConfig
ll128_auto_tune_bidirectional(size_t nbytes) {
  auto uni = ll128_auto_tune(nbytes);
  // Double the blocks to compensate for warp-halving in bidirectional mode,
  // but cap at 1024 (beyond which diminishing returns dominate).
  int bidir_blocks = uni.numBlocks * 2;
  if (bidir_blocks > 1024) {
    bidir_blocks = 1024;
  }
  // Ensure at least 2 blocks for bidirectional (1 per direction minimum).
  if (bidir_blocks < 2) {
    bidir_blocks = 2;
  }
  return {bidir_blocks, uni.numThreads};
}

/**
 * Return the recommended (numBlocks, numThreads) for an AllToAllv LL128 kernel.
 *
 * AllToAllv partitions warps into 2 * (nranks - 1) groups (send/recv x peers),
 * so each peer gets total_warps / (2 * (nranks - 1)) warps.
 *
 * The empirical lookup table below is derived from block-count sweep benchmarks
 * on 8x H100 NVLink.  The old linear formula (bidir * (nranks-1)) grossly
 * over-estimated block counts — e.g. 448 blocks for 64KB when the empirical
 * optimum is ~128.  All peers share the same SMs, so sub-linear scaling is
 * expected.
 *
 * For non-8-rank configurations a dampened heuristic is used as a conservative
 * fallback until more sweep data is available.
 *
 * @param nbytes_per_peer  Message size per peer in bytes.
 * @param nranks           Total number of ranks.
 * @return Recommended launch configuration.
 */
inline __host__ __device__ Ll128LaunchConfig
ll128_auto_tune_alltoallv(size_t nbytes_per_peer, int nranks) {
  if (nbytes_per_peer == 0 || nranks <= 1) {
    return {0, 0};
  }

  constexpr int kThreads = 512;

  // Empirical lookup table for 8 ranks (from block-count sweep benchmarks).
  // Each entry is the block count at or near the knee of the scaling curve.
  auto empirical_8rank = [](size_t nbytes) -> int {
    if (nbytes <= 4 * 1024) {
      return 16; // Sweep: 16 blocks -> 7.76 GB/s vs 18 -> 7.52 GB/s
    }
    if (nbytes <= 32 * 1024) {
      // 16KB benchmarked; 32KB interpolated (between 16KB=96 and 64KB=128)
      return 96;
    }
    // 64KB+: 128 blocks (sweep confirms optimal at 64KB, 256KB, 1MB).
    return 128;
  };

  int needed_blocks;

  if (nranks == 8) {
    // Direct lookup — this is the benchmarked configuration.
    needed_blocks = empirical_8rank(nbytes_per_peer);
  } else {
    // Dampened heuristic for other rank counts.
    // Use the 8-rank empirical value as a baseline, then scale by
    // sqrt(nranks-1) / sqrt(7) to account for more/fewer peers.
    // This is conservative: sqrt grows much slower than the old linear
    // formula, matching the observed sub-linear scaling.
    int base = empirical_8rank(nbytes_per_peer);
    auto bidir = ll128_auto_tune_bidirectional(nbytes_per_peer);
    // ceil(sqrt(nranks - 1))
    int sqrt_peers = 1;
    while (sqrt_peers * sqrt_peers < nranks - 1) {
      ++sqrt_peers;
    }
    int scaled = bidir.numBlocks * sqrt_peers;
    // Take the lesser of the scaled heuristic and the empirical baseline
    // (empirical data already accounts for SM saturation).
    needed_blocks = scaled < base ? scaled : base;
  }

  // Cap at 512 blocks (H100 has 132 SMs; beyond ~512 blocks diminishing
  // returns dominate due to wave scheduling overhead).
  if (needed_blocks > 512) {
    needed_blocks = 512;
  }
  // Minimum: 2 * (nranks - 1) blocks so each peer gets at least 1 warp
  // per direction.
  int min_blocks = 2 * (nranks - 1);
  if (needed_blocks < min_blocks) {
    needed_blocks = min_blocks;
  }

  return {needed_blocks, kThreads};
}

} // namespace comms::pipes
