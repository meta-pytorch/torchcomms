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
 *   - 256 threads (8 warps/block) consistently outperforms 128 threads
 *     (4 warps/block) at every message size, often by 50%+.
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

  // All configs use 256 threads (8 warps/block).
  //
  // Target: ~2-4 packets/warp.
  //   packets = ceil(nbytes / 120)
  //   warps   = numBlocks * 8
  //
  // The table below is intentionally simple — a series of message-size
  // thresholds.  Sizes are chosen so that the recommended block count sits
  // just past the "knee" of the scaling curve (i.e., the point where adding
  // more blocks yields <5% additional bandwidth).

  constexpr int kThreads = 256;

  if (nbytes <= 2 * 1024) {
    // 64B-2KB: 1 block, 8 warps.
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

} // namespace comms::pipes
