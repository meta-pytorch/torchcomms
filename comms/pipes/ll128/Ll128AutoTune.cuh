// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>

namespace comms::pipes {

/// Number of threads per block for LL128 kernels (16 warps/block).
constexpr int kLl128ThreadsPerBlock = 512;

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

  // All configs use kLl128ThreadsPerBlock threads (16 warps/block).
  //
  // Target: ~2-4 packets/warp.
  //   packets = ceil(nbytes / 120)
  //   warps   = numBlocks * 16
  //
  // The table below is intentionally simple — a series of message-size
  // thresholds.  Sizes are chosen so that the recommended block count sits
  // just past the "knee" of the scaling curve (i.e., the point where adding
  // more blocks yields <5% additional bandwidth).

  constexpr int kThreads = kLl128ThreadsPerBlock;

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
 * IMPORTANT: A sharp performance cliff exists at >~2048 total warps
 * (e.g. >128 blocks at 512 threads, >256 blocks at 256 threads).
 * Going from 128 to 192 blocks (512t) drops bandwidth by 18-32% due to
 * wave scheduling overhead on H100's 132 SMs. Do not increase the 64KB+
 * entry without re-running the block-count sweep.
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

  constexpr int kThreads = kLl128ThreadsPerBlock;

  int needed_blocks;

  if (nranks == 8) {
    // Sweep-validated lookup table for 8x H100 NVLink.
    // Each entry is at the knee of the block-count scaling curve.
    if (nbytes_per_peer <= 4 * 1024) {
      needed_blocks = 16; // Sweep: 16 blocks = 9.44 GB/s (knee); plateau 16-256
    } else if (nbytes_per_peer <= 32 * 1024) {
      // 16KB sweep: 48 blocks is the knee (35.6 GB/s); plateau extends to 128.
      // 32KB interpolated (no sweep data; 48 sits on the 16KB plateau).
      needed_blocks = 48;
    } else {
      // 64KB+: 128 blocks (sweep-confirmed optimal at 64KB, 256KB, 1MB).
      // CAUTION: sharp cliff at >~2048 total warps (e.g. 192 blocks at 512
      // threads drops BW by 18-32%). Do not increase without re-sweeping.
      needed_blocks = 128;
    }
  } else {
    // Conservative heuristic cap for non-8-rank configurations.
    // These values are intentionally higher than the 8-rank sweep optima
    // to avoid under-provisioning at higher rank counts (e.g., GB200 with
    // 72 NVLink ranks has 142 warp groups and needs more total warps).
    auto heuristic_cap = [](size_t nbytes) -> int {
      if (nbytes <= 4 * 1024) {
        return 16;
      }
      if (nbytes <= 32 * 1024) {
        return 96; // Conservative: 8-rank optimum is 48, but higher rank
                   // counts need headroom for more warp groups.
      }
      return 128;
    };

    // Dampened heuristic: scale by sqrt(nranks-1) / sqrt(7).
    // sqrt grows much slower than linear, matching observed sub-linear scaling.
    int base = heuristic_cap(nbytes_per_peer);
    auto bidir = ll128_auto_tune_bidirectional(nbytes_per_peer);
    // ceil(sqrt(nranks - 1))
    int sqrt_peers = 1;
    while (sqrt_peers * sqrt_peers < nranks - 1) {
      ++sqrt_peers;
    }
    int scaled = bidir.numBlocks * sqrt_peers;
    // Take the lesser of scaled heuristic and the conservative cap.
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
