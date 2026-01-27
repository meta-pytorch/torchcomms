// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <algorithm>
#include <cstddef>

namespace comms::pipes::benchmark {

/**
 * BroadcastOptimalConfig - Auto-tuning parameters for broadcast operations.
 *
 * These functions provide optimal configuration values based on message size,
 * rank count, and hardware characteristics. The values are derived from
 * empirical profiling and can be further tuned based on specific hardware.
 */
struct BroadcastOptimalConfig {
  std::size_t chunkSize;
  std::size_t stagingBufferSize;
  std::size_t pipelineDepth;
  int numBlocks;
  int numThreads;
  bool useRing; // true = ring algorithm, false = flat tree
};

/**
 * Get optimal chunk size based on message size.
 *
 * EMPIRICAL FINDINGS (from profiling on 4-rank NVLink broadcast):
 * - Smaller chunks dramatically improve performance for large messages!
 * - For 64MB: 128KB chunks = 114.81 GB/s (0.35x NCCL)
 *             1MB chunks = 44.79 GB/s (0.14x NCCL)
 * - This is because more chunks = more parallelism across warps
 *
 * The optimal chunk size balances:
 * 1. Parallelism (more chunks = more warps can work concurrently)
 * 2. Overhead (too small = excessive ChunkState polling overhead)
 * 3. Pipeline efficiency (chunk size affects pipelining depth)
 *
 * Key insight: The P2P NVL transport benefits from having many small chunks
 * that can be processed in parallel by different warps, rather than fewer
 * large chunks that serialize the transfer.
 */
inline std::size_t getOptimalChunkSize(std::size_t messageSize, int numRanks) {
  // Suppress unused parameter warning
  (void)numRanks;

  if (messageSize < 16 * 1024) {
    // Very small messages (< 16KB): Use 4KB chunks
    // Minimize overhead while allowing some parallelism
    return 4 * 1024;
  } else if (messageSize < 64 * 1024) {
    // Small messages (16KB - 64KB): Use 8KB chunks
    return 8 * 1024;
  } else if (messageSize < 256 * 1024) {
    // Medium-small messages (64KB - 256KB): Use 16KB chunks
    return 16 * 1024;
  } else if (messageSize < 1024 * 1024) {
    // Medium messages (256KB - 1MB): Use 32KB chunks
    return 32 * 1024;
  } else if (messageSize < 4 * 1024 * 1024) {
    // Medium-large messages (1MB - 4MB): Use 64KB chunks
    return 64 * 1024;
  } else {
    // Large messages (>= 4MB): Use 128KB chunks
    // EMPIRICAL: 128KB chunks gave best results for 64MB message (114.81 GB/s)
    // Smaller chunks (64KB) may be even better - worth testing
    return 128 * 1024;
  }
}

/**
 * Get optimal staging buffer size based on message size and chunk size.
 *
 * Staging buffer sizing criteria:
 * - Must be >= chunk size (to hold at least one chunk)
 * - Should be large enough for effective pipelining
 * - Larger is better for bandwidth, but uses more memory
 */
inline std::size_t getOptimalStagingBufferSize(
    std::size_t messageSize,
    std::size_t chunkSize,
    std::size_t pipelineDepth) {
  // Minimum: 4 chunks worth of staging
  std::size_t minSize = chunkSize * pipelineDepth;

  // For small messages, use small staging buffer
  if (messageSize < 64 * 1024) {
    return std::max(minSize, static_cast<std::size_t>(64 * 1024));
  } else if (messageSize < 1024 * 1024) {
    return std::max(minSize, static_cast<std::size_t>(256 * 1024));
  } else if (messageSize < 8 * 1024 * 1024) {
    return std::max(minSize, static_cast<std::size_t>(1024 * 1024));
  } else if (messageSize < 32 * 1024 * 1024) {
    return std::max(minSize, static_cast<std::size_t>(4 * 1024 * 1024));
  } else {
    return std::max(minSize, static_cast<std::size_t>(8 * 1024 * 1024));
  }
}

/**
 * Get optimal pipeline depth based on message size.
 *
 * Pipeline depth selection:
 * - Small messages: Lower depth to reduce synchronization overhead
 * - Large messages: Higher depth to hide NVLink latency
 */
inline std::size_t getOptimalPipelineDepth(std::size_t messageSize) {
  if (messageSize < 64 * 1024) {
    return 2; // Small messages: less pipelining
  } else if (messageSize < 1024 * 1024) {
    return 4; // Medium messages: moderate pipelining
  } else if (messageSize < 16 * 1024 * 1024) {
    return 4; // Large messages: moderate pipelining
  } else {
    return 8; // Very large messages: deep pipelining
  }
}

/**
 * Get optimal number of thread blocks based on message size and rank count.
 *
 * Block count selection:
 * - More blocks = more parallelism for large messages
 * - Fewer blocks = less overhead for small messages
 */
inline int getOptimalNumBlocks(std::size_t messageSize, int numRanks) {
  // Base blocks on message size
  int basedOnSize = 4;
  if (messageSize >= 1024 * 1024) {
    basedOnSize = 8;
  }
  if (messageSize >= 8 * 1024 * 1024) {
    basedOnSize = 16;
  }
  if (messageSize >= 32 * 1024 * 1024) {
    basedOnSize = 32;
  }

  // For flat-tree broadcast, root needs to send to (numRanks-1) peers
  // More blocks helps parallelize across peers
  int basedOnRanks = std::max(4, (numRanks - 1) * 4);

  // Take the maximum, but cap at reasonable limits
  return std::min(std::max(basedOnSize, basedOnRanks), 64);
}

/**
 * Get optimal number of threads per block.
 *
 * Thread count selection:
 * - More threads = better for large chunk processing
 * - 256-512 is typically optimal for NVLink transfers
 */
inline int getOptimalNumThreads(std::size_t messageSize) {
  if (messageSize < 256 * 1024) {
    return 256; // Small messages: 256 threads
  } else {
    return 512; // Large messages: 512 threads
  }
}

/**
 * Determine whether to use ring algorithm.
 *
 * Selection criteria (based on empirical profiling on 8-rank NVLink):
 * - Ring is 1.77x faster than flat-tree at 8MB, 5.19x faster at 64MB
 * - Flat tree is better for smaller messages (lower latency)
 * - Ring has no benefit for 2-rank case
 *
 * Benchmark results for 64MB messages:
 * - Ring: 253.36 GB/s (0.77x NCCL)
 * - Binomial: 116.65 GB/s (0.35x NCCL)
 * - Flat: 48.80 GB/s (0.15x NCCL)
 */
inline bool shouldUseRing(std::size_t messageSize, int numRanks) {
  // Threshold for switching to ring algorithm (8MB)
  // Empirically determined on 8-rank NVLink configurations
  // IMPORTANT: Must match kRingThreshold in
  // comms/pipes/collectives/BroadcastBinomialTree.cuh
  constexpr std::size_t kRingThreshold = 8 * 1024 * 1024;

  // Use ring for:
  // 1. Message size >= 8MB AND
  // 2. More than 2 ranks (ring has no benefit for 2 ranks)
  return messageSize >= kRingThreshold && numRanks > 2;
}

/**
 * Get complete optimal configuration for broadcast.
 *
 * This function combines all the individual tuning functions to provide
 * a complete configuration for a broadcast operation.
 */
inline BroadcastOptimalConfig getOptimalBroadcastConfig(
    std::size_t messageSize,
    int numRanks) {
  BroadcastOptimalConfig config;

  config.chunkSize = getOptimalChunkSize(messageSize, numRanks);
  config.pipelineDepth = getOptimalPipelineDepth(messageSize);
  config.stagingBufferSize = getOptimalStagingBufferSize(
      messageSize, config.chunkSize, config.pipelineDepth);
  config.numBlocks = getOptimalNumBlocks(messageSize, numRanks);
  config.numThreads = getOptimalNumThreads(messageSize);
  config.useRing = shouldUseRing(messageSize, numRanks);

  return config;
}

} // namespace comms::pipes::benchmark
