// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace ctran::allreduce::ring {

enum class GpuArch {
  Default, // GB200 / Blackwell (Phase 2 tuning)
  Hopper,  // H100 / SM 9.0
};

struct PipelineParams {
  size_t chunkSize;
  size_t numChunks;
};

// Auto-tune pipeline parameters (chunkSize, numChunks) based on message size
// and number of ranks. Values must satisfy chunkSize * numChunks <= maxBDP.
inline PipelineParams
getAutoTunedPipeline(size_t messageBytes, size_t maxBDP, int nRanks) {
  constexpr size_t kMinChunkSize = 256 * 1024; // 256KB
  constexpr size_t kMaxChunkSize = 16ULL * 1024 * 1024; // 16MB
  constexpr int kMinChunksPerRank = 4;

  size_t chunkSize = messageBytes / (kMinChunksPerRank * nRanks);
  chunkSize = std::clamp(chunkSize, kMinChunkSize, kMaxChunkSize);

  size_t numChunks = static_cast<size_t>(nRanks);

  // Ensure we don't exceed allocated buffer
  while (chunkSize * numChunks > maxBDP && chunkSize > kMinChunkSize) {
    chunkSize /= 2;
  }

  while (chunkSize * numChunks > maxBDP && numChunks > 1) {
    numChunks /= 2;
  }

  if (chunkSize * numChunks > maxBDP) {
    throw std::runtime_error(
        "AutoTune error: nRanks " + std::to_string(nRanks) + " maxBDP " +
        std::to_string(maxBDP) + " < chunks " + std::to_string(numChunks) +
        " x " + std::to_string(chunkSize) + "B");
  }

  return {chunkSize, numChunks};
}

// Auto-tune block count based on per-rank message size and GPU architecture.
//
// Default (GB200/Blackwell) thresholds from Phase 2 empirical sweep (n4, n8):
//   perRank < 64K: 1,  64K-256K: 2,  256K-512K: 4,  >=512K: 8
//
// Hopper (H100) thresholds from 8xH100 single-node sweep:
//   perRank < 32K: 1,  32K-1M: 2,  1M-16M: 4,  >=16M: 8
inline int getAutoTunedNumBlocks(
    size_t messageBytes,
    int nRanks,
    int maxOccupancyBlocks,
    GpuArch arch = GpuArch::Default) {
  size_t perRank = messageBytes / std::max(nRanks, 1);

  int targetBlocks;
  if (arch == GpuArch::Hopper) {
    constexpr size_t k32K = 32ULL * 1024;
    constexpr size_t k1M = 1024ULL * 1024;
    constexpr size_t k16M = 16ULL * 1024 * 1024;
    if (perRank < k32K) {
      targetBlocks = 1;
    } else if (perRank < k1M) {
      targetBlocks = 2;
    } else if (perRank < k16M) {
      targetBlocks = 4;
    } else {
      targetBlocks = 8;
    }
  } else {
    constexpr size_t k64K = 64ULL * 1024;
    constexpr size_t k256K = 256ULL * 1024;
    constexpr size_t k512K = 512ULL * 1024;
    if (perRank < k64K) {
      targetBlocks = 1;
    } else if (perRank < k256K) {
      targetBlocks = 2;
    } else if (perRank < k512K) {
      targetBlocks = 4;
    } else {
      targetBlocks = 8;
    }
  }
  return std::min(targetBlocks, maxOccupancyBlocks);
}

// Auto-tune thread block size based on per-rank message size and GPU arch.
//
// Default (GB200/Blackwell): returns defaultThreads (no tuning, uses
// cudaOccupancyMaxPotentialBlockSize result).
//
// Hopper (H100) thresholds from 8xH100 single-node sweep:
//   perRank < 1M: 384,  >=1M: 512
inline int getAutoTunedThreadBlockSize(
    size_t messageBytes,
    int nRanks,
    int defaultThreads,
    GpuArch arch = GpuArch::Default) {
  if (arch != GpuArch::Hopper) {
    return defaultThreads;
  }
  size_t perRank = messageBytes / std::max(nRanks, 1);
  constexpr size_t k1M = 1024ULL * 1024;
  return perRank < k1M ? 384 : 512;
}

} // namespace ctran::allreduce::ring
