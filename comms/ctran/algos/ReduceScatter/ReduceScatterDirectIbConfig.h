// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>

namespace ctran::reducescatter::direct_ib {

constexpr int kMaxNumBlocks = 8;
constexpr std::size_t kPerChannelSize = 512ULL * 1024;
constexpr int kPipelineDepth = 4;
constexpr int kQpsPerConnection = 1;
constexpr float kTimeoutMs = 30000.0f;
constexpr std::size_t kSignalChunkThreshold = 4ULL * 1024 * 1024;

inline int numBlocksForTotalBytes(std::size_t totalBytes) {
  if (totalBytes <= 32ULL * 1024) {
    return 1;
  }
  if (totalBytes <= 64ULL * 1024) {
    return 2;
  }
  if (totalBytes <= 128ULL * 1024) {
    return 4;
  }
  return kMaxNumBlocks;
}

inline std::size_t signalingDataSize(std::size_t chunkBytes) {
  return chunkBytes <= kSignalChunkThreshold ? kPerChannelSize / 2 : 0;
}

} // namespace ctran::reducescatter::direct_ib
