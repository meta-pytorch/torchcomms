// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <algorithm>
#include <cstddef>

namespace ctran::reducescatter::direct_ib {

constexpr std::size_t kSignalChunkThreshold = 4ULL * 1024 * 1024;
constexpr std::size_t kSignalingDataSize = 256ULL * 1024;

inline int numBlocksForTotalBytes(std::size_t totalBytes, int maxChannels) {
  if (totalBytes <= 32ULL * 1024) {
    return std::min(1, maxChannels);
  }
  if (totalBytes <= 64ULL * 1024) {
    return std::min(2, maxChannels);
  }
  if (totalBytes <= 128ULL * 1024) {
    return std::min(4, maxChannels);
  }
  return std::min(8, maxChannels);
}

inline std::size_t signalingDataSize(std::size_t chunkBytes) {
  return chunkBytes <= kSignalChunkThreshold ? kSignalingDataSize : 0;
}

} // namespace ctran::reducescatter::direct_ib
