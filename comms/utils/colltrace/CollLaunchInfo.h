// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <atomic>
#include <cstdint>

namespace meta::comms::colltrace {

// Launch geometry of a traced collective's kernel; 0 means unknown. Set by the
// launching thread and read by the CollTrace thread, hence the atomics.
class CollLaunchInfo {
 public:
  void set(uint32_t numBlocks, uint32_t blockSize, uint32_t blocksPerSm = 0) {
    numBlocks_.store(numBlocks, std::memory_order_relaxed);
    blockSize_.store(blockSize, std::memory_order_relaxed);
    blocksPerSm_.store(blocksPerSm, std::memory_order_relaxed);
  }

  void copyFrom(const CollLaunchInfo& other) {
    set(other.numBlocks(), other.blockSize(), other.blocksPerSm());
  }

  uint32_t numBlocks() const {
    return numBlocks_.load(std::memory_order_relaxed);
  }
  uint32_t blockSize() const {
    return blockSize_.load(std::memory_order_relaxed);
  }
  uint32_t blocksPerSm() const {
    return blocksPerSm_.load(std::memory_order_relaxed);
  }

 private:
  std::atomic<uint32_t> numBlocks_{0};
  std::atomic<uint32_t> blockSize_{0};
  std::atomic<uint32_t> blocksPerSm_{0};
};

} // namespace meta::comms::colltrace
