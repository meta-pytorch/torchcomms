// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/logger/LogTypes.h"

#include <atomic>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

namespace meta::comms::logger {
namespace {

TEST(LogTypesTest, SupportsConcurrentMaskPublicationAndReads) {
  /*
   * The race detector is the assertion for concurrent access to this shared
   * process-wide filter. The final checks retain ordinary functional coverage.
   */
  constexpr int kIterations = 100'000;
  constexpr int kWriterThreads = 2;
  constexpr int kReaderThreads = 6;

  std::atomic<bool> start{false};
  std::vector<std::thread> threads;
  threads.reserve(kWriterThreads + kReaderThreads);

  for (int writer = 0; writer < kWriterThreads; ++writer) {
    threads.emplace_back([&, writer] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (int iteration = 0; iteration < kIterations; ++iteration) {
        setSubSystemMask(
            (iteration + writer) % 2 == 0 ? static_cast<uint64_t>(INIT)
                                          : static_cast<uint64_t>(COLL));
      }
    });
  }

  for (int reader = 0; reader < kReaderThreads; ++reader) {
    threads.emplace_back([&] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (int iteration = 0; iteration < kIterations; ++iteration) {
        (void)isEnabledSubSystemBitwise(INIT | COLL);
      }
    });
  }

  start.store(true, std::memory_order_release);
  for (auto& thread : threads) {
    thread.join();
  }

  setSubSystemMask(INIT | COLL);
  EXPECT_TRUE(isEnabledSubSystemBitwise(INIT));
  EXPECT_TRUE(isEnabledSubSystemBitwise(COLL));
  EXPECT_FALSE(isEnabledSubSystemBitwise(P2P));
}

} // namespace
} // namespace meta::comms::logger
