// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/common/CollectiveStats.h"

#include <thread>
#include <vector>

#include <gtest/gtest.h>

namespace comms {
namespace {

TEST(CollectiveStatsTest, RecordAccumulatesTotalMinMaxCount) {
  CollectiveStats stats;
  stats.record("allreduce", "allreduce.ctring.1024", 10);
  stats.record("allreduce", "allreduce.ctring.1024", 30);
  stats.record("allreduce", "allreduce.ctring.1024", 20);

  const auto out = stats.getAndClear();
  const CollectiveStat expected{
      .count = 3, .total_us = 60, .min_us = 10, .max_us = 30};
  EXPECT_EQ(out.at("allreduce.ctring.1024"), expected);
}

TEST(CollectiveStatsTest, SeparateKeysTrackedIndependently) {
  CollectiveStats stats;
  stats.record("allreduce", "allreduce.ctring.1024", 5);
  stats.record("reducescatter", "reducescatter.ctring.2048", 7);

  const auto out = stats.getAndClear();
  EXPECT_EQ(
      out.at("allreduce.ctring.1024"),
      (CollectiveStat{.count = 1, .total_us = 5, .min_us = 5, .max_us = 5}));
  EXPECT_EQ(
      out.at("reducescatter.ctring.2048"),
      (CollectiveStat{.count = 1, .total_us = 7, .min_us = 7, .max_us = 7}));
}

// The same collective recorded under two algorithms rolls up into
// "<collective>.all"; everything rolls up into "all".
TEST(CollectiveStatsTest, GetAndClearDerivesCollectiveAndOverallAggregates) {
  CollectiveStats stats;
  stats.record("allreduce", "allreduce.ctring.1024", 10);
  stats.record("allreduce", "allreduce.ctdirect.4096", 30);
  stats.record("reducescatter", "reducescatter.ctring.2048", 4);

  const auto out = stats.getAndClear();
  EXPECT_EQ(
      out.at("allreduce.all"),
      (CollectiveStat{.count = 2, .total_us = 40, .min_us = 10, .max_us = 30}));
  EXPECT_EQ(
      out.at("reducescatter.all"),
      (CollectiveStat{.count = 1, .total_us = 4, .min_us = 4, .max_us = 4}));
  EXPECT_EQ(
      out.at("all"),
      (CollectiveStat{.count = 3, .total_us = 44, .min_us = 4, .max_us = 30}));
}

TEST(CollectiveStatsTest, GetAndClearResetsState) {
  CollectiveStats stats;
  stats.record("allreduce", "allreduce.ctring.1024", 10);
  EXPECT_FALSE(stats.getAndClear().empty());
  EXPECT_TRUE(stats.getAndClear().empty());
}

TEST(CollectiveStatsTest, ConcurrentRecordIsThreadSafe) {
  CollectiveStats stats;
  constexpr uint64_t kThreads = 8;
  constexpr uint64_t kPerThread = 1000;
  std::vector<std::thread> threads;
  threads.reserve(kThreads);
  for (uint64_t t = 0; t < kThreads; ++t) {
    threads.emplace_back([&] {
      for (uint64_t i = 0; i < kPerThread; ++i) {
        stats.record("allreduce", "allreduce.ctring.1024", 1);
      }
    });
  }
  for (auto& th : threads) {
    th.join();
  }

  const auto out = stats.getAndClear();
  const CollectiveStat expected{
      .count = kThreads * kPerThread,
      .total_us = kThreads * kPerThread,
      .min_us = 1,
      .max_us = 1};
  EXPECT_EQ(out.at("allreduce.ctring.1024"), expected);
  EXPECT_EQ(out.at("all"), expected);
}

} // namespace
} // namespace comms
