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

TEST(CollectiveStatsTest, RecordsLaunchGeometry) {
  CollectiveStats stats;
  stats.record(
      "allreduce",
      "allreduce.ctring.1024",
      10,
      /*numBlocks=*/8,
      /*blockSize=*/512,
      /*blocksPerSm=*/4);

  const auto out = stats.getAndClear();
  const CollectiveStat expected{
      .count = 1,
      .total_us = 10,
      .min_us = 10,
      .max_us = 10,
      .num_blocks = 8,
      .block_size = 512,
      .blocks_per_sm = 4,
      .total_sm_us = 20}; // ceil(8/4) * 10
  EXPECT_EQ(out.at("allreduce.ctring.1024"), expected);
}

TEST(CollectiveStatsTest, GeometryDefaultsToUnknown) {
  CollectiveStats stats;
  stats.record("allreduce", "allreduce.ctring.1024", 10);

  const auto out = stats.getAndClear();
  const CollectiveStat expected{
      .count = 1, .total_us = 10, .min_us = 10, .max_us = 10};
  EXPECT_EQ(out.at("allreduce.ctring.1024"), expected);
}

TEST(CollectiveStatsTest, RollUpOverMatchingGeometryKeepsIt) {
  CollectiveStats stats;
  stats.record("allreduce", "allreduce.ctring.1024", 10, 8, 512, 4);
  stats.record("allreduce", "allreduce.ctring.2048", 20, 8, 512, 4);

  const auto out = stats.getAndClear();
  const CollectiveStat expected{
      .count = 2,
      .total_us = 30,
      .min_us = 10,
      .max_us = 20,
      .num_blocks = 8,
      .block_size = 512,
      .blocks_per_sm = 4,
      .total_sm_us = 60};
  EXPECT_EQ(out.at("allreduce.all"), expected);
}

TEST(CollectiveStatsTest, RollUpOverDifferingGeometryReportsUnknown) {
  CollectiveStats stats;
  stats.record("allreduce", "allreduce.ctring.1024", 10, 1, 512, 4);
  stats.record("allreduce", "allreduce.ctring.1048576", 20, 8, 512, 4);

  const auto out = stats.getAndClear();
  const CollectiveStat expected{
      .count = 2,
      .total_us = 30,
      .min_us = 10,
      .max_us = 20,
      .num_blocks = 0,
      .block_size = 0,
      .blocks_per_sm = 0,
      .total_sm_us = 50}; // ceil(1/4)*10 + ceil(8/4)*20
  EXPECT_EQ(out.at("allreduce.all"), expected);
}

TEST(CollectiveStatsTest, GeometrylessRecordDoesNotClearReportedGeometry) {
  CollectiveStats stats;
  stats.record("allreduce", "allreduce.ctring.1024", 10, 8, 512, 4);
  stats.record("allreduce", "allreduce.ctring.1024", 20);

  const auto out = stats.getAndClear();
  const CollectiveStat expected{
      .count = 2,
      .total_us = 30,
      .min_us = 10,
      .max_us = 20,
      .num_blocks = 8,
      .block_size = 512,
      .blocks_per_sm = 4,
      .total_sm_us = 20}; // only the geometry-bearing record counts
  EXPECT_EQ(out.at("allreduce.ctring.1024"), expected);
}

TEST(CollectiveStatsTest, DisagreeingGeometryWithinOneBucketReportsUnknown) {
  CollectiveStats stats;
  // Variable-size ops share a "<op>.<algo>.0" bucket.
  stats.record("alltoallv", "alltoallv.ctran.0", 10, 4, 512, 2);
  stats.record("alltoallv", "alltoallv.ctran.0", 20, 16, 512, 2);

  const auto out = stats.getAndClear();
  const CollectiveStat expected{
      .count = 2,
      .total_us = 30,
      .min_us = 10,
      .max_us = 20,
      .num_blocks = 0,
      .block_size = 0,
      .blocks_per_sm = 0,
      .total_sm_us = 180}; // ceil(4/2)*10 + ceil(16/2)*20
  EXPECT_EQ(out.at("alltoallv.ctran.0"), expected);
}

TEST(CollectiveStatsTest, SmTimeFallsBackToGridWhenOccupancyUnknown) {
  CollectiveStats stats;
  stats.record("allreduce", "allreduce.ctdirect.8", 10, 32, 640, 0);

  const auto out = stats.getAndClear();
  EXPECT_EQ(out.at("allreduce.ctdirect.8").total_sm_us, 320u);
}

} // namespace
} // namespace comms
