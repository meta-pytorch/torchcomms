// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <thread>
#include <vector>

#include "meta/colltrace/AlgoStats.h"

using namespace ncclx::colltrace;

TEST(AlgoStatsTest, BasicRecordAndDump) {
  AlgoStats stats;
  stats.setCommInfo(0x12345678, "TP_PG");

  stats.record("ReduceScatter", "PAT");
  stats.record("ReduceScatter", "PAT");
  stats.record("ReduceScatter", "RING");
  stats.record("AllReduce", "TREE");

  auto dump = stats.dump();

  EXPECT_EQ(dump.commHash, 0x12345678);
  EXPECT_EQ(dump.commDesc, "TP_PG");
  EXPECT_EQ(dump.counts["ReduceScatter"]["PAT"], 2);
  EXPECT_EQ(dump.counts["ReduceScatter"]["RING"], 1);
  EXPECT_EQ(dump.counts["AllReduce"]["TREE"], 1);
}

TEST(AlgoStatsTest, Reset) {
  AlgoStats stats;
  stats.setCommInfo(0xABCD, "test_comm");

  stats.record("AllGather", "RING");
  stats.record("AllGather", "RING");

  auto dumpBefore = stats.dump();
  EXPECT_EQ(dumpBefore.counts["AllGather"]["RING"], 2);

  stats.reset();

  auto dumpAfter = stats.dump();
  EXPECT_TRUE(dumpAfter.counts.empty());
  // Comm info should be preserved after reset
  EXPECT_EQ(dumpAfter.commHash, 0xABCD);
  EXPECT_EQ(dumpAfter.commDesc, "test_comm");
}

TEST(AlgoStatsTest, ConcurrentRecording) {
  AlgoStats stats;
  stats.setCommInfo(0x1234, "concurrent_test");

  constexpr int kNumThreads = 4;
  constexpr int kRecordsPerThread = 1000;

  std::vector<std::thread> threads;
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([&stats, i]() {
      for (int j = 0; j < kRecordsPerThread; ++j) {
        stats.record("AllReduce", "PAT");
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  auto dump = stats.dump();
  EXPECT_EQ(dump.counts["AllReduce"]["PAT"], kNumThreads * kRecordsPerThread);
}

TEST(AlgoStatsTest, MultipleCollectivesAndAlgorithms) {
  AlgoStats stats;
  stats.setCommInfo(0x5678, "multi_test");

  // Record various collectives with different algorithms
  for (int i = 0; i < 30; ++i) {
    stats.record("ReduceScatter", "PAT");
  }
  for (int i = 0; i < 5; ++i) {
    stats.record("ReduceScatter", "RING");
  }
  for (int i = 0; i < 50; ++i) {
    stats.record("AllReduce", "TREE");
  }
  for (int i = 0; i < 10; ++i) {
    stats.record("AllGather", "RING");
  }

  auto dump = stats.dump();

  EXPECT_EQ(dump.counts.size(), 3); // 3 collectives
  EXPECT_EQ(dump.counts["ReduceScatter"].size(), 2); // 2 algorithms
  EXPECT_EQ(dump.counts["ReduceScatter"]["PAT"], 30);
  EXPECT_EQ(dump.counts["ReduceScatter"]["RING"], 5);
  EXPECT_EQ(dump.counts["AllReduce"]["TREE"], 50);
  EXPECT_EQ(dump.counts["AllGather"]["RING"], 10);
}

TEST(AlgoStatsTest, EmptyDump) {
  AlgoStats stats;
  stats.setCommInfo(0x9999, "empty_test");

  auto dump = stats.dump();

  EXPECT_EQ(dump.commHash, 0x9999);
  EXPECT_EQ(dump.commDesc, "empty_test");
  EXPECT_TRUE(dump.counts.empty());
}
