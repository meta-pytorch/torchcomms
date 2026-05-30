// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <thread>
#include <vector>

#include "meta/colltrace/AlgoStats.h"

using namespace ncclx::colltrace;

TEST(AlgoStatsTest, BasicRecordAndDump) {
  AlgoStats stats(0x12345678, "TP_PG");

  stats.record("ReduceScatter", "PAT");
  stats.record("ReduceScatter", "PAT");
  stats.record("ReduceScatter", "RING");
  stats.record("AllReduce", "TREE");

  auto dump = stats.dump();

  EXPECT_EQ(dump.commHash, 0x12345678);
  EXPECT_EQ(dump.commDesc, "TP_PG");
  EXPECT_EQ(dump.entries["ReduceScatter"]["PAT"][0], 2);
  EXPECT_EQ(dump.entries["ReduceScatter"]["RING"][0], 1);
  EXPECT_EQ(dump.entries["AllReduce"]["TREE"][0], 1);
}

TEST(AlgoStatsTest, Reset) {
  AlgoStats stats(0xABCD, "test_comm");

  stats.record("AllGather", "RING");
  stats.record("AllGather", "RING");

  auto dumpBefore = stats.dump();
  EXPECT_EQ(dumpBefore.entries["AllGather"]["RING"][0], 2);

  stats.reset();

  auto dumpAfter = stats.dump();
  EXPECT_TRUE(dumpAfter.entries.empty());
  // Comm info should be preserved after reset
  EXPECT_EQ(dumpAfter.commHash, 0xABCD);
  EXPECT_EQ(dumpAfter.commDesc, "test_comm");
}

TEST(AlgoStatsTest, ConcurrentRecording) {
  AlgoStats stats(0x1234, "concurrent_test");

  constexpr int kNumThreads = 4;
  constexpr int kRecordsPerThread = 1000;

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([&stats]() {
      for (int j = 0; j < kRecordsPerThread; ++j) {
        stats.record("AllReduce", "PAT");
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  auto dump = stats.dump();
  EXPECT_EQ(
      dump.entries["AllReduce"]["PAT"][0], kNumThreads * kRecordsPerThread);
}

TEST(AlgoStatsTest, MultipleCollectivesAndAlgorithms) {
  AlgoStats stats(0x5678, "multi_test");

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

  EXPECT_EQ(dump.entries.size(), 3); // 3 collectives
  EXPECT_EQ(dump.entries["ReduceScatter"].size(), 2); // 2 algorithms
  EXPECT_EQ(dump.entries["ReduceScatter"]["PAT"][0], 30);
  EXPECT_EQ(dump.entries["ReduceScatter"]["RING"][0], 5);
  EXPECT_EQ(dump.entries["AllReduce"]["TREE"][0], 50);
  EXPECT_EQ(dump.entries["AllGather"]["RING"][0], 10);
}

TEST(AlgoStatsTest, EmptyDump) {
  AlgoStats stats(0x9999, "empty_test");

  auto dump = stats.dump();

  EXPECT_EQ(dump.commHash, 0x9999);
  EXPECT_EQ(dump.commDesc, "empty_test");
  EXPECT_TRUE(dump.entries.empty());
}
