// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <folly/json/dynamic.h>
#include <folly/json/json.h>

#include "comms/utils/memtrace/MemoryTrace.h"

using namespace meta::comms::memtrace;

TEST(MemoryTraceTest, RecordAllocUpdatesStats) {
  auto trace = MemoryTrace::getOrCreate(0x1001);
  auto before = trace->getStats();
  trace->recordAlloc(0xAAAA, 1024);

  auto after = trace->getStats();
  EXPECT_EQ(after.totalAllocated, before.totalAllocated + 1024);
  EXPECT_EQ(after.currentUsage, before.currentUsage + 1024);
  EXPECT_GE(after.peakUsage, after.currentUsage);
}

TEST(MemoryTraceTest, RecordFreeUpdatesStats) {
  auto trace = MemoryTrace::getOrCreate(0x1002);
  trace->recordAlloc(0xBBBB, 2048);
  auto before = trace->getStats();

  trace->recordFree(0xBBBB, 2048);
  auto after = trace->getStats();
  EXPECT_EQ(after.totalFreed, before.totalFreed + 2048);
  EXPECT_EQ(after.currentUsage, before.currentUsage - 2048);
}

TEST(MemoryTraceTest, RecordFreeWithoutBytesLooksUpAllocMap) {
  auto trace = MemoryTrace::getOrCreate(0x1003);
  trace->recordAlloc(0xCCCC, 4096);
  auto before = trace->getStats();

  trace->recordFree(0xCCCC, std::nullopt);
  auto after = trace->getStats();
  EXPECT_EQ(after.totalFreed, before.totalFreed + 4096);
  EXPECT_EQ(after.currentUsage, before.currentUsage - 4096);
}

TEST(MemoryTraceTest, PeakUsageTracking) {
  auto trace = MemoryTrace::getOrCreate(0x1004);
  trace->recordAlloc(0xD001, 1000);
  trace->recordAlloc(0xD002, 2000);

  trace->recordFree(0xD001, 1000);
  auto stats = trace->getStats();
  EXPECT_GE(stats.peakUsage, 3000);
  EXPECT_EQ(stats.currentUsage, 2000);
}

TEST(MemoryTraceTest, GetOrCreateReturnsSameInstance) {
  auto t1 = MemoryTrace::getOrCreate(0x2001);
  auto t2 = MemoryTrace::getOrCreate(0x2001);
  EXPECT_EQ(t1.get(), t2.get());
}

TEST(MemoryTraceTest, GetOrCreateDifferentHash) {
  auto t1 = MemoryTrace::getOrCreate(0x3001);
  auto t2 = MemoryTrace::getOrCreate(0x3002);
  EXPECT_NE(t1.get(), t2.get());
}

TEST(MemoryTraceTest, DumpProducesValidJson) {
  auto trace = MemoryTrace::getOrCreate(0x4001);
  trace->recordAlloc(0xEEEE, 8192);

  auto jsonStr = trace->dump();
  auto parsed = folly::parseJson(jsonStr);
  EXPECT_TRUE(parsed.isObject());
  EXPECT_TRUE(parsed.count("totalAllocated"));
  EXPECT_TRUE(parsed.count("totalFreed"));
  EXPECT_TRUE(parsed.count("currentUsage"));
  EXPECT_TRUE(parsed.count("peakUsage"));
  EXPECT_GE(parsed["totalAllocated"].asInt(), 8192);
}
