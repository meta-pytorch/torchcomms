// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <folly/json/dynamic.h>
#include <folly/json/json.h>

#include "comms/utils/memtrace/MemoryTrace.h"

using namespace meta::comms::memtrace;

TEST(MemoryTraceTest, RecordAllocUpdatesStats) {
  auto trace = MemoryTrace::getOrCreate(0x1001);
  auto before = trace->getStats();
  trace->recordAlloc(0xAAAA, 1024, MemCallsite::Scope::kNccl);

  auto after = trace->getStats();
  EXPECT_EQ(after.totalAllocated, before.totalAllocated + 1024);
  EXPECT_EQ(after.currentUsage, before.currentUsage + 1024);
  EXPECT_GE(after.peakUsage, after.currentUsage);
}

TEST(MemoryTraceTest, RecordFreeUpdatesStats) {
  auto trace = MemoryTrace::getOrCreate(0x1002);
  trace->recordAlloc(0xBBBB, 2048, MemCallsite::Scope::kNccl);
  auto before = trace->getStats();

  trace->recordFree(0xBBBB, 2048, MemCallsite::Scope::kNccl);
  auto after = trace->getStats();
  EXPECT_EQ(after.totalFreed, before.totalFreed + 2048);
  EXPECT_EQ(after.currentUsage, before.currentUsage - 2048);
}

TEST(MemoryTraceTest, RecordFreeWithoutBytesLooksUpAllocMap) {
  auto trace = MemoryTrace::getOrCreate(0x1003);
  trace->recordAlloc(0xCCCC, 4096, MemCallsite::Scope::kNccl);
  auto before = trace->getStats();

  trace->recordFree(0xCCCC, std::nullopt, MemCallsite::Scope::kNccl);
  auto after = trace->getStats();
  EXPECT_EQ(after.totalFreed, before.totalFreed + 4096);
  EXPECT_EQ(after.currentUsage, before.currentUsage - 4096);
}

TEST(MemoryTraceTest, PeakUsageTracking) {
  auto trace = MemoryTrace::getOrCreate(0x1004);
  trace->recordAlloc(0xD001, 1000, MemCallsite::Scope::kNccl);
  trace->recordAlloc(0xD002, 2000, MemCallsite::Scope::kNccl);

  trace->recordFree(0xD001, 1000, MemCallsite::Scope::kNccl);
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
  trace->recordAlloc(0xEEEE, 8192, MemCallsite::Scope::kNccl);

  auto jsonStr = trace->dump();
  auto parsed = folly::parseJson(jsonStr);
  EXPECT_TRUE(parsed.isObject());
  EXPECT_TRUE(parsed.count("totalAllocated"));
  EXPECT_TRUE(parsed.count("totalFreed"));
  EXPECT_TRUE(parsed.count("currentUsage"));
  EXPECT_TRUE(parsed.count("peakUsage"));
  EXPECT_GE(parsed["totalAllocated"].asInt(), 8192);
}

// Tests the commDump integration pattern: adopt a MemoryTrace by commHash,
// record events, then dump to a string→string map (as commDump does).
TEST(MemoryTraceTest, DumpToCommDumpMap) {
  const uint64_t commHash = 0x5001;
  auto trace = MemoryTrace::getOrCreate(commHash);

  // Simulate ncclCommInit allocations
  trace->recordAlloc(0xA001, 1024 * 1024, MemCallsite::Scope::kNccl);
  trace->recordAlloc(0xA002, 512 * 1024, MemCallsite::Scope::kNccl);

  // commDump reads the trace into a map
  std::unordered_map<std::string, std::string> map;
  map["memory"] = trace->dump();

  auto parsed = folly::parseJson(map["memory"]);
  EXPECT_EQ(parsed["totalAllocated"].asInt(), 1024 * 1024 + 512 * 1024);
  EXPECT_EQ(parsed["currentUsage"].asInt(), 1024 * 1024 + 512 * 1024);
  EXPECT_EQ(parsed["peakUsage"].asInt(), 1024 * 1024 + 512 * 1024);

  // Simulate ncclCommDestroy: free without size (ncclCudaFree pattern)
  trace->recordFree(0xA001, std::nullopt, MemCallsite::Scope::kNccl);

  auto stats = trace->getStats();
  EXPECT_EQ(stats.totalFreed, 1024 * 1024);
  EXPECT_EQ(stats.currentUsage, 512 * 1024);
  EXPECT_EQ(stats.peakUsage, 1024 * 1024 + 512 * 1024);
}

// A MemCallsite built implicitly from a bare string is baseline (kNccl); the
// explicit two-arg form tags the scope (e.g. kCtran) while preserving the
// function name. This is how baseline call sites stay unchanged and ctran call
// sites opt into attribution.
TEST(MemoryTraceTest, MemCallsiteScopeClassification) {
  const MemCallsite baselineLiteral("initChannelDevRingUserRanks");
  EXPECT_EQ(baselineLiteral.scope, MemCallsite::Scope::kNccl);
  EXPECT_EQ(baselineLiteral.function, "initChannelDevRingUserRanks");

  const MemCallsite baselineString(std::string("commAlloc"));
  EXPECT_EQ(baselineString.scope, MemCallsite::Scope::kNccl);

  const MemCallsite ctranCallsite(MemCallsite::Scope::kCtran, "initTmpBufs");
  EXPECT_EQ(ctranCallsite.scope, MemCallsite::Scope::kCtran);
  EXPECT_EQ(ctranCallsite.function, "initTmpBufs");

  const MemCallsite mcclCallsite(
      MemCallsite::Scope::kMccl, "McclComm::commRegister");
  EXPECT_EQ(mcclCallsite.scope, MemCallsite::Scope::kMccl);
  EXPECT_EQ(mcclCallsite.function, "McclComm::commRegister");
}

// Allocations recorded under different scopes land in separate per-scope
// buckets, and the aggregate stat equals the sum across scopes.
TEST(MemoryTraceTest, PerScopeStatsSeparateAcrossScopes) {
  auto trace = MemoryTrace::getOrCreate(0x6001);
  const auto ncclBefore = trace->getStats(MemCallsite::Scope::kNccl);
  const auto ctranBefore = trace->getStats(MemCallsite::Scope::kCtran);
  const auto mcclBefore = trace->getStats(MemCallsite::Scope::kMccl);
  const auto aggBefore = trace->getStats();

  trace->recordAlloc(0x6A01, 100, MemCallsite::Scope::kNccl);
  trace->recordAlloc(0x6A02, 200, MemCallsite::Scope::kCtran);
  trace->recordAlloc(0x6A03, 400, MemCallsite::Scope::kMccl);

  const auto ncclAfter = trace->getStats(MemCallsite::Scope::kNccl);
  const auto ctranAfter = trace->getStats(MemCallsite::Scope::kCtran);
  const auto mcclAfter = trace->getStats(MemCallsite::Scope::kMccl);
  const auto aggAfter = trace->getStats();

  EXPECT_EQ(ncclAfter.totalAllocated - ncclBefore.totalAllocated, 100);
  EXPECT_EQ(ctranAfter.totalAllocated - ctranBefore.totalAllocated, 200);
  EXPECT_EQ(mcclAfter.totalAllocated - mcclBefore.totalAllocated, 400);
  EXPECT_EQ(ncclAfter.currentUsage - ncclBefore.currentUsage, 100);
  EXPECT_EQ(ctranAfter.currentUsage - ctranBefore.currentUsage, 200);
  EXPECT_EQ(mcclAfter.currentUsage - mcclBefore.currentUsage, 400);

  EXPECT_EQ(aggAfter.totalAllocated - aggBefore.totalAllocated, 700);
  EXPECT_EQ(
      aggAfter.totalAllocated - aggBefore.totalAllocated,
      (ncclAfter.totalAllocated - ncclBefore.totalAllocated) +
          (ctranAfter.totalAllocated - ctranBefore.totalAllocated) +
          (mcclAfter.totalAllocated - mcclBefore.totalAllocated));
}

// recordFree updates the bucket named by the passed scope; other scopes are
// untouched.
TEST(MemoryTraceTest, FreeUpdatesPassedScopeBucket) {
  auto trace = MemoryTrace::getOrCreate(0x6002);
  const auto ncclBefore = trace->getStats(MemCallsite::Scope::kNccl);
  const auto mcclBefore = trace->getStats(MemCallsite::Scope::kMccl);

  trace->recordAlloc(0x6B01, 4096, MemCallsite::Scope::kCtran);
  const auto ctranAfterAlloc = trace->getStats(MemCallsite::Scope::kCtran);

  trace->recordFree(0x6B01, 4096, MemCallsite::Scope::kCtran);
  const auto ctranAfterFree = trace->getStats(MemCallsite::Scope::kCtran);

  EXPECT_EQ(ctranAfterFree.totalFreed - ctranAfterAlloc.totalFreed, 4096);
  EXPECT_EQ(ctranAfterFree.currentUsage - ctranAfterAlloc.currentUsage, -4096);

  const auto ncclAfter = trace->getStats(MemCallsite::Scope::kNccl);
  const auto mcclAfter = trace->getStats(MemCallsite::Scope::kMccl);
  EXPECT_EQ(ncclAfter.totalFreed, ncclBefore.totalFreed);
  EXPECT_EQ(ncclAfter.currentUsage, ncclBefore.currentUsage);
  EXPECT_EQ(mcclAfter.totalFreed, mcclBefore.totalFreed);
  EXPECT_EQ(mcclAfter.currentUsage, mcclBefore.currentUsage);
}

// dump() surfaces a per-scope breakdown under "byScope" alongside the
// aggregate.
TEST(MemoryTraceTest, DumpContainsByScopeBreakdown) {
  auto trace = MemoryTrace::getOrCreate(0x6003);
  const auto beforeJson = folly::parseJson(trace->dump());
  const int64_t ncclBefore =
      beforeJson["byScope"]["nccl"]["totalAllocated"].asInt();
  const int64_t ctranBefore =
      beforeJson["byScope"]["ctran"]["totalAllocated"].asInt();
  const int64_t mcclBefore =
      beforeJson["byScope"]["mccl"]["totalAllocated"].asInt();
  const int64_t aggBefore = beforeJson["totalAllocated"].asInt();

  trace->recordAlloc(0x6C01, 1000, MemCallsite::Scope::kNccl);
  trace->recordAlloc(0x6C02, 2000, MemCallsite::Scope::kCtran);
  trace->recordAlloc(0x6C03, 4000, MemCallsite::Scope::kMccl);

  const auto afterJson = folly::parseJson(trace->dump());
  ASSERT_TRUE(afterJson.count("byScope"));
  EXPECT_EQ(
      afterJson["byScope"]["nccl"]["totalAllocated"].asInt() - ncclBefore,
      1000);
  EXPECT_EQ(
      afterJson["byScope"]["ctran"]["totalAllocated"].asInt() - ctranBefore,
      2000);
  EXPECT_EQ(
      afterJson["byScope"]["mccl"]["totalAllocated"].asInt() - mcclBefore,
      4000);
  EXPECT_EQ(afterJson["totalAllocated"].asInt() - aggBefore, 7000);
}
