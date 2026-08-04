// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/logger/EventMgr.h"

#include <cstdint>
#include <optional>
#include <string>

#include <folly/json/dynamic.h>
#include <folly/json/json.h>
#include <folly/portability/GTest.h>

#include "comms/utils/commSpecs.h"

namespace {

constexpr uint64_t kCommHash = 0xabc123;
constexpr const char* kCommDesc = "test_pg:0";
constexpr int kRank = 0;
constexpr int kNRanks = 8;
constexpr uintptr_t kMemoryAddr = 0xdeadbeef;
constexpr int64_t kBytes = 1024;
const std::string kMemType = "cudaMalloc";

CommLogData makeCommLogData() {
  return CommLogData{/*commId=*/0, kCommHash, kCommDesc, kRank, kNRanks};
}

MemoryEvent makeMemoryEvent(std::optional<std::string> memType) {
  return MemoryEvent(
      makeCommLogData(),
      /*callsite=*/"testCallsite",
      /*use=*/"testUse",
      kMemoryAddr,
      /*bytes=*/kBytes,
      /*numSegments=*/std::nullopt,
      /*durationUs=*/std::nullopt,
      std::move(memType));
}

MemoryEvent makeMemoryEventWithScope(
    meta::comms::memtrace::MemCallsite::Scope source) {
  return MemoryEvent(
      makeCommLogData(),
      /*callsite=*/"testCallsite",
      /*use=*/"testUse",
      kMemoryAddr,
      /*bytes=*/kBytes,
      /*numSegments=*/std::nullopt,
      /*durationUs=*/std::nullopt,
      /*memType=*/std::nullopt,
      /*isRegMemEvent=*/false,
      source);
}

} // namespace

// When a memType is supplied, toSample() emits a "memType" normal column
// carrying that exact value.
TEST(MemoryEventTest, ToSampleEmitsMemTypeWhenSet) {
  auto event = makeMemoryEvent(kMemType);

  auto sample = event.toSample();
  const auto json = folly::parseJson(sample.toJson());

  EXPECT_EQ(json["normal"]["memType"].asString(), kMemType);
}

// When memType is nullopt, the "memType" column must be absent entirely.
TEST(MemoryEventTest, ToSampleOmitsMemTypeWhenNullopt) {
  auto event = makeMemoryEvent(std::nullopt);

  auto sample = event.toSample();
  const auto json = folly::parseJson(sample.toJson());

  EXPECT_EQ(json["normal"].count("memType"), 0);
}

// A ctran allocation emits scope="ctran" so the nccl_memory_logging table can
// group ctran memory separately from baseline NCCL.
TEST(MemoryEventTest, ToSampleEmitsCtranScope) {
  auto event = makeMemoryEventWithScope(
      meta::comms::memtrace::MemCallsite::Scope::kCtran);

  auto sample = event.toSample();
  const auto json = folly::parseJson(sample.toJson());

  EXPECT_EQ(json["normal"]["callsite_scope"].asString(), "ctran");
}

// The default scope (baseline NCCL) emits scope="nccl".
TEST(MemoryEventTest, ToSampleEmitsNcclScopeByDefault) {
  auto event = makeMemoryEventWithScope(
      meta::comms::memtrace::MemCallsite::Scope::kNccl);

  auto sample = event.toSample();
  const auto json = folly::parseJson(sample.toJson());

  EXPECT_EQ(json["normal"]["callsite_scope"].asString(), "nccl");
}

// An mccl allocation emits scope="mccl" so the nccl_memory_logging table can
// group mccl memory separately from baseline NCCL and ctran.
TEST(MemoryEventTest, ToSampleEmitsMcclScope) {
  auto event = makeMemoryEventWithScope(
      meta::comms::memtrace::MemCallsite::Scope::kMccl);

  auto sample = event.toSample();
  const auto json = folly::parseJson(sample.toJson());

  EXPECT_EQ(json["normal"]["callsite_scope"].asString(), "mccl");
}

// The scope carried by a ctran MemCallsite surfaces as scope="ctran", while a
// bare-string (baseline) callsite surfaces as scope="nccl". Ties the
// MemCallsite scope to the scuba column memtrace populates.
TEST(MemoryEventTest, MemCallsiteScopeSurfacesInColumn) {
  const meta::comms::memtrace::MemCallsite ctranCallsite(
      meta::comms::memtrace::MemCallsite::Scope::kCtran, "initTmpBufs");
  auto ctranEvent = makeMemoryEventWithScope(ctranCallsite.scope);
  const auto ctranJson = folly::parseJson(ctranEvent.toSample().toJson());
  EXPECT_EQ(ctranJson["normal"]["callsite_scope"].asString(), "ctran");

  const meta::comms::memtrace::MemCallsite baselineCallsite(
      "initChannelDevRingUserRanks");
  auto ncclEvent = makeMemoryEventWithScope(baselineCallsite.scope);
  const auto ncclJson = folly::parseJson(ncclEvent.toSample().toJson());
  EXPECT_EQ(ncclJson["normal"]["callsite_scope"].asString(), "nccl");
}
