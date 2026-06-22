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
