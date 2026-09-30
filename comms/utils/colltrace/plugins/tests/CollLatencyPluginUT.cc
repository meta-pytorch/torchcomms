// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include <folly/dynamic.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/utils/colltrace/CollTraceEvent.h"
#include "comms/utils/colltrace/plugins/CollLatencyPlugin.h"
#include "comms/utils/colltrace/tests/MockTypes.h"

namespace meta::comms::colltrace {
namespace {

using ::comms::CollectiveStat;
using ::comms::CollectiveStatsMap;
using std::chrono::microseconds;
using ::testing::NiceMock;
using ::testing::Return;

const auto kEnqueueTs =
    std::chrono::system_clock::time_point{std::chrono::seconds{100}};

folly::dynamic opMetadata(const char* opName, const char* algoName = "ring") {
  return folly::dynamic::object("opName", opName)("algoName", algoName)(
      "dataType", "commFloat32")("count", 256);
}

CollTraceEvent makeEvent(
    folly::dynamic metadata,
    microseconds queueDelay,
    microseconds duration) {
  auto mock = std::make_unique<NiceMock<MockCollMetadata>>();
  ON_CALL(*mock, toDynamic()).WillByDefault(Return(std::move(metadata)));
  CollTraceEvent event{
      .collRecord = std::make_shared<CollRecord>(0, std::move(mock))};
  auto& timing = event.collRecord->getTimingInfo();
  timing.setCollEnqueueTs(kEnqueueTs);
  timing.setCollStartTs(kEnqueueTs + queueDelay);
  timing.setCollEndTs(kEnqueueTs + queueDelay + duration);
  return event;
}

void endCollectives(
    CollLatencyPlugin& plugin,
    std::vector<CollTraceEvent>& events) {
  for (auto& event : events) {
    ASSERT_TRUE(plugin.afterCollKernelEnd(event).hasValue());
  }
}

CollectiveStat
timing(uint64_t count, uint64_t totalUs, uint64_t minUs, uint64_t maxUs) {
  return CollectiveStat{
      .count = count, .total_us = totalUs, .min_us = minUs, .max_us = maxUs};
}

TEST(CollLatencyPluginTest, BucketsLikeTheGpe) {
  CollLatencyPlugin plugin;
  std::vector<CollTraceEvent> events;
  events.push_back(makeEvent(
      opMetadata("ReduceScatter", "direct"),
      microseconds{10},
      microseconds{900}));
  events.push_back(
      makeEvent(opMetadata("AllReduce"), microseconds{50}, microseconds{100}));
  events.push_back(
      makeEvent(opMetadata("AllReduce"), microseconds{50}, microseconds{300}));
  endCollectives(plugin, events);

  // 256 float32 elements are 1024 bytes.
  const CollectiveStatsMap expected{
      {"allreduce.ring.1024", timing(2, 400, 100, 300)},
      {"allreduce.all", timing(2, 400, 100, 300)},
      {"reducescatter.direct.1024", timing(1, 900, 900, 900)},
      {"reducescatter.all", timing(1, 900, 900, 900)},
      {"all", timing(3, 1300, 100, 900)},
  };
  EXPECT_EQ(plugin.takeCollectiveStats(), expected);
}

TEST(CollLatencyPluginTest, TakeStartsANewWindow) {
  CollLatencyPlugin plugin;
  std::vector<CollTraceEvent> events;
  events.push_back(
      makeEvent(opMetadata("AllReduce"), microseconds{50}, microseconds{200}));
  endCollectives(plugin, events);

  EXPECT_FALSE(plugin.takeCollectiveStats().empty());
  EXPECT_TRUE(plugin.takeCollectiveStats().empty());
}

TEST(CollLatencyPluginTest, FilteredCollectiveIsNotRecorded) {
  CollLatencyPlugin plugin{[](const folly::dynamic& metadata) {
    return metadata["algoName"] != "CtranAllReduceRing";
  }};
  std::vector<CollTraceEvent> events;
  events.push_back(makeEvent(
      opMetadata("AllReduce", "CtranAllReduceRing"),
      microseconds{50},
      microseconds{200}));
  endCollectives(plugin, events);

  EXPECT_TRUE(plugin.takeCollectiveStats().empty());
}

TEST(CollLatencyPluginTest, UnstartedCollectiveIsNotRecorded) {
  CollLatencyPlugin plugin;
  auto event =
      makeEvent(opMetadata("AllReduce"), microseconds{50}, microseconds{200});
  event.collRecord->getTimingInfo().setCollStartTs({});

  ASSERT_TRUE(plugin.afterCollKernelEnd(event).hasValue());

  EXPECT_TRUE(plugin.takeCollectiveStats().empty());
}

TEST(CollLatencyPluginTest, UnknownFieldsFallBack) {
  CollLatencyPlugin plugin;
  std::vector<CollTraceEvent> events;
  events.push_back(makeEvent(
      folly::dynamic::object("algoName", "ring"),
      microseconds{5},
      microseconds{30}));
  events.push_back(
      makeEvent(folly::dynamic{}, microseconds{5}, microseconds{30}));
  endCollectives(plugin, events);

  const CollectiveStatsMap expected{
      {"unknown.ring.0", timing(1, 30, 30, 30)},
      {"unknown.unknown.0", timing(1, 30, 30, 30)},
      {"unknown.all", timing(2, 60, 30, 30)},
      {"all", timing(2, 60, 30, 30)},
  };
  EXPECT_EQ(plugin.takeCollectiveStats(), expected);
}

TEST(CollLatencyPluginTest, EventWithoutRecordIsAnError) {
  CollLatencyPlugin plugin;
  CollTraceEvent event;

  EXPECT_TRUE(plugin.afterCollKernelEnd(event).hasError());
}

} // namespace
} // namespace meta::comms::colltrace
