// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <barrier>
#include <chrono>
#include <cstddef>
#include <memory>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "comms/utils/colltrace/CPUWaitEvent.h"
#include "comms/utils/colltrace/CollTraceEvent.h"
#include "comms/utils/colltrace/plugins/LifecycleEventFeedPlugin.h"
#include "comms/utils/colltrace/tests/MockTypes.h"

namespace meta::comms::colltrace {
namespace {

CollTraceEvent makeEvent(
    uint64_t collId,
    std::optional<uint64_t> replayId = std::nullopt,
    std::optional<uint64_t> capturedCollId = std::nullopt) {
  return CollTraceEvent{
      .collRecord = std::make_shared<CollRecord>(
          collId, std::make_unique<MockCollMetadata>()),
      .replayId = replayId,
      .capturedCollId = capturedCollId,
  };
}

TEST(LifecycleEventFeedPluginTest, DrainsUnreadLifecycleEvents) {
  constexpr uint64_t kCommId = 17;
  constexpr uint64_t kCollId = 23;
  constexpr uint64_t kCapturedCollId = 19;
  constexpr uint64_t kReplayId = 4;
  const auto enqueueTs =
      std::chrono::system_clock::time_point{std::chrono::milliseconds{100}};
  const auto startTs =
      std::chrono::system_clock::time_point{std::chrono::milliseconds{200}};
  const auto endTs =
      std::chrono::system_clock::time_point{std::chrono::milliseconds{300}};
  LifecycleEventFeedPlugin plugin{LifecycleEventFeedConfig{.commId = kCommId}};
  auto event = makeEvent(kCollId, kReplayId, kCapturedCollId);
  event.collRecord->getTimingInfo().setCollEnqueueTs(enqueueTs);
  event.collRecord->getTimingInfo().setCollStartTs(startTs);
  event.collRecord->getTimingInfo().setCollEndTs(endTs);

  EXPECT_TRUE(plugin.afterCollKernelScheduled(event).hasValue());
  EXPECT_TRUE(plugin.afterCollKernelStart(event).hasValue());
  EXPECT_TRUE(plugin.afterCollKernelEnd(event).hasValue());

  const std::vector<LifecycleEventRecord> expected{
      LifecycleEventRecord{
          .replayId = kReplayId,
          .commId = kCommId,
          .collId = kCollId,
          .capturedCollId = kCapturedCollId,
          .eventType = LifecycleEventType::kEnqueue,
          .timestamp = enqueueTs,
      },
      LifecycleEventRecord{
          .replayId = kReplayId,
          .commId = kCommId,
          .collId = kCollId,
          .capturedCollId = kCapturedCollId,
          .eventType = LifecycleEventType::kStart,
          .timestamp = startTs,
      },
      LifecycleEventRecord{
          .replayId = kReplayId,
          .commId = kCommId,
          .collId = kCollId,
          .capturedCollId = kCapturedCollId,
          .eventType = LifecycleEventType::kEnd,
          .timestamp = endTs,
      },
  };
  EXPECT_EQ(plugin.drainUnreadLifecycleEvents(), expected);
  EXPECT_TRUE(plugin.drainUnreadLifecycleEvents().empty());
}

TEST(LifecycleEventFeedPluginTest, ReadsLatestCollectiveIdWhileRecordsArrive) {
  constexpr uint64_t kNumColls = 1024;
  LifecycleEventFeedPlugin plugin;
  std::barrier phase{2};
  bool writerSucceeded = true;

  std::thread writer([&] {
    for (uint64_t collId = 0; collId < kNumColls; ++collId) {
      auto event = makeEvent(collId);
      phase.arrive_and_wait();
      writerSucceeded &= plugin.afterCollRecorded(event).hasValue();
      phase.arrive_and_wait();
    }
  });

  uint64_t previousCollId = 0;
  bool remainedMonotonic = true;
  for (uint64_t i = 0; i < kNumColls; ++i) {
    phase.arrive_and_wait();
    const auto collId = plugin.getLatestLifecycleCollectiveId();
    remainedMonotonic &= collId >= previousCollId;
    previousCollId = collId;
    phase.arrive_and_wait();
  }
  writer.join();

  EXPECT_TRUE(writerSucceeded);
  EXPECT_TRUE(remainedMonotonic);
  EXPECT_EQ(plugin.getLatestLifecycleCollectiveId(), kNumColls);
}

TEST(
    LifecycleEventFeedPluginTest,
    KeepsLargestCollectiveIdWhenRecordsArriveOutOfOrder) {
  constexpr uint64_t kCommId = 17;
  constexpr uint64_t kEarlierCollId = 23;
  constexpr uint64_t kLatestCollId = 29;
  LifecycleEventFeedPlugin plugin{LifecycleEventFeedConfig{.commId = kCommId}};
  auto earlierEvent = makeEvent(kEarlierCollId);
  auto latestEvent = makeEvent(kLatestCollId);

  EXPECT_EQ(plugin.getLatestLifecycleCollectiveId(), 0);
  EXPECT_TRUE(plugin.afterCollRecorded(latestEvent).hasValue());
  EXPECT_TRUE(plugin.afterCollRecorded(earlierEvent).hasValue());

  EXPECT_EQ(plugin.getLatestLifecycleCollectiveId(), kLatestCollId + 1);
}

TEST(LifecycleEventFeedPluginTest, PreservesBurstUntilDrained) {
  constexpr std::size_t kBurstSize = 10'240;
  const auto startTs = std::chrono::system_clock::now();
  LifecycleEventFeedPlugin plugin{LifecycleEventFeedConfig{.commId = 1}};
  auto event = makeEvent(2);
  event.collRecord->getTimingInfo().setCollStartTs(startTs);

  for (std::size_t i = 0; i < kBurstSize; ++i) {
    EXPECT_TRUE(plugin.afterCollKernelStart(event).hasValue());
  }

  const auto expected = LifecycleEventRecord{
      .commId = 1,
      .collId = 2,
      .eventType = LifecycleEventType::kStart,
      .timestamp = startTs,
  };
  EXPECT_EQ(
      plugin.drainUnreadLifecycleEvents(),
      std::vector<LifecycleEventRecord>(kBurstSize, expected));
}

TEST(LifecycleEventFeedPluginTest, UsesCurrentTimeForUnsetTimestamps) {
  LifecycleEventFeedPlugin plugin;
  auto event = makeEvent(2);
  const auto before = std::chrono::system_clock::now();

  EXPECT_TRUE(plugin.afterCollKernelScheduled(event).hasValue());
  EXPECT_TRUE(plugin.afterCollKernelStart(event).hasValue());
  EXPECT_TRUE(plugin.afterCollKernelEnd(event).hasValue());

  const auto after = std::chrono::system_clock::now();
  const auto events = plugin.drainUnreadLifecycleEvents();
  ASSERT_EQ(events.size(), 3);
  for (const auto& recordedEvent : events) {
    EXPECT_GE(recordedEvent.timestamp, before);
    EXPECT_LE(recordedEvent.timestamp, after);
  }
}

TEST(LifecycleEventFeedPluginTest, UsesWaitEventEnqueueTimeForEagerCollective) {
  LifecycleEventFeedPlugin plugin;
  auto event = makeEvent(2);
  auto waitEvent = std::make_unique<CPUWaitEvent>();
  const auto enqueueTs = waitEvent->getCollEnqueueTime().value();
  event.waitEvent = std::move(waitEvent);
  event.collRecord->getTimingInfo().setCollEnqueueTs(
      enqueueTs + std::chrono::hours{1});

  EXPECT_TRUE(plugin.afterCollKernelScheduled(event).hasValue());

  const auto events = plugin.drainUnreadLifecycleEvents();
  ASSERT_EQ(events.size(), 1);
  EXPECT_EQ(events.front().timestamp, enqueueTs);
}

TEST(LifecycleEventFeedPluginTest, SeparatesReplayAndCapturedCollIds) {
  constexpr uint64_t kReplayRecordId = 41;
  constexpr uint64_t kCapturedCollId = 9;
  LifecycleEventFeedPlugin plugin;
  auto event = makeEvent(kReplayRecordId, 2, kCapturedCollId);
  event.collRecord->getTimingInfo().setCollStartTs(
      std::chrono::system_clock::now());

  EXPECT_TRUE(plugin.afterCollKernelStart(event).hasValue());

  const auto events = plugin.drainUnreadLifecycleEvents();
  ASSERT_EQ(events.size(), 1);
  EXPECT_EQ(events.front().collId, kReplayRecordId);
  EXPECT_EQ(events.front().capturedCollId, kCapturedCollId);
  EXPECT_EQ(events.front().replayId, 2);
}

} // namespace
} // namespace meta::comms::colltrace
