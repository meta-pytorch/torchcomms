// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/colltrace/plugins/LifecycleEventFeedPlugin.h"

#include <chrono>
#include <cstddef>
#include <cstdint>

#include <folly/Unit.h>

#include "comms/utils/logger/SpdlogLogger.h"

namespace meta::comms::colltrace {

namespace {

constexpr auto kEpoch = ICollWaitEvent::system_clock_time_point{};
constexpr std::size_t kBacklogWarningThreshold = std::size_t{1} << 13;
constexpr uint32_t kBacklogCheckPeriod = 256;

} // namespace

uint64_t getNextLifecycleFeedCommId() noexcept {
  static std::atomic<uint64_t> nextCommId{1};
  return nextCommId.fetch_add(1, std::memory_order_relaxed);
}

LifecycleEventFeedPlugin::LifecycleEventFeedPlugin(
    const LifecycleEventFeedConfig& config)
    : commId_(config.commId),
      logger_(&logger::getSpdlogLogger(config.loggerName)),
      maxUnreadEvents_(config.maxUnreadEvents) {}

std::string_view LifecycleEventFeedPlugin::getName() const noexcept {
  return kLifecycleEventFeedPluginName;
}

CommsMaybeVoid LifecycleEventFeedPlugin::afterCollRecorded(
    CollTraceEvent& curEvent) noexcept {
  if (curEvent.collRecord == nullptr) {
    return folly::makeUnexpected(CommsError(
        "LifecycleEventFeedPlugin received an event without a collective record",
        commInternalError));
  }
  const auto collId =
      curEvent.capturedCollId.value_or(curEvent.collRecord->getCollId()) + 1;
  auto latestCollId = latestCollId_.load(std::memory_order_relaxed);
  while (latestCollId < collId &&
         !latestCollId_.compare_exchange_weak(
             latestCollId,
             collId,
             std::memory_order_relaxed,
             std::memory_order_relaxed)) {
  }
  return folly::unit;
}

CommsMaybeVoid LifecycleEventFeedPlugin::beforeCollKernelScheduled(
    CollTraceEvent&) noexcept {
  return folly::unit;
}

CommsMaybeVoid LifecycleEventFeedPlugin::afterCollKernelScheduled(
    CollTraceEvent& curEvent) noexcept {
  return recordEvent(curEvent, LifecycleEventType::kEnqueue);
}

CommsMaybeVoid LifecycleEventFeedPlugin::afterCollKernelStart(
    CollTraceEvent& curEvent) noexcept {
  return recordEvent(curEvent, LifecycleEventType::kStart);
}

CommsMaybeVoid LifecycleEventFeedPlugin::collEventProgressing(
    CollTraceEvent&) noexcept {
  return folly::unit;
}

CommsMaybeVoid LifecycleEventFeedPlugin::afterCollKernelEnd(
    CollTraceEvent& curEvent) noexcept {
  return recordEvent(curEvent, LifecycleEventType::kEnd);
}

CommsMaybeVoid LifecycleEventFeedPlugin::recordEvent(
    CollTraceEvent& curEvent,
    LifecycleEventType eventType) noexcept {
  if (curEvent.collRecord == nullptr) {
    return folly::makeUnexpected(CommsError(
        "LifecycleEventFeedPlugin received an event without a collective record",
        commInternalError));
  }

  auto timestamp = kEpoch;
  const auto& timingInfo = curEvent.collRecord->getTimingInfo();
  switch (eventType) {
    case LifecycleEventType::kEnqueue:
      if (curEvent.waitEvent != nullptr) {
        auto enqueueTime = curEvent.waitEvent->getCollEnqueueTime();
        if (enqueueTime.hasValue()) {
          timestamp = enqueueTime.value();
          break;
        }
      }
      timestamp = timingInfo.getCollEnqueueTs();
      break;
    case LifecycleEventType::kStart:
      timestamp = timingInfo.getCollStartTs();
      break;
    case LifecycleEventType::kEnd:
      timestamp = timingInfo.getCollEndTs();
      break;
  }
  if (timestamp == kEpoch) {
    timestamp = std::chrono::system_clock::now();
  }

  auto record = LifecycleEventRecord{
      .replayId = curEvent.replayId,
      .commId = commId_,
      .collId = curEvent.collRecord->getCollId(),
      .capturedCollId = curEvent.capturedCollId,
      .eventType = eventType,
      .timestamp = timestamp,
  };
  unreadEvents_.enqueue(std::move(record));
  discardOldestBeyondCap();
  static thread_local uint32_t backlogCheckCounter = 0;
  if (++backlogCheckCounter % kBacklogCheckPeriod == 0) {
    const auto backlog = unreadEvents_.size();
    const auto dropped = droppedEvents_.load(std::memory_order_relaxed);
    if (dropped != 0) [[unlikely]] {
      COMMS_LOGGER_STREAM_EVERY_MS(*logger_, WARN, 60000)
          << "LifecycleEventFeedPlugin has discarded " << dropped
          << " events for comm " << commId_
          << " to stay within its cap; the consumer is not draining and what "
             "it eventually reads will have gaps";
    } else if (backlog >= kBacklogWarningThreshold) [[unlikely]] {
      COMMS_LOGGER_STREAM_EVERY_MS(*logger_, WARN, 60000)
          << "LifecycleEventFeedPlugin estimated unread backlog is " << backlog
          << " events for comm " << commId_
          << "; consumer may be stalled and events will start being discarded";
    }
  }
  return folly::unit;
}

void LifecycleEventFeedPlugin::discardOldestBeyondCap() noexcept {
  if (maxUnreadEvents_ == 0) {
    return;
  }
  // Oldest first: a consumer that starts reading late wants recent history.
  // Dequeueing from a producer is safe because the queue is MPMC.
  uint64_t discarded = 0;
  LifecycleEventRecord dropped;
  while (unreadEvents_.size() > maxUnreadEvents_ &&
         unreadEvents_.try_dequeue(dropped)) {
    ++discarded;
  }
  if (discarded != 0) {
    droppedEvents_.fetch_add(discarded, std::memory_order_relaxed);
  }
}

std::vector<LifecycleEventRecord>
LifecycleEventFeedPlugin::drainUnreadLifecycleEvents() noexcept {
  std::vector<LifecycleEventRecord> events;
  LifecycleEventRecord event;
  while (unreadEvents_.try_dequeue(event)) {
    events.push_back(std::move(event));
  }
  return events;
}

uint64_t LifecycleEventFeedPlugin::getLatestLifecycleCollectiveId()
    const noexcept {
  return latestCollId_.load(std::memory_order_relaxed);
}

uint64_t LifecycleEventFeedPlugin::getCommId() const noexcept {
  return commId_;
}

uint64_t LifecycleEventFeedPlugin::getDroppedLifecycleEventCount()
    const noexcept {
  return droppedEvents_.load(std::memory_order_relaxed);
}

} // namespace meta::comms::colltrace
