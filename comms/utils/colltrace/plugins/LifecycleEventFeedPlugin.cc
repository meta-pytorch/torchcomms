// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/colltrace/plugins/LifecycleEventFeedPlugin.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>

#include <folly/Unit.h>

#include "comms/utils/logger/SpdlogLogger.h"

namespace meta::comms::colltrace {

namespace {

constexpr auto kEpoch = ICollWaitEvent::system_clock_time_point{};
} // namespace

uint64_t getNextLifecycleFeedCommId() noexcept {
  static std::atomic<uint64_t> nextCommId{1};
  return nextCommId.fetch_add(1, std::memory_order_relaxed);
}

LifecycleEventFeedPlugin::LifecycleEventFeedPlugin(
    const LifecycleEventFeedConfig& config)
    : commId_(config.commId),
      maxUnreadEvents_(std::max<std::size_t>(config.maxUnreadEvents, 1)),
      logger_(&logger::getSpdlogLogger(config.loggerName)) {
  if (config.maxUnreadEvents == 0) {
    COMMS_LOGGER_STREAM_FIRST_N(*logger_, WARN, 1)
        << "LifecycleEventFeedPlugin requires a nonzero queue capacity; using 1";
  }
}

std::string_view LifecycleEventFeedPlugin::getName() const noexcept {
  return kLifecycleEventFeedPluginName;
}

CommsMaybeVoid LifecycleEventFeedPlugin::afterCollRecorded(
    const CollTraceEvent& curEvent) {
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
    const CollTraceEvent&) {
  return folly::unit;
}

CommsMaybeVoid LifecycleEventFeedPlugin::afterCollKernelScheduled(
    const CollTraceEvent& curEvent) {
  return recordEvent(curEvent, LifecycleEventType::kEnqueue);
}

CommsMaybeVoid LifecycleEventFeedPlugin::afterCollKernelStart(
    const CollTraceEvent& curEvent) {
  return recordEvent(curEvent, LifecycleEventType::kStart);
}

CommsMaybeVoid LifecycleEventFeedPlugin::collEventProgressing(
    const CollTraceEvent&) {
  return folly::unit;
}

CommsMaybeVoid LifecycleEventFeedPlugin::afterCollKernelEnd(
    const CollTraceEvent& curEvent) {
  return recordEvent(curEvent, LifecycleEventType::kEnd);
}

CommsMaybeVoid LifecycleEventFeedPlugin::recordEvent(
    const CollTraceEvent& curEvent,
    LifecycleEventType eventType) {
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

  const auto sequence = nextSequence_.fetch_add(1, std::memory_order_relaxed);
  const auto reservedDepth = depth_.fetch_add(1, std::memory_order_relaxed) + 1;
  if (reservedDepth > maxUnreadEvents_) {
    depth_.fetch_sub(1, std::memory_order_relaxed);
    const auto dropped =
        droppedEventCount_.fetch_add(1, std::memory_order_relaxed) + 1;
    auto lowestDropped = lowestDroppedSequence_.load(std::memory_order_relaxed);
    while (lowestDropped > sequence &&
           !lowestDroppedSequence_.compare_exchange_weak(
               lowestDropped, sequence, std::memory_order_relaxed)) {
    }
    auto highestDropped =
        highestDroppedSequence_.load(std::memory_order_relaxed);
    while (highestDropped < sequence &&
           !highestDroppedSequence_.compare_exchange_weak(
               highestDropped, sequence, std::memory_order_relaxed)) {
    }
    COMMS_LOGGER_STREAM_EVERY_MS(*logger_, WARN, 60000)
        << "LifecycleEventFeedPlugin dropped event sequence " << sequence
        << " for comm " << commId_ << " because its " << maxUnreadEvents_
        << "-event queue is full; cumulative dropped events=" << dropped;
    return folly::unit;
  }

  auto queuedEvent = QueuedLifecycleEvent{
      .sequence = sequence,
      .record =
          LifecycleEventRecord{
              .replayId = curEvent.replayId,
              .commId = commId_,
              .collId = curEvent.collRecord->getCollId(),
              .capturedCollId = curEvent.capturedCollId,
              .eventType = eventType,
              .timestamp = timestamp,
          },
  };
  unreadEvents_.enqueue(std::move(queuedEvent));

  const auto retainedDepth = std::min(reservedDepth, maxUnreadEvents_);
  auto reservationHighWaterMark =
      reservationHighWaterMark_.load(std::memory_order_relaxed);
  while (
      reservationHighWaterMark < retainedDepth &&
      !reservationHighWaterMark_.compare_exchange_weak(
          reservationHighWaterMark, retainedDepth, std::memory_order_relaxed)) {
  }
  return folly::unit;
}

std::vector<LifecycleEventRecord>
LifecycleEventFeedPlugin::drainUnreadLifecycleEvents() noexcept {
  std::vector<LifecycleEventRecord> events;
  QueuedLifecycleEvent event;
  while (unreadEvents_.try_dequeue(event)) {
    depth_.fetch_sub(1, std::memory_order_relaxed);
    auto highestDrained =
        highestDrainedSequence_.load(std::memory_order_relaxed);
    while (highestDrained < event.sequence &&
           !highestDrainedSequence_.compare_exchange_weak(
               highestDrained, event.sequence, std::memory_order_relaxed)) {
    }
    events.push_back(std::move(event.record));
  }
  return events;
}

LifecycleEventFeedStats LifecycleEventFeedPlugin::getStats() const noexcept {
  const auto lowestDropped =
      lowestDroppedSequence_.load(std::memory_order_relaxed);
  const auto highestDropped =
      highestDroppedSequence_.load(std::memory_order_relaxed);
  return LifecycleEventFeedStats{
      .latestAssignedSequence =
          nextSequence_.load(std::memory_order_relaxed) - 1,
      .highestDrainedSequence =
          highestDrainedSequence_.load(std::memory_order_relaxed),
      .droppedEventCount = droppedEventCount_.load(std::memory_order_relaxed),
      .lowestDroppedSequence =
          lowestDropped == std::numeric_limits<uint64_t>::max()
          ? std::nullopt
          : std::optional<uint64_t>{lowestDropped},
      .highestDroppedSequence = highestDropped == 0
          ? std::nullopt
          : std::optional<uint64_t>{highestDropped},
      .depth = depth_.load(std::memory_order_relaxed),
      .reservationHighWaterMark =
          reservationHighWaterMark_.load(std::memory_order_relaxed),
  };
}

void LifecycleEventFeedPlugin::collectStats(CollTraceStats& stats) const {
  const auto pluginStats = getStats();
  stats.capabilities.lifecycleSubscriberAttached = true;
  stats.lifecycle = CollTraceLifecycleStats{
      .latestAssignedSequence = pluginStats.latestAssignedSequence,
      .highestDrainedSequence = pluginStats.highestDrainedSequence,
      .droppedEventCount = pluginStats.droppedEventCount,
      .lowestDroppedSequence = pluginStats.lowestDroppedSequence,
      .highestDroppedSequence = pluginStats.highestDroppedSequence,
      .depth = pluginStats.depth,
      .highWaterMark = pluginStats.reservationHighWaterMark,
  };
}

uint64_t LifecycleEventFeedPlugin::getLatestLifecycleCollectiveId()
    const noexcept {
  return latestCollId_.load(std::memory_order_relaxed);
}

uint64_t LifecycleEventFeedPlugin::getCommId() const noexcept {
  return commId_;
}

} // namespace meta::comms::colltrace
