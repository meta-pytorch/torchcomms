// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <folly/concurrency/UnboundedQueue.h>

#include "comms/utils/colltrace/CollTracePlugin.h"

namespace meta::comms::logger {
class CommsSpdlogLogger;
}

namespace meta::comms::colltrace {

enum class LifecycleEventType : uint8_t {
  kEnqueue,
  kStart,
  kEnd,
};

struct LifecycleEventRecord {
  std::optional<uint64_t> replayId;
  uint64_t commId{0};
  uint64_t collId{0};
  std::optional<uint64_t> capturedCollId;
  LifecycleEventType eventType{LifecycleEventType::kEnqueue};
  ICollWaitEvent::system_clock_time_point timestamp{};

  bool operator==(const LifecycleEventRecord&) const = default;
};

struct LifecycleEventFeedConfig {
  static constexpr std::size_t kDefaultMaxUnreadEvents{16'384};

  uint64_t commId{0};
  std::string loggerName{"comms"};
  std::size_t maxUnreadEvents{kDefaultMaxUnreadEvents};
};

struct LifecycleEventFeedStats {
  // Sequence counters and depth are independently sampled. Counts are exact
  // once producers and the consumer are quiescent; a live snapshot may span
  // concurrent updates. The reservation high-water mark includes producers
  // that reserved capacity but have not published their queue entry yet.
  uint64_t latestAssignedSequence{0};
  uint64_t highestDrainedSequence{0};
  uint64_t droppedEventCount{0};
  std::optional<uint64_t> lowestDroppedSequence;
  std::optional<uint64_t> highestDroppedSequence;
  std::size_t depth{0};
  std::size_t reservationHighWaterMark{0};
};

uint64_t getNextLifecycleFeedCommId() noexcept;

class LifecycleEventFeedPlugin : public ICollTracePlugin {
 public:
  explicit LifecycleEventFeedPlugin(
      const LifecycleEventFeedConfig& config = {});

  std::string_view getName() const noexcept override;

  CommsMaybeVoid afterCollRecorded(const CollTraceEvent& curEvent) override;
  CommsMaybeVoid beforeCollKernelScheduled(
      const CollTraceEvent& curEvent) override;
  CommsMaybeVoid afterCollKernelScheduled(
      const CollTraceEvent& curEvent) override;
  CommsMaybeVoid afterCollKernelStart(const CollTraceEvent& curEvent) override;
  CommsMaybeVoid collEventProgressing(const CollTraceEvent& curEvent) override;
  CommsMaybeVoid afterCollKernelEnd(const CollTraceEvent& curEvent) override;

  std::vector<LifecycleEventRecord> drainUnreadLifecycleEvents() noexcept;
  LifecycleEventFeedStats getStats() const noexcept;
  uint64_t getLatestLifecycleCollectiveId() const noexcept;
  uint64_t getCommId() const noexcept;
  static constexpr std::string_view kLifecycleEventFeedPluginName =
      "LifecycleEventFeedPlugin";

 private:
  CommsMaybeVoid recordEvent(
      const CollTraceEvent& curEvent,
      LifecycleEventType eventType);

  struct QueuedLifecycleEvent {
    uint64_t sequence{0};
    LifecycleEventRecord record;
  };

  uint64_t commId_{0};
  std::size_t maxUnreadEvents_{0};
  logger::CommsSpdlogLogger* logger_{nullptr};
  folly::UMPMCQueue<QueuedLifecycleEvent, false> unreadEvents_;
  std::atomic<uint64_t> latestCollId_{0};
  std::atomic<uint64_t> nextSequence_{1};
  std::atomic<uint64_t> highestDrainedSequence_{0};
  std::atomic<uint64_t> droppedEventCount_{0};
  std::atomic<uint64_t> lowestDroppedSequence_{
      std::numeric_limits<uint64_t>::max()};
  std::atomic<uint64_t> highestDroppedSequence_{0};
  std::atomic<std::size_t> depth_{0};
  std::atomic<std::size_t> reservationHighWaterMark_{0};
};

} // namespace meta::comms::colltrace
