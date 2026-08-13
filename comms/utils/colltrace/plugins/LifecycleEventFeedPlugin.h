// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <atomic>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include <folly/concurrency/UnboundedQueue.h>

#include "comms/utils/colltrace/CollTracePlugin.h"

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
  uint64_t commId{0};
};

class LifecycleEventFeedPlugin : public ICollTracePlugin {
 public:
  explicit LifecycleEventFeedPlugin(LifecycleEventFeedConfig config = {});

  std::string_view getName() const noexcept override;

  CommsMaybeVoid afterCollRecorded(CollTraceEvent& curEvent) noexcept override;
  CommsMaybeVoid beforeCollKernelScheduled(
      CollTraceEvent& curEvent) noexcept override;
  CommsMaybeVoid afterCollKernelScheduled(
      CollTraceEvent& curEvent) noexcept override;
  CommsMaybeVoid afterCollKernelStart(
      CollTraceEvent& curEvent) noexcept override;
  CommsMaybeVoid collEventProgressing(
      CollTraceEvent& curEvent) noexcept override;
  CommsMaybeVoid afterCollKernelEnd(CollTraceEvent& curEvent) noexcept override;

  std::vector<LifecycleEventRecord> drainUnreadLifecycleEvents() noexcept;
  uint64_t getLatestLifecycleCollectiveId() const noexcept;
  uint64_t getCommId() const noexcept;
  static constexpr std::string_view kLifecycleEventFeedPluginName =
      "LifecycleEventFeedPlugin";

 private:
  CommsMaybeVoid recordEvent(
      CollTraceEvent& curEvent,
      LifecycleEventType eventType) noexcept;

  uint64_t commId_{0};
  folly::UMPMCQueue<LifecycleEventRecord, false> unreadEvents_;
  std::atomic<uint64_t> latestCollId_{0};
};

} // namespace meta::comms::colltrace
