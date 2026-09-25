// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <atomic>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <folly/concurrency/UnboundedQueue.h>

#include "comms/observatory/colltrace/LifecycleFeedTypes.h"
#include "comms/utils/colltrace/CollTracePlugin.h"

namespace meta::comms::logger {
class CommsSpdlogLogger;
}

namespace meta::comms::colltrace {

struct LifecycleEventFeedConfig {
  uint64_t commId{0};
  std::string loggerName{"comms"};
};

class LifecycleEventFeedPlugin : public ICollTracePlugin {
 public:
  explicit LifecycleEventFeedPlugin(
      const LifecycleEventFeedConfig& config = {});

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
  logger::CommsSpdlogLogger* logger_{nullptr};
  folly::UMPMCQueue<LifecycleEventRecord, false> unreadEvents_;
  std::atomic<uint64_t> latestCollId_{0};
};

} // namespace meta::comms::colltrace
