// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <atomic>
#include <cstdint>
#include <optional>
#include <string_view>

#include <folly/dynamic.h>

#include "comms/utils/colltrace/CollRecord.h"
#include "comms/utils/colltrace/CollWaitEvent.h"

namespace meta::comms::colltrace {

enum class CollTraceTerminalReason : uint8_t {
  QueueRejected,
  SupersededBeforeSchedule,
  TrackingOverflow,
  GraphDestroyed,
  TraceDestroyed,
  PluginContention,
  Count,
};

constexpr std::string_view collTraceTerminalReasonToString(
    CollTraceTerminalReason reason) noexcept {
  switch (reason) {
    case CollTraceTerminalReason::QueueRejected:
      return "queue_rejected";
    case CollTraceTerminalReason::SupersededBeforeSchedule:
      return "superseded_before_schedule";
    case CollTraceTerminalReason::TrackingOverflow:
      return "tracking_overflow";
    case CollTraceTerminalReason::GraphDestroyed:
      return "graph_destroyed";
    case CollTraceTerminalReason::TraceDestroyed:
      return "trace_destroyed";
    case CollTraceTerminalReason::PluginContention:
      return "plugin_contention";
    case CollTraceTerminalReason::Count:
      break;
  }
  return "unknown";
}

// Should be accessible by CollTrace and all the plugin callbacks
struct CollTraceEvent {
  // Other plugin might also want to keep a pointer to the event, so we use
  // shared pointer here
  std::shared_ptr<CollRecord> collRecord;
  std::unique_ptr<ICollWaitEvent> waitEvent; // How we wait the event
  std::optional<uint64_t> replayId;
  // Graph replay CollRecords get new IDs; this preserves the capture-time ID.
  std::optional<uint64_t> capturedCollId;
  std::optional<CollTraceTerminalReason> terminalReason;
};

} // namespace meta::comms::colltrace
