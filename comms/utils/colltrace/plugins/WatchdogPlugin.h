// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/utils/colltrace/CollTracePlugin.h"

#include <chrono>
#include <optional>
#include <string>
#include <unordered_map>

#include <folly/stop_watch.h>

namespace meta::comms::logger {
class CommsSpdlogLogger;
}

namespace meta::comms::colltrace {

[[noreturn]] void logFatalError(
    CollTraceEvent& curEvent,
    std::string_view errorType,
    std::string_view loggerName = "comms");

struct WatchdogPluginConfig {
  std::string loggerName{"comms"};
  // Empty trigger callbacks are replaced by logger-aware defaults when the
  // WatchdogPlugin is constructed.
  // Async error config
  bool checkAsyncError{true};
  std::function<bool(void)> funcIfError{[]() { return false; }};
  std::function<void(CollTraceEvent&)> funcTriggerOnError;
  /*
   * funcMarkError runs synchronously before a deferred trigger. The default
   * marker is emitted synchronously so Analyzer can snapshot the failure state
   * before bounded history advances and process teardown can discard it.
   *
   * An unset deferErrorTrigger defers the default fatal handler and keeps a
   * custom handler synchronous. An explicit value always wins. The default
   * deferred path writes its diagnostic marker synchronously, then delays only
   * termination so process teardown cannot discard the marker.
   *
   * A custom deferred callback receives a point-in-time record snapshot plus
   * replay, capture, and terminal identifiers. Its waitEvent is intentionally
   * absent because wait events have unique ownership and no cloning contract.
   * Deferred callbacks must own anything they capture.
   */
  std::function<void(CollTraceEvent&)> funcMarkError;
  std::chrono::milliseconds asyncErrorDelay{std::chrono::seconds{60}};
  std::optional<bool> deferErrorTrigger;

  // Timeout config
  bool checkTimeout{false};
  std::chrono::milliseconds timeout{std::chrono::minutes{10}};
  std::function<void(CollTraceEvent&)> funcTriggerOnTimeout;
};

class WatchdogPlugin : public ICollTracePlugin {
 public:
  explicit WatchdogPlugin(WatchdogPluginConfig config);

  std::string_view getName() const noexcept override;

  CommsMaybeVoid beforeCollKernelScheduled(
      CollTraceEvent& curEvent) noexcept override;

  CommsMaybeVoid afterCollKernelScheduled(
      CollTraceEvent& curEvent) noexcept override;

  CommsMaybeVoid afterCollKernelStart(
      CollTraceEvent& curEvent) noexcept override;

  CommsMaybeVoid collEventProgressing(
      CollTraceEvent& curEvent) noexcept override;

  CommsMaybeVoid afterCollKernelEnd(CollTraceEvent& curEvent) noexcept override;

  CommsMaybeVoid afterCollTerminated(
      CollTraceEvent& curEvent,
      CollTraceTerminalReason reason) noexcept override;

  static constexpr std::string_view kWatchdogPluginName = "WatchdogPlugin";

 private:
  const WatchdogPluginConfig config_;
  logger::CommsSpdlogLogger* logger_{nullptr};

  // Per-event timeout tracking. Each in-flight event gets its own timer
  // so a stuck collective is detected even when others progress normally.
  // The startTs is used to detect new replays of graph collectives — when
  // the start timestamp changes, we know a new replay started and reset
  // the timer (even without seeing the previous replay's end event).
  struct EventTimer {
    folly::stop_watch<> timer;
    ICollWaitEvent::system_clock_time_point startTs{};
    bool timeoutTriggered{false};
  };
  std::unordered_map<CollTraceEvent*, EventTimer> eventTimers_;
  bool asyncErrorTriggered_{false};
  bool asyncErrorMarked_{false};

  CommsMaybeVoid dispatchAsyncError(CollTraceEvent& curEvent) noexcept;
};

} // namespace meta::comms::colltrace
