// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/utils/colltrace/CollTracePlugin.h"

#include <thread>
#include <unordered_map>

#include <folly/stop_watch.h>

namespace meta::comms::colltrace {

void logFatalError(CollTraceEvent& curEvent, std::string_view errorType);

struct WatchdogPluginConfig {
  // Async error config
  bool checkAsyncError{true};
  std::function<bool(void)> funcIfError{[]() { return false; }};
  std::function<void(CollTraceEvent&)> funcTriggerOnError{
      [](CollTraceEvent& event) {
        // Sleep for 60 seconds to allow Analyzer to collect the error.
        std::this_thread::sleep_for(std::chrono::seconds(60));
        logFatalError(event, "AsyncError");
      }};

  // Timeout config
  bool checkTimeout{false};
  std::chrono::milliseconds timeout{std::chrono::minutes{10}};
  std::function<void(CollTraceEvent&)> funcTriggerOnTimeout{
      [](CollTraceEvent& event) { logFatalError(event, "watchdog timeout"); }};
};

class WatchdogPlugin : public ICollTracePlugin {
 public:
  WatchdogPlugin(WatchdogPluginConfig config);

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

  static constexpr std::string_view kWatchdogPluginName = "WatchdogPlugin";

 private:
  const WatchdogPluginConfig config_;

  // Per-event timeout tracking. Each in-flight event gets its own timer
  // so a stuck collective is detected even when others progress normally.
  // The startTs is used to detect new replays of graph collectives — when
  // the start timestamp changes, we know a new replay started and reset
  // the timer (even without seeing the previous replay's end event).
  struct EventTimer {
    folly::stop_watch<> timer;
    ICollWaitEvent::system_clock_time_point startTs{};
  };
  std::unordered_map<CollTraceEvent*, EventTimer> eventTimers_;
};

} // namespace meta::comms::colltrace
