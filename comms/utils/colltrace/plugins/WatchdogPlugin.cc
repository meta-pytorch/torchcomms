// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/colltrace/plugins/WatchdogPlugin.h"

#include <string>
#include <thread>

#include <folly/Unit.h>
#include <folly/json.h>

#include "comms/utils/logger/SpdlogLogger.h"

namespace meta::comms::colltrace {

namespace {
std::string_view getCollectiveStateStr(CollTraceEvent& curEvent) {
  auto& timingInfo = curEvent.collRecord->getTimingInfo();
  // This should not happen for collectives with async error/timeout
  if (timingInfo.getCollEndTs().time_since_epoch().count() != 0) {
    return "Finished";
  }
  if (timingInfo.getCollStartTs().time_since_epoch().count() != 0) {
    return "Kernel Running";
  }
  if (timingInfo.getCollEnqueueTs().time_since_epoch().count() != 0) {
    return "Kernel Not Started";
  }
  // This should not happen... Just for completeness
  return "Not Scheduled";
}

WatchdogPluginConfig normalizeWatchdogConfig(WatchdogPluginConfig config) {
  if (!config.funcTriggerOnError) {
    config.funcTriggerOnError =
        [loggerName = std::string{config.loggerName}](CollTraceEvent& event) {
          /* Give Analyzer time to collect the error state before aborting. */
          // NOLINTNEXTLINE(facebook-hte-BadCall-sleep_for)
          std::this_thread::sleep_for(std::chrono::seconds(60));
          logFatalError(event, "AsyncError", loggerName);
        };
  }
  if (!config.funcTriggerOnTimeout) {
    config.funcTriggerOnTimeout =
        [loggerName = std::string{config.loggerName}](CollTraceEvent& event) {
          logFatalError(event, "watchdog timeout", loggerName);
        };
  }
  return config;
}
} // namespace

[[noreturn]] void logFatalError(
    CollTraceEvent& curEvent,
    std::string_view errorType,
    std::string_view loggerName) {
  const auto metadataDynamic = curEvent.collRecord->toDynamic();
  /*
   * Watchdog diagnostics consume this marker from NCCL_DEBUG_FILE. Keep it in
   * the payload because the owner logger's prefix is backend-specific.
   */
  const auto errorString = fmt::format(
      "COMM FATAL: FatalError: Collective (OpCount={}, OpType={}, Count={}, DataType={} CurrentState={}) for Comm {} raised {}",
      metadataDynamic.getDefault("opCount", "Unknown").asString(),
      metadataDynamic.getDefault("opName", "Unknown").asString(),
      metadataDynamic.getDefault("count", "N/A").asString(),
      metadataDynamic.getDefault("dataType", "Unknown").asString(),
      getCollectiveStateStr(curEvent),
      metadataDynamic.getDefault("commDesc", "Unknown").asString(),
      errorType);
  COMMS_LOG_NAMED(loggerName, FATAL, "{}", errorString);
}

WatchdogPlugin::WatchdogPlugin(WatchdogPluginConfig config)
    : config_(normalizeWatchdogConfig(std::move(config))),
      logger_(&logger::getSpdlogLogger(config_.loggerName)) {}

std::string_view WatchdogPlugin::getName() const noexcept {
  return kWatchdogPluginName;
}

CommsMaybeVoid WatchdogPlugin::beforeCollKernelScheduled(
    CollTraceEvent&) noexcept {
  return folly::unit;
}

CommsMaybeVoid WatchdogPlugin::afterCollKernelScheduled(
    CollTraceEvent&) noexcept {
  return folly::unit;
}

CommsMaybeVoid WatchdogPlugin::afterCollKernelStart(CollTraceEvent&) noexcept {
  return folly::unit;
}

CommsMaybeVoid WatchdogPlugin::collEventProgressing(
    CollTraceEvent& curEvent) noexcept {
  COMMS_LOG_IMPL(
      *logger_,
      ::spdlog::level::debug,
      COMMS_LOGGER_DEBUG,
      "WatchdogPlugin::collEventProgressing for CollTraceEvent {}",
      folly::toJson(curEvent.collRecord->toDynamic()));

  if (config_.checkAsyncError && config_.funcIfError()) {
    COMMS_LOGGER_STREAM(*logger_, DBG)
        << "WatchdogPlugin::collEventProgressing: triggering async error handling";

    config_.funcTriggerOnError(curEvent);
  }
  // Per-event timeout: each in-flight event gets its own timer so a stuck
  // collective is detected even when others are progressing normally.
  // If the start timestamp changed, a new replay started — reset the timer
  // so we don't falsely timeout on a fresh replay after data loss.
  if (config_.checkTimeout) {
    auto currentStartTs = curEvent.collRecord->getTimingInfo().getCollStartTs();
    auto [it, inserted] = eventTimers_.try_emplace(&curEvent);
    if (inserted || it->second.startTs != currentStartTs) {
      it->second.timer.reset();
      it->second.startTs = currentStartTs;
    } else if (it->second.timer.elapsed(config_.timeout)) {
      config_.funcTriggerOnTimeout(curEvent);
    }
  }
  return folly::unit;
}

CommsMaybeVoid WatchdogPlugin::afterCollKernelEnd(
    CollTraceEvent& curEvent) noexcept {
  eventTimers_.erase(&curEvent);
  return folly::unit;
}

} // namespace meta::comms::colltrace
