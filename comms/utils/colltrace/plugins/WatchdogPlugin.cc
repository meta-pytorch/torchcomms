// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/colltrace/plugins/WatchdogPlugin.h"

#include <atomic>
#include <string>

#include <folly/Indestructible.h>
#include <folly/Unit.h>
#include <folly/executors/FunctionScheduler.h>
#include <folly/json.h>

#include "comms/utils/logger/SpdlogLogger.h"

namespace meta::comms::colltrace {

namespace {
struct AsyncErrorScheduler {
  AsyncErrorScheduler() {
    scheduler.setThreadName("CollTraceWatchdog");
    scheduler.start();
  }

  folly::FunctionScheduler scheduler;
  std::atomic<uint64_t> nextTaskId{0};
};

AsyncErrorScheduler& getAsyncErrorScheduler() {
  static folly::Indestructible<AsyncErrorScheduler> scheduler;
  return *scheduler;
}

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
          logFatalError(event, "AsyncError", loggerName);
        };
    config.deferErrorTrigger = true;
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

CommsMaybeVoid WatchdogPlugin::dispatchAsyncError(
    CollTraceEvent& curEvent) noexcept {
  try {
    auto event = std::make_shared<CollTraceEvent>();
    event->collRecord = curEvent.collRecord;
    event->replayId = curEvent.replayId;
    event->capturedCollId = curEvent.capturedCollId;

    auto callback = config_.funcTriggerOnError;
    const auto delay = config_.asyncErrorDelay;
    auto loggerName = config_.loggerName;
    auto& scheduler = getAsyncErrorScheduler();
    const auto taskName = fmt::format(
        "colltrace_async_error_{}", scheduler.nextTaskId.fetch_add(1));
    scheduler.scheduler.addFunctionOnce(
        [callback = std::move(callback),
         event = std::move(event),
         loggerName = std::move(loggerName)]() mutable {
          try {
            callback(*event);
          } catch (const std::exception& ex) {
            COMMS_LOG_NAMED(
                loggerName,
                ERR,
                "Watchdog async-error callback threw an exception: {}",
                ex.what());
          } catch (...) {
            COMMS_LOG_NAMED(
                loggerName,
                ERR,
                "Watchdog async-error callback threw an unknown exception");
          }
        },
        taskName,
        std::chrono::duration_cast<std::chrono::microseconds>(delay));
  } catch (const std::exception& ex) {
    return folly::makeUnexpected(CommsError(
        fmt::format(
            "Failed to dispatch watchdog async-error callback: {}", ex.what()),
        commInternalError));
  }
  return folly::unit;
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

  if (config_.checkAsyncError && !asyncErrorTriggered_ &&
      config_.funcIfError()) {
    COMMS_LOGGER_STREAM(*logger_, DBG)
        << "WatchdogPlugin::collEventProgressing: triggering async error handling";

    asyncErrorTriggered_ = true;
    if (config_.deferErrorTrigger) {
      auto dispatchResult = dispatchAsyncError(curEvent);
      if (dispatchResult.hasError()) {
        asyncErrorTriggered_ = false;
        return dispatchResult;
      }
    } else {
      config_.funcTriggerOnError(curEvent);
    }
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
      it->second.timeoutTriggered = false;
    } else if (
        !it->second.timeoutTriggered &&
        it->second.timer.elapsed(config_.timeout)) {
      it->second.timeoutTriggered = true;
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

CommsMaybeVoid WatchdogPlugin::afterCollTerminated(
    CollTraceEvent& curEvent,
    CollTraceTerminalReason reason) noexcept {
  /*
   * These reasons are delivered by the producer thread before the event can
   * enter the poll thread, so they cannot have an associated watchdog timer.
   * Avoid touching the poll-thread-owned timer map from that thread.
   */
  if (reason == CollTraceTerminalReason::QueueRejected ||
      reason == CollTraceTerminalReason::SupersededBeforeSchedule) {
    return folly::unit;
  }
  eventTimers_.erase(&curEvent);
  return folly::unit;
}

} // namespace meta::comms::colltrace
