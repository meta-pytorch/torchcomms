// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/colltrace/plugins/WatchdogPlugin.h"

#include <folly/Unit.h>
#include <folly/json.h>
#include <folly/logging/xlog.h>

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
} // namespace

void logFatalError(CollTraceEvent& curEvent, std::string_view errorType) {
  auto metadataDynamic = curEvent.collRecord->toDynamic();
  auto errorString = fmt::format(
      "FatalError: Collective (OpCount={}, OpType={}, Count={}, DataType={} CurrentState={}) for Comm {} raised {}",
      metadataDynamic.getDefault("opCount", "Unknown").asString(),
      metadataDynamic.getDefault("opName", "Unknown").asString(),
      metadataDynamic.getDefault("count", "N/A").asString(),
      metadataDynamic.getDefault("dataType", "Unknown").asString(),
      getCollectiveStateStr(curEvent),
      metadataDynamic.getDefault("commDesc", "Unknown").asString(),
      errorType);
  XLOG(FATAL, errorString);
}

WatchdogPlugin::WatchdogPlugin(WatchdogPluginConfig config)
    : config_(std::move(config)) {}

std::string_view WatchdogPlugin::getName() const noexcept {
  return kWatchdogPluginName;
}

CommsMaybeVoid WatchdogPlugin::beforeCollKernelScheduled(
    CollTraceEvent& curEvent) noexcept {
  return folly::unit;
}

CommsMaybeVoid WatchdogPlugin::afterCollKernelScheduled(
    CollTraceEvent& curEvent) noexcept {
  return folly::unit;
}

CommsMaybeVoid WatchdogPlugin::afterCollKernelStart(
    CollTraceEvent& curEvent) noexcept {
  return folly::unit;
}

CommsMaybeVoid WatchdogPlugin::collEventProgressing(
    CollTraceEvent& curEvent) noexcept {
  XLOGF(
      DBG0,
      "WatchdogPlugin::collEventProgressing for CollTraceEvent {}",
      folly::toJson(curEvent.collRecord->toDynamic()));

  if (config_.checkAsyncError && config_.funcIfError()) {
    XLOG(DBG)
        << "WatchdogPlugin::collEventProgressing: triggering async error handling";

    config_.funcTriggerOnError(curEvent);
  }
  if (config_.checkTimeout) {
    if (&curEvent == lastEvent_ && timer_.elapsed(config_.timeout)) {
      config_.funcTriggerOnTimeout(curEvent);
    } else if (&curEvent != lastEvent_) {
      lastEvent_ = &curEvent;
      timer_.reset();
    }
  }
  return folly::unit;
}

CommsMaybeVoid WatchdogPlugin::afterCollKernelEnd(
    CollTraceEvent& curEvent) noexcept {
  return folly::unit;
}

} // namespace meta::comms::colltrace
