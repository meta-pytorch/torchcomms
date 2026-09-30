// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/colltrace/plugins/CollLatencyPlugin.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <utility>

#include <fmt/format.h>
#include <folly/String.h>
#include <folly/Unit.h>
#include <folly/dynamic.h>

#include "comms/utils/Conversion.h"
#include "comms/utils/commSpecs.h"

namespace meta::comms::colltrace {

namespace {

constexpr auto kUnset = ICollWaitEvent::system_clock_time_point{};

std::string stringField(const folly::dynamic& fields, const char* name) {
  const auto* value = fields.isObject() ? fields.get_ptr(name) : nullptr;
  return value != nullptr && value->isString() ? value->getString() : "unknown";
}

// 0 when the size is unknown, as the GPE does for variable-size operations.
uint64_t messageBytes(const folly::dynamic& fields) {
  const auto* count = fields.isObject() ? fields.get_ptr("count") : nullptr;
  if (count == nullptr || !count->isInt() || count->asInt() < 0) {
    return 0;
  }
  const int typeSize =
      commTypeSize(stringToCommsDatatype(stringField(fields, "dataType")));
  return typeSize > 0 ? static_cast<uint64_t>(count->asInt()) * typeSize : 0;
}

std::optional<double> elapsedUs(
    ICollWaitEvent::system_clock_time_point from,
    ICollWaitEvent::system_clock_time_point to) {
  if (from == kUnset || to == kUnset || to < from) {
    return std::nullopt;
  }
  return std::chrono::duration<double, std::micro>(to - from).count();
}

uint64_t roundUs(double us) {
  return static_cast<uint64_t>(std::llround(us));
}

} // namespace

CollLatencyPlugin::CollLatencyPlugin(Filter filter)
    : filter_{std::move(filter)} {}

std::string_view CollLatencyPlugin::getName() const noexcept {
  return kCollLatencyPluginName;
}

CommsMaybeVoid CollLatencyPlugin::beforeCollKernelScheduled(
    CollTraceEvent&) noexcept {
  return folly::unit;
}

CommsMaybeVoid CollLatencyPlugin::afterCollKernelScheduled(
    CollTraceEvent&) noexcept {
  return folly::unit;
}

CommsMaybeVoid CollLatencyPlugin::afterCollKernelStart(
    CollTraceEvent&) noexcept {
  return folly::unit;
}

CommsMaybeVoid CollLatencyPlugin::collEventProgressing(
    CollTraceEvent&) noexcept {
  return folly::unit;
}

CommsMaybeVoid CollLatencyPlugin::afterCollKernelEnd(
    CollTraceEvent& curEvent) noexcept {
  if (curEvent.collRecord == nullptr) {
    return folly::makeUnexpected(CommsError(
        "CollLatencyPlugin received an event without a collective record",
        commInternalError));
  }
  const auto& timing = curEvent.collRecord->getTimingInfo();
  const auto duration =
      elapsedUs(timing.getCollStartTs(), timing.getCollEndTs());
  if (!duration.has_value()) {
    return folly::unit;
  }

  const auto& metadata = curEvent.collRecord->getCollMetadata();
  const auto fields =
      metadata != nullptr ? metadata->toDynamic() : folly::dynamic{};
  if (filter_ && !filter_(fields)) {
    return folly::unit;
  }
  // CollTrace names operations "AllReduce"; the GPE keys use "allreduce".
  auto collective = stringField(fields, "opName");
  folly::toLowerAscii(collective);
  stats_.record(
      collective,
      fmt::format(
          "{}.{}.{}",
          collective,
          stringField(fields, "algoName"),
          messageBytes(fields)),
      roundUs(*duration));
  return folly::unit;
}

::comms::CollectiveStatsMap CollLatencyPlugin::takeCollectiveStats() noexcept {
  return stats_.getAndClear();
}

} // namespace meta::comms::colltrace
