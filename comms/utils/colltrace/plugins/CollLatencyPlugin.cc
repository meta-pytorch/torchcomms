// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/colltrace/plugins/CollLatencyPlugin.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <folly/Range.h>
#include <folly/String.h>
#include <folly/Unit.h>
#include <folly/dynamic.h>

#include "comms/utils/Conversion.h"
#include "comms/utils/commSpecs.h"

namespace meta::comms::colltrace {

namespace {

constexpr std::size_t kBufferSize = 512;
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

uint64_t quantileUs(const folly::TDigest& digest, double q) {
  return digest.empty() ? 0 : roundUs(digest.estimateQuantile(q));
}

// Skips keys with no recorded collective rather than adding empty roll-ups.
void setQuantiles(
    ::comms::CollectiveStatsMap& stats,
    const std::string& key,
    const folly::TDigest& duration,
    const folly::TDigest& queueDelay) {
  const auto it = stats.find(key);
  if (it == stats.end() || it->second.count == 0) {
    return;
  }
  auto& stat = it->second;
  stat.p50_us = quantileUs(duration, 0.5);
  stat.p90_us = quantileUs(duration, 0.9);
  stat.p99_us = quantileUs(duration, 0.99);
  stat.queue_p99_us = quantileUs(queueDelay, 0.99);
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
  const auto startTs = timing.getCollStartTs();
  const auto duration = elapsedUs(startTs, timing.getCollEndTs());
  if (!duration.has_value()) {
    return folly::unit;
  }
  const auto queueDelay = elapsedUs(timing.getCollEnqueueTs(), startTs);

  const auto& metadata = curEvent.collRecord->getCollMetadata();
  const auto fields =
      metadata != nullptr ? metadata->toDynamic() : folly::dynamic{};
  if (filter_ && !filter_(fields)) {
    return folly::unit;
  }
  // CollTrace names operations "AllReduce"; the GPE keys use "allreduce".
  auto collective = stringField(fields, "opName");
  folly::toLowerAscii(collective);
  const auto& launch = curEvent.collRecord->getLaunchInfo();
  stats_.record(
      collective,
      fmt::format(
          "{}.{}.{}",
          collective,
          stringField(fields, "algoName"),
          messageBytes(fields)),
      roundUs(*duration),
      launch.numBlocks(),
      launch.blockSize(),
      launch.blocksPerSm());

  auto window = window_.wlock();
  auto& opWindow = (*window)[std::move(collective)];
  const auto add = [](LatencySamples& samples, double us) {
    samples.buffer.push_back(us);
    if (samples.buffer.size() >= kBufferSize) {
      samples.digest = samples.digest.merge(folly::range(samples.buffer));
      samples.buffer.clear();
    }
  };
  add(opWindow.duration, *duration);
  if (queueDelay.has_value()) {
    add(opWindow.queueDelay, *queueDelay);
  }
  return folly::unit;
}

::comms::CollectiveStatsMap CollLatencyPlugin::takeCollectiveStats() noexcept {
  // Not atomic across the two: a collective ending in between is counted in
  // one window and sampled in the other, within the quantiles' error.
  auto stats = stats_.getAndClear();
  const auto window = std::exchange(*window_.wlock(), {});
  const auto flush = [](const LatencySamples& samples) {
    return samples.digest.merge(folly::range(samples.buffer));
  };
  std::vector<folly::TDigest> durations;
  std::vector<folly::TDigest> queueDelays;
  durations.reserve(window.size());
  queueDelays.reserve(window.size());
  for (const auto& [collective, opWindow] : window) {
    durations.push_back(flush(opWindow.duration));
    queueDelays.push_back(flush(opWindow.queueDelay));
    setQuantiles(
        stats,
        fmt::format("{}.{}", collective, ::comms::kCollectiveStatsAllKey),
        durations.back(),
        queueDelays.back());
  }
  setQuantiles(
      stats,
      std::string{::comms::kCollectiveStatsAllKey},
      folly::TDigest::merge(folly::range(durations)),
      folly::TDigest::merge(folly::range(queueDelays)));
  return stats;
}

} // namespace meta::comms::colltrace
