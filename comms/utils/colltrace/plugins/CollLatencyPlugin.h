// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include <folly/Synchronized.h>
#include <folly/dynamic.h>
#include <folly/stats/TDigest.h>

#include "comms/common/CollectiveStats.h"
#include "comms/utils/colltrace/CollTracePlugin.h"

namespace meta::comms::colltrace {

/*
 * Aggregates the device timing of every collective that ends normally into
 * the same buckets the CTRAN GPE uses, "<op>.<algo>.<bytes>", so collectives
 * that bypass the GPE report count, total, min and max too. Memory per
 * operation is bounded; the quantiles are TDigest estimates.
 */
class CollLatencyPlugin : public ICollTracePlugin {
 public:
  // Given a collective's metadata, returns whether to record it. Lets a
  // backend whose CollTrace also sees GPE-routed collectives skip them.
  using Filter = std::function<bool(const folly::dynamic& metadata)>;

  explicit CollLatencyPlugin(Filter filter = nullptr);

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

  /*
   * Returns the stats of collectives that ended since the previous call, with
   * the quantiles set on the "<op>.all" and "all" roll-ups, and starts
   * a new window. Meant for one reader.
   */
  ::comms::CollectiveStatsMap takeCollectiveStats() noexcept;

  static constexpr std::string_view kCollLatencyPluginName =
      "CollLatencyPlugin";

 private:
  // Samples in microseconds, buffered because TDigest merges are costly.
  struct LatencySamples {
    std::vector<double> buffer;
    folly::TDigest digest;
  };
  struct OpWindow {
    LatencySamples duration;
    LatencySamples queueDelay;
  };

  const Filter filter_;
  ::comms::CollectiveStats stats_;
  folly::Synchronized<std::map<std::string, OpWindow>> window_;
};

} // namespace meta::comms::colltrace
