// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <functional>
#include <string_view>

#include <folly/dynamic.h>

#include "comms/common/CollectiveStats.h"
#include "comms/utils/colltrace/CollTracePlugin.h"

namespace meta::comms::colltrace {

/*
 * Aggregates the device timing of every collective that ends normally into
 * the same buckets the CTRAN GPE uses, "<op>.<algo>.<bytes>", so collectives
 * that bypass the GPE report count, total, min and max too.
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

  // Returns the stats of collectives that ended since the previous call and
  // starts a new window.
  ::comms::CollectiveStatsMap takeCollectiveStats() noexcept;

  static constexpr std::string_view kCollLatencyPluginName =
      "CollLatencyPlugin";

 private:
  const Filter filter_;
  ::comms::CollectiveStats stats_;
};

} // namespace meta::comms::colltrace
