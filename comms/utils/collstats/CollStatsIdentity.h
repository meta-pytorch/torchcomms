// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <cstdint>
#include <optional>
#include <string>

// Stable identity + physical-placement join key for an exported readout window.
// Folly-free plain data, so the launch/comm layer can populate it without
// pulling the JSON serializer's dependencies. A process-local communicator
// value is not enough for fleet aggregation; the backend aggregates and dedups
// across restarts and ranks on this tuple, and joins the placement key (host,
// GPU, NIC/rail) against the topology inventory at `topologyVersion`.

namespace meta::comms::collstats {

// Every field must be able to say "not reported" distinguishably from a real
// value, because a producer sets only the ones it has a source for and the
// backend joins on what it receives. Strings use empty, the two rank/device
// ordinals use -1, and the counters below are optional: unlike an ordinal,
// 0 is a perfectly good generation or topology version, so a zero default
// would be indistinguishable from a field nobody filled in.
struct CollStatsExportIdentity {
  std::string job;
  std::string tenant;
  std::string processGroup;
  uint64_t commHash{0};
  std::optional<uint64_t> commGeneration;
  int32_t rank{-1};
  std::string softwareVersion;
  // Physical-placement join key, as seen at the rank.
  std::string host;
  int32_t gpu{-1};
  std::string nicRail;
  std::optional<uint64_t> topologyVersion;
};

} // namespace meta::comms::collstats
