// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <cstdint>
#include <string>

// Stable identity + physical-placement join key for an exported readout window.
// Folly-free plain data, so the launch/comm layer can populate it without
// pulling the JSON serializer's dependencies. A process-local communicator
// value is not enough for fleet aggregation; the backend aggregates and dedups
// across restarts and ranks on this tuple, and joins the placement key (host,
// GPU, NIC/rail) against the topology inventory at `topologyVersion`.

namespace meta::comms::collstats {

struct CollStatsExportIdentity {
  std::string job;
  std::string tenant;
  std::string processGroup;
  uint64_t commHash{0};
  uint64_t commGeneration{0};
  int32_t rank{-1};
  std::string softwareVersion;
  // Physical-placement join key, as seen at the rank.
  std::string host;
  int32_t gpu{-1};
  std::string nicRail;
  uint64_t topologyVersion{0};
};

} // namespace meta::comms::collstats
