// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <string>

namespace comms::fault_tolerance {

// Stable, append-only values shared by host and device abort state.
enum class AbortReason : int {
  NONE = 0,
  ABORTED = 1,
  TIMED_OUT = 2,
  BOOTSTRAP_POLL = 3,
  NETWORK_ERROR = 4,
  INTERNAL_ERROR = 5,
};

struct AbortInfo {
  // AbortInfo always describes an actual abort. Absence is represented by
  // std::nullopt at query boundaries.
  AbortReason reason{AbortReason::ABORTED};
  std::string context;

  bool operator==(const AbortInfo&) const = default;
};

} // namespace comms::fault_tolerance
