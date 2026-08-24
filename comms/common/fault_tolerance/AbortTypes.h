// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <string>
#include <string_view>

namespace comms::fault_tolerance {

// Device-side action to take after observing an abort.
enum class AbortBehavior : int {
  SKIP = 0,
  TRAP = 1,
};

// Stable, append-only values shared by host and device abort state.
enum class AbortReason : int {
  NONE = 0,
  ABORTED = 1,
  TIMED_OUT = 2,
  BOOTSTRAP_POLL = 3,
  NETWORK_ERROR = 4,
  INTERNAL_ERROR = 5,
};

constexpr bool isTerminalAbortReason(AbortReason reason) {
  switch (reason) {
    case AbortReason::ABORTED:
    case AbortReason::TIMED_OUT:
    case AbortReason::BOOTSTRAP_POLL:
    case AbortReason::NETWORK_ERROR:
    case AbortReason::INTERNAL_ERROR:
      return true;
    case AbortReason::NONE:
      return false;
  }
  return false;
}

constexpr std::string_view abortReasonToString(AbortReason reason) {
  switch (reason) {
    case AbortReason::NONE:
      return "none";
    case AbortReason::ABORTED:
      return "aborted";
    case AbortReason::TIMED_OUT:
      return "timed_out";
    case AbortReason::BOOTSTRAP_POLL:
      return "bootstrap_poll";
    case AbortReason::NETWORK_ERROR:
      return "network_error";
    case AbortReason::INTERNAL_ERROR:
      return "internal_error";
  }
  return "unknown";
}

struct AbortInfo {
  // AbortInfo always describes an actual abort. Absence is represented by
  // std::nullopt at query boundaries.
  AbortReason reason{AbortReason::ABORTED};
  std::string context;

  std::string_view reasonString() const {
    return abortReasonToString(reason);
  }

  bool operator==(const AbortInfo& other) const {
    return reason == other.reason && context == other.context;
  }
};

} // namespace comms::fault_tolerance
