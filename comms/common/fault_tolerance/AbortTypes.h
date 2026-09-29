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
  // The IBRC host CPU proxy stopped making progress and a device-side wait hit
  // its watchdog. Distinct from TIMED_OUT, which is the enclosing collective's
  // own deadline expiring: this one names the proxy thread as the stalled
  // component, which is a different fault with a different owner.
  IBRC_PROXY_TIMEOUT = 6,
};

constexpr bool isTerminalAbortReason(AbortReason reason) {
  switch (reason) {
    case AbortReason::ABORTED:
    case AbortReason::TIMED_OUT:
    case AbortReason::BOOTSTRAP_POLL:
    case AbortReason::NETWORK_ERROR:
    case AbortReason::INTERNAL_ERROR:
    case AbortReason::IBRC_PROXY_TIMEOUT:
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
    case AbortReason::IBRC_PROXY_TIMEOUT:
      return "ibrc_proxy_timeout";
  }
  return "unknown";
}

/*
 * Sentinel for `AbortState::originPeer` and `AbortInfo::originPeer`.
 *
 * Means the winning writer was not blocked on one specific rank, which is the
 * honest answer for aggregate waits -- a barrier, an LL decode, a signal wait
 * polling a bare counter -- and for every host writer with no peer in hand. It
 * is a distinct answer from "rank 0", so it needs a value outside rank space.
 */
inline constexpr int kNoAbortPeer = -1;

/*
 * Stable, append-only values naming which writer declared an abort.
 *
 * Coarse on purpose: these are the distinctions the fault-tolerance layer can
 * make on its own, without help from the wait that called it. A wait that
 * wants to name itself more precisely appends a value here and passes it to
 * `setAbort`; that is the intended growth path and is why this is append-only.
 */
enum class AbortSite : int {
  UNKNOWN = 0,
  // A host caller reached `Abort::setAbort` directly.
  HOST = 1,
  // The host active deadline lapsed and `Abort::isTimedOut` recorded it.
  HOST_DEADLINE = 2,
  // Device code called `AbortDevice::setAbort` or `AbortFlag::setAbort`.
  DEVICE = 3,
  // A device handle's local deadline lapsed inside `checkExpired()`.
  DEVICE_DEADLINE = 4,
};

constexpr std::string_view abortSiteToString(AbortSite site) {
  switch (site) {
    case AbortSite::UNKNOWN:
      return "unknown";
    case AbortSite::HOST:
      return "host";
    case AbortSite::HOST_DEADLINE:
      return "host_deadline";
    case AbortSite::DEVICE:
      return "device";
    case AbortSite::DEVICE_DEADLINE:
      return "device_deadline";
  }
  return "unknown";
}

struct AbortInfo {
  // AbortInfo always describes an actual abort. Absence is represented by
  // std::nullopt at query boundaries.
  AbortReason reason{AbortReason::ABORTED};
  std::string context;

  // Which writer won the reason CAS, and the global rank it was blocked on.
  //
  // These carry for device-originated aborts, where `context` cannot: a device
  // writer's `const char*` is consumed at the winning callsite and never
  // persisted in mapped state, so without these a device abort reaches the
  // host as a reason and nothing else.
  AbortSite site{AbortSite::UNKNOWN};
  int originPeer{kNoAbortPeer};

  std::string_view reasonString() const {
    return abortReasonToString(reason);
  }

  std::string_view siteString() const {
    return abortSiteToString(site);
  }

  bool operator==(const AbortInfo& other) const {
    return reason == other.reason && context == other.context &&
        site == other.site && originPeer == other.originPeer;
  }
};

} // namespace comms::fault_tolerance
