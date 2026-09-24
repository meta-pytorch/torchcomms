// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <type_traits>
#include <utility>

namespace comms::prims::detail {

template <typename Deregister, typename Release, typename Retain>
bool releaseAllocationAfterDeregistration(
    bool registered,
    Deregister&& deregister,
    Release&& release,
    Retain&& retain) noexcept {
  static_assert(
      std::is_nothrow_invocable_r_v<bool, Deregister>,
      "deregistration callback must be noexcept and report success");
  static_assert(
      std::is_nothrow_invocable_v<Release>,
      "release callback must be noexcept");
  static_assert(
      std::is_nothrow_invocable_v<Retain>,
      "retention callback must be noexcept");

  if (registered && !std::forward<Deregister>(deregister)()) {
    std::forward<Retain>(retain)();
    return false;
  }
  std::forward<Release>(release)();
  return true;
}

template <typename KeepAliveHolders, typename ReleaseResources>
bool releaseResourcesOrRetainKeepAlives(
    bool resourceLifetimeQuarantineRequired,
    bool keepAliveLifetimeQuarantineRequired,
    KeepAliveHolders& keepAlives,
    ReleaseResources&& releaseResources) noexcept {
  static_assert(
      std::is_nothrow_invocable_r_v<bool, ReleaseResources>,
      "cleanup callback must be noexcept and report success");

  const bool resourcesReleased = !resourceLifetimeQuarantineRequired &&
      std::forward<ReleaseResources>(releaseResources)();
  if (resourcesReleased && !keepAliveLifetimeQuarantineRequired) {
    return true;
  }

  for (auto& holder : keepAlives) {
    static_cast<void>(holder.release());
  }
  return false;
}

template <typename Mrs, typename Deregister>
bool tryDeregisterMrs(Mrs& mrs, int numNics, Deregister&& deregister) {
  bool success = true;
  for (int nic = 0; nic < numNics; ++nic) {
    if (mrs[nic] == nullptr) {
      continue;
    }
    if (deregister(nic, mrs[nic]) == 0) {
      mrs[nic] = nullptr;
    } else {
      success = false;
    }
  }
  return success;
}

template <typename Mrs>
bool registrationKeysAvailable(
    bool deregistrationFailed,
    const Mrs& mrs,
    int numNics) {
  if (deregistrationFailed) {
    return false;
  }
  for (int nic = 0; nic < numNics; ++nic) {
    if (mrs[nic] == nullptr) {
      return false;
    }
  }
  return true;
}

} // namespace comms::prims::detail
