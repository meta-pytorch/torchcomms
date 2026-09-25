// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <type_traits>
#include <utility>

namespace comms::prims::detail {

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

} // namespace comms::prims::detail
