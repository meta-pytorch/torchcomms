// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#ifndef __HIP_PLATFORM_AMD__

#include <cstddef>

#include <doca_gpunetio_host.h>

namespace comms::prims::detail {

void requireQpTransitionSuccess(
    doca_error_t status,
    const char* qpKind,
    std::size_t nicIndex,
    std::size_t qpIndex);

template <
    typename NicResources,
    typename TransitionQpGroup,
    typename TransitionQp,
    typename ReleaseGpuAllocations,
    typename ReleaseSendRecvBuffers>
void quiesceQpsThenReleaseBuffers(
    bool shouldQuiesceQps,
    const NicResources& nicResources,
    TransitionQpGroup transitionQpGroup,
    TransitionQp transitionQp,
    ReleaseGpuAllocations releaseGpuAllocations,
    ReleaseSendRecvBuffers releaseSendRecvBuffers) {
  if (shouldQuiesceQps) {
    for (std::size_t nicIndex = 0; nicIndex < nicResources.size(); ++nicIndex) {
      const auto& nic = nicResources[nicIndex];
      for (std::size_t qpIndex = 0; qpIndex < nic.qpSlots.size(); ++qpIndex) {
        const auto& resources = nic.qpSlots[qpIndex];
        if (resources.group != nullptr) {
          requireQpTransitionSuccess(
              transitionQpGroup(resources.group), "group", nicIndex, qpIndex);
        }
        if (resources.standaloneMain != nullptr) {
          requireQpTransitionSuccess(
              transitionQp(resources.standaloneMain),
              "standalone_main",
              nicIndex,
              qpIndex);
        }
        if (resources.loopback != nullptr) {
          requireQpTransitionSuccess(
              transitionQp(resources.loopback),
              "loopback_companion",
              nicIndex,
              qpIndex);
        }
      }
    }
  }

  releaseGpuAllocations();
  releaseSendRecvBuffers();
}

} // namespace comms::prims::detail

#endif
