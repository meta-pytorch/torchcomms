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
    typename QpSlotResources,
    typename TransitionQpGroup,
    typename TransitionQp>
void quiesceQpSlot(
    const QpSlotResources& resources,
    std::size_t nicIndex,
    std::size_t qpIndex,
    TransitionQpGroup transitionQpGroup,
    TransitionQp transitionQp) {
  resources.forEachQp(
      [&](auto* group, const char* qpKind) {
        requireQpTransitionSuccess(
            transitionQpGroup(group), qpKind, nicIndex, qpIndex);
      },
      [&](auto* qp, const char* qpKind) {
        requireQpTransitionSuccess(transitionQp(qp), qpKind, nicIndex, qpIndex);
      });
}

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
        quiesceQpSlot(
            nic.qpSlots[qpIndex],
            nicIndex,
            qpIndex,
            transitionQpGroup,
            transitionQp);
      }
    }
  }

  releaseGpuAllocations();
  releaseSendRecvBuffers();
}

template <
    typename NicResources,
    typename TransitionQpGroup,
    typename TransitionQp,
    typename ReleasePeerResources>
void quiescePeerQpsThenReleaseResources(
    bool shouldQuiesceQps,
    const NicResources& nicResources,
    std::size_t peerIndex,
    std::size_t slotsPerPeer,
    TransitionQpGroup transitionQpGroup,
    TransitionQp transitionQp,
    ReleasePeerResources releasePeerResources) {
  if (shouldQuiesceQps) {
    const std::size_t firstQpIndex = peerIndex * slotsPerPeer;
    for (std::size_t nicIndex = 0; nicIndex < nicResources.size(); ++nicIndex) {
      const auto& nic = nicResources[nicIndex];
      for (std::size_t slot = 0; slot < slotsPerPeer; ++slot) {
        const std::size_t qpIndex = firstQpIndex + slot;
        quiesceQpSlot(
            nic.qpSlots[qpIndex],
            nicIndex,
            qpIndex,
            transitionQpGroup,
            transitionQp);
      }
    }
  }

  releasePeerResources();
}

} // namespace comms::prims::detail

#endif
