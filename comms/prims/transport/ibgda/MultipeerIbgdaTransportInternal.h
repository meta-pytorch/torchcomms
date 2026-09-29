// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <exception>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>

#include "comms/prims/transport/ibgda/MultipeerIbgdaTransport.h"

#ifndef __HIP_PLATFORM_AMD__
#include <doca_gpunetio_host.h>
#endif

namespace comms::prims::detail {

template <typename ExchangePeerBuffers>
decltype(auto) exchangePeerBufferPayloadWithExposureTracking(
    PeerRkeyExposureState& exposureState,
    ExchangePeerBuffers&& exchangePeerBuffers) {
  return std::forward<ExchangePeerBuffers>(exchangePeerBuffers)(
      [&exposureState]() {
        exposureState = PeerRkeyExposureState::kPossiblyExposed;
      });
}

template <typename PeerRkeyExposureStates>
std::optional<std::size_t> findPossiblyExposedPeer(
    const PeerRkeyExposureStates& exposureStates) {
  for (std::size_t peerIndex = 0; peerIndex < exposureStates.size();
       ++peerIndex) {
    if (exposureStates[peerIndex] == PeerRkeyExposureState::kPossiblyExposed) {
      return peerIndex;
    }
  }
  return std::nullopt;
}

template <typename Transport>
Transport* releaseTransportForProcessLifetimeIfQuarantined(
    std::unique_ptr<Transport>& transport,
    bool processLifetimeQuarantineRequired) {
  return processLifetimeQuarantineRequired ? transport.release() : nullptr;
}

template <typename Transport, typename Operation, typename Quarantine>
decltype(auto) runWithProcessLifetimeQuarantineOnFailure(
    Transport& transport,
    Operation&& operation,
    Quarantine&& quarantine) {
  try {
    return std::forward<Operation>(operation)(transport);
  } catch (const std::exception& ex) {
    if (transport.requiresProcessLifetimeQuarantine()) {
      quarantine(ex.what());
    }
    throw;
  } catch (...) {
    if (transport.requiresProcessLifetimeQuarantine()) {
      quarantine("unknown IB transport failure requiring quarantine");
    }
    throw;
  }
}

template <typename Operation, typename Quarantine>
decltype(auto) runWithProcessLifetimeQuarantineOnFailureAfterRkeyExposure(
    bool& rkeysPossiblyExposed,
    Operation&& operation,
    Quarantine&& quarantine) {
  try {
    return std::forward<Operation>(operation)();
  } catch (const std::exception& ex) {
    if (rkeysPossiblyExposed) {
      quarantine(ex.what());
    }
    throw;
  } catch (...) {
    if (rkeysPossiblyExposed) {
      quarantine("unknown rkey exchange failure");
    }
    throw;
  }
}

template <
    typename Operation,
    typename QuarantineIbTransport,
    typename QuarantineCallerBuffer>
decltype(auto) runWithProcessLifetimeQuarantineOnFailureAfterWindowExposure(
    bool& ibgdaRkeysPossiblyExposed,
    bool& callerBufferPossiblyExposed,
    Operation&& operation,
    QuarantineIbTransport&& quarantineIbTransport,
    QuarantineCallerBuffer&& quarantineCallerBuffer) {
  try {
    return std::forward<Operation>(operation)();
  } catch (const std::exception& ex) {
    if (ibgdaRkeysPossiblyExposed) {
      quarantineIbTransport(ex.what());
    }
    if (callerBufferPossiblyExposed) {
      quarantineCallerBuffer(ex.what());
    }
    throw;
  } catch (...) {
    if (ibgdaRkeysPossiblyExposed) {
      quarantineIbTransport("unknown window exchange failure");
    }
    if (callerBufferPossiblyExposed) {
      quarantineCallerBuffer("unknown window exchange failure");
    }
    throw;
  }
}

template <typename ReleaseResources>
void releaseUnlessProcessLifetimeQuarantined(
    bool processLifetimeQuarantineRequired,
    ReleaseResources&& releaseResources) {
  if (!processLifetimeQuarantineRequired) {
    std::forward<ReleaseResources>(releaseResources)();
  }
}

#ifndef __HIP_PLATFORM_AMD__

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

#endif

} // namespace comms::prims::detail
