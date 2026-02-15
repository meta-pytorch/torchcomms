// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/window/WindowMemory.h"

#include <vector>

#include "comms/pipes/window/DeviceWindowMemory.cuh"
#include "comms/pipes/window/DeviceWindowSignal.cuh"
#include "comms/utils/checks.h"

namespace comms::pipes {

WindowMemory::WindowMemory(
    int myRank,
    int nRanks,
    std::shared_ptr<ctran::bootstrap::IBootstrap> bootstrap,
    const WindowMemoryConfig& config,
    MemSharingMode memSharingMode)
    : myRank_(myRank),
      nRanks_(nRanks),
      bootstrap_(std::move(bootstrap)),
      config_(config),
      memSharingMode_(memSharingMode) {
  // Calculate inbox size: signalCount * sizeof(SignalState)
  // Note: sizeof(SignalState) == 128 due to alignas(128) on the struct
  inboxSize_ = getSignalInboxBufferSize(static_cast<int>(config_.signalCount));

  // Allocate inbox buffer using GpuMemHandler (needs exchange with peers)
  inboxHandler_ = std::make_unique<GpuMemHandler>(
      bootstrap_, myRank_, nRanks_, inboxSize_, memSharingMode_);

  // Initialize inbox to zero
  std::vector<SignalState> initStates(config_.signalCount);
  CUDA_CHECK(cudaMemcpy(
      inboxHandler_->getLocalDeviceMemPtr(),
      initStates.data(),
      inboxSize_,
      cudaMemcpyDefault));

  // Allocate peer signal spans array (populate later in exchange())
  // Each entry is a DeviceSpan<SignalState> pointing to a peer's inbox
  // Size = nPeers (not nRanks) - excludes self
  int nPeers = nRanks_ - 1;
  std::size_t spanArraySize = nPeers * sizeof(DeviceSpan<SignalState>);
  peerInboxPtrsDevice_ =
      std::make_unique<meta::comms::DeviceBuffer>(spanArraySize);
}

void WindowMemory::exchange() {
  // Exchange inbox handles so peers can write to our inbox
  inboxHandler_->exchangeMemPtrs();

  // Build peer signal spans on host (peer-indexed, excludes self)
  // Each span wraps a peer's inbox pointer + signalCount
  int nPeers = nRanks_ - 1;
  auto signalCount = static_cast<uint32_t>(config_.signalCount);
  std::vector<DeviceSpan<SignalState>> peerSpans;
  peerSpans.reserve(nPeers);
  for (int peerIdx = 0; peerIdx < nPeers; ++peerIdx) {
    int rank = peerToRank(peerIdx);
    auto* ptr =
        static_cast<SignalState*>(inboxHandler_->getPeerDeviceMemPtr(rank));
    peerSpans.emplace_back(ptr, signalCount);
  }

  // Copy peer spans to device (buffer already allocated in constructor)
  CUDA_CHECK(cudaMemcpy(
      peerInboxPtrsDevice_->get(),
      peerSpans.data(),
      nPeers * sizeof(DeviceSpan<SignalState>),
      cudaMemcpyDefault));

  exchanged_ = true;
}

DeviceWindowSignal WindowMemory::getDeviceWindowSignal() const {
  if (!exchanged_) {
    throw std::runtime_error(
        "WindowMemory::getDeviceWindowSignal() called before exchange()");
  }

  // Build DeviceWindowSignal object
  SignalState* localInbox =
      static_cast<SignalState*>(inboxHandler_->getLocalDeviceMemPtr());
  auto* peerSpans =
      static_cast<DeviceSpan<SignalState>*>(peerInboxPtrsDevice_->get());

  // peerSpans has nPeers entries (not nRanks)
  int nPeers = nRanks_ - 1;
  return DeviceWindowSignal(
      myRank_,
      nRanks_,
      static_cast<int>(config_.signalCount),
      DeviceSpan<SignalState>(
          localInbox, static_cast<uint32_t>(config_.signalCount)),
      DeviceSpan<DeviceSpan<SignalState>>(peerSpans, nPeers));
}

DeviceWindowMemory WindowMemory::getDeviceWindowMemory() const {
  return DeviceWindowMemory(getDeviceWindowSignal());
}

} // namespace comms::pipes
