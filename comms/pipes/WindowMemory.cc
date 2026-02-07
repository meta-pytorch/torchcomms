// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/WindowMemory.h"

#include <vector>

#include "comms/pipes/DeviceBarrier.cuh"
#include "comms/pipes/DeviceCounter.cuh"
#include "comms/pipes/DeviceSignal.cuh"
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

  // Allocate peer inbox pointers array (populate later in exchange())
  // Size = nPeers (not nRanks) - excludes self
  int nPeers = nRanks_ - 1;
  std::size_t ptrArraySize = nPeers * sizeof(SignalState*);
  peerInboxPtrsDevice_ =
      std::make_unique<meta::comms::DeviceBuffer>(ptrArraySize);

  // Allocate counter buffer (local-only, no exchange needed)
  counterSize_ = getCounterBufferSize(static_cast<int>(config_.counterCount));
  counterDevice_ = std::make_unique<meta::comms::DeviceBuffer>(counterSize_);

  // Initialize counter buffer to zero
  std::vector<SignalState> counterInitStates(config_.counterCount);
  CUDA_CHECK(cudaMemcpy(
      counterDevice_->get(),
      counterInitStates.data(),
      counterSize_,
      cudaMemcpyDefault));

  // Allocate barrier inbox buffer (shared with peers via exchange)
  barrierInboxSize_ =
      getMultiPeerBarrierBufferSize(static_cast<int>(config_.barrierCount));
  barrierInboxHandler_ = std::make_unique<GpuMemHandler>(
      bootstrap_, myRank_, nRanks_, barrierInboxSize_, memSharingMode_);

  // Initialize barrier inbox to zero
  std::vector<BarrierState> barrierInitStates(config_.barrierCount);
  CUDA_CHECK(cudaMemcpy(
      barrierInboxHandler_->getLocalDeviceMemPtr(),
      barrierInitStates.data(),
      barrierInboxSize_,
      cudaMemcpyDefault));

  // Allocate peer barrier pointers array (populate later in exchange())
  std::size_t barrierPtrArraySize = nPeers * sizeof(BarrierState*);
  peerBarrierPtrsDevice_ =
      std::make_unique<meta::comms::DeviceBuffer>(barrierPtrArraySize);
}

void WindowMemory::exchange() {
  // Exchange inbox handles so peers can write to our inbox
  inboxHandler_->exchangeMemPtrs();

  // Build peer inbox pointers array on host (peer-indexed, excludes self)
  int nPeers = nRanks_ - 1;
  std::vector<SignalState*> peerPtrs(nPeers);
  for (int peerIdx = 0; peerIdx < nPeers; ++peerIdx) {
    int rank = peerToRank(peerIdx);
    peerPtrs[peerIdx] =
        static_cast<SignalState*>(inboxHandler_->getPeerDeviceMemPtr(rank));
  }

  // Copy peer pointers to device (buffer already allocated in constructor)
  CUDA_CHECK(cudaMemcpy(
      peerInboxPtrsDevice_->get(),
      peerPtrs.data(),
      nPeers * sizeof(SignalState*),
      cudaMemcpyDefault));

  // Exchange barrier inbox handles so peers can write arrive signals
  barrierInboxHandler_->exchangeMemPtrs();

  // Build peer barrier pointers array on host (peer-indexed, excludes self)
  std::vector<BarrierState*> peerBarrierPtrs(nPeers);
  for (int peerIdx = 0; peerIdx < nPeers; ++peerIdx) {
    int rank = peerToRank(peerIdx);
    peerBarrierPtrs[peerIdx] = static_cast<BarrierState*>(
        barrierInboxHandler_->getPeerDeviceMemPtr(rank));
  }

  // Copy peer barrier pointers to device
  CUDA_CHECK(cudaMemcpy(
      peerBarrierPtrsDevice_->get(),
      peerBarrierPtrs.data(),
      nPeers * sizeof(BarrierState*),
      cudaMemcpyDefault));

  exchanged_ = true;
}

DeviceSignal WindowMemory::getDeviceSignal() const {
  if (!exchanged_) {
    throw std::runtime_error(
        "WindowMemory::getDeviceSignal() called before exchange()");
  }

  // Build DeviceSignal object
  SignalState* localInbox =
      static_cast<SignalState*>(inboxHandler_->getLocalDeviceMemPtr());
  SignalState** peerInboxPtrs =
      static_cast<SignalState**>(peerInboxPtrsDevice_->get());

  // peerInboxPtrs has nPeers entries (not nRanks)
  int nPeers = nRanks_ - 1;
  return DeviceSignal(
      myRank_,
      nRanks_,
      static_cast<int>(config_.signalCount),
      DeviceSpan<SignalState>(localInbox, config_.signalCount),
      DeviceSpan<SignalState*>(peerInboxPtrs, nPeers));
}

DeviceCounter WindowMemory::getDeviceCounter() const {
  // No exchange check — counters are local-only, usable immediately
  auto* counters = static_cast<SignalState*>(counterDevice_->get());
  return DeviceCounter(DeviceSpan<SignalState>(counters, config_.counterCount));
}

DeviceBarrier WindowMemory::getDeviceBarrier() const {
  if (!exchanged_) {
    throw std::runtime_error(
        "WindowMemory::getDeviceBarrier() called before exchange()");
  }

  auto* localBarriers =
      static_cast<BarrierState*>(barrierInboxHandler_->getLocalDeviceMemPtr());
  auto** peerBarrierPtrs =
      static_cast<BarrierState**>(peerBarrierPtrsDevice_->get());

  int nPeers = nRanks_ - 1;
  return DeviceBarrier(
      myRank_,
      nRanks_,
      DeviceSpan<BarrierState>(localBarriers, config_.barrierCount),
      DeviceSpan<BarrierState*>(peerBarrierPtrs, nPeers));
}

} // namespace comms::pipes
