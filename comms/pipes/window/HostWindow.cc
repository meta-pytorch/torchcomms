// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/window/HostWindow.h"

#include <stdexcept>
#include <vector>

#include "comms/pipes/GpuMemHandler.h"
#include "comms/pipes/MultiPeerTransport.h"
#include "comms/pipes/SignalState.cuh"
#include "comms/pipes/window/DeviceWindow.cuh"
#include "comms/utils/checks.h"

namespace comms::pipes {

namespace {

// Allocate zeroed GPU memory via cudaMalloc and wrap in IbgdaLocalBuffer.
// The .ptr field holds the raw GPU pointer; .lkey is unset until
// registerIbgdaBuffer() is called in exchange().
IbgdaLocalBuffer allocateIbgdaBuffer(std::size_t size) {
  void* ptr = nullptr;
  CUDA_CHECK(cudaMalloc(&ptr, size));
  CUDA_CHECK(cudaMemset(ptr, 0, size));
  return IbgdaLocalBuffer(ptr, NetworkLKey{});
}

} // namespace

HostWindow::HostWindow(
    MultiPeerTransport& transport,
    const WindowConfig& config,
    void* userBuffer,
    std::size_t userBufferSize)
    : transport_(transport),
      myRank_(transport.my_rank()),
      nRanks_(transport.n_ranks()),
      config_(config),
      nvlPeerRanks_(transport.nvl_peer_ranks()),
      ibgdaPeerRanks_(transport.ibgda_peer_ranks()),
      nvlLocalRank_(transport.nvl_local_rank()),
      nvlNRanks_(transport.nvl_n_ranks()),
      userBuffer_(userBuffer),
      userBufferSize_(userBufferSize) {
  int nNvlPeers = static_cast<int>(nvlPeerRanks_.size());
  int nIbgdaPeers = static_cast<int>(ibgdaPeerRanks_.size());

  // ==========================================================================
  // Pre-computed peer index maps (O(1) rank → peer index lookup on device)
  // ==========================================================================
  if (nRanks_ > 0) {
    peerIndexMapsDevice_ =
        std::make_unique<meta::comms::DeviceBuffer>(nRanks_ * sizeof(int));

    std::vector<int> nvlMap(nRanks_, -1);

    for (int i = 0; i < nNvlPeers; ++i) {
      nvlMap[nvlPeerRanks_[i]] = i;
    }

    auto* base = static_cast<int*>(peerIndexMapsDevice_->get());
    CUDA_CHECK(cudaMemcpy(
        base, nvlMap.data(), nRanks_ * sizeof(int), cudaMemcpyDefault));
  }

  // ==========================================================================
  // Barrier buffers (dedicated, flat accumulation model)
  // ==========================================================================
  if (config_.barrierCount > 0) {
    if (nNvlPeers > 0) {
      auto nvlBarrierSize =
          getSignalBufferSize(static_cast<int>(config_.barrierCount));
      auto nvlBootstrap = transport_.nvl_bootstrap();
      if (nvlBootstrap) {
        nvlBarrierHandler_ = std::make_unique<GpuMemHandler>(
            nvlBootstrap, nvlLocalRank_, nvlNRanks_, nvlBarrierSize);
        CUDA_CHECK(cudaMemset(
            nvlBarrierHandler_->getLocalDeviceMemPtr(), 0, nvlBarrierSize));
      }

      nvlBarrierPeerPtrsDevice_ = std::make_unique<meta::comms::DeviceBuffer>(
          nNvlPeers * sizeof(SignalState*));
    }

    if (nIbgdaPeers > 0) {
      auto size = config_.barrierCount * sizeof(uint64_t);
      ibgdaBarrierLocalBuf_ = allocateIbgdaBuffer(size);

      ibgdaBarrierRemoteBufsDevice_ =
          std::make_unique<meta::comms::DeviceBuffer>(
              nIbgdaPeers * sizeof(IbgdaRemoteBuffer));
    }
  }

  // ==========================================================================
  // Per-peer signal buffers
  // ==========================================================================
  if (config_.peerSignalCount > 0) {
    if (nNvlPeers > 0) {
      auto nvlPeerSignalSize = getSignalBufferSize(
          static_cast<int>(nNvlPeers * config_.peerSignalCount));
      auto nvlBootstrap = transport_.nvl_bootstrap();
      if (nvlBootstrap) {
        nvlPeerSignalHandler_ = std::make_unique<GpuMemHandler>(
            nvlBootstrap, nvlLocalRank_, nvlNRanks_, nvlPeerSignalSize);
        CUDA_CHECK(cudaMemset(
            nvlPeerSignalHandler_->getLocalDeviceMemPtr(),
            0,
            nvlPeerSignalSize));
      }

      nvlPeerSignalSpansDevice_ = std::make_unique<meta::comms::DeviceBuffer>(
          nNvlPeers * sizeof(DeviceSpan<SignalState>));
    }

    if (nIbgdaPeers > 0) {
      auto size = nIbgdaPeers * config_.peerSignalCount * sizeof(uint64_t);
      ibgdaPeerSignalLocalBuf_ = allocateIbgdaBuffer(size);

      ibgdaPeerSignalRemoteBufsDevice_ =
          std::make_unique<meta::comms::DeviceBuffer>(
              nIbgdaPeers * sizeof(IbgdaRemoteBuffer));
    }
  }

  // ==========================================================================
  // Per-peer counter buffers (IBGDA-only)
  // ==========================================================================
  if (config_.peerCounterCount > 0 && nIbgdaPeers > 0) {
    auto size = nIbgdaPeers * config_.peerCounterCount * sizeof(uint64_t);
    ibgdaPeerCounterLocalBuf_ = allocateIbgdaBuffer(size);
  }
}

HostWindow::~HostWindow() {
  // Free IBGDA buffers: deregister (only if registered) then cudaFree.
  // lkey is only populated during exchange() via registerIbgdaBuffer(),
  // so check lkey != NetworkLKey{} to avoid deregistering unregistered buffers.
  if (ibgdaBarrierLocalBuf_.ptr) {
    if (ibgdaBarrierLocalBuf_.lkey != NetworkLKey{}) {
      transport_.localDeregisterIbgdaBuffer(ibgdaBarrierLocalBuf_.ptr);
    }
    cudaFree(ibgdaBarrierLocalBuf_.ptr);
  }
  if (ibgdaPeerSignalLocalBuf_.ptr) {
    if (ibgdaPeerSignalLocalBuf_.lkey != NetworkLKey{}) {
      transport_.localDeregisterIbgdaBuffer(ibgdaPeerSignalLocalBuf_.ptr);
    }
    cudaFree(ibgdaPeerSignalLocalBuf_.ptr);
  }
  if (ibgdaPeerCounterLocalBuf_.ptr) {
    if (ibgdaPeerCounterLocalBuf_.lkey != NetworkLKey{}) {
      transport_.localDeregisterIbgdaBuffer(ibgdaPeerCounterLocalBuf_.ptr);
    }
    cudaFree(ibgdaPeerCounterLocalBuf_.ptr);
  }

  // Clean up IBGDA buffer registrations
  for (auto* ptr : registeredLocalBuffers_) {
    transport_.localDeregisterIbgdaBuffer(ptr);
  }
  if (!exchangedNvlMappedPtrs_.empty()) {
    transport_.unmapNvlBuffers(exchangedNvlMappedPtrs_);
  }

  // NVL signal/barrier buffers are freed by GpuMemHandler destructors (RAII)
  // DeviceBuffers are freed by DeviceBuffer destructors (RAII)
}

void* HostWindow::get_nvlink_address(int peer, std::size_t offset) const {
  if (exchangedNvlMappedPtrs_.empty() || peer == myRank_ || peer < 0 ||
      peer >= nRanks_) {
    return nullptr;
  }
  // Find peer in NVL peer list (sorted by global rank).
  for (int i = 0; i < static_cast<int>(nvlPeerRanks_.size()); ++i) {
    if (nvlPeerRanks_[i] == peer) {
      // Map nvlPeerIdx → nvlLocalRank (skip self's slot).
      int nvlLocal = (i < nvlLocalRank_) ? i : (i + 1);
      auto* base = static_cast<char*>(exchangedNvlMappedPtrs_[nvlLocal]);
      return base ? base + offset : nullptr;
    }
  }
  return nullptr;
}

void HostWindow::exchange() {
  if (exchanged_) {
    throw std::runtime_error("HostWindow::exchange() called more than once");
  }

  int nNvlPeers = static_cast<int>(nvlPeerRanks_.size());
  int nIbgdaPeers = static_cast<int>(ibgdaPeerRanks_.size());

  // ==========================================================================
  // NVL barrier exchange
  // ==========================================================================
  if (nvlBarrierHandler_) {
    nvlBarrierHandler_->exchangeMemPtrs();

    std::vector<SignalState*> peerPtrs(nNvlPeers);
    for (int nvlLocalPeer = 0; nvlLocalPeer < nvlNRanks_; ++nvlLocalPeer) {
      if (nvlLocalPeer == nvlLocalRank_) {
        continue;
      }
      int peerIdx =
          (nvlLocalPeer < nvlLocalRank_) ? nvlLocalPeer : (nvlLocalPeer - 1);
      peerPtrs[peerIdx] = static_cast<SignalState*>(
          nvlBarrierHandler_->getPeerDeviceMemPtr(nvlLocalPeer));
    }

    CUDA_CHECK(cudaMemcpy(
        nvlBarrierPeerPtrsDevice_->get(),
        peerPtrs.data(),
        nNvlPeers * sizeof(SignalState*),
        cudaMemcpyDefault));
  }

  // ==========================================================================
  // NVL per-peer signal exchange
  // ==========================================================================
  if (nvlPeerSignalHandler_) {
    nvlPeerSignalHandler_->exchangeMemPtrs();

    auto signalCount = static_cast<int>(config_.peerSignalCount);
    std::vector<DeviceSpan<SignalState>> peerSpans;
    peerSpans.reserve(nNvlPeers);

    for (int nvlLocalPeer = 0; nvlLocalPeer < nvlNRanks_; ++nvlLocalPeer) {
      if (nvlLocalPeer == nvlLocalRank_) {
        continue;
      }

      auto* peerBase = static_cast<SignalState*>(
          nvlPeerSignalHandler_->getPeerDeviceMemPtr(nvlLocalPeer));
      int myIndexOnPeer =
          (nvlLocalRank_ < nvlLocalPeer) ? nvlLocalRank_ : (nvlLocalRank_ - 1);
      SignalState* myRowInPeer = peerBase + myIndexOnPeer * signalCount;
      peerSpans.emplace_back(myRowInPeer, signalCount);
    }

    CUDA_CHECK(cudaMemcpy(
        nvlPeerSignalSpansDevice_->get(),
        peerSpans.data(),
        nNvlPeers * sizeof(DeviceSpan<SignalState>),
        cudaMemcpyDefault));
  }

  // ==========================================================================
  // IBGDA barrier exchange
  // ==========================================================================
  if (ibgdaBarrierLocalBuf_.ptr) {
    auto size = config_.barrierCount * sizeof(uint64_t);
    ibgdaBarrierLocalBuf_ =
        transport_.localRegisterIbgdaBuffer(ibgdaBarrierLocalBuf_.ptr, size);
    auto remoteBufs = transport_.exchangeIbgdaBuffer(ibgdaBarrierLocalBuf_);

    CUDA_CHECK(cudaMemcpy(
        ibgdaBarrierRemoteBufsDevice_->get(),
        remoteBufs.data(),
        nIbgdaPeers * sizeof(IbgdaRemoteBuffer),
        cudaMemcpyDefault));
  }

  // ==========================================================================
  // IBGDA per-peer signal exchange
  // ==========================================================================
  if (ibgdaPeerSignalLocalBuf_.ptr) {
    auto size = static_cast<int>(ibgdaPeerRanks_.size()) *
        config_.peerSignalCount * sizeof(uint64_t);
    ibgdaPeerSignalLocalBuf_ =
        transport_.localRegisterIbgdaBuffer(ibgdaPeerSignalLocalBuf_.ptr, size);
    auto remoteBufs = transport_.exchangeIbgdaBuffer(ibgdaPeerSignalLocalBuf_);

    // Pre-offset each peer's remote buffer to point to "my row" in their
    // inbox. This moves the skip-self index computation from the GPU hot
    // path (every signal_peer call) to this one-time host exchange.
    auto signalCount = static_cast<int>(config_.peerSignalCount);
    for (int i = 0; i < nIbgdaPeers; ++i) {
      int peerRank = ibgdaPeerRanks_[i];
      int myIdxOnPeer = (myRank_ < peerRank) ? myRank_ : (myRank_ - 1);
      remoteBufs[i].ptr =
          static_cast<uint64_t*>(remoteBufs[i].ptr) + myIdxOnPeer * signalCount;
    }

    CUDA_CHECK(cudaMemcpy(
        ibgdaPeerSignalRemoteBufsDevice_->get(),
        remoteBufs.data(),
        nIbgdaPeers * sizeof(IbgdaRemoteBuffer),
        cudaMemcpyDefault));
  }

  // ==========================================================================
  // IBGDA counter registration (local only, no exchange)
  // ==========================================================================
  if (ibgdaPeerCounterLocalBuf_.ptr) {
    auto size = static_cast<int>(ibgdaPeerRanks_.size()) *
        config_.peerCounterCount * sizeof(uint64_t);
    ibgdaPeerCounterLocalBuf_ = transport_.localRegisterIbgdaBuffer(
        ibgdaPeerCounterLocalBuf_.ptr, size);
  }

  if (userBuffer_ && userBufferSize_ > 0) {
    registerAndExchangeBuffer(userBuffer_, userBufferSize_);
  }

  exchanged_ = true;
}

std::optional<NetworkLKey> HostWindow::registerLocalBuffer(
    void* ptr,
    std::size_t size) {
  if (ibgdaPeerRanks_.empty()) {
    return std::nullopt;
  }
  auto ibgdaBuf = transport_.localRegisterIbgdaBuffer(ptr, size);
  registeredLocalBuffers_.push_back(ptr);
  return ibgdaBuf.lkey;
}

void HostWindow::registerAndExchangeBuffer(void* ptr, std::size_t size) {
  if (userBufferRegistered_) {
    throw std::runtime_error(
        "HostWindow::registerAndExchangeBuffer() called more than once. "
        "Each DeviceWindow supports exactly one exchanged dst buffer.");
  }
  userBufferRegistered_ = true;

  int nIbgdaPeers = static_cast<int>(ibgdaPeerRanks_.size());
  int nNvlPeers = static_cast<int>(nvlPeerRanks_.size());

  // IBGDA side: register + exchange
  if (nIbgdaPeers > 0) {
    auto ibgdaBuf = transport_.localRegisterIbgdaBuffer(ptr, size);
    registeredLocalBuffers_.push_back(ptr);
    auto remoteBufs = transport_.exchangeIbgdaBuffer(ibgdaBuf);
    for (int i = 0; i < nIbgdaPeers; ++i) {
      remoteRegistrations_.push_back(
          RemoteBufferRegistration{
              remoteBufs[i].ptr, size, remoteBufs[i].rkey});
    }
  }

  // NVL side: IPC exchange
  if (nNvlPeers > 0) {
    exchangedNvlMappedPtrs_ = transport_.exchangeNvlBuffer(ptr, size);

    // Upload NVL peer pointers to device for offset-based put/put_signal.
    // exchangedNvlMappedPtrs_ is indexed by NVL local rank; extract peers
    // in nvlPeerIdx order (skip self).
    std::vector<void*> nvlPeerPtrs;
    nvlPeerPtrs.reserve(nNvlPeers);
    for (int nvlLocal = 0; nvlLocal < nvlNRanks_; ++nvlLocal) {
      if (nvlLocal == nvlLocalRank_) {
        continue;
      }
      nvlPeerPtrs.push_back(exchangedNvlMappedPtrs_[nvlLocal]);
    }
    userNvlPeerPtrsDevice_ =
        std::make_unique<meta::comms::DeviceBuffer>(nNvlPeers * sizeof(void*));
    CUDA_CHECK(cudaMemcpy(
        userNvlPeerPtrsDevice_->get(),
        nvlPeerPtrs.data(),
        nNvlPeers * sizeof(void*),
        cudaMemcpyDefault));
  }

  uploadRegistrationsToDevice();
}

void HostWindow::uploadRegistrationsToDevice() {
  int nIbgdaPeers = static_cast<int>(ibgdaPeerRanks_.size());

  if (nIbgdaPeers > 0 && !remoteRegistrations_.empty()) {
    remoteRegistrationsDevice_ = std::make_unique<meta::comms::DeviceBuffer>(
        remoteRegistrations_.size() * sizeof(RemoteBufferRegistration));
    CUDA_CHECK(cudaMemcpy(
        remoteRegistrationsDevice_->get(),
        remoteRegistrations_.data(),
        remoteRegistrations_.size() * sizeof(RemoteBufferRegistration),
        cudaMemcpyDefault));
  }
}

DeviceWindow HostWindow::getDeviceWindow() const {
  if (!exchanged_) {
    throw std::runtime_error(
        "HostWindow::getDeviceWindow() called before exchange()");
  }

  int nNvlPeers = static_cast<int>(nvlPeerRanks_.size());
  int nIbgdaPeers = static_cast<int>(ibgdaPeerRanks_.size());

  // DeviceSpan has deleted copy-assignment, so we use placement new
  // for all DeviceSpan-typed members.
  DeviceWindow dw;
  new (&dw.handle_) MultiPeerDeviceHandle(transport_.get_device_handle());
  dw.nNvlPeers_ = nNvlPeers;
  dw.nIbgdaPeers_ = nIbgdaPeers;

  // Pre-computed peer index maps (NVL only; IBGDA uses rank_to_peer_index())
  if (peerIndexMapsDevice_) {
    auto* base = static_cast<int*>(peerIndexMapsDevice_->get());
    new (&dw.rankToNvlPeerIndex_) DeviceSpan<int>(base, nRanks_);
  }

  // Per-peer signals
  dw.peerSignalCount_ = static_cast<int>(config_.peerSignalCount);
  if (nvlPeerSignalHandler_) {
    new (&dw.nvlPeerSignalInbox_) DeviceSpan<SignalState>(
        static_cast<SignalState*>(
            nvlPeerSignalHandler_->getLocalDeviceMemPtr()),
        nNvlPeers * static_cast<int>(config_.peerSignalCount));
  }
  if (nvlPeerSignalSpansDevice_) {
    new (&dw.nvlPeerSignalSpans_) DeviceSpan<DeviceSpan<SignalState>>(
        static_cast<DeviceSpan<SignalState>*>(nvlPeerSignalSpansDevice_->get()),
        nNvlPeers);
  }
  dw.ibgdaPeerSignalInbox_ =
      static_cast<uint64_t*>(ibgdaPeerSignalLocalBuf_.ptr);
  if (ibgdaPeerSignalRemoteBufsDevice_) {
    new (&dw.ibgdaPeerSignalRemoteBufs_) DeviceSpan<IbgdaRemoteBuffer>(
        static_cast<IbgdaRemoteBuffer*>(
            ibgdaPeerSignalRemoteBufsDevice_->get()),
        nIbgdaPeers);
  }

  // Per-peer counters
  dw.peerCounterCount_ = static_cast<int>(config_.peerCounterCount);
  dw.ibgdaPeerCounterBuf_ =
      static_cast<uint64_t*>(ibgdaPeerCounterLocalBuf_.ptr);
  dw.ibgdaPeerCounterLkey_ = ibgdaPeerCounterLocalBuf_.lkey;

  // Barrier
  dw.barrierCount_ = static_cast<int>(config_.barrierCount);
  if (nvlBarrierHandler_) {
    dw.nvlBarrierInbox_ =
        static_cast<SignalState*>(nvlBarrierHandler_->getLocalDeviceMemPtr());
  }
  if (nvlBarrierPeerPtrsDevice_) {
    new (&dw.nvlBarrierPeerPtrs_) DeviceSpan<SignalState*>(
        static_cast<SignalState**>(nvlBarrierPeerPtrsDevice_->get()),
        nNvlPeers);
  }
  dw.ibgdaBarrierInbox_ = static_cast<uint64_t*>(ibgdaBarrierLocalBuf_.ptr);
  if (ibgdaBarrierRemoteBufsDevice_) {
    new (&dw.ibgdaBarrierRemoteBufs_) DeviceSpan<IbgdaRemoteBuffer>(
        static_cast<IbgdaRemoteBuffer*>(ibgdaBarrierRemoteBufsDevice_->get()),
        nIbgdaPeers);
  }

  // Remote buffer registration table (for generic put/put_signal)
  if (remoteRegistrationsDevice_ && !remoteRegistrations_.empty()) {
    new (&dw.remoteBufferRegistry_) DeviceSpan<RemoteBufferRegistration>(
        static_cast<RemoteBufferRegistration*>(
            remoteRegistrationsDevice_->get()),
        nIbgdaPeers);
  }

  // Window buffer NVL peer pointers (for offset-based put/put_signal).
  // IBGDA uses remoteBufferRegistry_ (the window buffer is the exchanged buf).
  if (userNvlPeerPtrsDevice_ && nNvlPeers > 0) {
    new (&dw.windowNvlPeerPtrs_) DeviceSpan<void*>(
        static_cast<void**>(userNvlPeerPtrsDevice_->get()), nNvlPeers);
  }

  return dw;
}

} // namespace comms::pipes
