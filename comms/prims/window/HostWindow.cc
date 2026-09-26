// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/window/HostWindow.h"

#include <stdexcept>
#include <utility>
#include <vector>

#include "comms/prims/core/SignalState.cuh"
#include "comms/prims/memory/GpuMemHandler.h"
#include "comms/prims/transport/MultiPeerIbTransportInternal.h"
#include "comms/prims/transport/MultiPeerTransport.h"
#include "comms/prims/transport/ibgda/MultipeerIbgdaTransportInternal.h"
#include "comms/prims/window/DeviceWindow.cuh"
#include "comms/utils/checks.h"

namespace comms::prims {

namespace {

// Allocate zeroed GPU memory via cudaMalloc and wrap in IbgdaLocalBuffer.
// The .ptr field holds the raw GPU pointer; .lkey_per_device are unset until
// localRegisterIbgdaBuffer() is called in exchange().
IbgdaLocalBuffer allocateIbgdaBuffer(std::size_t size) {
  void* ptr = nullptr;
  CUDA_CHECK(cudaMalloc(&ptr, size));
  CUDA_CHECK(cudaMemset(ptr, 0, size));
  return IbgdaLocalBuffer(ptr, NetworkLKeys{});
}

void validateOwnedBuffer(
    const std::shared_ptr<void>& buffer,
    std::size_t size,
    bool allowEmpty) {
  if (allowEmpty && buffer == nullptr && size == 0) {
    return;
  }
  if (buffer == nullptr || size == 0) {
    throw std::invalid_argument(
        "HostWindow buffer ownership and nonzero size must be provided together");
  }
}

} // namespace

HostWindow::HostWindow(
    MultiPeerTransport& transport,
    const WindowConfig& config)
    : HostWindow(transport, config, std::shared_ptr<void>{}, 0) {}

HostWindow::HostWindow(
    MultiPeerTransport& transport,
    const WindowConfig& config,
    std::shared_ptr<void> userBuffer,
    std::size_t userBufferSize)
    : transport_(transport),
      myRank_(transport.my_rank()),
      nRanks_(transport.n_ranks()),
      config_(config),
      nvlPeerRanks_(transport.nvl_peer_ranks()),
      ibgdaPeerRanks_(transport.ib_peer_ranks()),
      nvlLocalRank_(transport.nvl_local_rank()),
      nvlNRanks_(transport.nvl_n_ranks()),
      userBuffer_(std::move(userBuffer)),
      userBufferSize_(userBufferSize) {
  validateOwnedBuffer(userBuffer_, userBufferSize_, /*allowEmpty=*/true);

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
      nvlPeerSignalInboxSize_ = nvlPeerSignalSize;
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
      ibgdaPeerSignalInboxSize_ = size;
      ibgdaPeerSignalLocalBuf_ = allocateIbgdaBuffer(size);

      ibgdaPeerSignalRemoteBufsDevice_ =
          std::make_unique<meta::comms::DeviceBuffer>(
              nIbgdaPeers * sizeof(IbgdaRemoteBuffer));
    }
  }

  // ==========================================================================
  // Per-peer counter buffers (IB-only)
  // ==========================================================================
  if (config_.peerCounterCount > 0 && nIbgdaPeers > 0) {
    auto size = nIbgdaPeers * config_.peerCounterCount * sizeof(uint64_t);
    ibgdaPeerCounterLocalBuf_ =
        transport_.allocateIbCounterBuffer(size, &ibgdaPeerCounterHostPtr_);
  }
}

HostWindow::~HostWindow() {
  static_cast<void>(detail::releaseResourcesOrRetainKeepAlives(
      transport_.ibgda_resources_quarantined(),
      callerBufferLifetimeQuarantineRequired_,
      callerBufferKeepAlives_,
      [this]() noexcept -> bool {
        bool allMrsDeregistered = true;
        const auto deregisterOwnedBuffer =
            [this, &allMrsDeregistered](IbgdaLocalBuffer& buffer) {
              if (buffer.ptr == nullptr || buffer.lkey_per_device.size == 0) {
                return;
              }
              const bool deregistered =
                  transport_.localDeregisterIbgdaBuffer(buffer.ptr);
              allMrsDeregistered = deregistered && allMrsDeregistered;
              if (deregistered) {
                buffer.lkey_per_device = NetworkLKeys{};
              }
            };

        deregisterOwnedBuffer(ibgdaBarrierLocalBuf_);
        deregisterOwnedBuffer(ibgdaPeerSignalLocalBuf_);
        deregisterOwnedBuffer(ibgdaPeerCounterLocalBuf_);
        for (auto* ptr : registeredLocalBuffers_) {
          const bool deregistered = transport_.localDeregisterIbgdaBuffer(ptr);
          allMrsDeregistered = deregistered && allMrsDeregistered;
        }

        if (!allMrsDeregistered) {
          return false;
        }

        if (ibgdaBarrierLocalBuf_.ptr != nullptr) {
          static_cast<void>(cudaFree(ibgdaBarrierLocalBuf_.ptr));
        }
        if (ibgdaPeerSignalLocalBuf_.ptr != nullptr) {
          static_cast<void>(cudaFree(ibgdaPeerSignalLocalBuf_.ptr));
        }
        if (ibgdaPeerCounterLocalBuf_.ptr != nullptr &&
            !transport_.freeIbCounterBuffer(
                ibgdaPeerCounterLocalBuf_, ibgdaPeerCounterHostPtr_)) {
          return false;
        }
        return true;
      }));

  if (!exchangedNvlMappedPtrs_.empty()) {
    transport_.unmapNvlBuffers(exchangedNvlMappedPtrs_);
  }

  // NVL signal/barrier buffers are freed by GpuMemHandler destructors (RAII)
  // DeviceBuffers are freed by DeviceBuffer destructors (RAII)
}

bool HostWindow::requiresProcessLifetimeQuarantine() const noexcept {
  return callerBufferLifetimeQuarantineRequired_ ||
      transport_.ibgda_resources_quarantined();
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

  detail::runWithProcessLifetimeQuarantineOnFailureAfterWindowExposure(
      ibgdaRkeysPossiblyExposed_,
      callerBufferPossiblyExposed_,
      [this]() { exchangeImpl(); },
      [this](std::string_view context) {
        transport_.quarantineIbgdaTransport(context);
      },
      [this](std::string_view) {
        callerBufferLifetimeQuarantineRequired_ = true;
      });
}

void HostWindow::exchangeImpl() {
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
    ibgdaRkeysPossiblyExposed_ = true;
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
    ibgdaRkeysPossiblyExposed_ = true;
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
    ibgdaPeerCounterLocalBuf_ =
        transport_.registerIbCounterBuffer(ibgdaPeerCounterLocalBuf_, size);
  }

  if (userBuffer_ && userBufferSize_ > 0) {
    registerAndExchangeBufferImpl(userBuffer_, userBufferSize_);
  }

  exchanged_ = true;
}

std::optional<NetworkLKeys> HostWindow::registerLocalBuffer(
    std::shared_ptr<void> buffer,
    std::size_t size) {
  validateOwnedBuffer(buffer, size, /*allowEmpty=*/false);
  if (ibgdaPeerRanks_.empty()) {
    return std::nullopt;
  }
  if (transport_.ibgda_resources_quarantined()) {
    throw std::runtime_error(
        "HostWindow::registerLocalBuffer: IB transport is quarantined");
  }
  callerBufferKeepAlives_.reserve(callerBufferKeepAlives_.size() + 1);
  auto holder = std::make_unique<std::shared_ptr<void>>(std::move(buffer));
  void* const ptr = holder->get();
  registeredLocalBuffers_.reserve(registeredLocalBuffers_.size() + 1);
  callerBufferKeepAlives_.push_back(std::move(holder));
  IbgdaLocalBuffer ibgdaBuf;
  try {
    ibgdaBuf = transport_.localRegisterIbgdaBuffer(ptr, size);
  } catch (...) {
    callerBufferKeepAlives_.pop_back();
    throw;
  }
  registeredLocalBuffers_.push_back(ptr);
  return ibgdaBuf.lkey_per_device;
}

void HostWindow::reset_signals(cudaStream_t stream) const {
  // Reset IBGDA signal inbox (written by remote NICs via RDMA atomics)
  if (ibgdaPeerSignalLocalBuf_.ptr && ibgdaPeerSignalInboxSize_ > 0) {
    CUDA_CHECK(cudaMemsetAsync(
        ibgdaPeerSignalLocalBuf_.ptr, 0, ibgdaPeerSignalInboxSize_, stream));
  }
  // Reset NVL signal inbox (written by NVLink peers via store)
  if (nvlPeerSignalHandler_ && nvlPeerSignalInboxSize_ > 0) {
    auto* ptr = nvlPeerSignalHandler_->getLocalDeviceMemPtr();
    if (ptr) {
      CUDA_CHECK(cudaMemsetAsync(ptr, 0, nvlPeerSignalInboxSize_, stream));
    }
  }
}

void HostWindow::registerAndExchangeBuffer(
    std::shared_ptr<void> buffer,
    std::size_t size) {
  validateOwnedBuffer(buffer, size, /*allowEmpty=*/false);
  if (userBufferRegistered_) {
    throw std::runtime_error(
        "HostWindow::registerAndExchangeBuffer() called more than once. "
        "Each DeviceWindow supports exactly one exchanged dst buffer.");
  }

  detail::runWithProcessLifetimeQuarantineOnFailureAfterWindowExposure(
      ibgdaRkeysPossiblyExposed_,
      callerBufferPossiblyExposed_,
      [this, buffer = std::move(buffer), size]() mutable {
        registerAndExchangeBufferImpl(std::move(buffer), size);
      },
      [this](std::string_view context) {
        transport_.quarantineIbgdaTransport(context);
      },
      [this](std::string_view) {
        callerBufferLifetimeQuarantineRequired_ = true;
      });
}

void HostWindow::registerAndExchangeBufferImpl(
    std::shared_ptr<void> buffer,
    std::size_t size) {
  int nIbgdaPeers = static_cast<int>(ibgdaPeerRanks_.size());
  int nNvlPeers = static_cast<int>(nvlPeerRanks_.size());
  if (nIbgdaPeers > 0 && transport_.ibgda_resources_quarantined()) {
    throw std::runtime_error(
        "HostWindow::registerAndExchangeBuffer: IB transport is quarantined");
  }
  userBufferRegistered_ = true;
  registeredLocalBuffers_.reserve(
      registeredLocalBuffers_.size() + (nIbgdaPeers > 0 ? 1 : 0));
  callerBufferKeepAlives_.reserve(callerBufferKeepAlives_.size() + 1);
  auto holder = std::make_unique<std::shared_ptr<void>>(std::move(buffer));
  void* const ptr = holder->get();
  callerBufferKeepAlives_.push_back(std::move(holder));

  try {
    // IBGDA side: register + exchange
    if (nIbgdaPeers > 0) {
      auto ibgdaBuf = transport_.localRegisterIbgdaBuffer(ptr, size);
      registeredLocalBuffers_.push_back(ptr);
      ibgdaRkeysPossiblyExposed_ = true;
      callerBufferPossiblyExposed_ = true;
      auto remoteBufs = transport_.exchangeIbgdaBuffer(ibgdaBuf);
      for (const auto& remoteBuf : remoteBufs) {
        remoteRegistrations_.emplace_back(
            remoteBuf.ptr, size, remoteBuf.rkey_per_device);
      }
    }

    // NVL side: IPC exchange
    if (nNvlPeers > 0) {
      exchangedNvlMappedPtrs_ =
          transport_.exchangeNvlBuffer(ptr, &callerBufferPossiblyExposed_);

      // Upload NVL peer pointers to device for offset-based put/put_signal.
      // exchangedNvlMappedPtrs_ is indexed by NVL local rank; extract peers
      // in nvlPeerIdx order (skip self).
      std::vector<void*> nvlPeerPtrs;
      nvlPeerPtrs.reserve(nNvlPeers);
      for (int nvlLocal = 0; nvlLocal < nvlNRanks_; ++nvlLocal) {
        if (nvlLocal == nvlLocalRank_) {
          continue;
        }
        nvlPeerPtrs.push_back(exchangedNvlMappedPtrs_.at(nvlLocal));
      }
      userNvlPeerPtrsDevice_ = std::make_unique<meta::comms::DeviceBuffer>(
          nNvlPeers * sizeof(void*));
      CUDA_CHECK(cudaMemcpy(
          userNvlPeerPtrsDevice_->get(),
          nvlPeerPtrs.data(),
          nNvlPeers * sizeof(void*),
          cudaMemcpyDefault));
    }

    uploadRegistrationsToDevice();
  } catch (...) {
    if (!callerBufferPossiblyExposed_) {
      callerBufferKeepAlives_.pop_back();
    }
    throw;
  }

  if (!callerBufferPossiblyExposed_) {
    callerBufferKeepAlives_.pop_back();
  }
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

DeviceWindow HostWindow::buildDeviceWindowImpl(
    MultiPeerDeviceHandle handle) const {
  if (!exchanged_) {
    throw std::runtime_error(
        "HostWindow::getDeviceWindow() called before exchange()");
  }

  int nNvlPeers = static_cast<int>(nvlPeerRanks_.size());
  int nIbgdaPeers = static_cast<int>(ibgdaPeerRanks_.size());

  // DeviceSpan has deleted copy-assignment, so we use placement new
  // for all DeviceSpan-typed members.
  DeviceWindow dw;
  new (&dw.handle_) MultiPeerDeviceHandle(handle);
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
  if (ibgdaPeerCounterLocalBuf_.ptr) {
    dw.ibgdaPeerCounterBuf_ =
        static_cast<uint64_t*>(ibgdaPeerCounterLocalBuf_.ptr);
    dw.ibgdaPeerCounterHostBuf_ =
        static_cast<uint64_t*>(ibgdaPeerCounterHostPtr_);
    dw.ibgdaPeerCounterLkeys_ = ibgdaPeerCounterLocalBuf_.lkey_per_device;
  }

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

  // Window buffer NVL peer pointers (for offset-based put/put_signal)
  if (userNvlPeerPtrsDevice_ && nNvlPeers > 0) {
    new (&dw.windowNvlPeerPtrs_) DeviceSpan<void*>(
        static_cast<void**>(userNvlPeerPtrsDevice_->get()), nNvlPeers);
  }

  return dw;
}

DeviceWindow HostWindow::getDeviceWindow() const {
  return buildDeviceWindowImpl(transport_.get_device_handle(ibgdaPeerRanks_));
}

DeviceWindow HostWindow::getDeviceWindow(const std::vector<int>& peers) {
  return buildDeviceWindowImpl(transport_.get_device_handle(peers));
}

} // namespace comms::prims
