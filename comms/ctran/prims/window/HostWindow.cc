// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/ctran/prims/window/HostWindow.h"

#include <stdexcept>
#include <string>
#include <vector>

#include "comms/ctran/prims/GpuMemHandler.h"
#include "comms/ctran/prims/MultiPeerTransport.h"
#include "comms/ctran/prims/SignalState.cuh"
#include "comms/ctran/prims/window/DeviceWindow.cuh"
#include "comms/utils/checks.h"

namespace ctran::prims {

namespace {

#ifdef __HIP_PLATFORM_AMD__
constexpr auto kGpuMemcpyDefault = hipMemcpyDefault;

void gpuCheck(hipError_t err, const char* expr, const char* file, int line) {
  if (err != hipSuccess) {
    throw std::runtime_error(
        std::string("HIP error: ") + file + ":" + std::to_string(line) + " " +
        expr + " " + hipGetErrorString(err));
  }
}

#define HOST_WINDOW_GPU_CHECK(expr) gpuCheck((expr), #expr, __FILE__, __LINE__)

hipError_t gpuMalloc(void** ptr, std::size_t size) {
  return hipMalloc(ptr, size);
}

hipError_t gpuMemset(void* ptr, int value, std::size_t size) {
  return hipMemset(ptr, value, size);
}

hipError_t
gpuMemcpy(void* dst, const void* src, std::size_t count, hipMemcpyKind kind) {
  return hipMemcpy(dst, src, count, kind);
}

hipError_t gpuMemsetAsync(
    void* ptr,
    int value,
    std::size_t size,
    HostWindowStream stream) {
  return hipMemsetAsync(ptr, value, size, stream);
}

void gpuFree(void* ptr) {
  (void)hipFree(ptr);
}
#else
constexpr auto kGpuMemcpyDefault = cudaMemcpyDefault;

#define HOST_WINDOW_GPU_CHECK(expr) CUDA_CHECK(expr)

cudaError_t gpuMalloc(void** ptr, std::size_t size) {
  return cudaMalloc(ptr, size);
}

cudaError_t gpuMemset(void* ptr, int value, std::size_t size) {
  return cudaMemset(ptr, value, size);
}

cudaError_t
gpuMemcpy(void* dst, const void* src, std::size_t count, cudaMemcpyKind kind) {
  return cudaMemcpy(dst, src, count, kind);
}

cudaError_t gpuMemsetAsync(
    void* ptr,
    int value,
    std::size_t size,
    HostWindowStream stream) {
  return cudaMemsetAsync(ptr, value, size, stream);
}

void gpuFree(void* ptr) {
  (void)cudaFree(ptr);
}
#endif

// Allocate zeroed GPU memory and wrap in IbgdaLocalBuffer.
// The .ptr field holds the raw GPU pointer; .lkey_per_device are unset until
// localRegisterIbgdaBuffer() is called in exchange().
IbgdaLocalBuffer allocateIbgdaBuffer(std::size_t size) {
  void* ptr = nullptr;
  HOST_WINDOW_GPU_CHECK(gpuMalloc(&ptr, size));
  HOST_WINDOW_GPU_CHECK(gpuMemset(ptr, 0, size));
  return IbgdaLocalBuffer(ptr, NetworkLKeys{});
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
    HOST_WINDOW_GPU_CHECK(gpuMemcpy(
        base, nvlMap.data(), nRanks_ * sizeof(int), kGpuMemcpyDefault));
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
        HOST_WINDOW_GPU_CHECK(gpuMemset(
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
        HOST_WINDOW_GPU_CHECK(gpuMemset(
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
  // Per-peer counter buffers (IBGDA-only)
  // ==========================================================================
  if (config_.peerCounterCount > 0 && nIbgdaPeers > 0) {
    auto size = nIbgdaPeers * config_.peerCounterCount * sizeof(uint64_t);
    ibgdaPeerCounterLocalBuf_ = allocateIbgdaBuffer(size);
  }
}

HostWindow::~HostWindow() {
  // Free IBGDA buffers: deregister (only if registered) then free GPU memory.
  // lkey is only populated during exchange() via registerIbgdaBuffer(),
  // so check lkey != NetworkLKey{} to avoid deregistering unregistered buffers.
  if (ibgdaBarrierLocalBuf_.ptr) {
    if (ibgdaBarrierLocalBuf_.lkey_per_device.size > 0) {
      transport_.localDeregisterIbgdaBuffer(ibgdaBarrierLocalBuf_.ptr);
    }
    gpuFree(ibgdaBarrierLocalBuf_.ptr);
  }
  if (ibgdaPeerSignalLocalBuf_.ptr) {
    if (ibgdaPeerSignalLocalBuf_.lkey_per_device.size > 0) {
      transport_.localDeregisterIbgdaBuffer(ibgdaPeerSignalLocalBuf_.ptr);
    }
    gpuFree(ibgdaPeerSignalLocalBuf_.ptr);
  }
  if (ibgdaPeerCounterLocalBuf_.ptr) {
    if (ibgdaPeerCounterLocalBuf_.lkey_per_device.size > 0) {
      transport_.localDeregisterIbgdaBuffer(ibgdaPeerCounterLocalBuf_.ptr);
    }
    gpuFree(ibgdaPeerCounterLocalBuf_.ptr);
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

    HOST_WINDOW_GPU_CHECK(gpuMemcpy(
        nvlBarrierPeerPtrsDevice_->get(),
        peerPtrs.data(),
        nNvlPeers * sizeof(SignalState*),
        kGpuMemcpyDefault));
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

    HOST_WINDOW_GPU_CHECK(gpuMemcpy(
        nvlPeerSignalSpansDevice_->get(),
        peerSpans.data(),
        nNvlPeers * sizeof(DeviceSpan<SignalState>),
        kGpuMemcpyDefault));
  }

  // ==========================================================================
  // IBGDA barrier exchange
  // ==========================================================================
  if (ibgdaBarrierLocalBuf_.ptr) {
    auto size = config_.barrierCount * sizeof(uint64_t);
    ibgdaBarrierLocalBuf_ =
        transport_.localRegisterIbgdaBuffer(ibgdaBarrierLocalBuf_.ptr, size);
    auto remoteBufs = transport_.exchangeIbgdaBuffer(ibgdaBarrierLocalBuf_);

    HOST_WINDOW_GPU_CHECK(gpuMemcpy(
        ibgdaBarrierRemoteBufsDevice_->get(),
        remoteBufs.data(),
        nIbgdaPeers * sizeof(IbgdaRemoteBuffer),
        kGpuMemcpyDefault));
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

    HOST_WINDOW_GPU_CHECK(gpuMemcpy(
        ibgdaPeerSignalRemoteBufsDevice_->get(),
        remoteBufs.data(),
        nIbgdaPeers * sizeof(IbgdaRemoteBuffer),
        kGpuMemcpyDefault));
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

std::optional<NetworkLKeys> HostWindow::registerLocalBuffer(
    void* ptr,
    std::size_t size) {
  if (ibgdaPeerRanks_.empty()) {
    return std::nullopt;
  }
  auto ibgdaBuf = transport_.localRegisterIbgdaBuffer(ptr, size);
  registeredLocalBuffers_.push_back(ptr);
  return ibgdaBuf.lkey_per_device;
}

void HostWindow::reset_signals(HostWindowStream stream) const {
  // Reset IBGDA signal inbox (written by remote NICs via RDMA atomics)
  if (ibgdaPeerSignalLocalBuf_.ptr && ibgdaPeerSignalInboxSize_ > 0) {
    HOST_WINDOW_GPU_CHECK(gpuMemsetAsync(
        ibgdaPeerSignalLocalBuf_.ptr, 0, ibgdaPeerSignalInboxSize_, stream));
  }
  // Reset NVL signal inbox (written by NVLink peers via store)
  if (nvlPeerSignalHandler_ && nvlPeerSignalInboxSize_ > 0) {
    auto* ptr = nvlPeerSignalHandler_->getLocalDeviceMemPtr();
    if (ptr) {
      HOST_WINDOW_GPU_CHECK(
          gpuMemsetAsync(ptr, 0, nvlPeerSignalInboxSize_, stream));
    }
  }
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
    for (const auto& remoteBuf : remoteBufs) {
      remoteRegistrations_.emplace_back(
          remoteBuf.ptr, size, remoteBuf.rkey_per_device);
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
    HOST_WINDOW_GPU_CHECK(gpuMemcpy(
        userNvlPeerPtrsDevice_->get(),
        nvlPeerPtrs.data(),
        nNvlPeers * sizeof(void*),
        kGpuMemcpyDefault));
  }

  uploadRegistrationsToDevice();
}

void HostWindow::uploadRegistrationsToDevice() {
  int nIbgdaPeers = static_cast<int>(ibgdaPeerRanks_.size());

  if (nIbgdaPeers > 0 && !remoteRegistrations_.empty()) {
    remoteRegistrationsDevice_ = std::make_unique<meta::comms::DeviceBuffer>(
        remoteRegistrations_.size() * sizeof(RemoteBufferRegistration));
    HOST_WINDOW_GPU_CHECK(gpuMemcpy(
        remoteRegistrationsDevice_->get(),
        remoteRegistrations_.data(),
        remoteRegistrations_.size() * sizeof(RemoteBufferRegistration),
        kGpuMemcpyDefault));
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
  return buildDeviceWindowImpl(transport_.get_device_handle());
}

DeviceWindow HostWindow::getDeviceWindow(const std::vector<int>& peers) {
  return buildDeviceWindowImpl(transport_.get_device_handle(peers));
}

} // namespace ctran::prims
