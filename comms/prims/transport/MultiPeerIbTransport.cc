// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/transport/MultiPeerIbTransport.h"

#include <cerrno>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

#include <fmt/core.h>
#include <folly/ScopeGuard.h>
#include <glog/logging.h>

#include "comms/ctran/ibverbx/Ibverbx.h"
#include "comms/ctran/ibverbx/IbverbxSymbols.h"
#include "comms/prims/transport/rdma/NicDiscovery.h"
// GPU DMA-BUF export for MR registration. Generic (no DOCA context): on NVIDIA
// it is cuMemGetHandleForAddressRange via DocaHostUtils (with the CUDA driver
// address-range lookup from CudaDriverLazy); on AMD it is the HSA path provided
// through DocaCompat.
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>

#include "comms/prims/transport/amd/DocaCompat.h"
// meta::comms::DeviceBuffer (HIP shim) for the send/recv staging bulks.
#include "comms/prims/transport/amd/HipHostCompat.h"
#else
#include <cuda_runtime.h>

#include "comms/prims/platform/CudaDriverLazy.h"
#include "comms/prims/platform/DocaHostUtils.h"
// meta::comms::DeviceBuffer (CUDA RAII) for the send/recv staging bulks.
#include "comms/utils/CudaRAII.h"
#endif

namespace comms::prims {

namespace {
constexpr int kDefaultGidIndex = 3; // Default RoCE GID index

#ifdef __HIP_PLATFORM_AMD__
using SlotGpuError = hipError_t;
constexpr SlotGpuError kSlotGpuSuccess = hipSuccess;

const char* slotGpuGetErrorString(SlotGpuError err) {
  return hipGetErrorString(err);
}

SlotGpuError slotGpuMalloc(void** ptr, std::size_t bytes) {
  return hipMalloc(ptr, bytes);
}

SlotGpuError slotGpuFree(void* ptr) {
  return hipFree(ptr);
}

SlotGpuError slotGpuMemset(void* ptr, int value, std::size_t bytes) {
  return hipMemset(ptr, value, bytes);
}

SlotGpuError slotHostPinnedAlloc(void** ptr, std::size_t bytes) {
  return hipHostMalloc(ptr, bytes, hipHostMallocMapped);
}

SlotGpuError slotHostGetDevicePointer(void** devicePtr, void* hostPtr) {
  return hipHostGetDevicePointer(devicePtr, hostPtr, 0);
}

SlotGpuError slotHostFree(void* ptr) {
  return hipHostFree(ptr);
}
#else
using SlotGpuError = cudaError_t;
constexpr SlotGpuError kSlotGpuSuccess = cudaSuccess;

const char* slotGpuGetErrorString(SlotGpuError err) {
  return cudaGetErrorString(err);
}

SlotGpuError slotGpuMalloc(void** ptr, std::size_t bytes) {
  return cudaMalloc(ptr, bytes);
}

SlotGpuError slotGpuFree(void* ptr) {
  return cudaFree(ptr);
}

SlotGpuError slotGpuMemset(void* ptr, int value, std::size_t bytes) {
  return cudaMemset(ptr, value, bytes);
}

SlotGpuError slotHostPinnedAlloc(void** ptr, std::size_t bytes) {
  return cudaHostAlloc(ptr, bytes, cudaHostAllocMapped);
}

SlotGpuError slotHostGetDevicePointer(void** devicePtr, void* hostPtr) {
  return cudaHostGetDevicePointer(devicePtr, hostPtr, 0);
}

SlotGpuError slotHostFree(void* ptr) {
  return cudaFreeHost(ptr);
}
#endif

void checkSlotGpu(SlotGpuError err, const std::string& what) {
  if (err != kSlotGpuSuccess) {
    throw std::runtime_error(
        fmt::format("{}: {}", what, slotGpuGetErrorString(err)));
  }
}

// Allocate via allocFn, check the GPU error, and verify the result is non-null,
// so callers never observe a null pointer on success (clears the nullability
// lint and centralizes the checks). Throws on any failure.
void* checkedSlotAlloc(
    SlotGpuError (*allocFn)(void**, std::size_t),
    std::size_t bytes,
    const std::string& what) {
  void* ptr = nullptr;
  checkSlotGpu(allocFn(&ptr, bytes), what);
  if (ptr == nullptr) {
    throw std::runtime_error(fmt::format("{}: allocation returned null", what));
  }
  return ptr;
}
} // namespace

MultiPeerIbTransportBase::MultiPeerIbTransportBase(
    int myRank,
    int nRanks,
    std::shared_ptr<meta::comms::IBootstrap> bootstrap,
    MultipeerIbTransportConfig config)
    : myRank_(myRank),
      nRanks_(nRanks),
      bootstrap_(std::move(bootstrap)),
      config_(std::move(config)) {
  if (config_.sendRecv.has_value() && config_.sendRecv->maxGroups == 0) {
    config_.sendRecv->maxGroups = config_.maxGroups;
  }
  if (myRank_ < 0 || myRank_ >= nRanks_) {
    throw std::invalid_argument("Invalid rank");
  }
  if (nRanks_ < 2) {
    throw std::invalid_argument("Need at least 2 ranks");
  }
  // RoCE GID index: config override, else the RoCEv2 default. Read by
  // openNics() (query_gid) and by backends when building address handles.
  gidIndex_ = config_.gidIndex.value_or(kDefaultGidIndex);

  // Resolve numNics_ from the available NIC sources. No numeric knob — the
  // count is implied by what the caller / topology actually provides:
  //   1. config.gpuNicMap[cudaDevice] populated → use its NIC list.
  //   2. Otherwise auto-discover via GpuNicDiscovery — every NIC at the
  //      best-affinity tier (same pathType + bandwidth + isDataDirect as
  //      the top candidate).
  // No silent fallback to 1: if a GPU is wired to N best-affinity NICs, the
  // transport must use all N. H100 (1 NIC) and GB200/GB300 (2 NICs) both get
  // the right count automatically; an unexpected count throws with a clear
  // hint.
  auto it = config_.gpuNicMap.find(config_.cudaDevice);
  int n = 0;
  const char* source = nullptr;
  if (it != config_.gpuNicMap.end() && !it->second.empty()) {
    n = static_cast<int>(it->second.size());
    source = "config.gpuNicMap";
  } else {
    // On AMD, the `DataDirectMode::Only` default triggers `ibv_reg_dmabuf_mr`
    // inside `augmentWithDataDirect()`, which is not exercised on AMD's
    // libibverbs path here. Force `Disabled` to skip the DataDirect probe.
#ifdef __HIP_PLATFORM_AMD__
    GpuNicDiscovery discovery(
        config_.cudaDevice, config_.ibHca, DataDirectMode::Disabled);
#else
    GpuNicDiscovery discovery(config_.cudaDevice, config_.ibHca);
#endif
    auto bestNics = discovery.getBestAffinityNics();
    if (bestNics.empty()) {
      throw std::runtime_error(
          fmt::format(
              "MultiPeerIbTransport: NIC auto-discovery returned no candidates "
              "for GPU {}; set config.gpuNicMap or config.ibHca to expose at "
              "least one NIC",
              config_.cudaDevice));
    }
    n = static_cast<int>(bestNics.size());
    source = "auto-discovery (best-affinity tier)";
  }
  if (n > kMaxNicsPerGpu) {
    throw std::runtime_error(
        fmt::format(
            "MultiPeerIbTransport: {} found {} NIC(s) for GPU {} but "
            "kMaxNicsPerGpu={}; raise kMaxNicsPerGpu or trim the source",
            source,
            n,
            config_.cudaDevice,
            kMaxNicsPerGpu));
  }
  numNics_ = n;
  VLOG(1) << "MultiPeerIbTransport: numNics_=" << numNics_
          << " (source=" << source << ")";
}

// Out-of-line so the unique_ptr<DeviceBuffer> send/recv members destruct
// against a complete type (DeviceBuffer is only forward-declared in the
// header).
MultiPeerIbTransportBase::~MultiPeerIbTransportBase() = default;

// ---- shared send/recv staging-ring lifecycle (eager mode) ----

const MultipeerIbTransportConfig::SendRecvConfig&
MultiPeerIbTransportBase::sendRecvConfig() const {
  if (!config_.sendRecv.has_value()) {
    throw std::runtime_error("MultiPeerIbTransport: send/recv not configured");
  }
  return *config_.sendRecv;
}

void MultiPeerIbTransportBase::validateSendRecvConfig() const {
  const auto& sr = sendRecvConfig();
  if (sr.pipelineDepth < 1) {
    throw std::invalid_argument(
        "MultiPeerIbTransport: sendRecv.pipelineDepth must be >= 1");
  }
  if (sr.maxGroups < 1) {
    throw std::invalid_argument(
        "MultiPeerIbTransport: sendRecv.maxGroups must be >= 1");
  }
  if (config_.dataBufferSize == 0) {
    throw std::invalid_argument(
        "MultiPeerIbTransport: dataBufferSize must be > 0 when sendRecv is "
        "enabled");
  }
  if ((config_.dataBufferSize / static_cast<std::size_t>(sr.maxGroups)) < 16) {
    throw std::invalid_argument(
        fmt::format(
            "MultiPeerIbTransport: dataBufferSize / maxGroups must be >= 16, "
            "got {} / {} = {}",
            config_.dataBufferSize,
            sr.maxGroups,
            config_.dataBufferSize / sr.maxGroups));
  }
}

std::size_t MultiPeerIbTransportBase::sendRecvStagingBytesPerPeer() const {
  const auto& sr = sendRecvConfig();
  return static_cast<std::size_t>(sr.pipelineDepth) * config_.dataBufferSize;
}

std::size_t MultiPeerIbTransportBase::sendRecvSignalBytesPerPeer() const {
  const auto& sr = sendRecvConfig();
  return 2 * static_cast<std::size_t>(sr.maxGroups) * sizeof(uint64_t);
}

std::size_t MultiPeerIbTransportBase::sendRecvCounterBytesPerPeer() const {
  const auto& sr = sendRecvConfig();
  return static_cast<std::size_t>(sr.maxGroups) * sizeof(uint64_t);
}

std::size_t MultiPeerIbTransportBase::sendRecvStateBytesPerPeer() const {
  const auto& sr = sendRecvConfig();
  return 2 * static_cast<std::size_t>(sr.maxGroups) *
      sizeof(IbSendRecvState::ProgressSlot);
}

IbSendRecvState MultiPeerIbTransportBase::sendRecvStateForPeer(
    int peerIndex) const {
  if (!config_.sendRecv.has_value() || sendRecvPeerBuffers_.empty() ||
      peerIndex < 0 ||
      peerIndex >= static_cast<int>(sendRecvPeerBuffers_.size())) {
    return {};
  }
  const auto& pb = sendRecvPeerBuffers_[peerIndex];
  return IbSendRecvState{
      .sendStagingBuf = pb.sendStaging,
      .recvStagingBuf = pb.remoteRecvStaging,
      .sendStagingPtr = static_cast<char*>(pb.sendStaging.ptr),
      .recvStagingPtr = static_cast<char*>(pb.recvStaging.ptr),
      .localSignalBuf = pb.signal,
      .remoteSignalBuf = pb.remoteSignal,
      .localCounterBuf = pb.counter,
      .localCounterCompletionBuf = pb.counterCompletion,
      .state = pb.state.value_or(DeviceSpan<IbSendRecvState::ProgressSlot>()),
      .maxGroups = config_.sendRecv->maxGroups,
      .pipelineDepth = config_.sendRecv->pipelineDepth,
      .dataBufferSize = config_.dataBufferSize,
  };
}

void MultiPeerIbTransportBase::allocateSendRecvBuffersEager(
    IbCounterStorage counterStorage) {
  if (!config_.sendRecv.has_value()) {
    return;
  }
  validateSendRecvConfig();

  const int numPeers = nRanks_ - 1;
  if (numPeers <= 0) {
    return;
  }
  sendRecvCounterStorage_ = counterStorage;

  const std::size_t stagingPerPeer = sendRecvStagingBytesPerPeer();
  const std::size_t signalPerPeer = sendRecvSignalBytesPerPeer();
  const std::size_t counterPerPeer = sendRecvCounterBytesPerPeer();
  const std::size_t statePerPeer = sendRecvStateBytesPerPeer();
  const auto stateSlotsPerPeer =
      static_cast<DeviceSpan<IbSendRecvState::ProgressSlot>::size_type>(
          2 * config_.sendRecv->maxGroups);

  auto allocateBulk = [&](std::size_t perPeer, const char* label) {
    auto buf = std::make_unique<meta::comms::DeviceBuffer>(perPeer * numPeers);
    checkSlotGpu(
        slotGpuMemset(buf->get(), 0, perPeer * numPeers),
        fmt::format("MultiPeerIbTransport: zero send/recv {}", label));
    return buf;
  };

  sendRecvPeerBuffers_.resize(numPeers);

  sendRecvSendStagingBulk_ = allocateBulk(stagingPerPeer, "send staging bulk");
  sendRecvRecvStagingBulk_ = allocateBulk(stagingPerPeer, "recv staging bulk");
  sendRecvSignalBulk_ = allocateBulk(signalPerPeer, "signal bulk");
  sendRecvStateBulk_ = allocateBulk(statePerPeer, "state bulk");

  auto sendStagingBulkReg = registerBuffer(
      sendRecvSendStagingBulk_->get(), stagingPerPeer * numPeers);
  sendRecvRecvStagingBulkReg_ = registerBuffer(
      sendRecvRecvStagingBulk_->get(), stagingPerPeer * numPeers);
  sendRecvSignalBulkReg_ =
      registerBuffer(sendRecvSignalBulk_->get(), signalPerPeer * numPeers);

  IbgdaLocalBuffer counterBulkBuf;
  IbgdaLocalBuffer counterCompletionBulkBuf;
  if (counterStorage == IbCounterStorage::Device) {
    // Device counter: transport-allocated + registered. The NIC bumps it via a
    // loopback RDMA atomic (IBGDA).
    sendRecvCounterBulk_ = allocateBulk(counterPerPeer, "counter bulk");
    sendRecvCounterBulkReg_ =
        registerBuffer(sendRecvCounterBulk_->get(), counterPerPeer * numPeers);
    counterBulkBuf = sendRecvCounterBulkReg_;
    counterCompletionBulkBuf = sendRecvCounterBulkReg_;
  } else {
    // Host counter: transport-allocated host-mapped (cudaHostAllocMapped). The
    // CPU proxy writes the host alias on CQE; the device reads via the mapped
    // pointer (IBRC). lkeys are unused (no RDMA target), so wrap with empty
    // keys.
    sendRecvHostCounterAllocation_ = allocateCounterSlotAllocation(
        IbCounterStorage::HostPinned,
        counterPerPeer * numPeers,
        "send/recv host counter");
    counterBulkBuf = IbgdaLocalBuffer(
        sendRecvHostCounterAllocation_.devicePtr, NetworkLKeys{});
    counterCompletionBulkBuf = IbgdaLocalBuffer(
        sendRecvHostCounterAllocation_.hostPtr, NetworkLKeys{});
    sendRecvCounterBulkReg_ = counterBulkBuf;
  }

  for (int i = 0; i < numPeers; ++i) {
    auto& pb = sendRecvPeerBuffers_[i];
    pb.sendStaging = sendStagingBulkReg.subBuffer(i * stagingPerPeer);
    pb.recvStaging = sendRecvRecvStagingBulkReg_.subBuffer(i * stagingPerPeer);
    pb.signal = sendRecvSignalBulkReg_.subBuffer(i * signalPerPeer);
    pb.counter = counterBulkBuf.subBuffer(i * counterPerPeer);
    pb.counterCompletion =
        counterCompletionBulkBuf.subBuffer(i * counterPerPeer);
    auto* statePtr = reinterpret_cast<IbSendRecvState::ProgressSlot*>(
        static_cast<char*>(sendRecvStateBulk_->get()) + i * statePerPeer);
    pb.state.emplace(statePtr, stateSlotsPerPeer);
  }

  VLOG(1) << "MultiPeerIbTransport: rank " << myRank_
          << " allocated send/recv staging for " << numPeers
          << " peers (staging=" << stagingPerPeer << "B per peer, counter="
          << (counterStorage == IbCounterStorage::Device ? "device" : "host")
          << ")";
}

void MultiPeerIbTransportBase::exchangeSendRecvBuffersEager() {
  if (!config_.sendRecv.has_value() || sendRecvPeerBuffers_.empty()) {
    return;
  }

  const int numPeers = nRanks_ - 1;
  const std::size_t stagingPerPeer = sendRecvStagingBytesPerPeer();
  const std::size_t signalPerPeer = sendRecvSignalBytesPerPeer();

  auto recvStagingRemotes = exchangeBuffer(sendRecvRecvStagingBulkReg_);
  auto signalRemotes = exchangeBuffer(sendRecvSignalBulkReg_);

  for (int i = 0; i < numPeers; ++i) {
    const int peerRank = peerIndexToRank(i);
    const int remotePeerIndex = (myRank_ < peerRank) ? myRank_ : (myRank_ - 1);
    sendRecvPeerBuffers_[i].remoteRecvStaging =
        recvStagingRemotes[i].subBuffer(remotePeerIndex * stagingPerPeer);
    sendRecvPeerBuffers_[i].remoteSignal =
        signalRemotes[i].subBuffer(remotePeerIndex * signalPerPeer);
  }

  VLOG(1) << "MultiPeerIbTransport: rank " << myRank_
          << " exchanged send/recv staging with " << numPeers << " peers";
}

void MultiPeerIbTransportBase::cleanupSendRecvBuffers() noexcept {
  auto deregisterNoexcept = [&](void* ptr) noexcept {
    if (ptr == nullptr) {
      return;
    }
    try {
      deregisterBuffer(ptr);
    } catch (const std::exception& ex) {
      LOG(ERROR) << "MultiPeerIbTransport: failed to deregister send/recv "
                    "buffer: "
                 << ex.what();
    }
  };

  deregisterNoexcept(
      sendRecvSendStagingBulk_ ? sendRecvSendStagingBulk_->get() : nullptr);
  deregisterNoexcept(
      sendRecvRecvStagingBulk_ ? sendRecvRecvStagingBulk_->get() : nullptr);
  deregisterNoexcept(
      sendRecvSignalBulk_ ? sendRecvSignalBulk_->get() : nullptr);
  // Device counter is transport-owned + registered; host counter is host-mapped
  // and freed via freeCounterSlotAllocation below (never registered).
  if (sendRecvCounterStorage_ == IbCounterStorage::Device) {
    deregisterNoexcept(
        sendRecvCounterBulk_ ? sendRecvCounterBulk_->get() : nullptr);
  }

  sendRecvSendStagingBulk_.reset();
  sendRecvRecvStagingBulk_.reset();
  sendRecvSignalBulk_.reset();
  sendRecvStateBulk_.reset();
  sendRecvCounterBulk_.reset();
  freeCounterSlotAllocation(sendRecvHostCounterAllocation_);
  sendRecvRecvStagingBulkReg_ = IbgdaLocalBuffer{};
  sendRecvSignalBulkReg_ = IbgdaLocalBuffer{};
  sendRecvCounterBulkReg_ = IbgdaLocalBuffer{};
  // Lazy per-peer allocations (empty in eager mode).
  for (auto& buf : lazyPeerBufs_) {
    deregisterNoexcept(buf ? buf->get() : nullptr);
    buf.reset();
  }
  lazyPeerBufs_.clear();
  for (auto& counter : lazySendRecvHostCounters_) {
    freeCounterSlotAllocation(counter);
  }
  lazySendRecvHostCounters_.clear();
  sendRecvCounterStorage_ = IbCounterStorage::Device;
  sendRecvPeerBuffers_.clear();
}

void MultiPeerIbTransportBase::allocateSendRecvBufferForPeer(
    int peerIndex,
    PeerBufferPayload& payload,
    IbCounterStorage counterStorage) {
  if (!config_.sendRecv.has_value()) {
    return;
  }
  validateSendRecvConfig();
  const int numPeers = nRanks_ - 1;
  if (peerIndex < 0 || peerIndex >= numPeers) {
    throw std::invalid_argument(
        fmt::format(
            "allocateSendRecvBufferForPeer: invalid peerIndex={}", peerIndex));
  }
  sendRecvPeerBuffers_.resize(numPeers);
  lazyPeerBufs_.resize(numPeers);
  lazySendRecvHostCounters_.resize(numPeers);
  sendRecvCounterStorage_ = counterStorage;

  const std::size_t stagingPerPeer = sendRecvStagingBytesPerPeer();
  const std::size_t signalPerPeer = sendRecvSignalBytesPerPeer();
  const std::size_t counterPerPeer = sendRecvCounterBytesPerPeer();
  const std::size_t statePerPeer = sendRecvStateBytesPerPeer();
  const auto stateSlots =
      static_cast<DeviceSpan<IbSendRecvState::ProgressSlot>::size_type>(
          2 * config_.sendRecv->maxGroups);
  const bool deviceCounter = (counterStorage == IbCounterStorage::Device);

  // One contiguous device buffer: sendStaging | recvStaging | signal | state,
  // plus the counter when it is device-resident. A HostPinned counter is
  // allocated separately (host-mapped, never RDMA-registered).
  std::size_t total = 2 * stagingPerPeer + signalPerPeer + statePerPeer;
  if (deviceCounter) {
    total += counterPerPeer;
  }
  auto buf = std::make_unique<meta::comms::DeviceBuffer>(total);
  checkSlotGpu(
      slotGpuMemset(buf->get(), 0, total),
      "MultiPeerIbTransport: zero per-peer send/recv buffer");
  auto reg = registerBuffer(buf->get(), total);

  char* p = static_cast<char*>(buf->get());
  std::size_t off = 0;
  auto& pb = sendRecvPeerBuffers_[peerIndex];
  pb.sendStaging = IbgdaLocalBuffer(p + off, reg.lkey_per_device);
  off += stagingPerPeer;
  void* recvStagingPtr = p + off;
  pb.recvStaging = IbgdaLocalBuffer(recvStagingPtr, reg.lkey_per_device);
  off += stagingPerPeer;
  void* signalPtr = p + off;
  pb.signal = IbgdaLocalBuffer(signalPtr, reg.lkey_per_device);
  off += signalPerPeer;
  auto* statePtr = reinterpret_cast<IbSendRecvState::ProgressSlot*>(p + off);
  off += statePerPeer;
  pb.state.emplace(statePtr, stateSlots);
  if (deviceCounter) {
    pb.counter = IbgdaLocalBuffer(p + off, reg.lkey_per_device);
    pb.counterCompletion = pb.counter;
  } else {
    auto alloc = allocateCounterSlotAllocation(
        IbCounterStorage::HostPinned,
        counterPerPeer,
        "lazy send/recv host counter");
    pb.counter = IbgdaLocalBuffer(alloc.devicePtr, NetworkLKeys{});
    pb.counterCompletion = IbgdaLocalBuffer(alloc.hostPtr, NetworkLKeys{});
    lazySendRecvHostCounters_[peerIndex] = std::move(alloc);
  }

  // The peer RDMA-writes into our recvStaging ring and signal inbox; publish
  // their addr + per-NIC rkeys (whole per-peer regions, no slicing).
  payload.recvStaging = registeredSlotMemoryExchInfo(recvStagingPtr);
  payload.srSignal = registeredSlotMemoryExchInfo(signalPtr);
  lazyPeerBufs_[peerIndex] = std::move(buf);
}

void MultiPeerIbTransportBase::applyRemoteSendRecvBuffer(
    int peerIndex,
    const PeerBufferPayload& remotePayload) {
  if (!config_.sendRecv.has_value() || peerIndex < 0 ||
      peerIndex >= static_cast<int>(sendRecvPeerBuffers_.size())) {
    return;
  }
  auto& pb = sendRecvPeerBuffers_[peerIndex];
  pb.remoteRecvStaging = remotePayload.recvStaging.toRemoteBuffer();
  pb.remoteSignal = remotePayload.srSignal.toRemoteBuffer();
}

void MultiPeerIbTransportBase::cleanupSendRecvBufferForPeer(
    int peerIndex) noexcept {
  if (peerIndex < 0 ||
      peerIndex >= static_cast<int>(sendRecvPeerBuffers_.size())) {
    return;
  }
  if (peerIndex < static_cast<int>(lazyPeerBufs_.size()) &&
      lazyPeerBufs_[peerIndex]) {
    try {
      deregisterBuffer(lazyPeerBufs_[peerIndex]->get());
    } catch (const std::exception& ex) {
      LOG(ERROR) << "MultiPeerIbTransport: failed to deregister per-peer "
                    "send/recv buffer: "
                 << ex.what();
    }
    lazyPeerBufs_[peerIndex].reset();
  }
  if (peerIndex < static_cast<int>(lazySendRecvHostCounters_.size())) {
    freeCounterSlotAllocation(lazySendRecvHostCounters_[peerIndex]);
  }
  // Reset the per-peer views field-wise (IbSendRecvPeerBuffers is not
  // copy-assignable due to its optional<DeviceSpan> state member).
  auto& pb = sendRecvPeerBuffers_[peerIndex];
  pb.sendStaging = IbgdaLocalBuffer{};
  pb.recvStaging = IbgdaLocalBuffer{};
  pb.signal = IbgdaLocalBuffer{};
  pb.counter = IbgdaLocalBuffer{};
  pb.remoteRecvStaging = IbgdaRemoteBuffer{};
  pb.remoteSignal = IbgdaRemoteBuffer{};
  pb.state.reset();
}

void MultiPeerIbTransportBase::openNics() {
  nics_.resize(numNics_);
  auto initResult = ibverbx::ibvInit();
  if (!initResult) {
    throw std::runtime_error(
        "Failed to initialize ibverbx: " + initResult.error().errStr);
  }
  auto& symbols = ibverbx::ibvSymbols;

  // Get all IB devices via ibverbx's dynamically loaded libibverbs symbols.
  int numDevices = 0;
  ibverbx::ibv_device** deviceList =
      symbols.ibv_internal_get_device_list(&numDevices);
  if (!deviceList || numDevices == 0) {
    throw std::runtime_error("No IB devices found");
  }

  // Free the device list on every exit; on failure also close any ctx/PD opened
  // for earlier NICs, so openNics() never leaks on a partial open — this covers
  // the manual throws below and exceptions from callees (e.g. GpuNicDiscovery).
  SCOPE_EXIT {
    symbols.ibv_internal_free_device_list(deviceList);
  };
  SCOPE_FAIL {
    for (auto& nic : nics_) {
      if (nic.ibvPd != nullptr) {
        symbols.ibv_internal_dealloc_pd(nic.ibvPd);
        nic.ibvPd = nullptr;
      }
      if (nic.ibvCtx != nullptr) {
        symbols.ibv_internal_close_device(nic.ibvCtx);
        nic.ibvCtx = nullptr;
      }
    }
  };

  // Resolve nics_[0..numNics_).deviceName — config override first, then
  // topology-aware auto-discovery.
  //
  // Priority 1: Explicit GPU-to-NIC mapping from config (entries [0..numNics_)
  // used in order — first is preferred).
  auto it = config_.gpuNicMap.find(config_.cudaDevice);
  if (it != config_.gpuNicMap.end() && !it->second.empty()) {
    const auto& names = it->second;
    if (static_cast<int>(names.size()) < numNics_) {
      throw std::runtime_error(
          fmt::format(
              "config.gpuNicMap[{}] supplies {} NIC(s) but numNics_={}; "
              "provide at least numNics_ NIC names",
              config_.cudaDevice,
              names.size(),
              numNics_));
    }
    for (int n = 0; n < numNics_; ++n) {
      nics_[n].deviceName = names[n];
    }
    VLOG(1) << "MultiPeerIbTransport: using config.gpuNicMap for GPU "
            << config_.cudaDevice << " -> " << nics_[0].deviceName
            << (numNics_ > 1 ? " (+ " + std::to_string(numNics_ - 1) +
                        " more for multi-NIC)"
                             : "");
  }

  // Priority 2: Auto-discovery (top-numNics_ candidates by NUMA affinity).
  if (nics_[0].deviceName.empty()) {
    // On AMD, the `DataDirectMode::Only` default triggers `ibv_reg_dmabuf_mr`
    // inside `augmentWithDataDirect()`, which is not exercised on AMD's
    // libibverbs path here. Force `Disabled` to skip the DataDirect probe.
#ifdef __HIP_PLATFORM_AMD__
    auto discovery = GpuNicDiscovery(
        config_.cudaDevice, config_.ibHca, DataDirectMode::Disabled);
#else
    auto discovery = GpuNicDiscovery(config_.cudaDevice, config_.ibHca);
#endif
    const auto& candidates = discovery.getCandidates();
    if (static_cast<int>(candidates.size()) < numNics_) {
      throw std::runtime_error(
          fmt::format(
              "NIC auto-discovery found {} candidate(s) for GPU {} but "
              "numNics_={}; set config.gpuNicMap or config.ibHca to expose "
              "additional NICs",
              candidates.size(),
              config_.cudaDevice,
              numNics_));
    }
    for (int n = 0; n < numNics_; ++n) {
      nics_[n].deviceName = candidates[n].name;
    }
    VLOG(1) << "MultiPeerIbTransport: auto-discovered NIC "
            << nics_[0].deviceName << " for GPU device " << config_.cudaDevice;
  }

  // Open + setup each NIC: find by name, open ctx, alloc PD, query GID + port.
  for (int n = 0; n < numNics_; ++n) {
    int nicIdx = -1;
    for (int i = 0; i < numDevices; i++) {
      const char* devName = symbols.ibv_internal_get_device_name(deviceList[i]);
      if (devName && nics_[n].deviceName == devName) {
        nicIdx = i;
        break;
      }
    }
    if (nicIdx < 0) {
      throw std::runtime_error(
          "Specified NIC not found: " + nics_[n].deviceName);
    }
    VLOG(1) << "MultiPeerIbTransport: NIC " << n << " = " << nics_[n].deviceName
            << " at device-list index " << nicIdx;

    nics_[n].ibvCtx = symbols.ibv_internal_open_device(deviceList[nicIdx]);
    if (!nics_[n].ibvCtx) {
      throw std::runtime_error(
          "Failed to open IB device: " + nics_[n].deviceName);
    }

    nics_[n].ibvPd = symbols.ibv_internal_alloc_pd(nics_[n].ibvCtx);
    if (!nics_[n].ibvPd) {
      throw std::runtime_error(
          "Failed to allocate protection domain on NIC " + nics_[n].deviceName);
    }

    if (symbols.ibv_internal_query_gid(
            nics_[n].ibvCtx, 1, gidIndex_, &nics_[n].localGid) != 0) {
      throw std::runtime_error(
          "Failed to query GID at index " + std::to_string(gidIndex_) +
          " on NIC " + nics_[n].deviceName);
    }

    auto gidStr = fmt::format(
        "{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}:"
        "{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}:{:02x}{:02x}",
        nics_[n].localGid.raw[0],
        nics_[n].localGid.raw[1],
        nics_[n].localGid.raw[2],
        nics_[n].localGid.raw[3],
        nics_[n].localGid.raw[4],
        nics_[n].localGid.raw[5],
        nics_[n].localGid.raw[6],
        nics_[n].localGid.raw[7],
        nics_[n].localGid.raw[8],
        nics_[n].localGid.raw[9],
        nics_[n].localGid.raw[10],
        nics_[n].localGid.raw[11],
        nics_[n].localGid.raw[12],
        nics_[n].localGid.raw[13],
        nics_[n].localGid.raw[14],
        nics_[n].localGid.raw[15]);
    VLOG(1) << "MultiPeerIbTransport: NIC " << n << " GID[" << gidIndex_
            << "] = " << gidStr;

    ibverbx::ibv_port_attr portAttr{};
    if (symbols.ibv_internal_query_port(nics_[n].ibvCtx, 1, &portAttr) != 0) {
      throw std::runtime_error(
          "Failed to query port attributes on NIC " + nics_[n].deviceName);
    }

    VLOG(1) << "MultiPeerIbTransport: NIC " << n
            << " port 1 state=" << portAttr.state
            << " link_layer=" << (int)portAttr.link_layer
            << " (1=IB, 2=Ethernet) active_mtu=" << portAttr.active_mtu;

    if (portAttr.state != ibverbx::IBV_PORT_ACTIVE) {
      throw std::runtime_error(
          "NIC " + nics_[n].deviceName + " port 1 is not active (state=" +
          std::to_string(portAttr.state) + ")");
    }

    nics_[n].linkLayer = portAttr.link_layer;

    // MTU is common across NICs (same fabric/HCA generation assumed). Capture
    // from NIC 0; cross-check the rest match.
    if (n == 0) {
      localMtu_ = portAttr.active_mtu;
    } else if (portAttr.active_mtu != localMtu_) {
      LOG(WARNING) << "MultiPeerIbTransport: NIC " << n << " ("
                   << nics_[n].deviceName
                   << ") active_mtu=" << portAttr.active_mtu
                   << " differs from NIC 0 active_mtu=" << localMtu_
                   << "; using NIC 0's MTU for negotiation";
    }
  }
  // Success: SCOPE_EXIT frees the device list; SCOPE_FAIL is skipped, so the
  // opened ctx/PD are kept for the transport's lifetime.
}

IbgdaLocalBuffer MultiPeerIbTransportBase::registerBuffer(
    void* ptr,
    std::size_t size) {
  if (ptr == nullptr || size == 0) {
    throw std::invalid_argument("Invalid buffer pointer or size");
  }

  // Fast path: containment lookup — if [ptr, ptr+size) falls entirely within an
  // existing registration, return the cached per-NIC lkeys with no driver call.
  const auto addr = reinterpret_cast<uintptr_t>(ptr);
  auto it = registeredBuffers_.upper_bound(addr);
  if (it != registeredBuffers_.begin()) {
    --it;
    if (addr + size <= it->first + it->second.allocSize) {
      it->second.refs++;
      VLOG(1) << "MultiPeerIbTransport: cache hit for ptr=" << ptr
              << " allocBase=0x" << std::hex << it->first << std::dec
              << " refs=" << it->second.refs;
      NetworkLKeys keys(numNics_);
      for (int n = 0; n < numNics_; ++n) {
        keys[n] = NetworkLKey(HostLKey(it->second.mrs[n]->lkey));
      }
      return IbgdaLocalBuffer(ptr, keys);
    }
  }

  // Cache miss: resolve the GPU allocation base and register one MR per NIC
  // (DMABUF-first, ibv_reg_mr fallback). Generic — no DOCA, no backend hook.
#ifdef __HIP_PLATFORM_AMD__
  // HIP doesn't expose an exact cuMemGetAddressRange equivalent; for the common
  // case where the caller passes the allocation base, register the requested
  // range.
  uintptr_t allocBase = reinterpret_cast<uintptr_t>(ptr);
  std::size_t allocSize = size;
#else
  if (cuda_driver_lazy_init() != 0 || pfn_cuMemGetAddressRange == nullptr) {
    throw std::runtime_error(
        "registerBuffer: failed to initialize CUDA driver API");
  }
  CUdeviceptr allocBase = 0;
  std::size_t allocSize = 0;
  CUresult cuRes =
      pfn_cuMemGetAddressRange(&allocBase, &allocSize, (CUdeviceptr)ptr);
  if (cuRes != CUDA_SUCCESS || allocBase == 0) {
    throw std::runtime_error(
        "registerBuffer: cuMemGetAddressRange failed for ptr");
  }
#endif
  auto& symbols = ibverbx::ibvSymbols;
  int accessFlags = ibverbx::IBV_ACCESS_LOCAL_WRITE |
      ibverbx::IBV_ACCESS_REMOTE_WRITE | ibverbx::IBV_ACCESS_REMOTE_READ |
      ibverbx::IBV_ACCESS_REMOTE_ATOMIC;

  CachedMr cached;
  cached.allocSize = allocSize;
  cached.refs = 1;

  // Try DMABUF first per NIC, fall back to plain reg_mr per NIC. If any NIC's
  // registration fails, deregister everything already registered and propagate.
  for (int n = 0; n < numNics_; ++n) {
    ibverbx::ibv_mr* mr = nullptr;
    auto dmabuf = export_gpu_dmabuf_aligned(
        reinterpret_cast<void*>(allocBase), allocSize);
    if (dmabuf) {
      if (symbols.ibv_internal_reg_dmabuf_mr != nullptr) {
        mr = symbols.ibv_internal_reg_dmabuf_mr(
            nics_[n].ibvPd,
            dmabuf->alignment.dmabufOffset,
            allocSize,
            static_cast<uint64_t>(allocBase),
            dmabuf->fd,
            accessFlags);
      }
      close(dmabuf->fd);
    }
    if (!mr) {
      errno = 0;
      mr = symbols.ibv_internal_reg_mr(
          nics_[n].ibvPd,
          reinterpret_cast<void*>(allocBase),
          allocSize,
          accessFlags);
      if (!mr) {
        const int savedErrno = errno;
        for (int j = 0; j < n; ++j) {
          symbols.ibv_internal_dereg_mr(cached.mrs[j]);
        }
        throw std::runtime_error(
            fmt::format(
                "Failed to register buffer with RDMA on NIC {} "
                "(allocBase=0x{:x} allocSize={} errno={} ({}))",
                n,
                allocBase,
                allocSize,
                savedErrno,
                std::strerror(savedErrno)));
      }
    }
    cached.mrs[n] = mr;
  }

  registeredBuffers_.emplace(static_cast<uintptr_t>(allocBase), cached);

  NetworkLKeys keys(numNics_);
  for (int n = 0; n < numNics_; ++n) {
    keys[n] = NetworkLKey(HostLKey(cached.mrs[n]->lkey));
  }
  return IbgdaLocalBuffer(ptr, keys);
}

void MultiPeerIbTransportBase::deregisterBuffer(void* ptr) {
  // Containment lookup on the ordered map avoids resolving the allocation range
  // again (which fails once the underlying memory is freed).
  const auto addr = reinterpret_cast<uintptr_t>(ptr);
  auto it = registeredBuffers_.upper_bound(addr);
  if (it != registeredBuffers_.begin()) {
    --it;
    if (addr < it->first + it->second.allocSize) {
      it->second.refs--;
      VLOG(1) << "MultiPeerIbTransport: deregister ptr=" << ptr
              << " allocBase=0x" << std::hex << it->first << std::dec
              << " refs=" << it->second.refs;
      if (it->second.refs <= 0) {
        // Deregistration is backend-agnostic (no PD/DOCA needed).
        for (int n = 0; n < numNics_; ++n) {
          ibverbx::ibvSymbols.ibv_internal_dereg_mr(it->second.mrs[n]);
        }
        registeredBuffers_.erase(it);
      }
      return;
    }
  }
  LOG(WARNING) << "MultiPeerIbTransport: buffer not registered: " << ptr;
}

std::vector<IbgdaRemoteBuffer> MultiPeerIbTransportBase::exchangeBuffer(
    const IbgdaLocalBuffer& localBuf) {
  const int numPeers = nRanks_ - 1;

  // Containment lookup (same as deregisterBuffer): find the registered
  // allocation covering localBuf.ptr. Avoids re-resolving the allocation base;
  // sub-buffers resolve correctly via the ordered map.
  const auto addr = reinterpret_cast<uintptr_t>(localBuf.ptr);
  auto it = registeredBuffers_.upper_bound(addr);
  if (it == registeredBuffers_.begin()) {
    throw std::runtime_error(
        "Buffer not registered - call registerBuffer() first");
  }
  --it;
  if (addr >= it->first + it->second.allocSize) {
    throw std::runtime_error(
        "Buffer not registered - call registerBuffer() first");
  }

  // allGather addr + per-NIC rkeys; one entry per rank.
  std::vector<IbgdaBufferExchInfo> allInfo(nRanks_);
  allInfo[myRank_].addr = reinterpret_cast<uint64_t>(localBuf.ptr);
  allInfo[myRank_].numNics = numNics_;
  for (int n = 0; n < numNics_; ++n) {
    allInfo[myRank_].rkey_per_device[n] = HostRKey(it->second.mrs[n]->rkey);
  }

  auto result =
      bootstrap_
          ->allGather(
              allInfo.data(), sizeof(IbgdaBufferExchInfo), myRank_, nRanks_)
          .get();
  if (result != 0) {
    throw std::runtime_error(
        "MultiPeerIbTransport::exchangeBuffer allGather failed");
  }

  std::vector<IbgdaRemoteBuffer> peerBuffers(numPeers);
  for (int peerIndex = 0; peerIndex < numPeers; peerIndex++) {
    const int peerRank = peerIndexToRank(peerIndex);
    peerBuffers[peerIndex] = allInfo[peerRank].toRemoteBuffer();
  }

  VLOG(1) << "MultiPeerIbTransport: exchanged buffer info with " << numPeers
          << " peers";
  return peerBuffers;
}

MultiPeerIbTransportBase::DeviceSlotAllocation
MultiPeerIbTransportBase::allocateDeviceSlotAllocation(
    std::size_t bytes,
    const char* label) {
  if (bytes == 0) {
    throw std::invalid_argument(
        fmt::format("MultiPeerIbTransport: {} size must be non-zero", label));
  }
  DeviceSlotAllocation allocation;
  allocation.bytes = bytes;
  // Free whatever was allocated if any step below throws (free is null-safe).
  SCOPE_FAIL {
    freeDeviceSlotAllocation(allocation);
  };
#ifdef __HIP_PLATFORM_AMD__
  // On AMD, GPU-memory MR registration relies on amdgpu's peer-mem
  // integration which is unreliable on test hosts; host-pinned memory
  // registers with ibv_reg_mr (no peer_mem) and is GPU-accessible.
  allocation.ptr = checkedSlotAlloc(
      slotHostPinnedAlloc,
      bytes,
      fmt::format(
          "MultiPeerIbTransport: host-pinned allocation for {}", label));
  allocation.isHostPinned = true;
  std::memset(allocation.ptr, 0, bytes);
#else
  allocation.ptr = checkedSlotAlloc(
      slotGpuMalloc,
      bytes,
      fmt::format("MultiPeerIbTransport: device allocation for {}", label));
  checkSlotGpu(
      slotGpuMemset(allocation.ptr, 0, bytes),
      fmt::format(
          "MultiPeerIbTransport: zero device allocation for {}", label));
#endif
  return allocation;
}

MultiPeerIbTransportBase::CounterSlotAllocation
MultiPeerIbTransportBase::allocateCounterSlotAllocation(
    IbCounterStorage storage,
    std::size_t bytes,
    const char* label) {
  if (bytes == 0) {
    throw std::invalid_argument(
        fmt::format("MultiPeerIbTransport: {} size must be non-zero", label));
  }
  CounterSlotAllocation allocation;
  allocation.bytes = bytes;
  // Free whatever was allocated if any step below throws (free is null-safe).
  SCOPE_FAIL {
    freeCounterSlotAllocation(allocation);
  };
  switch (storage) {
    case IbCounterStorage::Device:
      // NIC loopback atomic target: device memory, zeroed.
      allocation.devicePtr = checkedSlotAlloc(
          slotGpuMalloc,
          bytes,
          fmt::format("MultiPeerIbTransport: device allocation for {}", label));
      checkSlotGpu(
          slotGpuMemset(allocation.devicePtr, 0, bytes),
          fmt::format(
              "MultiPeerIbTransport: zero device allocation for {}", label));
      break;
    case IbCounterStorage::HostPinned:
      // CPU-proxy counter: host-mapped memory; the device reads via the mapped
      // device pointer.
      allocation.hostPtr = checkedSlotAlloc(
          slotHostPinnedAlloc,
          bytes,
          fmt::format(
              "MultiPeerIbTransport: host-pinned allocation for {}", label));
      std::memset(allocation.hostPtr, 0, bytes);
      checkSlotGpu(
          slotHostGetDevicePointer(&allocation.devicePtr, allocation.hostPtr),
          fmt::format(
              "MultiPeerIbTransport: host-pinned device pointer lookup for {}",
              label));
      break;
  }
  return allocation;
}

IbgdaLocalBuffer MultiPeerIbTransportBase::registerSlotMemory(
    void* registrationPtr,
    void* devicePtr,
    std::size_t bytes,
    bool& registered) {
  if (registrationPtr == nullptr || devicePtr == nullptr || bytes == 0) {
    throw std::invalid_argument(
        "MultiPeerIbTransport: invalid slot memory registration");
  }
  if (!registered) {
    (void)registerBuffer(registrationPtr, bytes);
    registered = true;
  }

  NetworkLKeys keys(numNics_);
  const auto addr = reinterpret_cast<uintptr_t>(registrationPtr);
  auto it = registeredBuffers_.upper_bound(addr);
  CHECK(it != registeredBuffers_.begin())
      << "slot allocation MR not found after registration";
  --it;
  CHECK(addr < it->first + it->second.allocSize)
      << "slot allocation MR does not cover registration pointer";
  for (int n = 0; n < numNics_; ++n) {
    keys[n] = NetworkLKey(HostLKey(it->second.mrs[n]->lkey));
  }
  return IbgdaLocalBuffer(devicePtr, keys);
}

IbgdaBufferExchInfo MultiPeerIbTransportBase::registeredSlotMemoryExchInfo(
    void* registrationPtr) const {
  if (registrationPtr == nullptr) {
    throw std::invalid_argument(
        "MultiPeerIbTransport: invalid slot memory exchange info");
  }
  const auto addr = reinterpret_cast<uintptr_t>(registrationPtr);
  auto it = registeredBuffers_.upper_bound(addr);
  CHECK(it != registeredBuffers_.begin())
      << "slot allocation MR not found after registration";
  --it;
  CHECK(addr < it->first + it->second.allocSize)
      << "slot allocation MR does not cover registration pointer";

  IbgdaBufferExchInfo info;
  info.addr = reinterpret_cast<uint64_t>(registrationPtr);
  info.numNics = numNics_;
  for (int n = 0; n < numNics_; ++n) {
    info.rkey_per_device[n] = HostRKey(it->second.mrs[n]->rkey);
  }
  return info;
}

void MultiPeerIbTransportBase::freeDeviceSlotAllocation(
    DeviceSlotAllocation& allocation) noexcept {
  if (allocation.ptr == nullptr) {
    return;
  }

  if (allocation.registered) {
    try {
      deregisterBuffer(allocation.ptr);
    } catch (const std::exception& ex) {
      LOG(WARNING) << "MultiPeerIbTransport: failed to deregister device slot "
                   << "allocation: " << ex.what();
    }
    allocation.registered = false;
  }

  SlotGpuError err = allocation.isHostPinned ? slotHostFree(allocation.ptr)
                                             : slotGpuFree(allocation.ptr);
  if (err != kSlotGpuSuccess) {
    LOG(WARNING) << "MultiPeerIbTransport: failed to free device slot "
                 << "allocation: " << slotGpuGetErrorString(err);
  }
  allocation = DeviceSlotAllocation{};
}

void MultiPeerIbTransportBase::freeCounterSlotAllocation(
    CounterSlotAllocation& allocation) noexcept {
  if (allocation.devicePtr == nullptr && allocation.hostPtr == nullptr) {
    return;
  }

  if (allocation.registered && allocation.devicePtr != nullptr) {
    try {
      void* registerPtr = allocation.hostPtr != nullptr ? allocation.hostPtr
                                                        : allocation.devicePtr;
      deregisterBuffer(registerPtr);
    } catch (const std::exception& ex) {
      LOG(WARNING) << "MultiPeerIbTransport: failed to deregister slot "
                   << "allocation: " << ex.what();
    }
    allocation.registered = false;
  }

  SlotGpuError err = kSlotGpuSuccess;
  if (allocation.hostPtr != nullptr) {
    err = slotHostFree(allocation.hostPtr);
  } else if (allocation.devicePtr != nullptr) {
    err = slotGpuFree(allocation.devicePtr);
  }
  if (err != kSlotGpuSuccess) {
    LOG(WARNING) << "MultiPeerIbTransport: failed to free counter slot "
                 << "allocation: " << slotGpuGetErrorString(err);
  }
  allocation = CounterSlotAllocation{};
}

void MultiPeerIbTransportBase::allocateSignalCounterResources(
    IbCounterStorage counterStorage,
    bool allocateDiscardSignal) {
  cleanupSignalCounterResources();

  const int numPeers = nRanks_ - 1;
  slotRemoteSignalViews_.assign(numPeers, IbgdaRemoteBuffer{});
  slotLocalSignalViews_.assign(numPeers, IbgdaLocalBuffer{});
  slotCounterDeviceViews_.assign(numPeers, IbgdaLocalBuffer{});
  slotCounterHostViews_.assign(numPeers, IbgdaLocalBuffer{});
  slotDiscardSignalRemoteViews_.assign(numPeers, IbgdaRemoteBuffer{});

  if (config_.numSignalSlots > 0) {
    const auto slotsPerPeer = static_cast<std::size_t>(config_.numSignalSlots);
    const std::size_t totalSignalBytes =
        static_cast<std::size_t>(numPeers) * slotsPerPeer * sizeof(uint64_t);
    slotSignalAllocation_ =
        allocateDeviceSlotAllocation(totalSignalBytes, "slot signal buffer");
    auto localSignalBuf = registerSlotMemory(
        slotSignalAllocation_.ptr,
        slotSignalAllocation_.ptr,
        slotSignalAllocation_.bytes,
        slotSignalAllocation_.registered);
    auto remoteSignalBufs = exchangeBuffer(localSignalBuf);
    for (int peerIndex = 0; peerIndex < numPeers; ++peerIndex) {
      const int peerRank = peerIndexToRank(peerIndex);
      const int myPeerIndexOnPeer =
          (myRank_ < peerRank) ? myRank_ : (myRank_ - 1);
      slotRemoteSignalViews_[peerIndex] = remoteSignalBufs[peerIndex].subBuffer(
          static_cast<std::size_t>(myPeerIndexOnPeer) * slotsPerPeer *
          sizeof(uint64_t));
      slotLocalSignalViews_[peerIndex] = localSignalBuf.subBuffer(
          static_cast<std::size_t>(peerIndex) * slotsPerPeer *
          sizeof(uint64_t));
    }
  }

  if (config_.numCounterSlots > 0) {
    const auto slotsPerPeer = static_cast<std::size_t>(config_.numCounterSlots);
    const std::size_t totalCounterBytes =
        static_cast<std::size_t>(numPeers) * slotsPerPeer * sizeof(uint64_t);
    slotCounterAllocation_ = allocateCounterSlotAllocation(
        counterStorage, totalCounterBytes, "slot counter buffer");
    if (counterStorage == IbCounterStorage::HostPinned) {
      IbgdaLocalBuffer deviceCounterBuf(
          slotCounterAllocation_.devicePtr, NetworkLKeys{});
      IbgdaLocalBuffer hostCounterBuf(
          slotCounterAllocation_.hostPtr, NetworkLKeys{});
      for (int peerIndex = 0; peerIndex < numPeers; ++peerIndex) {
        const auto offset = static_cast<std::size_t>(peerIndex) * slotsPerPeer *
            sizeof(uint64_t);
        slotCounterDeviceViews_[peerIndex] = deviceCounterBuf.subBuffer(offset);
        slotCounterHostViews_[peerIndex] = hostCounterBuf.subBuffer(offset);
      }
    } else {
      auto localCounterBuf = registerSlotMemory(
          slotCounterAllocation_.devicePtr,
          slotCounterAllocation_.devicePtr,
          slotCounterAllocation_.bytes,
          slotCounterAllocation_.registered);
      for (int peerIndex = 0; peerIndex < numPeers; ++peerIndex) {
        const auto offset = static_cast<std::size_t>(peerIndex) * slotsPerPeer *
            sizeof(uint64_t);
        slotCounterDeviceViews_[peerIndex] = localCounterBuf.subBuffer(offset);
        slotCounterHostViews_[peerIndex] = localCounterBuf.subBuffer(offset);
      }
    }
  }

  if (allocateDiscardSignal && config_.numCounterSlots > 0) {
    const std::size_t totalDiscardBytes =
        static_cast<std::size_t>(numPeers) * sizeof(uint64_t);
    slotDiscardSignalAllocation_ = allocateDeviceSlotAllocation(
        totalDiscardBytes, "slot discard-signal buffer");
    auto localDiscardBuf = registerSlotMemory(
        slotDiscardSignalAllocation_.ptr,
        slotDiscardSignalAllocation_.ptr,
        slotDiscardSignalAllocation_.bytes,
        slotDiscardSignalAllocation_.registered);
    auto remoteDiscardBufs = exchangeBuffer(localDiscardBuf);
    for (int peerIndex = 0; peerIndex < numPeers; ++peerIndex) {
      const int peerRank = peerIndexToRank(peerIndex);
      const int myPeerIndexOnPeer =
          (myRank_ < peerRank) ? myRank_ : (myRank_ - 1);
      slotDiscardSignalRemoteViews_[peerIndex] =
          remoteDiscardBufs[peerIndex].subBuffer(
              static_cast<std::size_t>(myPeerIndexOnPeer) * sizeof(uint64_t));
    }
  }
}

void MultiPeerIbTransportBase::cleanupSignalCounterResources() noexcept {
  freeDeviceSlotAllocation(slotSignalAllocation_);
  freeCounterSlotAllocation(slotCounterAllocation_);
  freeDeviceSlotAllocation(slotDiscardSignalAllocation_);
  for (auto& allocation : lazySlotSignalAllocations_) {
    freeDeviceSlotAllocation(allocation);
  }
  for (auto& allocation : lazySlotCounterAllocations_) {
    freeCounterSlotAllocation(allocation);
  }
  for (auto& allocation : lazySlotDiscardSignalAllocations_) {
    freeDeviceSlotAllocation(allocation);
  }
  lazySlotSignalAllocations_.clear();
  lazySlotCounterAllocations_.clear();
  lazySlotDiscardSignalAllocations_.clear();
  slotRemoteSignalViews_.clear();
  slotLocalSignalViews_.clear();
  slotCounterDeviceViews_.clear();
  slotCounterHostViews_.clear();
  slotDiscardSignalRemoteViews_.clear();
}

void MultiPeerIbTransportBase::cleanupPeerSignalCounterResources(
    int peerIndex) noexcept {
  if (peerIndex < 0 || peerIndex >= nRanks_ - 1) {
    return;
  }
  if (peerIndex < static_cast<int>(lazySlotSignalAllocations_.size())) {
    freeDeviceSlotAllocation(lazySlotSignalAllocations_[peerIndex]);
  }
  if (peerIndex < static_cast<int>(lazySlotCounterAllocations_.size())) {
    freeCounterSlotAllocation(lazySlotCounterAllocations_[peerIndex]);
  }
  if (peerIndex < static_cast<int>(lazySlotDiscardSignalAllocations_.size())) {
    freeDeviceSlotAllocation(lazySlotDiscardSignalAllocations_[peerIndex]);
  }
  if (peerIndex < static_cast<int>(slotRemoteSignalViews_.size())) {
    slotRemoteSignalViews_[peerIndex] = IbgdaRemoteBuffer{};
  }
  if (peerIndex < static_cast<int>(slotLocalSignalViews_.size())) {
    slotLocalSignalViews_[peerIndex] = IbgdaLocalBuffer{};
  }
  if (peerIndex < static_cast<int>(slotCounterDeviceViews_.size())) {
    slotCounterDeviceViews_[peerIndex] = IbgdaLocalBuffer{};
  }
  if (peerIndex < static_cast<int>(slotCounterHostViews_.size())) {
    slotCounterHostViews_[peerIndex] = IbgdaLocalBuffer{};
  }
  if (peerIndex < static_cast<int>(slotDiscardSignalRemoteViews_.size())) {
    slotDiscardSignalRemoteViews_[peerIndex] = IbgdaRemoteBuffer{};
  }
}

void MultiPeerIbTransportBase::allocatePeerSignalCounterResources(
    int peerIndex,
    PeerBufferPayload& payload,
    IbCounterStorage counterStorage,
    bool allocateDiscardSignal) {
  const int numPeers = nRanks_ - 1;
  if (peerIndex < 0 || peerIndex >= numPeers) {
    throw std::invalid_argument(
        fmt::format(
            "allocatePeerSignalCounterResources: invalid peerIndex={}",
            peerIndex));
  }

  slotRemoteSignalViews_.resize(numPeers);
  slotLocalSignalViews_.resize(numPeers);
  slotCounterDeviceViews_.resize(numPeers);
  slotCounterHostViews_.resize(numPeers);
  slotDiscardSignalRemoteViews_.resize(numPeers);
  lazySlotSignalAllocations_.resize(numPeers);
  lazySlotCounterAllocations_.resize(numPeers);
  lazySlotDiscardSignalAllocations_.resize(numPeers);

  if (config_.numSignalSlots > 0) {
    const std::size_t signalBytes =
        static_cast<std::size_t>(config_.numSignalSlots) * sizeof(uint64_t);
    freeDeviceSlotAllocation(lazySlotSignalAllocations_[peerIndex]);
    auto allocation =
        allocateDeviceSlotAllocation(signalBytes, "lazy slot signal buffer");
    auto localSignalBuf = registerSlotMemory(
        allocation.ptr,
        allocation.ptr,
        allocation.bytes,
        allocation.registered);
    payload.slotSignal = registeredSlotMemoryExchInfo(allocation.ptr);
    slotLocalSignalViews_[peerIndex] = localSignalBuf;
    lazySlotSignalAllocations_[peerIndex] = std::move(allocation);
  }

  if (config_.numCounterSlots > 0) {
    const std::size_t counterBytes =
        static_cast<std::size_t>(config_.numCounterSlots) * sizeof(uint64_t);
    freeCounterSlotAllocation(lazySlotCounterAllocations_[peerIndex]);
    auto allocation = allocateCounterSlotAllocation(
        counterStorage, counterBytes, "lazy slot counter buffer");
    if (counterStorage == IbCounterStorage::HostPinned) {
      slotCounterDeviceViews_[peerIndex] =
          IbgdaLocalBuffer(allocation.devicePtr, NetworkLKeys{});
      slotCounterHostViews_[peerIndex] =
          IbgdaLocalBuffer(allocation.hostPtr, NetworkLKeys{});
      lazySlotCounterAllocations_[peerIndex] = std::move(allocation);
    } else {
      auto localCounterBuf = registerSlotMemory(
          allocation.devicePtr,
          allocation.devicePtr,
          allocation.bytes,
          allocation.registered);
      slotCounterDeviceViews_[peerIndex] = localCounterBuf;
      slotCounterHostViews_[peerIndex] = localCounterBuf;
      lazySlotCounterAllocations_[peerIndex] = std::move(allocation);
    }
  }

  if (allocateDiscardSignal && config_.numCounterSlots > 0) {
    freeDeviceSlotAllocation(lazySlotDiscardSignalAllocations_[peerIndex]);
    auto allocation = allocateDeviceSlotAllocation(
        sizeof(uint64_t), "lazy slot discard-signal buffer");
    (void)registerSlotMemory(
        allocation.ptr,
        allocation.ptr,
        allocation.bytes,
        allocation.registered);
    payload.slotDiscard = registeredSlotMemoryExchInfo(allocation.ptr);
    lazySlotDiscardSignalAllocations_[peerIndex] = std::move(allocation);
  }
}

void MultiPeerIbTransportBase::applyRemoteSignalCounterResources(
    int peerIndex,
    const PeerBufferPayload& remotePayload,
    bool hasDiscardSignal) {
  const int numPeers = nRanks_ - 1;
  if (peerIndex < 0 || peerIndex >= numPeers) {
    throw std::invalid_argument(
        fmt::format(
            "applyRemoteSignalCounterResources: invalid peerIndex={}",
            peerIndex));
  }
  slotRemoteSignalViews_.resize(numPeers);
  slotDiscardSignalRemoteViews_.resize(numPeers);
  if (config_.numSignalSlots > 0) {
    slotRemoteSignalViews_[peerIndex] =
        remotePayload.slotSignal.toRemoteBuffer();
  }
  if (hasDiscardSignal && config_.numCounterSlots > 0) {
    slotDiscardSignalRemoteViews_[peerIndex] =
        remotePayload.slotDiscard.toRemoteBuffer();
  }
}

IbgdaRemoteBuffer MultiPeerIbTransportBase::slotRemoteSignalView(
    int peerIndex) const {
  return slotRemoteSignalViews_.at(peerIndex);
}

IbgdaLocalBuffer MultiPeerIbTransportBase::slotLocalSignalView(
    int peerIndex) const {
  return slotLocalSignalViews_.at(peerIndex);
}

IbgdaLocalBuffer MultiPeerIbTransportBase::slotCounterDeviceView(
    int peerIndex) const {
  return slotCounterDeviceViews_.at(peerIndex);
}

IbgdaLocalBuffer MultiPeerIbTransportBase::slotCounterHostView(
    int peerIndex) const {
  return slotCounterHostViews_.at(peerIndex);
}

IbgdaRemoteBuffer MultiPeerIbTransportBase::slotDiscardSignalRemoteView(
    int peerIndex) const {
  return slotDiscardSignalRemoteViews_.at(peerIndex);
}

bool MultiPeerIbTransportBase::isPeerMaterialized(int peerRank) const {
  if (peerRank == myRank_ || peerRank < 0 || peerRank >= nRanks_) {
    throw std::invalid_argument(
        fmt::format(
            "isPeerMaterialized: invalid peerRank={} (myRank={}, nRanks={})",
            peerRank,
            myRank_,
            nRanks_));
  }
  if (!config_.ibLazyConnect) {
    return true;
  }
  return peerMaterialized_[rankToPeerIndex(peerRank)];
}

void MultiPeerIbTransportBase::queuePeerForMaterialization(int peerRank) {
  if (!config_.ibLazyConnect) {
    return;
  }
  if (materializationFailed_) {
    throw std::runtime_error(
        "MultiPeerIbTransport: lazy peer materialization previously failed; "
        "retry is not supported");
  }
  if (peerRank == myRank_ || peerRank < 0 || peerRank >= nRanks_) {
    throw std::invalid_argument(
        fmt::format(
            "queuePeerForMaterialization: invalid peerRank={} (myRank={}, "
            "nRanks={})",
            peerRank,
            myRank_,
            nRanks_));
  }
  if (isPeerMaterialized(peerRank)) {
    return;
  }
  for (int p : pendingPeers_) {
    if (p == peerRank) {
      return;
    }
  }
  pendingPeers_.push_back(peerRank);
}

std::vector<IbTransportExchInfoAll> MultiPeerIbTransportBase::allGatherExchInfo(
    const IbTransportExchInfoAll& localInfo) {
  if (nRanks_ > kMaxRanksForAllGather) {
    throw std::runtime_error(
        fmt::format(
            "Too many ranks ({}) for allGather-based exchange, max is {}",
            nRanks_,
            kMaxRanksForAllGather));
  }
  std::vector<IbTransportExchInfoAll> allInfo(nRanks_);
  allInfo[myRank_] = localInfo;
  auto result =
      bootstrap_
          ->allGather(
              allInfo.data(), sizeof(IbTransportExchInfoAll), myRank_, nRanks_)
          .get();
  if (result != 0) {
    throw std::runtime_error(
        "MultiPeerIbTransport::allGatherExchInfo allGather failed");
  }
  return allInfo;
}

void MultiPeerIbTransportBase::validatePeerTopology(
    const std::vector<IbTransportExchInfoAll>& allInfo) const {
  const int numPeers = nRanks_ - 1;
  for (int peerIndex = 0; peerIndex < numPeers; ++peerIndex) {
    const int peerRank = peerIndexToRank(peerIndex);
    const auto& peerInfo = allInfo[peerRank];
    // Same-rail pairing relies on the symmetric (myRank+peerRank) % numNics
    // offset, which only makes sense when both sides agree on numNics.
    if (peerInfo.numNics != numNics_) {
      throw std::runtime_error(
          fmt::format(
              "Peer rank {} reports numNics={} but my numNics={}; all ranks "
              "must agree on numNics for same-rail pairing",
              peerRank,
              peerInfo.numNics,
              numNics_));
    }
    const int expectedNumQpsPerPeerPerNic = config_.numQpsPerPeerPerNic();
    if (peerInfo.numQpsPerPeerPerNic != expectedNumQpsPerPeerPerNic) {
      throw std::runtime_error(
          fmt::format(
              "Peer rank {} reports numQpsPerPeerPerNic={} but mine is {}; all "
              "ranks must use the same numQpsPerPeerPerNic",
              peerRank,
              peerInfo.numQpsPerPeerPerNic,
              expectedNumQpsPerPeerPerNic));
    }
    if (peerInfo.maxGroups != config_.maxGroups ||
        peerInfo.qpsPerBlockPerNic != config_.qpsPerBlockPerNic) {
      throw std::runtime_error(
          fmt::format(
              "Peer rank {} reports maxGroups={} qpsPerBlockPerNic={} but "
              "mine are {} {}; all ranks must use the same IB QP shape",
              peerRank,
              peerInfo.maxGroups,
              peerInfo.qpsPerBlockPerNic,
              config_.maxGroups,
              config_.qpsPerBlockPerNic));
    }
  }
}

void MultiPeerIbTransportBase::exchangeRawWithPeer(
    int peerRank,
    const void* localPayload,
    void* remotePayload,
    std::size_t bytes,
    int tag) {
  auto timeoutUs = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::milliseconds(config_.materializePeerTimeoutMs));
  auto waitFuture = [&](auto&& future, const char* op) -> int {
    try {
      return std::move(future).get(timeoutUs);
    } catch (const std::exception&) {
      throw std::runtime_error(
          fmt::format(
              "materializePeer: rank {} {} with peer {} timed out ({}ms)",
              myRank_,
              op,
              peerRank,
              config_.materializePeerTimeoutMs));
    }
  };

  // Lower rank recvs first to avoid deadlock with blocking bootstrap
  // implementations (e.g. MpiBootstrap uses blocking MPI_Send/MPI_Recv).
  if (myRank_ < peerRank) {
    auto recvFuture =
        bootstrap_->recv(remotePayload, bytes, peerRank, /*tag=*/tag);
    int recvResult = waitFuture(std::move(recvFuture), "recv");
    if (recvResult != 0) {
      throw std::runtime_error(
          fmt::format(
              "materializePeer: rank {} recv from peer {} failed (error {})",
              myRank_,
              peerRank,
              recvResult));
    }
    auto sendFuture = bootstrap_->send(
        const_cast<void*>(localPayload), bytes, peerRank, /*tag=*/tag);
    int sendResult = waitFuture(std::move(sendFuture), "send");
    if (sendResult != 0) {
      throw std::runtime_error(
          fmt::format(
              "materializePeer: rank {} send to peer {} failed (error {})",
              myRank_,
              peerRank,
              sendResult));
    }
  } else {
    auto sendFuture = bootstrap_->send(
        const_cast<void*>(localPayload), bytes, peerRank, /*tag=*/tag);
    int sendResult = waitFuture(std::move(sendFuture), "send");
    if (sendResult != 0) {
      throw std::runtime_error(
          fmt::format(
              "materializePeer: rank {} send to peer {} failed (error {})",
              myRank_,
              peerRank,
              sendResult));
    }
    auto recvFuture =
        bootstrap_->recv(remotePayload, bytes, peerRank, /*tag=*/tag);
    int recvResult = waitFuture(std::move(recvFuture), "recv");
    if (recvResult != 0) {
      throw std::runtime_error(
          fmt::format(
              "materializePeer: rank {} recv from peer {} failed (error {})",
              myRank_,
              peerRank,
              recvResult));
    }
  }
}

} // namespace comms::prims
