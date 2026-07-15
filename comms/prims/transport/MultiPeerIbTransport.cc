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
#include <folly/String.h>
#include <glog/logging.h>

#include "comms/ctran/ibverbx/Ibverbx.h"
#include "comms/ctran/ibverbx/IbverbxSymbols.h"
#include "comms/ctran/ibverbx/Mlx5core.h"
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
#if defined(NIC_IONIC)
// ionic (AMD Pensando) routable RoCEv2 GID discovery — see call site below.
#include "comms/prims/transport/amd/nic/ionic/IonicGidDiscovery.h"
#endif
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

// A NIC is Data-Direct-capable iff all three hold: (1) the mlx5dv provider
// supports the device, (2) the mlx5 Data-Direct DMA-BUF verb is usable, and
// (3) the driver exposes a data-direct sysfs path for its context. This mirrors
// the gate NIC discovery (augmentWithDataDirect) and NCCL
// (ncclMlx5dvDmaBufCapable + the sysfs check) use, so the explicit gpuNicMap
// path -- which has no discovery candidate to read capability from -- matches
// the auto-discovery path rather than relying on the sysfs check alone. Always
// false on AMD (no mlx5 Data-Direct).
bool nicSupportsDataDirect(
    [[maybe_unused]] ibverbx::ibv_device* device,
    [[maybe_unused]] ibverbx::ibv_context* ctx,
    [[maybe_unused]] ibverbx::ibv_pd* pd) {
#ifdef __HIP_PLATFORM_AMD__
  return false;
#else
  // Precondition: an opened device/context/PD. Guard so a future caller that
  // probes before opening degrades to "not DD-capable" rather than
  // dereferencing null inside the mlx5 driver. DCHECK surfaces misuse in
  // debug/test builds; release falls back safely.
  DCHECK(device != nullptr && ctx != nullptr && pd != nullptr)
      << "nicSupportsDataDirect called with null device/ctx/pd";
  if (device == nullptr || ctx == nullptr || pd == nullptr) {
    return false;
  }
  const auto& symbols = ibverbx::ibvSymbols;

  // (1) mlx5dv provider supports this device.
  if (symbols.mlx5dv_internal_is_supported == nullptr ||
      !symbols.mlx5dv_internal_is_supported(device)) {
    return false;
  }

  // (2) The mlx5 Data-Direct DMA-BUF verb is usable. Probe with an invalid fd:
  // the driver rejects an unsupported verb with EOPNOTSUPP/EPROTONOSUPPORT,
  // while any other errno (e.g. EBADF) means the verb exists and would have
  // proceeded. This is the predictor NCCL uses (ncclMlx5dvDmaBufCapable) and is
  // the relevant one, since DD registration goes through this exact verb.
  if (symbols.mlx5dv_internal_reg_dmabuf_mr == nullptr) {
    return false;
  }
  errno = 0;
  ibverbx::ibv_mr* probeMr = symbols.mlx5dv_internal_reg_dmabuf_mr(
      pd,
      /*offset=*/0,
      /*length=*/0,
      /*iova=*/0,
      /*fd=*/-1,
      /*access=*/0,
      ibverbx::MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT);
  const bool dmabufUnsupported =
      (errno == EOPNOTSUPP) || (errno == EPROTONOSUPPORT);
  if (probeMr != nullptr) {
    symbols.ibv_internal_dereg_mr(probeMr);
  }
  if (dmabufUnsupported) {
    return false;
  }

  // (3) Data-Direct sysfs path resolves (mlx5dv contract: rc == 0 on success).
  if (symbols.mlx5dv_internal_get_data_direct_sysfs_path == nullptr) {
    return false;
  }
  char ddSysfsPath[4096];
  return symbols.mlx5dv_internal_get_data_direct_sysfs_path(
             ctx, ddSysfsPath, sizeof(ddSysfsPath)) == 0;
#endif
}

// A NIC supports PCIe Relaxed Ordering iff its driver accepts
// IBV_ACCESS_RELAXED_ORDERING on a registration. Probe once by registering a
// tiny host buffer with the flag (the same shape NCCL uses): if the driver
// rejects it, applying the flag to the real (GPU) data MRs would fail every
// registration and throw, breaking transport setup. RO is a TLP attribute
// negotiated via the access flag, independent of the buffer's memory type, so a
// host probe is a valid proxy for flag acceptance. registerBuffer() gates the
// flag on this so an unsupporting NIC falls back to strict ordering.
bool nicSupportsRelaxedOrdering(ibverbx::ibv_pd* pd) {
  // Precondition: pd is an allocated protection domain (openNics throws on a
  // failed alloc before reaching here). Guard defensively so a future caller
  // that probes before allocation degrades to "not RO-capable" rather than
  // dereferencing null inside the driver.
  DCHECK(pd != nullptr) << "nicSupportsRelaxedOrdering called with null pd";
  if (pd == nullptr) {
    return false;
  }
  const auto& symbols = ibverbx::ibvSymbols;
  if (symbols.ibv_internal_reg_mr == nullptr ||
      symbols.ibv_internal_dereg_mr == nullptr) {
    return false;
  }
  alignas(64) char probe[64] = {};
  ibverbx::ibv_mr* mr = symbols.ibv_internal_reg_mr(
      pd,
      probe,
      sizeof(probe),
      ibverbx::IBV_ACCESS_LOCAL_WRITE | ibverbx::IBV_ACCESS_RELAXED_ORDERING);
  if (mr == nullptr) {
    return false;
  }
  symbols.ibv_internal_dereg_mr(mr);
  return true;
}

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

std::size_t alignUp(std::size_t x, std::size_t a) {
  return ((x + a - 1) / a) * a;
}

void checkSendRecvSignalAlignment(const void* ptr, const char* label) {
  if (ptr != nullptr &&
      reinterpret_cast<std::uintptr_t>(ptr) % alignof(SignalState) != 0) {
    throw std::runtime_error(
        fmt::format("{} must be {}-byte aligned", label, alignof(SignalState)));
  }
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
  if (config_.perChannelSize > 0) {
    if (config_.max_num_channels <= 0) {
      throw std::invalid_argument(
          "max_num_channels must be positive when perChannelSize is set");
    }
    if (config_.perChannelSize < 16) {
      throw std::invalid_argument(
          "IB fixed-channel perChannelSize must be >= 16");
    }
    if (config_.perChannelSize % 16 != 0) {
      throw std::invalid_argument(
          "IB fixed-channel perChannelSize must be 16-byte aligned");
    }
    config_.dataBufferSize = config_.fixedChannelDataBufferSize();
    config_.maxGroups = config_.max_num_channels;
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
    // Pass the configured Data-Direct mode through so discovery's DD probing
    // honors config.enableDataDirect (Disabled here yields no DD candidates).
    // On AMD, force Disabled: augmentWithDataDirect()'s ibv_reg_dmabuf_mr probe
    // is not exercised on AMD's libibverbs path.
#ifdef __HIP_PLATFORM_AMD__
    GpuNicDiscovery discovery(
        config_.cudaDevice, config_.ibHca, DataDirectMode::Disabled);
#else
    GpuNicDiscovery discovery(
        config_.cudaDevice, config_.ibHca, config_.enableDataDirect);
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

void MultiPeerIbTransportBase::validateSendRecvConfig() const {
  if (!sendRecvBuffersEnabled()) {
    throw std::runtime_error("MultiPeerIbTransport: send/recv not configured");
  }
  if (config_.pipelineDepth < 1) {
    throw std::invalid_argument(
        "MultiPeerIbTransport: pipelineDepth must be >= 1");
  }
  if (config_.max_num_channels < 1) {
    throw std::invalid_argument(
        "MultiPeerIbTransport: max_num_channels must be >= 1");
  }
  if (config_.dataBufferSize == 0) {
    throw std::invalid_argument(
        "MultiPeerIbTransport: dataBufferSize must be > 0 when send/recv is "
        "enabled");
  }
  if ((config_.dataBufferSize /
       static_cast<std::size_t>(config_.max_num_channels)) < 16) {
    throw std::invalid_argument(
        fmt::format(
            "MultiPeerIbTransport: dataBufferSize / max_num_channels must be >= 16, "
            "got {} / {} = {}",
            config_.dataBufferSize,
            config_.max_num_channels,
            config_.dataBufferSize / config_.max_num_channels));
  }
}

std::size_t MultiPeerIbTransportBase::sendRecvStagingBytesPerPeer() const {
  return static_cast<std::size_t>(config_.pipelineDepth) *
      config_.dataBufferSize;
}

std::size_t MultiPeerIbTransportBase::sendRecvSignalBytesPerPeer() const {
  // Slots are cacheline-strided (kSendRecvSignalSlotStride), not packed, to
  // avoid cross-group false sharing of the send/recv sync flags.
  return 2 * static_cast<std::size_t>(config_.max_num_channels) *
      kSendRecvSignalSlotStride;
}

std::size_t MultiPeerIbTransportBase::sendRecvCounterBytesPerPeer() const {
  return static_cast<std::size_t>(config_.max_num_channels) *
      kSendRecvSignalSlotStride;
}

IbChannelLayout MultiPeerIbTransportBase::channelLayoutForPeer(
    int peerIndex) const {
  if (!sendRecvBuffersEnabled() || sendRecvPeerBuffers_.empty() ||
      peerIndex < 0 ||
      peerIndex >= static_cast<int>(sendRecvPeerBuffers_.size())) {
    return {};
  }
  const auto& pb = sendRecvPeerBuffers_[peerIndex];
  return IbChannelLayout{
      .sendStagingBuf = pb.sendStaging,
      .recvStagingBuf = pb.remoteRecvStaging,
      .sendStagingPtr = static_cast<char*>(pb.sendStaging.ptr),
      .recvStagingPtr = static_cast<char*>(pb.recvStaging.ptr),
      .localSignalBuf = pb.signal,
      .remoteSignalBuf = pb.remoteSignal,
      .localCounterBuf = pb.counter,
      .localCounterCompletionBuf = pb.counterCompletion,
      .maxChannels = config_.max_num_channels,
      .pipelineDepth = config_.pipelineDepth,
      .perChannelSize = config_.perChannelSize,
  };
}

void MultiPeerIbTransportBase::allocateSendRecvBuffersEager(
    IbCounterStorage counterStorage) {
  if (!sendRecvBuffersEnabled()) {
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

  // Align every GPU bulk allocation to the CUDA VMM allocation granularity so
  // that any buffer which is later mlx5 Data-Direct (BAR1) registered has a 0
  // DMA-BUF offset: GB300 rejects a non-zero offset with EOPNOTSUPP, and small
  // cudaMalloc bulks (signal/counter/state) otherwise land unaligned (staging
  // is large enough to already be aligned). Done unconditionally -- not gated
  // on enableDataDirect -- so alignment is decoupled from the DD config and
  // cannot silently break if a buffer is DD-registered; the off-DD cost is only
  // a few MB of rounding on the small bulks. Granularity is queried from the
  // driver (2 MiB fallback); AMD (no Data-Direct) keeps the natural allocation
  // size.
  std::size_t ddAlign = 1;
#ifndef __HIP_PLATFORM_AMD__
  ddAlign = std::size_t{2} << 20; // fallback if the query below fails
  if (cuda_driver_lazy_init() == 0 && pfn_cuDeviceGet != nullptr &&
      pfn_cuMemGetAllocationGranularity != nullptr) {
    CUdevice dev = 0;
    if (pfn_cuDeviceGet(&dev, config_.cudaDevice) == CUDA_SUCCESS) {
      CUmemAllocationProp prop = {};
      prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
      prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      prop.location.id = dev;
      std::size_t granularity = 0;
      if (pfn_cuMemGetAllocationGranularity(
              &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) ==
              CUDA_SUCCESS &&
          granularity > 0) {
        ddAlign = granularity;
      }
    }
  }
#endif
  auto allocateBulk = [&](std::size_t perPeer, const char* label) {
    const std::size_t used = perPeer * numPeers;
    const std::size_t allocBytes = ((used + ddAlign - 1) / ddAlign) * ddAlign;
    auto buf = std::make_unique<meta::comms::DeviceBuffer>(allocBytes);
    checkSlotGpu(
        slotGpuMemset(buf->get(), 0, allocBytes),
        fmt::format("MultiPeerIbTransport: zero send/recv {}", label));
    return buf;
  };

  sendRecvPeerBuffers_.resize(numPeers);

  sendRecvSendStagingBulk_ = allocateBulk(stagingPerPeer, "send staging bulk");
  sendRecvRecvStagingBulk_ = allocateBulk(stagingPerPeer, "recv staging bulk");

  // Signal and the device counter are the small RDMA-registered control
  // buffers; pack them into ONE granularity-aligned allocation so they cost a
  // single granularity unit and share one aligned Data-Direct MR (offset 0),
  // instead of one aligned unit each. Region layout: [signal | (device
  // counter)], each SignalState-aligned; the whole allocation rounded to the
  // VMM granularity so the DD export offset is 0.
  const std::size_t signalTotal = signalPerPeer * numPeers;
  const bool deviceCounter = (counterStorage == IbCounterStorage::Device);
  const std::size_t counterTotal =
      deviceCounter ? counterPerPeer * numPeers : 0;
  const std::size_t counterOff = alignUp(signalTotal, alignof(SignalState));
  const std::size_t controlBytes = alignUp(counterOff + counterTotal, ddAlign);
  sendRecvControlBulk_ =
      std::make_unique<meta::comms::DeviceBuffer>(controlBytes);
  checkSlotGpu(
      slotGpuMemset(sendRecvControlBulk_->get(), 0, controlBytes),
      "MultiPeerIbTransport: zero send/recv control bulk");
  char* controlBase = static_cast<char*>(sendRecvControlBulk_->get());
  checkSendRecvSignalAlignment(
      controlBase, "MultiPeerIbTransport: send/recv signal base");

  // Staging is bulk data: opt into Relaxed Ordering. Signal/counter stay strict
  // so a flag write can't be reordered ahead of the data on the shared route.
  // (Data-Direct, when active, applies to all of these automatically.)
  auto sendStagingBulkReg = registerBuffer(
      sendRecvSendStagingBulk_->get(),
      stagingPerPeer * numPeers,
      /*relaxedOrdering=*/true);
  sendRecvRecvStagingBulkReg_ = registerBuffer(
      sendRecvRecvStagingBulk_->get(),
      stagingPerPeer * numPeers,
      /*relaxedOrdering=*/true);
  // One registration covers the whole control allocation (registerBuffer
  // registers the entire underlying allocation regardless of the size arg; the
  // base is granularity-aligned so the DMA-BUF offset is 0). The signal and
  // device-counter handles are then just views into this single MR (same lkey),
  // so there is no second registration / refcount to balance.
  sendRecvSignalBulkReg_ = registerBuffer(controlBase, controlBytes);

  IbgdaLocalBuffer counterBulkBuf;
  IbgdaLocalBuffer counterCompletionBulkBuf;
  if (deviceCounter) {
    // Device counter is a view into the control MR (same lkey). The NIC bumps
    // it via a loopback RDMA atomic (IBGDA).
    checkSendRecvSignalAlignment(
        controlBase + counterOff,
        "MultiPeerIbTransport: send/recv counter base");
    sendRecvCounterBulkReg_ = sendRecvSignalBulkReg_.subBuffer(counterOff);
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
    checkSendRecvSignalAlignment(
        sendRecvHostCounterAllocation_.devicePtr,
        "MultiPeerIbTransport: send/recv host counter device base");
    checkSendRecvSignalAlignment(
        sendRecvHostCounterAllocation_.hostPtr,
        "MultiPeerIbTransport: send/recv host counter host base");
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
  }

  VLOG(1) << "MultiPeerIbTransport: rank " << myRank_
          << " allocated send/recv staging for " << numPeers
          << " peers (staging=" << stagingPerPeer << "B per peer, counter="
          << (counterStorage == IbCounterStorage::Device ? "device" : "host")
          << ")";
}

void MultiPeerIbTransportBase::exchangeSendRecvBuffersEager() {
  if (!sendRecvBuffersEnabled() || sendRecvPeerBuffers_.empty()) {
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
  // The control bulk was registered exactly once (signal + device-counter are
  // views into that single MR); state was never registered.
  deregisterNoexcept(
      sendRecvControlBulk_ ? sendRecvControlBulk_->get() : nullptr);

  sendRecvSendStagingBulk_.reset();
  sendRecvRecvStagingBulk_.reset();
  sendRecvControlBulk_.reset();
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
  if (!sendRecvBuffersEnabled()) {
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
  const bool deviceCounter = (counterStorage == IbCounterStorage::Device);

  // One contiguous device buffer: sendStaging | recvStaging | signal,
  // plus the counter when it is device-resident. SignalState-backed regions are
  // padded before their starts so the first slot and every strided slot are
  // aligned. A HostPinned counter is allocated separately (host-mapped, never
  // RDMA-registered).
  std::size_t off = 0;
  const std::size_t sendStagingOff = off;
  off += stagingPerPeer;
  const std::size_t recvStagingOff = off;
  off += stagingPerPeer;
  off = alignUp(off, alignof(SignalState));
  const std::size_t signalOff = off;
  off += signalPerPeer;
  std::size_t counterOff = 0;
  if (deviceCounter) {
    off = alignUp(off, alignof(SignalState));
    counterOff = off;
    off += counterPerPeer;
  }
  const std::size_t total = off;
  auto buf = std::make_unique<meta::comms::DeviceBuffer>(total);
  checkSlotGpu(
      slotGpuMemset(buf->get(), 0, total),
      "MultiPeerIbTransport: zero per-peer send/recv buffer");
  auto reg = registerBuffer(buf->get(), total);

  char* p = static_cast<char*>(buf->get());
  auto& pb = sendRecvPeerBuffers_[peerIndex];
  pb.sendStaging = IbgdaLocalBuffer(p + sendStagingOff, reg.lkey_per_device);
  void* recvStagingPtr = p + recvStagingOff;
  pb.recvStaging = IbgdaLocalBuffer(recvStagingPtr, reg.lkey_per_device);
  void* signalPtr = p + signalOff;
  checkSendRecvSignalAlignment(
      signalPtr, "MultiPeerIbTransport: lazy send/recv signal base");
  pb.signal = IbgdaLocalBuffer(signalPtr, reg.lkey_per_device);
  if (deviceCounter) {
    checkSendRecvSignalAlignment(
        p + counterOff, "MultiPeerIbTransport: lazy send/recv counter base");
    pb.counter = IbgdaLocalBuffer(p + counterOff, reg.lkey_per_device);
    pb.counterCompletion = pb.counter;
  } else {
    auto alloc = allocateCounterSlotAllocation(
        IbCounterStorage::HostPinned,
        counterPerPeer,
        "lazy send/recv host counter");
    checkSendRecvSignalAlignment(
        alloc.devicePtr,
        "MultiPeerIbTransport: lazy send/recv host counter device base");
    checkSendRecvSignalAlignment(
        alloc.hostPtr,
        "MultiPeerIbTransport: lazy send/recv host counter host base");
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
  if (!sendRecvBuffersEnabled() || peerIndex < 0 ||
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
  // Reset the per-peer views field-wise.
  auto& pb = sendRecvPeerBuffers_[peerIndex];
  pb.sendStaging = IbgdaLocalBuffer{};
  pb.recvStaging = IbgdaLocalBuffer{};
  pb.signal = IbgdaLocalBuffer{};
  pb.counter = IbgdaLocalBuffer{};
  pb.counterCompletion = IbgdaLocalBuffer{};
  pb.remoteRecvStaging = IbgdaRemoteBuffer{};
  pb.remoteSignal = IbgdaRemoteBuffer{};
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
  const bool usingGpuNicMap =
      it != config_.gpuNicMap.end() && !it->second.empty();
  if (usingGpuNicMap) {
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
    // Pass the configured Data-Direct mode through so discovery's DD probing
    // honors config.enableDataDirect (Disabled here yields no DD candidates).
    // On AMD, force Disabled: augmentWithDataDirect()'s ibv_reg_dmabuf_mr probe
    // is not exercised on AMD's libibverbs path.
#ifdef __HIP_PLATFORM_AMD__
    auto discovery = GpuNicDiscovery(
        config_.cudaDevice, config_.ibHca, DataDirectMode::Disabled);
#else
    auto discovery = GpuNicDiscovery(
        config_.cudaDevice, config_.ibHca, config_.enableDataDirect);
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
      // Discovery already probed Data-Direct capability for each candidate;
      // read it cheaply (no extra sysfs probe needed on this path).
      nics_[n].isDataDirect = candidates[n].isDataDirect;
    }
    VLOG(1) << "MultiPeerIbTransport: auto-discovered NIC "
            << nics_[0].deviceName << " for GPU device " << config_.cudaDevice;
  }

  // Data-Direct active per NIC = requested (config Auto) AND capable. RO mode
  // string for logging by enum value (Disabled/Enabled/Auto) so autodetection
  // vs explicit opt-in stays distinguishable. Computed once before the bring-up
  // loop so each NIC can log its resolved status inline.
  const bool ddEnabled = config_.enableDataDirect != DataDirectMode::Disabled;
  const char* roMode = "auto";
  switch (config_.enablePciRelaxedOrdering) {
    case MultipeerIbTransportConfig::PciRelaxedOrderingMode::Disabled:
      roMode = "disabled";
      break;
    case MultipeerIbTransportConfig::PciRelaxedOrderingMode::Enabled:
      roMode = "enabled";
      break;
    case MultipeerIbTransportConfig::PciRelaxedOrderingMode::Auto:
      roMode = "auto";
      break;
  }
  // RO must be uniform across NICs (the MR cache keys on one effective-ordering
  // bool per allocation); AND in each NIC's capability as it is brought up.
  relaxedOrderingCapable_ = numNics_ > 0;

  // Open + setup each NIC: find by name, open ctx, alloc PD, query GID + port.
  // Each NIC resolves its GID starting from the caller-configured default, not
  // a previous NIC's discovered index; captured once so per-NIC discovery below
  // cannot leak across iterations.
  const int callerGidIndex = gidIndex_;
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

    // Probe PCIe Relaxed Ordering support once per NIC. registerBuffer() gates
    // IBV_ACCESS_RELAXED_ORDERING on this; on a driver that rejects the flag,
    // applying it would fail every data-MR registration and break setup, so an
    // unsupporting NIC falls back to strict ordering instead.
    nics_[n].relaxedOrderingCapable =
        nicSupportsRelaxedOrdering(nics_[n].ibvPd);

    // Detect Data-Direct capability for the explicit gpuNicMap path.
    // registerBuffer() auto-selects the Data-Direct (BAR1) path when
    // nics_[n].isDataDirect is set. The auto-discovery path sets that flag for
    // free from the discovery candidate; the gpuNicMap path bypasses discovery,
    // so run the same capability gate here on the just-opened device.
    if (usingGpuNicMap) {
      nics_[n].isDataDirect = nicSupportsDataDirect(
          deviceList[nicIdx], nics_[n].ibvCtx, nics_[n].ibvPd);
    }

    int nicGidIndex = callerGidIndex;
    if (symbols.ibv_internal_query_gid(
            nics_[n].ibvCtx, 1, nicGidIndex, &nics_[n].localGid) != 0) {
      throw std::runtime_error(
          "Failed to query GID at index " + std::to_string(nicGidIndex) +
          " on NIC " + nics_[n].deviceName);
    }

#if defined(NIC_IONIC)
    // ionic's routable RoCEv2 GID is not at the shared default index 3 (that
    // slot is empty), so auto-discover it; see IonicGidDiscovery.h.
    resolveRoceGidIndex(
        symbols,
        nics_[n].ibvCtx,
        nics_[n].deviceName,
        /*port=*/1,
        /*callerPinnedIndex=*/config_.gidIndex.has_value(),
        nicGidIndex,
        nics_[n].localGid);
#endif
    // The transport uses one shared sgid_index for all NICs' address handles;
    // ionic deployments are homogeneous, so each NIC resolves the same index.
    gidIndex_ = nicGidIndex;

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

    // NIC fully brought up: fold its RO capability into the cross-NIC aggregate
    // and log its resolved Data-Direct / Relaxed-Ordering status inline.
    relaxedOrderingCapable_ =
        relaxedOrderingCapable_ && nics_[n].relaxedOrderingCapable;
    LOG(INFO) << "MultiPeerIbTransport: NIC " << n << " ("
              << nics_[n].deviceName << ") Data-Direct enabled=" << ddEnabled
              << " nicCapable=" << nics_[n].isDataDirect << " -> "
              << ((ddEnabled && nics_[n].isDataDirect) ? "ACTIVE" : "inactive")
              << "; relaxedOrdering=" << roMode
              << " nicCapable=" << nics_[n].relaxedOrderingCapable;
  }

  // PCIe Relaxed Ordering is applied only when every NIC accepts the flag
  // (aggregated above). Surface an explicit Enabled request that can't be met
  // (it falls back to strict ordering rather than throwing); otherwise just
  // record the resolved setting: config, capability, and the ordering in
  // effect.
  const bool useRelaxedOrdering =
      relaxedOrderingActiveForNic(config_, relaxedOrderingCapable_);
  if (config_.enablePciRelaxedOrdering ==
          MultipeerIbTransportConfig::PciRelaxedOrderingMode::Enabled &&
      !relaxedOrderingCapable_) {
    LOG(WARNING) << "MultiPeerIbTransport: PCIe Relaxed Ordering requested "
                    "(Enabled) but not supported on all NICs; falling back to "
                    "strict ordering on data MRs";
  } else {
    LOG(INFO) << "MultiPeerIbTransport: PCIe Relaxed Ordering config=" << roMode
              << " allNicsCapable=" << relaxedOrderingCapable_ << " -> "
              << (useRelaxedOrdering ? "ACTIVE" : "strict");
  }

  // Success: SCOPE_EXIT frees the device list; SCOPE_FAIL is skipped, so the
  // opened ctx/PD are kept for the transport's lifetime.
}

IbgdaLocalBuffer MultiPeerIbTransportBase::registerBuffer(
    void* ptr,
    std::size_t size,
    bool relaxedOrdering) {
  if (ptr == nullptr || size == 0) {
    throw std::invalid_argument("Invalid buffer pointer or size");
  }

  // Resolve the effective Relaxed Ordering once, up front: the caller's request
  // gated by config (NCCL_IB_PCI_RELAXED_ORDERING) AND by NIC capability probed
  // during openNics. Gating on capability means a NIC whose driver rejects
  // IBV_ACCESS_RELAXED_ORDERING falls back to strict ordering here rather than
  // failing every data-MR registration below. This is the actual MR access
  // flag, so it is part of the cache identity (key) below and is reused for the
  // access flags — keeping the two from drifting apart.
  const bool useRelaxedOrdering = relaxedOrdering &&
      relaxedOrderingActiveForNic(config_, relaxedOrderingCapable_);

  // Fast path: containment lookup — if [ptr, ptr+size) falls entirely within an
  // existing registration with the same effective ordering, return the cached
  // per-NIC lkeys with no driver call.
  const auto addr = reinterpret_cast<uintptr_t>(ptr);
  auto it = registeredBuffers_.upper_bound(addr);
  if (it != registeredBuffers_.begin()) {
    --it;
    if (addr + size <= it->first + it->second.allocSize) {
      // The cache holds one MR set per allocation; its access flags (including
      // Relaxed Ordering) are fixed at registration, so the effective ordering
      // is part of the cache key. A containment hit resolving to different
      // ordering would silently get the wrong semantics.
      if (it->second.relaxedOrdering != useRelaxedOrdering) {
        throw std::runtime_error(
            fmt::format(
                "registerBuffer: ptr={} is contained in an existing registration "
                "(allocBase=0x{:x}) registered with relaxedOrdering={} but "
                "requested relaxedOrdering={}",
                ptr,
                it->first,
                it->second.relaxedOrdering,
                useRelaxedOrdering));
      }
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
  bool isMultiSegment = false;
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
  // cuMemGetAddressRange returns a single physical segment for cuMem VMM
  // (cuMemCreate/cuMemMap) buffers. When the caller's range spans multiple
  // segments (expandable-segment / disjoint allocations), the returned range
  // covers only the first segment. Widen to the caller's range so the MR
  // covers the full contiguous VA — ibv_reg_dmabuf_mr handles the underlying
  // physical discontinuity transparently.
  {
    const auto requestedEnd = reinterpret_cast<uintptr_t>(ptr) + size;
    const auto allocEnd = static_cast<uintptr_t>(allocBase) + allocSize;
    if (requestedEnd > allocEnd) {
      allocBase = reinterpret_cast<CUdeviceptr>(ptr);
      allocSize = size;
      isMultiSegment = true;
    }
  }
#endif
  auto& symbols = ibverbx::ibvSymbols;
  int accessFlags = ibverbx::IBV_ACCESS_LOCAL_WRITE |
      ibverbx::IBV_ACCESS_REMOTE_WRITE | ibverbx::IBV_ACCESS_REMOTE_READ |
      ibverbx::IBV_ACCESS_REMOTE_ATOMIC;
  // PCIe Relaxed Ordering (resolved above as useRelaxedOrdering): on bulk data
  // MRs only, let NIC<->HBM DMA TLPs pipeline instead of strict-ordering to
  // ~half rate. Signal/counter MRs stay strict so a strict flag write cannot be
  // reordered ahead of the data on the shared route.
  if (useRelaxedOrdering) {
    accessFlags |= ibverbx::IBV_ACCESS_RELAXED_ORDERING;
  }
  // mlx5 Data-Direct (config.enableDataDirect): on a DD-capable NIC, register
  // through the data-direct (BAR1) PCIe path for ~2x NIC<->HBM write BW on
  // GB300 (NCCL's GDAKI path). Applied to every MR registered here so data and
  // signal/counter share the same route on the same QP -- preserving
  // data-before-flag ordering without a flush. Autodetected per NIC.

  CachedMr cached;
  cached.allocSize = allocSize;
  cached.refs = 1;
  cached.relaxedOrdering = useRelaxedOrdering;

  // Per NIC, register the MR in priority order, each path falling through to
  // the next on failure:
  //   1. Data-Direct: PCIe-mapped (BAR1) dmabuf + mlx5dv DATA_DIRECT reg. Only
  //      on a DD-capable NIC with DD enabled. The regular C2C dmabuf is NOT
  //      used here -- DD needs its own PCIe-mapped dmabuf.
  //   2. Regular DMABUF: default (C2C) dmabuf + ibv_reg_dmabuf_mr.
  //   3. Plain ibv_reg_mr.
  // If any NIC ultimately fails, deregister everything already done and throw.
  for (int n = 0; n < numNics_; ++n) {
    ibverbx::ibv_mr* mr = nullptr;
    // 1. Data-Direct. When selected for this NIC it is mandatory: every MR on
    //    the NIC must share the Data-Direct route so data and signal/counter
    //    stay ordered on one QP without a flush (see the per-MR-uniformity note
    //    above). A per-MR fallback would mix DD and non-DD MRs on the same QP
    //    and break that ordering, so any Data-Direct failure here is fatal --
    //    deregister the MRs done so far and throw, rather than silently
    //    downgrading this one MR to the regular path.
    if (dataDirectActiveForNic(config_, nics_[n].isDataDirect) &&
        symbols.mlx5dv_internal_reg_dmabuf_mr != nullptr) {
      auto ddDmabuf = export_gpu_dmabuf_aligned(
          reinterpret_cast<void*>(allocBase),
          allocSize,
          DmaBufExportKind::Pcie);
      if (!ddDmabuf) {
        for (int j = 0; j < n; ++j) {
          symbols.ibv_internal_dereg_mr(cached.mrs[j]);
        }
        throw std::runtime_error(
            fmt::format(
                "Data-Direct selected for NIC {} but PCIe DMA-BUF export failed "
                "(allocSize={}); refusing to mix DD and non-DD MRs on one QP "
                "(PCIe DMA-BUF export needs CUDA >= 12.8 and a capable driver)",
                n,
                allocSize));
      }
      errno = 0;
      mr = symbols.mlx5dv_internal_reg_dmabuf_mr(
          nics_[n].ibvPd,
          ddDmabuf->alignment.dmabufOffset,
          allocSize,
          static_cast<uint64_t>(allocBase),
          ddDmabuf->fd,
          accessFlags,
          ibverbx::MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT);
      // Capture the registration errno before close(): a failing (or even
      // successful) close() may clobber errno, which would mask the real
      // mlx5dv_reg_dmabuf_mr failure reason in the message below.
      const int regErrno = errno;
      close(ddDmabuf->fd);
      if (!mr) {
        for (int j = 0; j < n; ++j) {
          symbols.ibv_internal_dereg_mr(cached.mrs[j]);
        }
        throw std::runtime_error(
            fmt::format(
                "Data-Direct mlx5dv_reg_dmabuf_mr failed for NIC {} "
                "(allocSize={} errno={} ({})); refusing to mix DD and non-DD "
                "MRs on one QP",
                n,
                allocSize,
                regErrno,
                folly::errnoStr(regErrno)));
      }
    }
    // 2. Regular DMABUF (default C2C mapping).
    if (!mr) {
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
    }
    // 3. Plain reg_mr. ibv_reg_mr cannot handle physically disjoint pages
    //    behind contiguous VA (cuMem VMM multi-segment buffers), so reject
    //    rather than silently producing a broken MR.
    if (!mr) {
      if (isMultiSegment) {
        for (int j = 0; j < n; ++j) {
          symbols.ibv_internal_dereg_mr(cached.mrs[j]);
        }
        throw std::runtime_error(
            fmt::format(
                "registerBuffer: buffer spans multiple cuMem VMM segments "
                "(allocSize={}) and DMA-BUF registration failed on NIC {}; "
                "ibv_reg_mr cannot handle disjoint physical pages",
                allocSize,
                n));
      }
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
    const int expectedNumQpsPerPeerPerNic =
        config_.fixedChannelMainQpsPerPeerPerNic();
    if (peerInfo.numQpsPerPeerPerNic != expectedNumQpsPerPeerPerNic) {
      throw std::runtime_error(
          fmt::format(
              "Peer rank {} reports numQpsPerPeerPerNic={} but mine is {}; all "
              "ranks must use the same numQpsPerPeerPerNic",
              peerRank,
              peerInfo.numQpsPerPeerPerNic,
              expectedNumQpsPerPeerPerNic));
    }
    if (peerInfo.maxGroups != config_.max_num_channels ||
        peerInfo.qpsPerBlockPerNic != config_.qpsPerConnection) {
      throw std::runtime_error(
          fmt::format(
              "Peer rank {} reports maxGroups={} qpsPerBlockPerNic={} but "
              "mine are {} {}; all ranks must use the same IB QP shape",
              peerRank,
              peerInfo.maxGroups,
              peerInfo.qpsPerBlockPerNic,
              config_.max_num_channels,
              config_.qpsPerConnection));
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
