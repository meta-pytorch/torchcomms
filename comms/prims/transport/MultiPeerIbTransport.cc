// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/transport/MultiPeerIbTransport.h"

#include <cerrno>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
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
#include "comms/prims/transport/amd/DocaCompat.h"
#else
#include "comms/prims/platform/CudaDriverLazy.h"
#include "comms/prims/platform/DocaHostUtils.h"
#endif

namespace comms::prims {

namespace {
constexpr int kDefaultGidIndex = 3; // Default RoCE GID index
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
    if (peerInfo.numQpsPerPeerPerNic != config_.numQpsPerPeerPerNic) {
      throw std::runtime_error(
          fmt::format(
              "Peer rank {} reports numQpsPerPeerPerNic={} but mine is {}; all "
              "ranks must use the same numQpsPerPeerPerNic",
              peerRank,
              peerInfo.numQpsPerPeerPerNic,
              config_.numQpsPerPeerPerNic));
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
