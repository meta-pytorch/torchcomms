// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/transport/MultiPeerTransport.h"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef __HIP_PLATFORM_AMD__
// On AMD, HIPify renames `cuda*` runtime calls to `hip*`; pull in the HIP
// runtime so those symbols resolve. The CUDA driver-API path
// (`CudaDriverLazy.h` + `cuMem*`) is unavailable; the corresponding code
// paths in this file are guarded by `#ifndef __HIP_PLATFORM_AMD__`.
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>

#include "comms/prims/platform/CudaDriverLazy.h"
#endif

#include <glog/logging.h>

#include "comms/prims/bootstrap/NvlBootstrapAdapter.h"
#include "comms/prims/memory/CuMemAllocation.h"
#include "comms/prims/topology/TopologyDiscovery.h"
#include "comms/prims/transport/MultiPeerDeviceHandle.cuh"

namespace comms::prims {

namespace {

#define CUDA_CHECK(cmd)                                                    \
  do {                                                                     \
    cudaError_t err = (cmd);                                               \
    if (err != cudaSuccess) {                                              \
      throw std::runtime_error(                                            \
          std::string("CUDA error: ") + cudaGetErrorString(err) + " at " + \
          __FILE__ + ":" + std::to_string(__LINE__));                      \
    }                                                                      \
  } while (0)

#define CU_CHECK(cmd)                                                          \
  do {                                                                         \
    CUresult err = (cmd);                                                      \
    if (err != CUDA_SUCCESS) {                                                 \
      const char* errStr = nullptr;                                            \
      pfn_cuGetErrorString(err, &errStr);                                      \
      throw std::runtime_error(                                                \
          std::string("CUDA driver error: ") + (errStr ? errStr : "unknown") + \
          " at " + __FILE__ + ":" + std::to_string(__LINE__));                 \
    }                                                                          \
  } while (0)

} // namespace

namespace detail {

void validateChannelProtocolRecords(
    std::span<const ChannelProtocolRecord> records) {
  if (records.empty()) {
    throw std::invalid_argument(
        "channel protocol validation requires at least one rank");
  }
  for (size_t rank = 0; rank < records.size(); ++rank) {
    const auto& record = records[rank];
    if (record.mode != PrimsChannelMode::kEager &&
        record.mode != PrimsChannelMode::kLazyPrefix) {
      throw std::runtime_error(
          "invalid channel protocol record from rank " + std::to_string(rank));
    }
  }
  for (size_t rank = 1; rank < records.size(); ++rank) {
    if (records[rank] != records.front()) {
      throw std::runtime_error(
          "channel protocol mismatch between rank 0 and rank " +
          std::to_string(rank));
    }
  }
}

void exchangeAndValidateChannelProtocol(
    meta::comms::IBootstrap& bootstrap,
    int rank,
    int nRanks,
    const ChannelProtocolRecord& localRecord) {
  std::vector<ChannelProtocolRecord> records(nRanks);
  records.at(rank) = localRecord;
  const int rc =
      bootstrap
          .allGather(
              records.data(), sizeof(ChannelProtocolRecord), rank, nRanks)
          .get();
  if (rc != 0) {
    throw std::runtime_error(
        "channel protocol allGather failed with error " + std::to_string(rc));
  }
  validateChannelProtocolRecords(records);
}

void validatePrimsTransportRoutes(
    std::span<const PrimsTransportRoute> routeMatrix,
    int nRanks) {
  const auto ranks = static_cast<size_t>(nRanks);
  if (nRanks <= 0 || routeMatrix.size() != ranks * ranks) {
    throw std::invalid_argument("invalid PRIMS transport route matrix");
  }
  const auto route = [&](int from, int to) {
    return routeMatrix[static_cast<size_t>(from) * ranks + to];
  };
  for (int rank = 0; rank < nRanks; ++rank) {
    if (route(rank, rank) != PrimsTransportRoute::kSelf) {
      throw std::runtime_error(
          "PRIMS transport route matrix has an invalid self edge");
    }
    for (int peer = rank + 1; peer < nRanks; ++peer) {
      if (route(rank, peer) != route(peer, rank)) {
        throw std::runtime_error(
            "PRIMS transport route mismatch between rank " +
            std::to_string(rank) + " and rank " + std::to_string(peer));
      }
    }
  }
}

void exchangeAndValidatePrimsTransportRoutes(
    meta::comms::IBootstrap& bootstrap,
    int rank,
    int nRanks,
    std::span<const PrimsTransportRoute> localRoutes) {
  if (localRoutes.size() != static_cast<size_t>(nRanks)) {
    throw std::invalid_argument("invalid local PRIMS transport route row");
  }
  std::vector<PrimsTransportRoute> routeMatrix(
      static_cast<size_t>(nRanks) * nRanks);
  std::copy(
      localRoutes.begin(),
      localRoutes.end(),
      routeMatrix.begin() + static_cast<size_t>(rank) * nRanks);
  const int rc = bootstrap
                     .allGather(
                         routeMatrix.data(),
                         nRanks * sizeof(PrimsTransportRoute),
                         rank,
                         nRanks)
                     .get();
  if (rc != 0) {
    throw std::runtime_error(
        "PRIMS transport route allGather failed with error " +
        std::to_string(rc));
  }
  validatePrimsTransportRoutes(routeMatrix, nRanks);
}

} // namespace detail

MultiPeerTransport::MultiPeerTransport(
    int myRank,
    int nRanks,
    int deviceId,
    std::shared_ptr<meta::comms::IBootstrap> bootstrap,
    const MultiPeerTransportConfig& config,
    std::optional<TopologyResult> topo)
    : myRank_(myRank),
      nRanks_(nRanks),
      deviceId_(deviceId),
      bootstrap_(std::move(bootstrap)) {
  if (!topo.has_value()) {
    TopologyDiscovery topoDiscovery;
    topo = topoDiscovery.discover(
        myRank_, nRanks_, deviceId_, *bootstrap_, config.topoConfig);
  }
  initFromTopology(std::move(*topo), config);
}

void MultiPeerTransport::initFromTopology(
    TopologyResult topo,
    const MultiPeerTransportConfig& config) {
  nvlPeerRanks_ = std::move(topo.nvlPeerRanks);
  globalToNvlLocal_ = std::move(topo.globalToNvlLocal);

  // Derive fields from the slim TopologyResult.
  nvlNRanks_ = static_cast<int>(nvlPeerRanks_.size()) + 1;
  nvlLocalRank_ = globalToNvlLocal_.at(myRank_);

  typePerRank_.resize(nRanks_);

  if (config.disableIb) {
    // NVL-only mode: validate all non-self peers are NVL-reachable, then
    // force every non-self rank to P2P_NVL. IBGDA is never constructed.
    LOG(INFO) << "MultiPeerTransport: rank " << myRank_
              << " IBGDA disabled by config, NVL-only mode";

    for (int r = 0; r < nRanks_; ++r) {
      if (r == myRank_) {
        typePerRank_.at(r) = TransportType::SELF;
      } else if (globalToNvlLocal_.count(r)) {
        typePerRank_.at(r) = TransportType::P2P_NVL;
      } else {
        throw std::runtime_error(
            "MultiPeerTransport: IBGDA disabled but rank " + std::to_string(r) +
            " is not NVL-reachable from rank " + std::to_string(myRank_) +
            ". All ranks must be in the same NVL domain when "
            "NCCL_CTRAN_PIPES_DISABLE_IB=1.");
      }
    }
    // ibPeerRanks_ stays empty; ibgdaTransport_ stays nullptr.
  } else {
    const auto ibTransportType = config.ibMode == IbBackendMode::kIbrc
        ? TransportType::P2P_IBRC
        : TransportType::P2P_IBGDA;
    for (int r = 0; r < nRanks_; ++r) {
      if (r == myRank_) {
        typePerRank_.at(r) = TransportType::SELF;
      } else if (globalToNvlLocal_.count(r)) {
        typePerRank_.at(r) = TransportType::P2P_NVL;
      } else {
        typePerRank_.at(r) = ibTransportType;
      }
    }

    for (int r = 0; r < nRanks_; ++r) {
      if (typePerRank_.at(r) == TransportType::P2P_IBGDA ||
          typePerRank_.at(r) == TransportType::P2P_IBRC) {
        ibPeerRanks_.push_back(r);
      }
    }
  }

  // Log topology summary (init-time, once per communicator).
  {
    int nvlCount = 0;
    int ibgdaCount = 0;
    int ibrcCount = 0;
    for (int r = 0; r < nRanks_; ++r) {
      if (typePerRank_[r] == TransportType::P2P_NVL) {
        ++nvlCount;
      } else if (typePerRank_[r] == TransportType::P2P_IBGDA) {
        ++ibgdaCount;
      } else if (typePerRank_[r] == TransportType::P2P_IBRC) {
        ++ibrcCount;
      }
    }
    LOG(INFO) << "MultiPeerTransport: rank " << myRank_ << "/" << nRanks_
              << " topology: " << nvlCount << " NVL peers, " << ibgdaCount
              << " IBGDA peers, " << ibrcCount << " IBRC peers";
  }
  for (int r = 0; r < nRanks_; ++r) {
    VLOG(1) << "MultiPeerTransport: rank " << myRank_ << " -> rank " << r
            << ": " << transport_type_name(typePerRank_[r]);
  }

  // Create NVLink sub-transport with NvlBootstrapAdapter
  if (!nvlPeerRanks_.empty()) {
    std::vector<int> localRankToCommRank(nvlNRanks_);
    for (const auto& [globalRank, nvlLocal] : globalToNvlLocal_) {
      localRankToCommRank[nvlLocal] = globalRank;
    }

    nvlBootstrapAdapter_ = std::make_shared<NvlBootstrapAdapter>(
        bootstrap_, std::move(localRankToCommRank));

    nvlTransport_ = std::make_unique<MultiPeerNvlTransport>(
        nvlLocalRank_,
        nvlNRanks_,
        deviceId_,
        nvlBootstrapAdapter_,
        config.nvlConfig);
    VLOG(1) << "MultiPeerTransport: rank " << myRank_
            << " created NVL sub-transport, nvlNRanks=" << nvlNRanks_
            << " nvlLocalRank=" << nvlLocalRank_;
  }

  // Create the selected IB backend when at least one peer prefers IB. Its
  // global-rank table can also serve NVL-preferred peers when an algorithm
  // explicitly requests an IBGDA fallback.
  auto ibConfig = config.ibConfig.normalizedChannelGeometry();
  channelMode_ = ibConfig.lazyChannels ? detail::PrimsChannelMode::kLazyPrefix
                                       : detail::PrimsChannelMode::kEager;
  ibChannelCapacity_ = static_cast<uint32_t>(ibConfig.max_num_channels);
  if (!config.disableIb && !ibPeerRanks_.empty()) {
    ibConfig.cudaDevice = deviceId_;
    if (config.ibMode == IbBackendMode::kIbrc) {
      ibrcTransport_ = std::make_unique<MultipeerIbrcTransport>(
          myRank_, nRanks_, bootstrap_, ibConfig);
      VLOG(1) << "MultiPeerTransport: rank " << myRank_
              << " created IBRC sub-transport for " << ibPeerRanks_.size()
              << " peers";
    } else {
      ibgdaTransport_ = std::make_unique<MultipeerIbgdaTransport>(
          myRank_, nRanks_, bootstrap_, ibConfig);
      VLOG(1) << "MultiPeerTransport: rank " << myRank_
              << " created IBGDA sub-transport for " << ibPeerRanks_.size()
              << " peers";
    }
  }
}

MultiPeerTransport::~MultiPeerTransport() {
  free_device_handle();
}

void MultiPeerTransport::setExternalNvlDataBuffers(
    ExternalStagingBuffers externalStagingBuffers) {
  if (nvlTransport_) {
    nvlTransport_->setExternalDataBuffers(std::move(externalStagingBuffers));
  }
}

void MultiPeerTransport::exchange() {
  const detail::ChannelProtocolRecord channelProtocol{
      .mode = channelMode_,
      .channelCapacity = ibChannelCapacity_,
  };
  detail::exchangeAndValidateChannelProtocol(
      *bootstrap_, myRank_, nRanks_, channelProtocol);
  std::vector<detail::PrimsTransportRoute> localRoutes;
  localRoutes.reserve(typePerRank_.size());
  for (const auto type : typePerRank_) {
    switch (type) {
      case TransportType::SELF:
        localRoutes.push_back(detail::PrimsTransportRoute::kSelf);
        break;
      case TransportType::P2P_NVL:
        localRoutes.push_back(detail::PrimsTransportRoute::kNvl);
        break;
      case TransportType::P2P_IBGDA:
        localRoutes.push_back(detail::PrimsTransportRoute::kIbgda);
        break;
      case TransportType::P2P_IBRC:
        localRoutes.push_back(detail::PrimsTransportRoute::kIbrc);
        break;
    }
  }
  detail::exchangeAndValidatePrimsTransportRoutes(
      *bootstrap_, myRank_, nRanks_, localRoutes);
#ifndef __HIP_PLATFORM_AMD__
  // CUDA driver-API init is required for the cuMem-based fabric / POSIX-FD
  // exchange paths. On AMD only the cudaIpc (hipIpc) path is available, so
  // no driver-API init is needed.
  if (cuda_driver_lazy_init() != 0) {
    throw std::runtime_error(
        "MultiPeerTransport::exchange: failed to initialize CUDA driver API");
  }
#endif

  VLOG(1) << "MultiPeerTransport: rank " << myRank_ << " exchange()"
          << " nvl=" << (nvlTransport_ ? "yes" : "no")
          << " ibgda=" << (ibgdaTransport_ ? "yes" : "no")
          << " ibrc=" << (ibrcTransport_ ? "yes" : "no");

  if (nvlTransport_) {
    nvlTransport_->exchange();
  }
  if (ibgdaTransport_) {
    ibgdaTransport_->exchange();
  }
  if (ibrcTransport_) {
    ibrcTransport_->exchange();
  }

  build_device_handle();
}

TransportType MultiPeerTransport::get_transport_type(int peerRank) const {
  return typePerRank_[peerRank];
}

bool MultiPeerTransport::is_nvl_peer(int peerRank) const {
  return typePerRank_[peerRank] == TransportType::P2P_NVL;
}

bool MultiPeerTransport::is_ibgda_peer(int peerRank) const {
  return typePerRank_[peerRank] == TransportType::P2P_IBGDA;
}

P2pNvlTransportDevice MultiPeerTransport::get_p2p_nvl_transport_device(
    int globalPeerRank) const {
  if (!nvlTransport_) {
    throw std::runtime_error(
        "get_p2p_nvl_transport_device: NVL transport not available");
  }
  int nvlLocalPeerRank = globalToNvlLocal_.at(globalPeerRank);
  return nvlTransport_->getP2pTransportDevice(nvlLocalPeerRank);
}

P2pIbgdaTransportDevice* MultiPeerTransport::get_p2p_ibgda_transport_device(
    int globalPeerRank) const {
  if (!ibgdaTransport_) {
    throw std::runtime_error(
        "get_p2p_ibgda_transport_device: IBGDA transport not available (nRanks == 1?)");
  }
  if (ibgdaTransport_->materializedChannelCount(globalPeerRank) == 0) {
    throw std::runtime_error(
        "get_p2p_ibgda_transport_device: peer is not ready; call "
        "get_device_handle(demands) before accessing its device transport");
  }
  return ibgdaTransport_->getP2pTransportDeviceSlot(globalPeerRank);
}

Transport* /*nullable*/ MultiPeerTransport::get_nvl_transports_array() const {
  if (!nvlTransport_) {
    return nullptr;
  }
  return nvlTransport_->getDeviceTransports().data();
}

bool MultiPeerTransport::has_multimem_nvl_transport() const {
  // nvlTransport_ is legitimately null in normal builds without an NVL domain
  // (e.g. nRanks == 1), same as get_nvl_transports_array() above; the null
  // check is not masking an invariant.
  return nvlTransport_ && nvlTransport_->hasMultimemNvlTransport();
}

bool MultiPeerTransport::initialize_multimem_nvl_transport() const {
  return nvlTransport_ &&
      nvlTransport_->initializeMultimemNvlTransportIfEligible();
}

MultimemNvlTransportDevice
MultiPeerTransport::get_multimem_nvl_transport_device() const {
  // The getter is local after collective initialization succeeds.
  if (!has_multimem_nvl_transport()) {
    throw std::runtime_error(
        "MultiPeerTransport: multimem NVL transport is not initialized");
  }
  return nvlTransport_->getMultimemNvlTransportDevice();
}

P2pSelfTransportDevice MultiPeerTransport::get_p2p_self_transport_device()
    const {
  return P2pSelfTransportDevice{};
}

MultiPeerDeviceHandle MultiPeerTransport::make_device_handle() const {
  return MultiPeerDeviceHandle{
      myRank_,
      nRanks_,
      {transportsGpu_, static_cast<uint32_t>(nRanks_)},
      static_cast<int>(nvlPeerRanks_.size()),
      static_cast<int>(ibPeerRanks_.size()),
  };
}

MultiPeerDeviceHandle MultiPeerTransport::get_device_handle(
    const std::vector<int>& peers) {
  std::vector<PeerChannelDemand> demands;
  demands.reserve(peers.size());
  for (const int peer : peers) {
    if (peer < 0 || peer >= nRanks_ || peer == myRank_) {
      continue;
    }
    const auto type = typePerRank_[peer];
    if (type == TransportType::P2P_IBGDA || type == TransportType::P2P_IBRC) {
      demands.push_back({.peerRank = peer, .ibChannels = ibChannelCapacity_});
    }
  }
  return get_device_handle(demands);
}

MultiPeerDeviceHandle MultiPeerTransport::get_device_handle(
    std::span<const PeerChannelDemand> demands) {
  if (!deviceHandleBuilt_) {
    throw std::runtime_error(
        "MultiPeerTransport::get_device_handle called before exchange()");
  }
  for (const auto& demand : demands) {
    if (demand.peerRank < 0 || demand.peerRank >= nRanks_ ||
        demand.peerRank == myRank_) {
      throw std::invalid_argument(
          "peer channel demand contains an invalid peer rank");
    }
    if (demand.ibChannels > ibChannelCapacity_) {
      throw std::invalid_argument(
          "peer channel demand exceeds the configured IB capacity");
    }
    if (demand.ibChannels != 0) {
      const auto type = typePerRank_[demand.peerRank];
      const bool prefersIb =
          type == TransportType::P2P_IBGDA || type == TransportType::P2P_IBRC;
      if (!prefersIb && !ibgdaTransport_) {
        throw std::invalid_argument(
            "positive channel demand requires a preferred IB peer or an "
            "IBGDA fallback");
      }
    }
  }

  std::vector<PeerChannelDemand> sortedDemands(demands.begin(), demands.end());
  std::sort(
      sortedDemands.begin(),
      sortedDemands.end(),
      [](const auto& lhs, const auto& rhs) {
        return lhs.peerRank < rhs.peerRank;
      });

  std::vector<PeerChannelDemand> mergedDemands;
  mergedDemands.reserve(sortedDemands.size());
  for (size_t i = 0; i < sortedDemands.size();) {
    const int peerRank = sortedDemands[i].peerRank;
    uint32_t ibChannels = 0;
    do {
      ibChannels = std::max(ibChannels, sortedDemands[i].ibChannels);
      ++i;
    } while (i < sortedDemands.size() && sortedDemands[i].peerRank == peerRank);
    if (ibChannels != 0) {
      mergedDemands.push_back({.peerRank = peerRank, .ibChannels = ibChannels});
    }
  }
  if (!mergedDemands.empty()) {
    materializePeerChannels(mergedDemands);
  }
  return make_device_handle();
}

void MultiPeerTransport::materializePeerChannels(
    std::span<const PeerChannelDemand> demands) {
  auto materializeOn = [&](auto& ibTransport) {
    for (const auto& demand : demands) {
      ibTransport->queuePeerForMaterialization(
          demand.peerRank, demand.ibChannels);
    }
    ibTransport->connectPeers();
  };
  if (ibgdaTransport_) {
    materializeOn(ibgdaTransport_);
  } else if (ibrcTransport_) {
    materializeOn(ibrcTransport_);
  }
}

IbgdaLocalBuffer MultiPeerTransport::localRegisterIbgdaBuffer(
    void* ptr,
    size_t size) {
  if (ibgdaTransport_) {
    return ibgdaTransport_->registerBuffer(ptr, size);
  }
  if (ibrcTransport_) {
    return ibrcTransport_->registerBuffer(ptr, size);
  }
  throw std::runtime_error(
      "localRegisterIbgdaBuffer: IB transport not available");
}

void MultiPeerTransport::localDeregisterIbgdaBuffer(void* ptr) {
  if (ibgdaTransport_) {
    ibgdaTransport_->deregisterBuffer(ptr);
  } else if (ibrcTransport_) {
    ibrcTransport_->deregisterBuffer(ptr);
  }
}

std::vector<IbgdaRemoteBuffer> MultiPeerTransport::exchangeIbgdaBuffer(
    const IbgdaLocalBuffer& localBuf) {
  if (ibgdaTransport_) {
    return ibgdaTransport_->exchangeBuffer(localBuf);
  }
  if (ibrcTransport_) {
    return ibrcTransport_->exchangeBuffer(localBuf);
  }
  throw std::runtime_error("exchangeIbgdaBuffer: IB transport not available");
}

IbgdaLocalBuffer MultiPeerTransport::allocateIbCounterBuffer(
    std::size_t size,
    void** hostPtr) {
  *hostPtr = nullptr;
  if (ibrcTransport_) {
    void* host = nullptr;
    void* device = nullptr;
    CUDA_CHECK(cudaHostAlloc(&host, size, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&device, host, 0));
    std::memset(host, 0, size);
    *hostPtr = host;
    return IbgdaLocalBuffer(device, NetworkLKeys{});
  }
  if (ibgdaTransport_) {
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    CUDA_CHECK(cudaMemset(ptr, 0, size));
    return IbgdaLocalBuffer(ptr, NetworkLKeys{});
  }
  throw std::runtime_error(
      "allocateIbCounterBuffer: IB transport not available");
}

IbgdaLocalBuffer MultiPeerTransport::registerIbCounterBuffer(
    const IbgdaLocalBuffer& buffer,
    std::size_t size) {
  if (ibgdaTransport_) {
    return ibgdaTransport_->registerBuffer(buffer.ptr, size);
  }
  if (ibrcTransport_) {
    return buffer;
  }
  throw std::runtime_error(
      "registerIbCounterBuffer: IB transport not available");
}

void MultiPeerTransport::freeIbCounterBuffer(
    IbgdaLocalBuffer& buffer,
    void*& hostPtr) noexcept {
  if (buffer.ptr == nullptr) {
    return;
  }
  if (buffer.lkey_per_device.size > 0 && ibgdaTransport_) {
    ibgdaTransport_->deregisterBuffer(buffer.ptr);
  }
  if (hostPtr != nullptr) {
    (void)cudaFreeHost(hostPtr);
    hostPtr = nullptr;
  } else {
    (void)cudaFree(buffer.ptr);
  }
  buffer = IbgdaLocalBuffer{};
}

MultiPeerTransport::NvlMemMode MultiPeerTransport::detectNvlMemMode(
    void* ptr) const {
#if !defined(__HIP_PLATFORM_AMD__) && CUDART_VERSION >= 12030
  if (cuda_driver_lazy_init() != 0) {
    throw std::runtime_error("detectNvlMemMode: CUDA driver not available");
  }

  CUmemGenericAllocationHandle handle;
  CUresult ret = pfn_cuMemRetainAllocationHandle(&handle, ptr);
  if (ret == CUDA_ERROR_INVALID_VALUE) {
    return NvlMemMode::kCudaIpc;
  }
  if (ret != CUDA_SUCCESS) {
    const char* errStr = nullptr;
    pfn_cuGetErrorString(ret, &errStr);
    throw std::runtime_error(
        std::string("detectNvlMemMode: cuMemRetainAllocationHandle failed: ") +
        (errStr ? errStr : "unknown"));
  }

  CUmemAllocationProp prop = {};
  CUresult propRet = pfn_cuMemGetAllocationPropertiesFromHandle(&prop, handle);
  pfn_cuMemRelease(handle);
  if (propRet != CUDA_SUCCESS) {
    const char* errStr = nullptr;
    pfn_cuGetErrorString(propRet, &errStr);
    throw std::runtime_error(
        std::string(
            "detectNvlMemMode: cuMemGetAllocationPropertiesFromHandle failed: ") +
        (errStr ? errStr : "unknown"));
  }

  if (prop.requestedHandleTypes & CU_MEM_HANDLE_TYPE_FABRIC) {
    return NvlMemMode::kFabric;
  }
  if (prop.requestedHandleTypes & CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
    return NvlMemMode::kPosixFd;
  }
  throw std::runtime_error(
      "exchangeNvlBuffer: cuMem buffer lacks both CU_MEM_HANDLE_TYPE_FABRIC "
      "and CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR. "
      "Allocate with at least one shareable handle type.");
#else
  return NvlMemMode::kCudaIpc;
#endif
}

std::vector<void*> MultiPeerTransport::exchangeNvlBuffer(void* localPtr) {
  if (!nvlBootstrapAdapter_ || nvlNRanks_ <= 1) {
    throw std::runtime_error(
        "exchangeNvlBuffer: NVL transport not available or single rank");
  }

  NvlMemMode mode = detectNvlMemMode(localPtr);

  if (mode == NvlMemMode::kFabric || mode == NvlMemMode::kPosixFd) {
#if defined(__HIP_PLATFORM_AMD__) || CUDART_VERSION < 12030
    throw std::runtime_error("exchangeNvlBuffer: VMM path requires CUDA 12.3+");
#else
    if (cuda_driver_lazy_init() != 0) {
      throw std::runtime_error("exchangeNvlBuffer: CUDA driver not available");
    }

    int cudaDev = 0;
    CUdevice cuDev;
    CUDA_CHECK(cudaGetDevice(&cudaDev));
    CU_CHECK(pfn_cuDeviceGet(&cuDev, cudaDev));

    // Retain the external VMM allocation's physical handle via
    // CuMemAllocation::retain -- its destructor releases the retain reference
    // on both the success path and any exception unwind from
    // nvlMemExchangeVmm (allGather / import / map failures), preventing a
    // silent driver-refcount leak. The factory also queries the buffer's
    // real allocated size via cuMemGetAddressRange -- nvlMemExchangeVmm
    // needs a granularity-multiple size for peer-side cuMemAddressReserve.
    auto phys = CuMemAllocation::retain(localPtr);

    auto pm = nvlMemExchangeVmm(
        *nvlBootstrapAdapter_,
        nvlLocalRank_,
        nvlNRanks_,
        cuDev,
        phys->handle(),
        localPtr,
        phys->size(),
        /*preferFabric=*/mode == NvlMemMode::kFabric);

    std::vector<void*> mappedPtrs = pm.peerPtrs;
    nvlExchangeRecords_[localPtr] = NvlExchangeRecord{mode, std::move(pm)};
    return mappedPtrs;
#endif
  }

  auto pm = nvlMemExchangeCudaIpc(
      *nvlBootstrapAdapter_, nvlLocalRank_, nvlNRanks_, localPtr);
  std::vector<void*> mappedPtrs = pm.peerPtrs;
  nvlExchangeRecords_[localPtr] =
      NvlExchangeRecord{NvlMemMode::kCudaIpc, std::move(pm)};
  return mappedPtrs;
}

void MultiPeerTransport::unmapNvlBuffers(const std::vector<void*>& mappedPtrs) {
  // Find the exchange record by the self entry (localPtr).
  void* localPtr = (nvlLocalRank_ >= 0 &&
                    nvlLocalRank_ < static_cast<int>(mappedPtrs.size()))
      ? mappedPtrs[nvlLocalRank_]
      : nullptr;

  auto it =
      localPtr ? nvlExchangeRecords_.find(localPtr) : nvlExchangeRecords_.end();
  if (it == nvlExchangeRecords_.end()) {
    return;
  }

  auto& record = it->second;
  if (record.mode == NvlMemMode::kFabric ||
      record.mode == NvlMemMode::kPosixFd) {
#if !defined(__HIP_PLATFORM_AMD__) && CUDART_VERSION >= 12030
    if (cuda_driver_lazy_init() == 0) {
      // Tear down the peer VAs. Each CuMemMapping co-owns its imported peer
      // CuMemAllocation (via keepAlive), so clearing the mappings runs
      // cuMemUnmap + cuMemAddressFree and then releases the imported physical
      // handle -- the unmap-VA-then-release-handle ordering is preserved
      // without separate handle bookkeeping.
      record.mem.vmmMappings.clear();
    }
#endif
  } else {
    // cudaIpc path: close every non-self peer handle.
    for (int rank = 0; rank < static_cast<int>(record.mem.peerPtrs.size());
         ++rank) {
      if (rank == nvlLocalRank_ || record.mem.peerPtrs[rank] == nullptr) {
        continue;
      }
      cudaError_t err = cudaIpcCloseMemHandle(record.mem.peerPtrs[rank]);
      if (err != cudaSuccess) {
        fprintf(
            stderr,
            "MultiPeerTransport::unmapNvlBuffers: "
            "cudaIpcCloseMemHandle failed for rank %d: %s\n",
            rank,
            cudaGetErrorString(err));
      }
    }
  }

  nvlExchangeRecords_.erase(it);
}

void MultiPeerTransport::build_device_handle() {
  if (deviceHandleBuilt_) {
    free_device_handle();
  }

  // Build a host-side Transport array indexed by global rank, then cudaMemcpy
  // it to GPU. Since Transport has deleted copy constructor, we allocate raw
  // memory and use placement new.
  const size_t arrayBytes = nRanks_ * sizeof(Transport);
  auto* transportsHost = static_cast<Transport*>(
      std::aligned_alloc(alignof(Transport), arrayBytes));
  if (!transportsHost) {
    throw std::runtime_error("Failed to allocate host Transport array");
  }

  for (int r = 0; r < nRanks_; ++r) {
    switch (typePerRank_[r]) {
      case TransportType::SELF:
        new (&transportsHost[r]) Transport(P2pSelfTransportDevice{});
        break;

      case TransportType::P2P_NVL: {
        int nvlLocal = globalToNvlLocal_.at(r);
        P2pNvlTransportDevice nvlDev =
            nvlTransport_->buildP2pTransportDevice(nvlLocal);
        new (&transportsHost[r]) Transport(nvlDev);
        break;
      }

      case TransportType::P2P_IBGDA: {
        P2pIbgdaTransportDevice* devPtr = ibgdaTransport_
            ? ibgdaTransport_->getP2pTransportDeviceSlot(r)
            : nullptr;
        new (&transportsHost[r]) Transport(devPtr);
        break;
      }

      case TransportType::P2P_IBRC: {
        P2pIbrcTransportDevice* devPtr = ibrcTransport_
            ? ibrcTransport_->getP2pTransportDeviceSlot(r)
            : nullptr;
        new (&transportsHost[r]) Transport(devPtr);
        break;
      }
    }
  }

  // Allocate GPU memory and raw-copy the Transport array.
  // Transport union members are standard-layout + trivially destructible,
  // so raw byte copy via cudaMemcpy produces valid device-side objects.
  CUDA_CHECK(cudaMalloc(&transportsGpu_, arrayBytes));
  CUDA_CHECK(cudaMemcpy(
      transportsGpu_, transportsHost, arrayBytes, cudaMemcpyHostToDevice));

  // Destroy host-side Transport objects and free
  for (int r = 0; r < nRanks_; ++r) {
    transportsHost[r].~Transport();
  }
  std::free(transportsHost);

  deviceHandleBuilt_ = true;
}

void MultiPeerTransport::free_device_handle() {
  if (transportsGpu_) {
    (void)cudaFree(transportsGpu_);
    transportsGpu_ = nullptr;
  }
  deviceHandleBuilt_ = false;
}

} // namespace comms::prims
