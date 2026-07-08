// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/transport/MultiPeerTransport.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <stdexcept>

#include <sys/syscall.h>
#include <unistd.h>

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
      ibLazyConnect_(config.ibConfig.ibLazyConnect),
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
        nvlLocalRank_, nvlNRanks_, nvlBootstrapAdapter_, config.nvlConfig);
    VLOG(1) << "MultiPeerTransport: rank " << myRank_
            << " created NVL sub-transport, nvlNRanks=" << nvlNRanks_
            << " nvlLocalRank=" << nvlLocalRank_;
  }

  // Create the IB sub-transport — the universal fallback for all non-NVL peers.
  // Exactly one backend is built, selected by config.ibMode (kIbgda default;
  // kIbrc selects the CPU-proxy backend).
  if (!config.disableIb && !ibPeerRanks_.empty()) {
    auto ibConfig = config.ibConfig;
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
  return ibgdaTransport_->getP2pTransportDevice(globalPeerRank);
}

P2pIbTransportDevice MultiPeerTransport::get_p2p_ib_transport_device(
    int globalPeerRank) const {
  const auto transportType = typePerRank_.at(globalPeerRank);
  if (transportType == TransportType::P2P_IBGDA) {
    if (!ibgdaTransport_) {
      throw std::runtime_error(
          "get_p2p_ib_transport_device: IBGDA transport not available");
    }
    return P2pIbTransportDevice(
        ibgdaTransport_->getP2pTransportDevice(globalPeerRank));
  }
  if (transportType == TransportType::P2P_IBRC) {
    if (!ibrcTransport_) {
      throw std::runtime_error(
          "get_p2p_ib_transport_device: IBRC transport not available");
    }
    return P2pIbTransportDevice(
        ibrcTransport_->getP2pTransportDevice(globalPeerRank));
  }
  throw std::runtime_error(
      "get_p2p_ib_transport_device: peer " + std::to_string(globalPeerRank) +
      " is not an IB peer");
}

Transport* /*nullable*/ MultiPeerTransport::get_nvl_transports_array() const {
  if (!nvlTransport_) {
    return nullptr;
  }
  return nvlTransport_->getDeviceTransports().data();
}

P2pSelfTransportDevice MultiPeerTransport::get_p2p_self_transport_device()
    const {
  return P2pSelfTransportDevice{};
}

MultiPeerDeviceHandle MultiPeerTransport::get_device_handle() const {
  if (!deviceHandleBuilt_) {
    throw std::runtime_error(
        "MultiPeerTransport::get_device_handle() called before exchange()");
  }
  if (ibLazyConnect_) {
    throw std::runtime_error(
        "get_device_handle() cannot be used with lazy mode (ibLazyConnect=true). "
        "Use get_device_handle(peers) or getP2pTransportDevice(peerRank).");
  }

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
  if (!deviceHandleBuilt_) {
    throw std::runtime_error(
        "MultiPeerTransport::get_device_handle(peers) called before exchange()");
  }
  materializePeers(peers);
  return MultiPeerDeviceHandle{
      myRank_,
      nRanks_,
      {transportsGpu_, static_cast<uint32_t>(nRanks_)},
      static_cast<int>(nvlPeerRanks_.size()),
      static_cast<int>(ibPeerRanks_.size()),
  };
}

bool MultiPeerTransport::is_lazy_mode() const {
  return ibLazyConnect_;
}

void MultiPeerTransport::materializePeers(const std::vector<int>& peers) {
  auto materializeOn = [&](auto& ibTransport) {
    for (int peer : peers) {
      if (peer >= 0 && peer < nRanks_ && peer != myRank_ &&
          (typePerRank_[peer] == TransportType::P2P_IBGDA ||
           typePerRank_[peer] == TransportType::P2P_IBRC)) {
        ibTransport->queuePeerForMaterialization(peer);
      }
    }
    ibTransport->connectPeers();
  };
  if (ibgdaTransport_) {
    materializeOn(ibgdaTransport_);
  } else if (ibrcTransport_) {
    materializeOn(ibrcTransport_);
  }
}

void MultiPeerTransport::connectPeers() {
  if (ibgdaTransport_) {
    ibgdaTransport_->connectPeers();
  } else if (ibrcTransport_) {
    ibrcTransport_->connectPeers();
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
    cudaFreeHost(hostPtr);
    hostPtr = nullptr;
  } else {
    cudaFree(buffer.ptr);
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

std::vector<void*> MultiPeerTransport::exchangeNvlBufferCudaIpc(
    void* localPtr) {
  cudaIpcMemHandle_t localHandle{};
  CUDA_CHECK(cudaIpcGetMemHandle(&localHandle, localPtr));

  std::vector<cudaIpcMemHandle_t> allHandles(nvlNRanks_);
  allHandles[nvlLocalRank_] = localHandle;

  auto result = nvlBootstrapAdapter_
                    ->allGather(
                        allHandles.data(),
                        sizeof(cudaIpcMemHandle_t),
                        nvlLocalRank_,
                        nvlNRanks_)
                    .get();
  if (result != 0) {
    throw std::runtime_error("exchangeNvlBufferCudaIpc: allGather failed");
  }

  std::vector<void*> mappedPtrs(nvlNRanks_, nullptr);
  mappedPtrs[nvlLocalRank_] = localPtr;

  for (int rank = 0; rank < nvlNRanks_; ++rank) {
    if (rank == nvlLocalRank_) {
      continue;
    }
    CUDA_CHECK(cudaIpcOpenMemHandle(
        &mappedPtrs[rank], allHandles[rank], cudaIpcMemLazyEnablePeerAccess));
  }

  return mappedPtrs;
}

std::vector<void*> MultiPeerTransport::exchangeNvlBufferFabric(
    void* localPtr,
    std::size_t size) {
#if defined(__HIP_PLATFORM_AMD__) || CUDART_VERSION < 12030
  throw std::runtime_error("Fabric handles require CUDA 12.3+");
#else
  if (cuda_driver_lazy_init() != 0) {
    throw std::runtime_error(
        "exchangeNvlBufferFabric: CUDA driver not available");
  }

  // Retain allocation handle and export fabric handle
  CUmemGenericAllocationHandle allocHandle;
  CU_CHECK(pfn_cuMemRetainAllocationHandle(&allocHandle, localPtr));

  FabricHandle localFabricHandle{};
  CU_CHECK(pfn_cuMemExportToShareableHandle(
      &localFabricHandle, allocHandle, CU_MEM_HANDLE_TYPE_FABRIC, 0));
  CU_CHECK(pfn_cuMemRelease(allocHandle));

  // Get actual allocated size (may be larger due to granularity)
  CUdeviceptr basePtr;
  size_t allocatedSize = 0;
  CU_CHECK(pfn_cuMemGetAddressRange(
      &basePtr, &allocatedSize, (CUdeviceptr)localPtr));

  // Exchange fabric handles + allocated sizes
  struct ExchangeData {
    FabricHandle handle;
    size_t allocatedSize;
  };

  std::vector<ExchangeData> allData(nvlNRanks_);
  allData[nvlLocalRank_].handle = localFabricHandle;
  allData[nvlLocalRank_].allocatedSize = allocatedSize;

  auto result =
      nvlBootstrapAdapter_
          ->allGather(
              allData.data(), sizeof(ExchangeData), nvlLocalRank_, nvlNRanks_)
          .get();
  if (result != 0) {
    throw std::runtime_error("exchangeNvlBufferFabric: allGather failed");
  }

  // Import peer fabric handles
  int cudaDev = 0;
  CUdevice cuDev;
  CUDA_CHECK(cudaGetDevice(&cudaDev));
  CU_CHECK(pfn_cuDeviceGet(&cuDev, cudaDev));

  NvlExchangeRecord record;
  record.mode = NvlMemMode::kFabric;
  record.cuMemPeerPtrs.resize(nvlNRanks_, 0);
  record.cuMemPeerAllocHandles.resize(nvlNRanks_, 0);
  record.cuMemPeerSizes.resize(nvlNRanks_, 0);

  std::vector<void*> mappedPtrs(nvlNRanks_, nullptr);
  mappedPtrs[nvlLocalRank_] = localPtr;

  for (int rank = 0; rank < nvlNRanks_; ++rank) {
    if (rank == nvlLocalRank_) {
      continue;
    }

    size_t peerAllocatedSize = allData[rank].allocatedSize;
    record.cuMemPeerSizes[rank] = peerAllocatedSize;

    CU_CHECK(pfn_cuMemImportFromShareableHandle(
        &record.cuMemPeerAllocHandles[rank],
        const_cast<void*>(static_cast<const void*>(&allData[rank].handle)),
        CU_MEM_HANDLE_TYPE_FABRIC));

    CUmemAllocationProp prop = {};
    CU_CHECK(pfn_cuMemGetAllocationPropertiesFromHandle(
        &prop, record.cuMemPeerAllocHandles[rank]));

    size_t granularity = 0;
    CU_CHECK(pfn_cuMemGetAllocationGranularity(
        &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

    CU_CHECK(pfn_cuMemAddressReserve(
        &record.cuMemPeerPtrs[rank], peerAllocatedSize, granularity, 0, 0));

    CU_CHECK(pfn_cuMemMap(
        record.cuMemPeerPtrs[rank],
        peerAllocatedSize,
        0,
        record.cuMemPeerAllocHandles[rank],
        0));

    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = cuDev;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CU_CHECK(pfn_cuMemSetAccess(
        record.cuMemPeerPtrs[rank], peerAllocatedSize, &accessDesc, 1));

    mappedPtrs[rank] = reinterpret_cast<void*>(record.cuMemPeerPtrs[rank]);
  }

  // Store record keyed by the local pointer for cleanup
  nvlExchangeRecords_[localPtr] = std::move(record);

  return mappedPtrs;
#endif
}

std::vector<void*> MultiPeerTransport::exchangeNvlBufferPosixFd(
    void* localPtr,
    std::size_t size) {
#if defined(__HIP_PLATFORM_AMD__) || CUDART_VERSION < 12030
  throw std::runtime_error("POSIX FD cuMem handles require CUDA 12.3+");
#else
  if (cuda_driver_lazy_init() != 0) {
    throw std::runtime_error(
        "exchangeNvlBufferPosixFd: CUDA driver not available");
  }

  // Retain allocation handle and export as POSIX file descriptor
  CUmemGenericAllocationHandle allocHandle;
  CU_CHECK(pfn_cuMemRetainAllocationHandle(&allocHandle, localPtr));

  int localFd = -1;
  CU_CHECK(pfn_cuMemExportToShareableHandle(
      &localFd, allocHandle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
  CU_CHECK(pfn_cuMemRelease(allocHandle));

  // Get actual allocated size (may be larger due to granularity)
  CUdeviceptr basePtr;
  size_t allocatedSize = 0;
  CU_CHECK(pfn_cuMemGetAddressRange(
      &basePtr, &allocatedSize, (CUdeviceptr)localPtr));

  // Exchange {pid, fd, allocatedSize} with NVL peers.
  // Peers will use pidfd_getfd to duplicate our fd into their fd table.
  struct ExchangeData {
    pid_t pid;
    int fd;
    size_t allocatedSize;
  };

  std::vector<ExchangeData> allData(nvlNRanks_);
  allData[nvlLocalRank_] = {getpid(), localFd, allocatedSize};

  auto result =
      nvlBootstrapAdapter_
          ->allGather(
              allData.data(), sizeof(ExchangeData), nvlLocalRank_, nvlNRanks_)
          .get();
  if (result != 0) {
    close(localFd);
    throw std::runtime_error("exchangeNvlBufferPosixFd: allGather failed");
  }

  // Import peer handles via pidfd_open + pidfd_getfd (Linux 5.6+)
  int cudaDev = 0;
  CUdevice cuDev;
  CUDA_CHECK(cudaGetDevice(&cudaDev));
  CU_CHECK(pfn_cuDeviceGet(&cuDev, cudaDev));

  NvlExchangeRecord record;
  record.mode = NvlMemMode::kPosixFd;
  record.localExportedFd = localFd;
  record.cuMemPeerPtrs.resize(nvlNRanks_, 0);
  record.cuMemPeerAllocHandles.resize(nvlNRanks_, 0);
  record.cuMemPeerSizes.resize(nvlNRanks_, 0);

  std::vector<void*> mappedPtrs(nvlNRanks_, nullptr);
  mappedPtrs[nvlLocalRank_] = localPtr;

  for (int rank = 0; rank < nvlNRanks_; ++rank) {
    if (rank == nvlLocalRank_) {
      continue;
    }

    // Duplicate the remote process's fd into this process
    int pidfd = static_cast<int>(syscall(SYS_pidfd_open, allData[rank].pid, 0));
    if (pidfd < 0) {
      throw std::runtime_error(
          "exchangeNvlBufferPosixFd: pidfd_open failed for rank " +
          std::to_string(rank) + ": " + strerror(errno));
    }

    int importedFd =
        static_cast<int>(syscall(SYS_pidfd_getfd, pidfd, allData[rank].fd, 0));
    close(pidfd);
    if (importedFd < 0) {
      throw std::runtime_error(
          "exchangeNvlBufferPosixFd: pidfd_getfd failed for rank " +
          std::to_string(rank) + ": " + strerror(errno));
    }

    size_t peerAllocatedSize = allData[rank].allocatedSize;
    record.cuMemPeerSizes[rank] = peerAllocatedSize;

    CU_CHECK(pfn_cuMemImportFromShareableHandle(
        &record.cuMemPeerAllocHandles[rank],
        reinterpret_cast<void*>(static_cast<uintptr_t>(importedFd)),
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));

    // fd can be closed immediately after import
    close(importedFd);

    CUmemAllocationProp prop = {};
    CU_CHECK(pfn_cuMemGetAllocationPropertiesFromHandle(
        &prop, record.cuMemPeerAllocHandles[rank]));

    size_t granularity = 0;
    CU_CHECK(pfn_cuMemGetAllocationGranularity(
        &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

    CU_CHECK(pfn_cuMemAddressReserve(
        &record.cuMemPeerPtrs[rank], peerAllocatedSize, granularity, 0, 0));

    CU_CHECK(pfn_cuMemMap(
        record.cuMemPeerPtrs[rank],
        peerAllocatedSize,
        0,
        record.cuMemPeerAllocHandles[rank],
        0));

    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = cuDev;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CU_CHECK(pfn_cuMemSetAccess(
        record.cuMemPeerPtrs[rank], peerAllocatedSize, &accessDesc, 1));

    mappedPtrs[rank] = reinterpret_cast<void*>(record.cuMemPeerPtrs[rank]);
  }

  nvlExchangeRecords_[localPtr] = std::move(record);

  return mappedPtrs;
#endif
}

std::vector<void*> MultiPeerTransport::exchangeNvlBuffer(
    void* localPtr,
    std::size_t size) {
  if (!nvlBootstrapAdapter_ || nvlNRanks_ <= 1) {
    throw std::runtime_error(
        "exchangeNvlBuffer: NVL transport not available or single rank");
  }

  NvlMemMode mode = detectNvlMemMode(localPtr);
  if (mode == NvlMemMode::kFabric) {
    return exchangeNvlBufferFabric(localPtr, size);
  }
  if (mode == NvlMemMode::kPosixFd) {
    return exchangeNvlBufferPosixFd(localPtr, size);
  }

  auto mappedPtrs = exchangeNvlBufferCudaIpc(localPtr);

  // Store a cudaIpc record for cleanup dispatch
  NvlExchangeRecord record;
  record.mode = NvlMemMode::kCudaIpc;
  nvlExchangeRecords_[localPtr] = std::move(record);

  return mappedPtrs;
}

void MultiPeerTransport::unmapNvlBuffers(const std::vector<void*>& mappedPtrs) {
  // Find the exchange record by the self entry (localPtr)
  void* localPtr = (nvlLocalRank_ >= 0 &&
                    nvlLocalRank_ < static_cast<int>(mappedPtrs.size()))
      ? mappedPtrs[nvlLocalRank_]
      : nullptr;

  auto it =
      localPtr ? nvlExchangeRecords_.find(localPtr) : nvlExchangeRecords_.end();

  bool isCuMem =
      (it != nvlExchangeRecords_.end() &&
       (it->second.mode == NvlMemMode::kFabric ||
        it->second.mode == NvlMemMode::kPosixFd));

  if (isCuMem) {
#if !defined(__HIP_PLATFORM_AMD__) && CUDART_VERSION >= 12030
    if (cuda_driver_lazy_init() != 0) {
      return;
    }

    auto& record = it->second;
    for (int rank = 0; rank < static_cast<int>(mappedPtrs.size()); ++rank) {
      if (rank == nvlLocalRank_) {
        continue;
      }
      if (record.cuMemPeerPtrs[rank] != 0) {
        pfn_cuMemUnmap(record.cuMemPeerPtrs[rank], record.cuMemPeerSizes[rank]);
        pfn_cuMemAddressFree(
            record.cuMemPeerPtrs[rank], record.cuMemPeerSizes[rank]);
      }
      if (record.cuMemPeerAllocHandles[rank] != 0) {
        pfn_cuMemRelease(record.cuMemPeerAllocHandles[rank]);
      }
    }
    if (record.localExportedFd >= 0) {
      close(record.localExportedFd);
    }
#endif
  } else {
    // cudaIpc path
    for (int rank = 0; rank < static_cast<int>(mappedPtrs.size()); ++rank) {
      if (rank == nvlLocalRank_ || mappedPtrs[rank] == nullptr) {
        continue;
      }
      cudaError_t err = cudaIpcCloseMemHandle(mappedPtrs[rank]);
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

  if (it != nvlExchangeRecords_.end()) {
    nvlExchangeRecords_.erase(it);
  }
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
    cudaFree(transportsGpu_);
    transportsGpu_ = nullptr;
  }
  deviceHandleBuilt_ = false;
}

} // namespace comms::prims
