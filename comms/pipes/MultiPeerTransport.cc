// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/MultiPeerTransport.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <stdexcept>

#include <sys/syscall.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <glog/logging.h>

#include "comms/pipes/CudaDriverLazy.h"
#include "comms/pipes/MultiPeerDeviceHandle.cuh"
#include "comms/pipes/MultipeerIbgdaDeviceTransport.cuh"
#include "comms/pipes/TopologyDiscovery.h"
#include "comms/pipes/bootstrap/NvlBootstrapAdapter.h"

namespace comms::pipes {

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
    // ibgdaPeerRanks_ stays empty; ibgdaTransport_ stays nullptr.
  } else {
    for (int r = 0; r < nRanks_; ++r) {
      if (r == myRank_) {
        typePerRank_.at(r) = TransportType::SELF;
      } else if (globalToNvlLocal_.count(r)) {
        typePerRank_.at(r) = TransportType::P2P_NVL;
      } else {
        typePerRank_.at(r) = TransportType::P2P_IBGDA;
      }
    }

    for (int r = 0; r < nRanks_; ++r) {
      if (r != myRank_) {
        ibgdaPeerRanks_.push_back(r);
      }
    }
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
  }

  // Always create IBGDA transport — it is the universal fallback for all peers.
  // NVL is preferred when available, but IBGDA covers every non-self rank.
  if (!config.disableIb && nRanks_ > 1) {
    auto ibgdaConfig = config.ibgdaConfig;
    ibgdaConfig.cudaDevice = deviceId_;
    ibgdaTransport_ = std::make_unique<MultipeerIbgdaTransport>(
        myRank_, nRanks_, bootstrap_, ibgdaConfig);
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
  if (cuda_driver_lazy_init() != 0) {
    throw std::runtime_error(
        "MultiPeerTransport::exchange: failed to initialize CUDA driver API");
  }

  if (nvlTransport_) {
    nvlTransport_->exchange();
  }
  if (ibgdaTransport_) {
    ibgdaTransport_->exchange();
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

  return MultiPeerDeviceHandle{
      myRank_,
      nRanks_,
      {transportsGpu_, static_cast<uint32_t>(nRanks_)},
      static_cast<int>(nvlPeerRanks_.size()),
      static_cast<int>(ibgdaPeerRanks_.size()),
  };
}

IbgdaLocalBuffer MultiPeerTransport::localRegisterIbgdaBuffer(
    void* ptr,
    size_t size) {
  if (!ibgdaTransport_) {
    throw std::runtime_error(
        "localRegisterIbgdaBuffer: IBGDA transport not available");
  }
  return ibgdaTransport_->registerBuffer(ptr, size);
}

void MultiPeerTransport::localDeregisterIbgdaBuffer(void* ptr) {
  if (ibgdaTransport_) {
    ibgdaTransport_->deregisterBuffer(ptr);
  }
}

std::vector<IbgdaRemoteBuffer> MultiPeerTransport::exchangeIbgdaBuffer(
    const IbgdaLocalBuffer& localBuf) {
  if (!ibgdaTransport_) {
    throw std::runtime_error(
        "exchangeIbgdaBuffer: IBGDA transport not available");
  }
  return ibgdaTransport_->exchangeBuffer(localBuf);
}

MultiPeerTransport::NvlMemMode MultiPeerTransport::detectNvlMemMode(
    void* ptr) const {
#if CUDART_VERSION >= 12030
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
#if CUDART_VERSION < 12030
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
#if CUDART_VERSION < 12030
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
#if CUDART_VERSION >= 12030
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

  // Get IBGDA GPU pointers per-peer via getP2pTransportDevice()
  // which returns device-memory addresses suitable for Transport.p2p_ibgda

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
            ? ibgdaTransport_->getP2pTransportDevice(r)
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

} // namespace comms::pipes
