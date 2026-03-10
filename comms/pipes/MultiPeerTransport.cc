// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/MultiPeerTransport.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include <cuda_runtime.h>

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
      cuGetErrorString(err, &errStr);                                          \
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
    const MultiPeerTransportConfig& config)
    : myRank_(myRank),
      nRanks_(nRanks),
      deviceId_(deviceId),
      bootstrap_(std::move(bootstrap)) {
  TopologyDiscovery topoDiscovery;
  auto topo = topoDiscovery.discover(
      myRank_, nRanks_, deviceId_, *bootstrap_, config.topoConfig);
  nvlPeerRanks_ = std::move(topo.nvlPeerRanks);
  globalToNvlLocal_ = std::move(topo.globalToNvlLocal);

  // Derive fields from the slim TopologyResult.
  nvlNRanks_ = static_cast<int>(nvlPeerRanks_.size()) + 1;
  nvlLocalRank_ = globalToNvlLocal_.at(myRank_);

  typePerRank_.resize(nRanks_);
  for (int r = 0; r < nRanks_; ++r) {
    if (r == myRank_) {
      typePerRank_[r] = TransportType::SELF;
    } else if (globalToNvlLocal_.count(r)) {
      typePerRank_[r] = TransportType::P2P_NVL;
    } else {
      typePerRank_[r] = TransportType::P2P_IBGDA;
    }
  }

  for (int r = 0; r < nRanks_; ++r) {
    if (r != myRank_) {
      ibgdaPeerRanks_.push_back(r);
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
  if (nRanks_ > 1) {
    auto ibgdaConfig = config.ibgdaConfig;
    ibgdaConfig.cudaDevice = deviceId_;
    ibgdaTransport_ = std::make_unique<MultipeerIbgdaTransport>(
        myRank_, nRanks_, bootstrap_, ibgdaConfig);
  }
}

MultiPeerTransport::~MultiPeerTransport() {
  free_device_handle();
}

void MultiPeerTransport::exchange() {
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
  CUmemGenericAllocationHandle handle;
  CUresult ret = cuMemRetainAllocationHandle(&handle, ptr);
  if (ret == CUDA_ERROR_INVALID_VALUE) {
    return NvlMemMode::kCudaIpc;
  }
  if (ret != CUDA_SUCCESS) {
    const char* errStr = nullptr;
    cuGetErrorString(ret, &errStr);
    throw std::runtime_error(
        std::string("detectNvlMemMode: cuMemRetainAllocationHandle failed: ") +
        (errStr ? errStr : "unknown"));
  }

  CUmemAllocationProp prop = {};
  CU_CHECK(cuMemGetAllocationPropertiesFromHandle(&prop, handle));
  CU_CHECK(cuMemRelease(handle));

  if (!(prop.requestedHandleTypes & CU_MEM_HANDLE_TYPE_FABRIC)) {
    throw std::runtime_error(
        "exchangeNvlBuffer: cuMem buffer lacks CU_MEM_HANDLE_TYPE_FABRIC. "
        "Use ncclMemAlloc or allocate with fabric handle support.");
  }
  return NvlMemMode::kFabric;
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
  // Retain allocation handle and export fabric handle
  CUmemGenericAllocationHandle allocHandle;
  CU_CHECK(cuMemRetainAllocationHandle(&allocHandle, localPtr));

  FabricHandle localFabricHandle{};
  CU_CHECK(cuMemExportToShareableHandle(
      &localFabricHandle, allocHandle, CU_MEM_HANDLE_TYPE_FABRIC, 0));
  CU_CHECK(cuMemRelease(allocHandle));

  // Get actual allocated size (may be larger due to granularity)
  CUdeviceptr basePtr;
  size_t allocatedSize = 0;
  CU_CHECK(
      cuMemGetAddressRange(&basePtr, &allocatedSize, (CUdeviceptr)localPtr));

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
  CU_CHECK(cuDeviceGet(&cuDev, cudaDev));

  NvlExchangeRecord record;
  record.mode = NvlMemMode::kFabric;
  record.fabricPeerPtrs.resize(nvlNRanks_, 0);
  record.fabricPeerAllocHandles.resize(nvlNRanks_, 0);
  record.fabricPeerSizes.resize(nvlNRanks_, 0);

  std::vector<void*> mappedPtrs(nvlNRanks_, nullptr);
  mappedPtrs[nvlLocalRank_] = localPtr;

  for (int rank = 0; rank < nvlNRanks_; ++rank) {
    if (rank == nvlLocalRank_) {
      continue;
    }

    size_t peerAllocatedSize = allData[rank].allocatedSize;
    record.fabricPeerSizes[rank] = peerAllocatedSize;

    CU_CHECK(cuMemImportFromShareableHandle(
        &record.fabricPeerAllocHandles[rank],
        const_cast<void*>(static_cast<const void*>(&allData[rank].handle)),
        CU_MEM_HANDLE_TYPE_FABRIC));

    CUmemAllocationProp prop = {};
    CU_CHECK(cuMemGetAllocationPropertiesFromHandle(
        &prop, record.fabricPeerAllocHandles[rank]));

    size_t granularity = 0;
    CU_CHECK(cuMemGetAllocationGranularity(
        &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

    CU_CHECK(cuMemAddressReserve(
        &record.fabricPeerPtrs[rank], peerAllocatedSize, granularity, 0, 0));

    CU_CHECK(cuMemMap(
        record.fabricPeerPtrs[rank],
        peerAllocatedSize,
        0,
        record.fabricPeerAllocHandles[rank],
        0));

    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = cuDev;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CU_CHECK(cuMemSetAccess(
        record.fabricPeerPtrs[rank], peerAllocatedSize, &accessDesc, 1));

    mappedPtrs[rank] = reinterpret_cast<void*>(record.fabricPeerPtrs[rank]);
  }

  // Store record keyed by the local pointer for cleanup
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

  bool isFabric =
      (it != nvlExchangeRecords_.end() &&
       it->second.mode == NvlMemMode::kFabric);

  if (isFabric) {
#if CUDART_VERSION >= 12030
    auto& record = it->second;
    for (int rank = 0; rank < static_cast<int>(mappedPtrs.size()); ++rank) {
      if (rank == nvlLocalRank_) {
        continue;
      }
      if (record.fabricPeerPtrs[rank] != 0) {
        cuMemUnmap(record.fabricPeerPtrs[rank], record.fabricPeerSizes[rank]);
        cuMemAddressFree(
            record.fabricPeerPtrs[rank], record.fabricPeerSizes[rank]);
      }
      if (record.fabricPeerAllocHandles[rank] != 0) {
        cuMemRelease(record.fabricPeerAllocHandles[rank]);
      }
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
