// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/GpuMemHandler.h"

#include <glog/logging.h>
#include <mutex>
#include <stdexcept>

namespace comms::pipes {

namespace {

void checkCudaError(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
  }
}

void checkCuError(CUresult err, const char* msg) {
  if (err != CUDA_SUCCESS) {
    const char* errStr = nullptr;
    cuGetErrorString(err, &errStr);
    throw std::runtime_error(
        std::string(msg) + ": " + (errStr ? errStr : "unknown error"));
  }
}

// Minimum allocation size for trial allocation (matches ctran)
constexpr size_t kTrialAllocSize = 2097152UL; // 2MB

// Helper function that performs the actual fabric handle support check.
// This is called once and the result is cached by isFabricHandleSupported().
bool checkFabricHandleSupportedImpl() {
#if CUDART_VERSION < 12030
  return false;
#else
  int cudaDev = 0;
  CUdevice cuDev;

  // 1. Check basic CUDA setup
  cudaError_t cudaErr = cudaGetDevice(&cudaDev);
  if (cudaErr != cudaSuccess) {
    return false;
  }

  CUresult cuErr = cuDeviceGet(&cuDev, cudaDev);
  if (cuErr != CUDA_SUCCESS) {
    return false;
  }

  // 2. Check device attribute for fabric handle support
  int fabricSupported = 0;
  cuErr = cuDeviceGetAttribute(
      &fabricSupported,
      CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
      cuDev);

  if (cuErr != CUDA_SUCCESS || !fabricSupported) {
    return false;
  }

  // 3. Trial allocation to verify fabric handles actually work
  //    (attribute may be true but allocation/export could still fail)
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = cuDev;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

  size_t granularity = 0;
  cuErr = cuMemGetAllocationGranularity(
      &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  if (cuErr != CUDA_SUCCESS) {
    return false;
  }

  size_t allocSize =
      ((kTrialAllocSize + granularity - 1) / granularity) * granularity;

  CUmemGenericAllocationHandle handle;
  cuErr = cuMemCreate(&handle, allocSize, &prop, 0);
  if (cuErr != CUDA_SUCCESS) {
    return false;
  }

  CUdeviceptr ptr;
  cuErr = cuMemAddressReserve(&ptr, allocSize, granularity, 0, 0);
  if (cuErr != CUDA_SUCCESS) {
    cuMemRelease(handle);
    return false;
  }

  cuErr = cuMemMap(ptr, allocSize, 0, handle, 0);
  if (cuErr != CUDA_SUCCESS) {
    cuMemAddressFree(ptr, allocSize);
    cuMemRelease(handle);
    return false;
  }

  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = cuDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  cuErr = cuMemSetAccess(ptr, allocSize, &accessDesc, 1);
  if (cuErr != CUDA_SUCCESS) {
    cuMemUnmap(ptr, allocSize);
    cuMemAddressFree(ptr, allocSize);
    cuMemRelease(handle);
    return false;
  }

  // 4. Trial export to fabric handle
  CUmemFabricHandle fabricHandle;
  cuErr = cuMemExportToShareableHandle(
      &fabricHandle, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0);
  if (cuErr != CUDA_SUCCESS) {
    cuMemUnmap(ptr, allocSize);
    cuMemAddressFree(ptr, allocSize);
    cuMemRelease(handle);
    return false;
  }

  // 5. Trial import from fabric handle
  CUmemGenericAllocationHandle importedHandle;
  cuErr = cuMemImportFromShareableHandle(
      &importedHandle, &fabricHandle, CU_MEM_HANDLE_TYPE_FABRIC);
  if (cuErr != CUDA_SUCCESS) {
    cuMemUnmap(ptr, allocSize);
    cuMemAddressFree(ptr, allocSize);
    cuMemRelease(handle);
    return false;
  }

  // Import increases ref count, release it
  cuMemRelease(importedHandle);

  // Cleanup trial allocation
  cuMemUnmap(ptr, allocSize);
  cuMemAddressFree(ptr, allocSize);
  cuMemRelease(handle);

  return true;
#endif
}

} // namespace

bool GpuMemHandler::isFabricHandleSupported() {
  static std::once_flag onceFlag;
  static bool cachedResult = false;

  std::call_once(
      onceFlag, []() { cachedResult = checkFabricHandleSupportedImpl(); });

  return cachedResult;
}

MemSharingMode GpuMemHandler::detectBestMode() {
  if (isFabricHandleSupported()) {
    return MemSharingMode::kFabric;
  }
  return MemSharingMode::kCudaIpc;
}

GpuMemHandler::GpuMemHandler(
    std::shared_ptr<ctran::bootstrap::IBootstrap> bootstrap,
    int32_t selfRank,
    int32_t nRanks,
    size_t size)
    : GpuMemHandler(
          std::move(bootstrap),
          selfRank,
          nRanks,
          size,
          detectBestMode()) {}

GpuMemHandler::GpuMemHandler(
    std::shared_ptr<ctran::bootstrap::IBootstrap> bootstrap,
    int32_t selfRank,
    int32_t nRanks,
    size_t size,
    MemSharingMode mode)
    : bootstrap_(std::move(bootstrap)),
      selfRank_(selfRank),
      nRanks_(nRanks),
      mode_(mode),
      fabricPeerPtrs_(nRanks, 0),
      fabricPeerAllocHandles_(nRanks, 0),
      fabricPeerAllocatedSizes_(nRanks, 0),
      cudaIpcPeerPtrs_(nRanks, nullptr) {
  if (mode_ == MemSharingMode::kFabric && !isFabricHandleSupported()) {
    throw std::runtime_error(
        "Fabric handle mode requested but not supported on this system. "
        "Requires Hopper (H100) or newer GPU with CUDA 12.3+.");
  }

  init(size);
}

GpuMemHandler::~GpuMemHandler() {
  if (mode_ == MemSharingMode::kFabric) {
    cleanupFabric();
  } else {
    cleanupCudaIpc();
  }
}

void GpuMemHandler::init(size_t size) {
  if (mode_ == MemSharingMode::kFabric) {
    allocateFabricMemory(size);
  } else {
    allocateCudaIpcMemory(size);
  }
}

void* GpuMemHandler::getLocalDeviceMemPtr() const {
  if (mode_ == MemSharingMode::kFabric) {
    // NOLINTNEXTLINE(performance-no-int-to-ptr): CUdeviceptr is an integer type
    return reinterpret_cast<void*>(fabricLocalPtr_);
  } else {
    return cudaIpcLocalPtr_;
  }
}

void* GpuMemHandler::getPeerDeviceMemPtr(int32_t rank) const {
  if (rank < 0 || rank >= nRanks_) {
    throw std::runtime_error(
        "GpuMemHandler::getPeerDeviceMemPtr: rank out of bounds");
  }

  if (!exchanged_ && rank != selfRank_) {
    throw std::runtime_error(
        "GpuMemHandler: Must call exchangeMemPtrs() before accessing peer memory");
  }

  if (mode_ == MemSharingMode::kFabric) {
    // NOLINTNEXTLINE(performance-no-int-to-ptr): CUdeviceptr is an integer type
    return reinterpret_cast<void*>(fabricPeerPtrs_[rank]);
  } else {
    return cudaIpcPeerPtrs_[rank];
  }
}

void GpuMemHandler::exchangeMemPtrs() {
  if (exchanged_) {
    return;
  }

  // Single rank case: nothing to exchange, just mark as exchanged
  if (nRanks_ == 1) {
    exchanged_ = true;
    return;
  }

  if (mode_ == MemSharingMode::kFabric) {
    exchangeFabricHandles();
  } else {
    exchangeCudaIpcHandles();
  }

  exchanged_ = true;
}

// ============================================================================
// Fabric Mode Implementation
// ============================================================================

void GpuMemHandler::allocateFabricMemory(size_t size) {
#if CUDART_VERSION < 12030
  throw std::runtime_error("Fabric handles require CUDA 12.3+");
#else
  int cudaDev = 0;
  CUdevice cuDev;

  checkCudaError(cudaGetDevice(&cudaDev), "cudaGetDevice failed");
  checkCuError(cuDeviceGet(&cuDev, cudaDev), "cuDeviceGet failed");

  // Set up allocation properties with fabric handle support
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = cuDev;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

  // Check for GPUDirect RDMA support
  int rdmaSupported = 0;
  cuDeviceGetAttribute(
      &rdmaSupported, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED, cuDev);
  if (rdmaSupported) {
    prop.allocFlags.gpuDirectRDMACapable = 1;
  }

  // Get allocation granularity
  size_t granularity = 0;
  checkCuError(
      cuMemGetAllocationGranularity(
          &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
      "cuMemGetAllocationGranularity failed");

  // Round up size to granularity
  allocatedSize_ = ((size + granularity - 1) / granularity) * granularity;

  // Create the physical memory allocation
  checkCuError(
      cuMemCreate(&fabricLocalAllocHandle_, allocatedSize_, &prop, 0),
      "cuMemCreate failed");

  // Reserve virtual address space
  checkCuError(
      cuMemAddressReserve(&fabricLocalPtr_, allocatedSize_, granularity, 0, 0),
      "cuMemAddressReserve failed");

  // Map the physical memory to virtual address
  checkCuError(
      cuMemMap(fabricLocalPtr_, allocatedSize_, 0, fabricLocalAllocHandle_, 0),
      "cuMemMap failed");

  // Set access permissions
  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = cuDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  checkCuError(
      cuMemSetAccess(fabricLocalPtr_, allocatedSize_, &accessDesc, 1),
      "cuMemSetAccess failed");

  // Export to fabric handle for sharing with peers
  checkCuError(
      cuMemExportToShareableHandle(
          &fabricLocalHandle_,
          fabricLocalAllocHandle_,
          CU_MEM_HANDLE_TYPE_FABRIC,
          0),
      "cuMemExportToShareableHandle failed");

  // Store local pointer in peer array for uniform access
  fabricPeerPtrs_[selfRank_] = fabricLocalPtr_;
  fabricPeerAllocHandles_[selfRank_] = fabricLocalAllocHandle_;
#endif
}

void GpuMemHandler::exchangeFabricHandles() {
#if CUDART_VERSION < 12030
  throw std::runtime_error("Fabric handles require CUDA 12.3+");
#else
  // Prepare data for allGather: fabric handle + allocated size
  struct ExchangeData {
    FabricHandle handle;
    size_t allocatedSize;
  };

  std::vector<ExchangeData> allData(nRanks_);
  allData[selfRank_].handle = fabricLocalHandle_;
  allData[selfRank_].allocatedSize = allocatedSize_;

  // Exchange fabric handles with all ranks
  auto result =
      bootstrap_
          ->allGather(allData.data(), sizeof(ExchangeData), selfRank_, nRanks_)
          .get();
  if (result != 0) {
    throw std::runtime_error(
        "GpuMemHandler::exchangeFabricHandles allGather failed");
  }

  // Import peer memory from received fabric handles
  for (int32_t rank = 0; rank < nRanks_; ++rank) {
    if (rank == selfRank_) {
      continue;
    }
    fabricPeerAllocatedSizes_[rank] = allData[rank].allocatedSize;
    importFabricPeerMemory(
        rank, allData[rank].handle, allData[rank].allocatedSize);
  }
#endif
}

void GpuMemHandler::importFabricPeerMemory(
    int32_t rank,
    const FabricHandle& handle,
    size_t peerAllocatedSize) {
#if CUDART_VERSION < 12030
  throw std::runtime_error("Fabric handles require CUDA 12.3+");
#else
  int cudaDev = 0;
  CUdevice cuDev;

  checkCudaError(cudaGetDevice(&cudaDev), "cudaGetDevice failed");
  checkCuError(cuDeviceGet(&cuDev, cudaDev), "cuDeviceGet failed");

  // Import the fabric handle to get allocation handle
  checkCuError(
      cuMemImportFromShareableHandle(
          &fabricPeerAllocHandles_[rank],
          const_cast<void*>(static_cast<const void*>(&handle)),
          CU_MEM_HANDLE_TYPE_FABRIC),
      "cuMemImportFromShareableHandle failed");

  // Get allocation properties for granularity
  CUmemAllocationProp prop = {};
  checkCuError(
      cuMemGetAllocationPropertiesFromHandle(
          &prop, fabricPeerAllocHandles_[rank]),
      "cuMemGetAllocationPropertiesFromHandle failed");

  size_t granularity = 0;
  checkCuError(
      cuMemGetAllocationGranularity(
          &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
      "cuMemGetAllocationGranularity failed");

  // Reserve virtual address space for peer memory
  checkCuError(
      cuMemAddressReserve(
          &fabricPeerPtrs_[rank], peerAllocatedSize, granularity, 0, 0),
      "cuMemAddressReserve for peer failed");

  // Map peer's physical memory to our virtual address
  checkCuError(
      cuMemMap(
          fabricPeerPtrs_[rank],
          peerAllocatedSize,
          0,
          fabricPeerAllocHandles_[rank],
          0),
      "cuMemMap for peer failed");

  // Set access permissions for peer memory
  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = cuDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  checkCuError(
      cuMemSetAccess(fabricPeerPtrs_[rank], peerAllocatedSize, &accessDesc, 1),
      "cuMemSetAccess for peer failed");
#endif
}

void GpuMemHandler::cleanupFabric() {
#if CUDART_VERSION >= 12030
  // Check if CUDA context is still valid
  CUcontext ctx = nullptr;
  if (cuCtxGetCurrent(&ctx) != CUDA_SUCCESS || ctx == nullptr) {
    return;
  }

  // Release peer mappings
  for (int32_t rank = 0; rank < nRanks_; ++rank) {
    if (rank == selfRank_) {
      continue;
    }
    if (fabricPeerPtrs_[rank] != 0) {
      cuMemUnmap(fabricPeerPtrs_[rank], fabricPeerAllocatedSizes_[rank]);
      cuMemAddressFree(fabricPeerPtrs_[rank], fabricPeerAllocatedSizes_[rank]);
      fabricPeerPtrs_[rank] = 0;
    }
    if (fabricPeerAllocHandles_[rank] != 0) {
      cuMemRelease(fabricPeerAllocHandles_[rank]);
      fabricPeerAllocHandles_[rank] = 0;
    }
  }

  // Release local allocation
  if (fabricLocalPtr_ != 0) {
    cuMemUnmap(fabricLocalPtr_, allocatedSize_);
    cuMemAddressFree(fabricLocalPtr_, allocatedSize_);
    fabricLocalPtr_ = 0;
  }
  if (fabricLocalAllocHandle_ != 0) {
    cuMemRelease(fabricLocalAllocHandle_);
    fabricLocalAllocHandle_ = 0;
  }
#endif
}

// ============================================================================
// CudaIpc Mode Implementation
// ============================================================================

void GpuMemHandler::allocateCudaIpcMemory(size_t size) {
  checkCudaError(cudaMalloc(&cudaIpcLocalPtr_, size), "cudaMalloc failed");
  allocatedSize_ = size;

  // Get IPC handle for local memory
  checkCudaError(
      cudaIpcGetMemHandle(&cudaIpcLocalHandle_, cudaIpcLocalPtr_),
      "cudaIpcGetMemHandle failed");

  // Store local pointer in peer array
  cudaIpcPeerPtrs_[selfRank_] = cudaIpcLocalPtr_;
}

void GpuMemHandler::exchangeCudaIpcHandles() {
  // Exchange IPC handles with all ranks
  std::vector<cudaIpcMemHandle_t> allHandles(nRanks_);
  allHandles[selfRank_] = cudaIpcLocalHandle_;

  auto result =
      bootstrap_
          ->allGather(
              allHandles.data(), sizeof(cudaIpcMemHandle_t), selfRank_, nRanks_)
          .get();
  if (result != 0) {
    throw std::runtime_error(
        "GpuMemHandler::exchangeCudaIpcHandles allGather failed");
  }

  // Open peer memory handles
  for (int32_t rank = 0; rank < nRanks_; ++rank) {
    if (rank == selfRank_) {
      continue;
    }
    checkCudaError(
        cudaIpcOpenMemHandle(
            &cudaIpcPeerPtrs_[rank],
            allHandles[rank],
            cudaIpcMemLazyEnablePeerAccess),
        "cudaIpcOpenMemHandle failed");
  }
}

void GpuMemHandler::cleanupCudaIpc() {
  // Close peer handles
  for (int32_t rank = 0; rank < nRanks_; ++rank) {
    if (rank == selfRank_) {
      continue;
    }
    if (cudaIpcPeerPtrs_[rank] != nullptr) {
      cudaError_t err = cudaIpcCloseMemHandle(cudaIpcPeerPtrs_[rank]);
      if (err != cudaSuccess) {
        LOG(ERROR) << "cudaIpcCloseMemHandle failed for rank " << rank << ": "
                   << cudaGetErrorString(err);
      }
      cudaIpcPeerPtrs_[rank] = nullptr;
    }
  }

  // Free local allocation
  if (cudaIpcLocalPtr_ != nullptr) {
    cudaError_t err = cudaFree(cudaIpcLocalPtr_);
    if (err != cudaSuccess) {
      LOG(ERROR) << "cudaFree failed: " << cudaGetErrorString(err);
    }
    cudaIpcLocalPtr_ = nullptr;
  }
}

} // namespace comms::pipes
