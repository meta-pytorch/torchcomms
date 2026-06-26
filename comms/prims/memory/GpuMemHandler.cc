// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/memory/GpuMemHandler.h"

// CUDA driver API (`cuda.h` / `cudaTypedefs.h`) is NVIDIA-only. On AMD,
// fabric handles aren't supported, so all `cuMem*` driver-API code paths
// in this file are guarded by `#ifndef __HIP_PLATFORM_AMD__` and the cudaIpc
// fallback path (which HIPify rewrites to hipIpc) is the only available
// mechanism.
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#else
#include "comms/prims/platform/CudaDriverLazy.h"
#endif

#include "comms/prims/core/Checks.h"

#include <glog/logging.h>
#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace comms::prims {

namespace {

// Minimum allocation size for trial allocation (matches ctran)
constexpr size_t kTrialAllocSize = 2097152UL; // 2MB

// Helper function that performs the actual fabric handle support check.
// This is called once and the result is cached by isFabricHandleSupported().
bool checkFabricHandleSupportedImpl() {
#if defined(__HIP_PLATFORM_AMD__) || CUDART_VERSION < 12030
  return false;
#else
  if (cuda_driver_lazy_init() != 0) {
    return false;
  }

  int cudaDev = 0;
  CUdevice cuDev;

  // 1. Check basic CUDA setup
  cudaError_t cudaErr = cudaGetDevice(&cudaDev);
  if (cudaErr != cudaSuccess) {
    return false;
  }

  CUresult cuErr = pfn_cuDeviceGet(&cuDev, cudaDev);
  if (cuErr != CUDA_SUCCESS) {
    return false;
  }

  // 2. Check device attribute for fabric handle support
  int fabricSupported = 0;
  cuErr = pfn_cuDeviceGetAttribute(
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
  cuErr = pfn_cuMemGetAllocationGranularity(
      &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  if (cuErr != CUDA_SUCCESS) {
    return false;
  }

  size_t allocSize =
      ((kTrialAllocSize + granularity - 1) / granularity) * granularity;

  CUmemGenericAllocationHandle handle;
  cuErr = pfn_cuMemCreate(&handle, allocSize, &prop, 0);
  if (cuErr != CUDA_SUCCESS) {
    return false;
  }

  CUdeviceptr ptr;
  cuErr = pfn_cuMemAddressReserve(&ptr, allocSize, granularity, 0, 0);
  if (cuErr != CUDA_SUCCESS) {
    pfn_cuMemRelease(handle);
    return false;
  }

  cuErr = pfn_cuMemMap(ptr, allocSize, 0, handle, 0);
  if (cuErr != CUDA_SUCCESS) {
    pfn_cuMemAddressFree(ptr, allocSize);
    pfn_cuMemRelease(handle);
    return false;
  }

  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = cuDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  cuErr = pfn_cuMemSetAccess(ptr, allocSize, &accessDesc, 1);
  if (cuErr != CUDA_SUCCESS) {
    pfn_cuMemUnmap(ptr, allocSize);
    pfn_cuMemAddressFree(ptr, allocSize);
    pfn_cuMemRelease(handle);
    return false;
  }

  // 4. Trial export to fabric handle
  CUmemFabricHandle fabricHandle;
  cuErr = pfn_cuMemExportToShareableHandle(
      &fabricHandle, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0);
  if (cuErr != CUDA_SUCCESS) {
    pfn_cuMemUnmap(ptr, allocSize);
    pfn_cuMemAddressFree(ptr, allocSize);
    pfn_cuMemRelease(handle);
    return false;
  }

  // 5. Trial import from fabric handle
  CUmemGenericAllocationHandle importedHandle;
  cuErr = pfn_cuMemImportFromShareableHandle(
      &importedHandle, &fabricHandle, CU_MEM_HANDLE_TYPE_FABRIC);
  if (cuErr != CUDA_SUCCESS) {
    pfn_cuMemUnmap(ptr, allocSize);
    pfn_cuMemAddressFree(ptr, allocSize);
    pfn_cuMemRelease(handle);
    return false;
  }

  // Import increases ref count, release it
  pfn_cuMemRelease(importedHandle);

  // Cleanup trial allocation
  pfn_cuMemUnmap(ptr, allocSize);
  pfn_cuMemAddressFree(ptr, allocSize);
  pfn_cuMemRelease(handle);

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
    std::shared_ptr<meta::comms::IBootstrap> bootstrap,
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
    std::shared_ptr<meta::comms::IBootstrap> bootstrap,
    int32_t selfRank,
    int32_t nRanks,
    size_t size,
    MemSharingMode mode,
    std::size_t alignFloor)
    : bootstrap_(std::move(bootstrap)),
      selfRank_(selfRank),
      nRanks_(nRanks),
      mode_(mode) {
  if (mode_ == MemSharingMode::kFabric && !isFabricHandleSupported()) {
    throw std::runtime_error(
        "Fabric handle mode requested but not supported on this system. "
        "Requires Hopper (H100) or newer GPU with CUDA 12.3+.");
  }

  init(size, alignFloor);
}

GpuMemHandler::~GpuMemHandler() {
  if (isVmmMode()) {
    cleanupVmm();
  } else {
    cleanupCudaIpc();
  }
}

void GpuMemHandler::init(size_t size, std::size_t alignFloor) {
  if (isVmmMode()) {
    allocateVmmMemory(size, alignFloor);
  } else {
    allocateCudaIpcMemory(size);
  }
}

void* GpuMemHandler::getLocalDeviceMemPtr() const {
  if (isVmmMode()) {
    // NOLINTNEXTLINE(performance-no-int-to-ptr): CUdeviceptr is an integer type
    return reinterpret_cast<void*>(unicastMapping_->devicePtr());
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

  // The self pointer is always available (even before exchange / single-rank);
  // peers_.peerPtrs is only populated by exchangeMemPtrs().
  if (rank == selfRank_) {
    return getLocalDeviceMemPtr();
  }
  return peers_.peerPtrs[static_cast<std::size_t>(rank)];
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

  if (isVmmMode()) {
    exchangeVmmHandles();
  } else {
    exchangeCudaIpcHandles();
  }

  exchanged_ = true;
}

const cudaIpcMemHandle_t& GpuMemHandler::getLocalIpcHandle() const {
  if (isVmmMode()) {
    throw std::runtime_error(
        "GpuMemHandler::getLocalIpcHandle: not available in VMM (fabric/posix-fd) mode");
  }
  return cudaIpcLocalHandle_;
}

// ============================================================================
// VMM Mode Implementation (kFabric)
// ============================================================================

void GpuMemHandler::allocateVmmMemory(size_t size, std::size_t alignFloor) {
#if defined(__HIP_PLATFORM_AMD__) || CUDART_VERSION < 12030
  (void)size;
  (void)alignFloor;
  throw std::runtime_error("VMM fabric handles require CUDA 12.3+");
#else
  if (cuda_driver_lazy_init() != 0) {
    throw std::runtime_error("CUDA driver not available");
  }

  int cudaDev = 0;
  CUdevice cuDev;

  checkCudaError(cudaGetDevice(&cudaDev), "cudaGetDevice failed");
  checkCuError(pfn_cuDeviceGet(&cuDev, cudaDev), "cuDeviceGet failed");

  // CuMemAllocation::create allocates the physical handle requesting fabric
  // support. The shareable-handle export is deferred to exchangeMemPtrs() so a
  // handler used purely as a multicast backing (no P2P) does no export.
  //
  // create() returns unique_ptr; the implicit promotion to shared_ptr keeps
  // exception safety -- if the control-block allocation throws, the
  // unique_ptr's destructor releases the physical handle.
  allocation_ = CuMemAllocation::create(
      cuDev, size, CU_MEM_HANDLE_TYPE_FABRIC, alignFloor);
  allocatedSize_ = allocation_->size();

  // CuMemMapping reserves and maps the unicast VA and grants access. It co-owns
  // allocation_ so the physical handle outlives the VA.
  unicastMapping_ = std::make_unique<CuMemMapping>(CuMemMapping::overAllocation(
      allocation_, allocation_->size(), allocation_->granularity()));
#endif
}

void GpuMemHandler::exchangeVmmHandles() {
#if defined(__HIP_PLATFORM_AMD__) || CUDART_VERSION < 12030
  throw std::runtime_error("VMM fabric handles require CUDA 12.3+");
#else
  if (cuda_driver_lazy_init() != 0) {
    throw std::runtime_error("CUDA driver not available");
  }

  int cudaDev = 0;
  CUdevice cuDev;
  checkCudaError(cudaGetDevice(&cudaDev), "cudaGetDevice failed");
  checkCuError(pfn_cuDeviceGet(&cuDev, cudaDev), "cuDeviceGet failed");

  peers_ = nvlMemExchangeVmm(
      *bootstrap_,
      selfRank_,
      nRanks_,
      cuDev,
      allocation_->handle(),
      getLocalDeviceMemPtr(),
      allocation_->size());
#endif
}

void GpuMemHandler::cleanupVmm() {
#if !defined(__HIP_PLATFORM_AMD__) && CUDART_VERSION >= 12030
  // RAII teardown: unicast VA first, then peer VAs (each co-owns its imported
  // allocation), then the shared physical allocation (released when the last
  // shared_ptr owner drops).
  unicastMapping_.reset();
  peers_.vmmMappings.clear();
  allocation_.reset();
#endif
}

// ============================================================================
// CudaIpc Mode Implementation
// ============================================================================

void GpuMemHandler::allocateCudaIpcMemory(size_t size) {
  if (mode_ == MemSharingMode::kCudaIpcUncached) {
    // GPU-uncached alloc on AMD; see MemSharingMode::kCudaIpcUncached.
#ifdef __HIP_PLATFORM_AMD__
    checkCudaError(
        hipExtMallocWithFlags(&cudaIpcLocalPtr_, size, hipDeviceMallocUncached),
        "hipExtMallocWithFlags(hipDeviceMallocUncached) failed");
#else
    checkCudaError(cudaMalloc(&cudaIpcLocalPtr_, size), "cudaMalloc failed");
#endif
  } else {
    checkCudaError(cudaMalloc(&cudaIpcLocalPtr_, size), "cudaMalloc failed");
  }
  // Cache the local IPC handle for getLocalIpcHandle(). It depends only on the
  // local allocation, so deriving it here makes it valid before exchange too.
  checkCudaError(
      cudaIpcGetMemHandle(&cudaIpcLocalHandle_, cudaIpcLocalPtr_),
      "cudaIpcGetMemHandle failed");
  allocatedSize_ = size;
}

void GpuMemHandler::exchangeCudaIpcHandles() {
  peers_ =
      nvlMemExchangeCudaIpc(*bootstrap_, selfRank_, nRanks_, cudaIpcLocalPtr_);
}

void GpuMemHandler::cleanupCudaIpc() {
  // Close peer handles opened by cudaIpcOpenMemHandle. The self slot holds the
  // local pointer (freed below), not an opened handle.
  for (int32_t rank = 0; rank < nRanks_; ++rank) {
    if (rank == selfRank_) {
      continue;
    }
    void* peerPtr = peers_.peerPtrs.empty()
        ? nullptr
        : peers_.peerPtrs[static_cast<std::size_t>(rank)];
    if (peerPtr != nullptr) {
      cudaError_t err = cudaIpcCloseMemHandle(peerPtr);
      if (err != cudaSuccess) {
        LOG(ERROR) << "cudaIpcCloseMemHandle failed for rank " << rank << ": "
                   << cudaGetErrorString(err);
      }
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

} // namespace comms::prims
