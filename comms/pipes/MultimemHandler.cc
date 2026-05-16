// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/MultimemHandler.h"

#include "comms/pipes/CudaDriverLazy.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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
    pfn_cuGetErrorString(err, &errStr);
    throw std::runtime_error(
        std::string(msg) + ": " + (errStr ? errStr : "unknown error"));
  }
}

std::size_t alignUp(std::size_t value, std::size_t alignment) {
  if (alignment == 0) {
    return value;
  }
  return ((value + alignment - 1) / alignment) * alignment;
}

constexpr int kCachedMultimemSupportDevices = 8;
constexpr std::size_t kTrialAllocSize = 2 * 1024 * 1024;

enum class MultimemSupportCacheState : uint8_t {
  kUnknown,
  kUnsupported,
  kSupported,
};

bool multicastDriverApisAvailable() {
#if CUDART_VERSION < 12030
  return false;
#else
  return pfn_cuMulticastCreate != nullptr &&
      pfn_cuMulticastAddDevice != nullptr &&
      pfn_cuMulticastBindMem != nullptr && pfn_cuMulticastUnbind != nullptr &&
      pfn_cuMulticastGetGranularity != nullptr;
#endif
}

bool multicastSupported(CUdevice cuDev) {
#if CUDART_VERSION < 12030
  (void)cuDev;
  return false;
#else
  int multicastSupported = 0;
  auto err = pfn_cuDeviceGetAttribute(
      &multicastSupported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, cuDev);
  return err == CUDA_SUCCESS && multicastSupported != 0;
#endif
}

CUmemAllocationProp makeLocalAllocationProp(CUdevice cuDev) {
  CUmemAllocationProp localProp = {};
#if CUDART_VERSION >= 12030
  localProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  localProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  localProp.location.id = cuDev;
  localProp.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

  int rdmaSupported = 0;
  pfn_cuDeviceGetAttribute(
      &rdmaSupported, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED, cuDev);
  if (rdmaSupported) {
    localProp.allocFlags.gpuDirectRDMACapable = 1;
  }
#else
  (void)cuDev;
#endif
  return localProp;
}

bool fabricHandleSupported(CUdevice cuDev) {
#if CUDART_VERSION < 12030
  (void)cuDev;
  return false;
#else
  int fabricSupported = 0;
  auto err = pfn_cuDeviceGetAttribute(
      &fabricSupported,
      CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
      cuDev);
  if (err != CUDA_SUCCESS || fabricSupported == 0) {
    return false;
  }

  auto localProp = makeLocalAllocationProp(cuDev);
  std::size_t granularity = 0;
  err = pfn_cuMemGetAllocationGranularity(
      &granularity, &localProp, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  if (err != CUDA_SUCCESS) {
    return false;
  }

  const std::size_t allocSize = alignUp(kTrialAllocSize, granularity);
  CUmemGenericAllocationHandle handle = 0;
  err = pfn_cuMemCreate(&handle, allocSize, &localProp, 0);
  if (err != CUDA_SUCCESS) {
    return false;
  }

  FabricHandle fabricHandle = {};
  err = pfn_cuMemExportToShareableHandle(
      &fabricHandle, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0);
  if (err != CUDA_SUCCESS) {
    pfn_cuMemRelease(handle);
    return false;
  }

  CUmemGenericAllocationHandle importedHandle = 0;
  err = pfn_cuMemImportFromShareableHandle(
      &importedHandle, &fabricHandle, CU_MEM_HANDLE_TYPE_FABRIC);
  if (err != CUDA_SUCCESS) {
    pfn_cuMemRelease(handle);
    return false;
  }

  pfn_cuMemRelease(importedHandle);
  pfn_cuMemRelease(handle);
  return true;
#endif
}

int32_t findNvlRankForCommRank(
    int32_t commRank,
    const std::vector<int>& nvlRankToCommRank) {
  for (std::size_t nvlRank = 0; nvlRank < nvlRankToCommRank.size(); ++nvlRank) {
    if (nvlRankToCommRank[nvlRank] == commRank) {
      return static_cast<int32_t>(nvlRank);
    }
  }
  return -1;
}

bool isMultimemSupportedImpl(int cudaDevice) {
#if CUDART_VERSION < 12030
  return false;
#else
  if (cudaDevice < 0) {
    return false;
  }
  if (cuda_driver_lazy_init() != 0) {
    return false;
  }
  if (!multicastDriverApisAvailable()) {
    return false;
  }

  CUdevice cuDev = 0;
  if (pfn_cuDeviceGet(&cuDev, cudaDevice) != CUDA_SUCCESS) {
    return false;
  }

  if (!multicastSupported(cuDev)) {
    return false;
  }

  return fabricHandleSupported(cuDev);
#endif
}

} // namespace

MultimemHandler::MultimemHandler(
    std::shared_ptr<meta::comms::IBootstrap> bootstrap,
    int32_t commRank,
    std::vector<int> nvlRankToCommRank,
    int cudaDevice,
    std::size_t size)
    : bootstrap_(std::move(bootstrap)),
      commRank_(commRank),
      nvlRank_(findNvlRankForCommRank(commRank, nvlRankToCommRank)),
      nvlRanks_(static_cast<int32_t>(nvlRankToCommRank.size())),
      nvlRankToCommRank_(std::move(nvlRankToCommRank)),
      cudaDevice_(cudaDevice),
      requestedSize_(size) {
  if (nvlRanks_ <= 0) {
    throw std::runtime_error(
        "MultimemHandler: nvlRankToCommRank must be non-empty");
  }
  if (nvlRank_ < 0 || nvlRank_ >= nvlRanks_) {
    throw std::runtime_error(
        "MultimemHandler: commRank must appear in nvlRankToCommRank");
  }
  for (int rank = 0; rank < nvlRanks_; ++rank) {
    if (nvlRankToCommRank_[rank] < 0) {
      throw std::runtime_error(
          "MultimemHandler: nvlRankToCommRank contains a negative rank");
    }
    for (int prev = 0; prev < rank; ++prev) {
      if (nvlRankToCommRank_[prev] == nvlRankToCommRank_[rank]) {
        throw std::runtime_error(
            "MultimemHandler: nvlRankToCommRank contains duplicate ranks");
      }
    }
  }
  if (requestedSize_ == 0) {
    throw std::runtime_error("MultimemHandler: size must be non-zero");
  }
  if (!bootstrap_) {
    throw std::runtime_error("MultimemHandler: bootstrap must be non-null");
  }
  if (!isMultimemSupported(cudaDevice_)) {
    throw std::runtime_error(
        "MultimemHandler: multicast multimem is not supported. Requires CUDA "
        "12.3+, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, and fabric allocation "
        "handles.");
  }
}

MultimemHandler::~MultimemHandler() {
  cleanup();
}

bool MultimemHandler::isMultimemSupported(int cudaDevice) {
  if (cudaDevice < 0) {
    return false;
  }

  static std::array<
      std::atomic<MultimemSupportCacheState>,
      kCachedMultimemSupportDevices>
      cachedResults{};
  if (cudaDevice >= kCachedMultimemSupportDevices) {
    return isMultimemSupportedImpl(cudaDevice);
  }

  auto& cachedResult = cachedResults[static_cast<std::size_t>(cudaDevice)];
  const auto cachedState = cachedResult.load(std::memory_order_acquire);
  if (cachedState != MultimemSupportCacheState::kUnknown) {
    return cachedState == MultimemSupportCacheState::kSupported;
  }

  const bool supported = isMultimemSupportedImpl(cudaDevice);
  cachedResult.store(
      supported ? MultimemSupportCacheState::kSupported
                : MultimemSupportCacheState::kUnsupported,
      std::memory_order_release);
  return supported;
}

void MultimemHandler::exchange() {
  if (exchanged_) {
    return;
  }

#if CUDART_VERSION < 12030
  throw std::runtime_error("MultimemHandler requires CUDA 12.3+");
#else
  const char* failedPhase = "initializeDevice";
  const char* completedPhase = "none";

  try {
    // Multicast setup is host-synchronous locally, but it still has ordering
    // requirements across ranks:
    //
    // 1. initializeDevice() resolves cudaDevice_ to a CUdevice and builds the
    //    access descriptor used for both mappings. The caller must have already
    //    made cudaDevice_ current; this handler does not call cudaSetDevice().
    //
    // 2. computeAllocationSize() aligns the requested size to both the local
    //    allocation granularity and the multicast granularity. The local
    //    backing allocation and multicast object must agree on the final size.
    //
    // 3. create/export/exchange/import shares only the multicast object
    //    handle. Team rank 0 owns object creation, exports a fabric handle for
    //    it, all ranks exchange handles, and the other ranks import rank 0's
    //    object before joining the multicast team.
    //
    // 4. addLocalDeviceToMulticast() registers this rank's device with the
    //    shared object. CUDA requires every participating device to be added
    //    before memory can be bound to the multicast object.
    //
    // 5. allocateLocalMemory() creates, maps, grants access to, and zeroes this
    //    rank's local backing allocation. This handle stays private to the
    //    rank; peers do not need it because each rank binds its own allocation.
    //
    // 6. The pre-bind barrier is a control-plane robustness barrier. NCCL keeps
    //    the same barrier to avoid possible cuMulticastBindMem hangs during
    //    uneven setup or abort paths; it is not a GPU stream synchronization.
    //
    // 7. bindLocalMemoryToMulticast() attaches this rank's local allocation at
    //    offset 0 in the shared object. After all ranks bind the same range,
    //    multimem stores to that range can be replicated by NVSwitch.
    //
    // 8. mapMulticastMemory() reserves and maps this rank's multicast VA, then
    //    grants device access. Kernels use this VA for multimem instructions.
    //
    // 9. The post-map barrier is the "ready to use" contract for exchange():
    //    no rank returns and launches a kernel using multimem while another
    //    rank is still binding or mapping the multicast VA.
    const auto device = initializeDevice();
    cuDev_ = device.cuDev;
    deviceInitialized_ = true;
    completedPhase = failedPhase;

    failedPhase = "computeAllocationLayout";
    const auto layout = computeAllocationLayout(device);
    allocatedSize_ = layout.allocatedSize;
    completedPhase = failedPhase;

    failedPhase = "createMulticastHandle";
    createMulticastHandle(layout.allocatedSize);
    completedPhase = failedPhase;

    FabricHandle multicastFabricHandle = {};
    if (nvlRank_ == 0) {
      failedPhase = "exportMulticastHandle";
      multicastFabricHandle = exportMulticastHandle();
      completedPhase = failedPhase;
    }
    failedPhase = "exchangeMulticastHandle";
    multicastFabricHandle = exchangeMulticastHandle(multicastFabricHandle);
    completedPhase = failedPhase;

    // nvlRank_ is the rank inside this NVLink multicast team. Team rank 0
    // already owns the CUDA multicast handle it created; every other team rank
    // imports rank 0's exchanged fabric handle.
    if (nvlRank_ != 0) {
      failedPhase = "importMulticastHandle";
      importMulticastHandle(multicastFabricHandle);
      completedPhase = failedPhase;
    }

    // Every participating rank, including rank 0, must add its local device to
    // the shared multicast object before any rank binds memory to the object.
    failedPhase = "addLocalDeviceToMulticast";
    addLocalDeviceToMulticast(device.cuDev);
    completedPhase = failedPhase;

    // Only the multicast object is shared. Each rank binds its own local
    // allocation into that object.
    failedPhase = "allocateLocalMemory";
    allocateLocalMemory(device, layout);
    completedPhase = failedPhase;

    // Keep all ranks out of the driver's blocking cuMulticastBindMem path
    // until every rank has joined the multicast object and allocated its local
    // backing memory. NCCL uses the same barrier to avoid bind hangs during
    // uneven setup or abort paths.
    failedPhase = "synchronizeRanks:pre-bind";
    synchronizeRanks("pre-bind");
    completedPhase = failedPhase;

    failedPhase = "bindLocalMemoryToMulticast";
    bindLocalMemoryToMulticast(layout.allocatedSize);
    completedPhase = failedPhase;

    failedPhase = "mapMulticastMemory";
    mapMulticastMemory(device, layout);
    completedPhase = failedPhase;

    // exchange() returns ready-to-use pointers. This barrier prevents an early
    // rank from launching kernels against the multicast VA while another rank
    // is still binding or mapping that VA.
    failedPhase = "synchronizeRanks:post-map";
    synchronizeRanks("post-map");

    exchanged_ = true;
  } catch (const std::exception& ex) {
    const auto state = describeState(failedPhase, completedPhase);
    std::string message = std::string("MultimemHandler::exchange failed: ") +
        ex.what() + "\n" + state;
    std::fprintf(stderr, "%s\n", message.c_str());
    cleanup();
    throw std::runtime_error(message);
  } catch (...) {
    const auto state = describeState(failedPhase, completedPhase);
    std::string message =
        std::string(
            "MultimemHandler::exchange failed with unknown exception\n") +
        state;
    std::fprintf(stderr, "%s\n", message.c_str());
    cleanup();
    throw std::runtime_error(message);
  }
#endif
}

void* MultimemHandler::getLocalDeviceMemPtr() const {
  if (!exchanged_ || localPtr_ == 0) {
    throw std::runtime_error(
        "MultimemHandler: exchange() must complete before using local ptr");
  }
  // NOLINTNEXTLINE(performance-no-int-to-ptr): CUdeviceptr is an integer type
  return reinterpret_cast<void*>(localPtr_);
}

void* MultimemHandler::getMultimemDeviceMemPtr() const {
  if (!exchanged_ || multicastPtr_ == 0) {
    throw std::runtime_error(
        "MultimemHandler: exchange() must complete before using multimem ptr");
  }
  // NOLINTNEXTLINE(performance-no-int-to-ptr): CUdeviceptr is an integer type
  return reinterpret_cast<void*>(multicastPtr_);
}

MultimemHandler::DeviceContext MultimemHandler::initializeDevice() const {
  DeviceContext device;
#if CUDART_VERSION >= 12030
  checkCuError(
      pfn_cuDeviceGet(&device.cuDev, cudaDevice_), "cuDeviceGet failed");

  device.accessDesc = {};
  device.accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  device.accessDesc.location.id = device.cuDev;
  device.accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
#endif
  return device;
}

MultimemHandler::AllocationLayout MultimemHandler::computeAllocationLayout(
    const DeviceContext& device) const {
  AllocationLayout layout;
#if CUDART_VERSION >= 12030
  auto localProp = makeLocalAllocationProp(device.cuDev);

  checkCuError(
      pfn_cuMemGetAllocationGranularity(
          &layout.localGranularity,
          &localProp,
          CU_MEM_ALLOC_GRANULARITY_MINIMUM),
      "cuMemGetAllocationGranularity failed");

  CUmulticastObjectProp multicastProp = {};
  multicastProp.numDevices = static_cast<unsigned int>(nvlRanks_);
  multicastProp.size = alignUp(requestedSize_, layout.localGranularity);
  multicastProp.handleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
  multicastProp.flags = 0;

  checkCuError(
      pfn_cuMulticastGetGranularity(
          &layout.multicastGranularity,
          &multicastProp,
          CU_MULTICAST_GRANULARITY_MINIMUM),
      "cuMulticastGetGranularity failed");

  layout.allocatedSize = alignUp(
      alignUp(requestedSize_, layout.localGranularity),
      layout.multicastGranularity);
#endif
  return layout;
}

void MultimemHandler::createMulticastHandle(std::size_t allocatedSize) {
#if CUDART_VERSION >= 12030
  if (nvlRank_ != 0) {
    return;
  }

  CUmulticastObjectProp multicastProp = {};
  multicastProp.numDevices = static_cast<unsigned int>(nvlRanks_);
  multicastProp.size = allocatedSize;
  multicastProp.handleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
  multicastProp.flags = 0;

  checkCuError(
      pfn_cuMulticastCreate(&multicastHandle_, &multicastProp),
      "cuMulticastCreate failed");
  multicastHandleValid_ = true;
#endif
}

FabricHandle MultimemHandler::exportMulticastHandle() {
  FabricHandle fabricHandle = {};
#if CUDART_VERSION >= 12030
  if (nvlRank_ == 0) {
    checkCuError(
        pfn_cuMemExportToShareableHandle(
            &fabricHandle, multicastHandle_, CU_MEM_HANDLE_TYPE_FABRIC, 0),
        "cuMemExportToShareableHandle for multicast fabric handle failed");
  }
#endif
  return fabricHandle;
}

FabricHandle MultimemHandler::exchangeMulticastHandle(
    FabricHandle fabricHandle) {
  std::vector<FabricHandle> allHandles(nvlRanks_, FabricHandle{});
  allHandles[nvlRank_] = fabricHandle;

  auto result = bootstrap_
                    ->allGatherNvlDomain(
                        allHandles.data(),
                        sizeof(FabricHandle),
                        nvlRank_,
                        nvlRanks_,
                        nvlRankToCommRank_)
                    .get();
  if (result != 0) {
    throw std::runtime_error(
        "MultimemHandler::exchangeMulticastHandle fabric allGatherNvlDomain failed");
  }
  return allHandles[0];
}

void MultimemHandler::importMulticastHandle(const FabricHandle& fabricHandle) {
#if CUDART_VERSION >= 12030
  auto importedFabricHandle = fabricHandle;
  checkCuError(
      pfn_cuMemImportFromShareableHandle(
          &multicastHandle_, &importedFabricHandle, CU_MEM_HANDLE_TYPE_FABRIC),
      "cuMemImportFromShareableHandle for multicast fabric handle failed");
  multicastHandleValid_ = true;
#endif
}

void MultimemHandler::addLocalDeviceToMulticast(CUdevice cuDev) {
#if CUDART_VERSION >= 12030
  checkCuError(
      pfn_cuMulticastAddDevice(multicastHandle_, cuDev),
      "cuMulticastAddDevice failed");
#endif
}

void MultimemHandler::allocateLocalMemory(
    const DeviceContext& device,
    const AllocationLayout& layout) {
#if CUDART_VERSION >= 12030
  auto localProp = makeLocalAllocationProp(device.cuDev);

  checkCuError(
      pfn_cuMemAddressReserve(
          &localPtr_, layout.allocatedSize, layout.localGranularity, 0, 0),
      "cuMemAddressReserve for local multimem backing failed");

  checkCuError(
      pfn_cuMemCreate(&localAllocHandle_, layout.allocatedSize, &localProp, 0),
      "cuMemCreate for local multimem backing failed");
  localHandleValid_ = true;

  checkCuError(
      pfn_cuMemMap(localPtr_, layout.allocatedSize, 0, localAllocHandle_, 0),
      "cuMemMap for local multimem backing failed");
  localMapped_ = true;

  checkCuError(
      pfn_cuMemSetAccess(
          localPtr_, layout.allocatedSize, &device.accessDesc, 1),
      "cuMemSetAccess for local multimem backing failed");

  // NOLINTNEXTLINE(performance-no-int-to-ptr): CUdeviceptr is an integer type
  auto* localPtr = reinterpret_cast<void*>(localPtr_);
  checkCudaError(
      cudaMemset(localPtr, 0, layout.allocatedSize),
      "cudaMemset for local multimem backing failed");
  checkCudaError(
      cudaDeviceSynchronize(),
      "cudaDeviceSynchronize after local multimem backing memset failed");
#endif
}

void MultimemHandler::bindLocalMemoryToMulticast(std::size_t allocatedSize) {
#if CUDART_VERSION >= 12030
  checkCuError(
      pfn_cuMulticastBindMem(
          multicastHandle_, 0, localAllocHandle_, 0, allocatedSize, 0),
      "cuMulticastBindMem failed");
  multicastBound_ = true;
#endif
}

void MultimemHandler::mapMulticastMemory(
    const DeviceContext& device,
    const AllocationLayout& layout) {
#if CUDART_VERSION >= 12030
  checkCuError(
      pfn_cuMemAddressReserve(
          &multicastPtr_,
          layout.allocatedSize,
          layout.multicastGranularity,
          0,
          0),
      "cuMemAddressReserve for multimem VA failed");

  checkCuError(
      pfn_cuMemMap(multicastPtr_, layout.allocatedSize, 0, multicastHandle_, 0),
      "cuMemMap for multimem VA failed");
  multicastMapped_ = true;

  checkCuError(
      pfn_cuMemSetAccess(
          multicastPtr_, layout.allocatedSize, &device.accessDesc, 1),
      "cuMemSetAccess for multimem VA failed");
#endif
}

void MultimemHandler::synchronizeRanks(const char* phase) {
  auto barrierResult =
      bootstrap_->barrierNvlDomain(nvlRank_, nvlRanks_, nvlRankToCommRank_)
          .get();
  if (barrierResult != 0) {
    throw std::runtime_error(
        std::string("MultimemHandler::synchronizeRanks failed during ") +
        phase);
  }
}

std::string MultimemHandler::describeState(
    const char* failedPhase,
    const char* completedPhase) const {
  std::ostringstream out;
  out << "MultimemHandler state:"
      << " failedPhase=" << (failedPhase != nullptr ? failedPhase : "unknown")
      << " completedPhase="
      << (completedPhase != nullptr ? completedPhase : "unknown")
      << " commRank=" << commRank_ << " nvlRank=" << nvlRank_
      << " nvlRanks=" << nvlRanks_ << " cudaDevice=" << cudaDevice_
      << " requestedSize=" << requestedSize_
      << " allocatedSize=" << allocatedSize_ << " rankMap=[";
  for (std::size_t i = 0; i < nvlRankToCommRank_.size(); ++i) {
    if (i != 0) {
      out << ",";
    }
    out << nvlRankToCommRank_[i];
  }
  out << "] flags={exchanged=" << exchanged_
      << ",deviceInitialized=" << deviceInitialized_
      << ",multicastHandleValid=" << multicastHandleValid_
      << ",localHandleValid=" << localHandleValid_
      << ",localMapped=" << localMapped_
      << ",multicastBound=" << multicastBound_
      << ",multicastMapped=" << multicastMapped_
      << "} handles={cuDev=" << cuDev_ << ",localPtr=0x" << std::hex
      << localPtr_ << ",multicastPtr=0x" << multicastPtr_
      << ",localAllocHandle=0x" << localAllocHandle_ << ",multicastHandle=0x"
      << multicastHandle_ << std::dec << "}";
  return out.str();
}

void MultimemHandler::cleanup() {
#if CUDART_VERSION >= 12030
  if (!deviceInitialized_ || cuda_driver_lazy_init() != 0) {
    return;
  }

  CUcontext ctx = nullptr;
  if (pfn_cuCtxGetCurrent == nullptr ||
      pfn_cuCtxGetCurrent(&ctx) != CUDA_SUCCESS || ctx == nullptr) {
    return;
  }

  if (multicastMapped_) {
    pfn_cuMemUnmap(multicastPtr_, allocatedSize_);
    pfn_cuMemAddressFree(multicastPtr_, allocatedSize_);
    multicastPtr_ = 0;
    multicastMapped_ = false;
  } else if (multicastPtr_ != 0) {
    pfn_cuMemAddressFree(multicastPtr_, allocatedSize_);
    multicastPtr_ = 0;
  }

  if (multicastBound_ && pfn_cuMulticastUnbind != nullptr) {
    pfn_cuMulticastUnbind(multicastHandle_, cuDev_, 0, allocatedSize_);
    multicastBound_ = false;
  }

  if (multicastHandleValid_) {
    pfn_cuMemRelease(multicastHandle_);
    multicastHandle_ = 0;
    multicastHandleValid_ = false;
  }

  if (localMapped_) {
    pfn_cuMemUnmap(localPtr_, allocatedSize_);
    pfn_cuMemAddressFree(localPtr_, allocatedSize_);
    localPtr_ = 0;
    localMapped_ = false;
  } else if (localPtr_ != 0) {
    pfn_cuMemAddressFree(localPtr_, allocatedSize_);
    localPtr_ = 0;
  }

  if (localHandleValid_) {
    pfn_cuMemRelease(localAllocHandle_);
    localAllocHandle_ = 0;
    localHandleValid_ = false;
  }
#endif
}

} // namespace comms::pipes
