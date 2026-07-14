// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/memory/CuMemAllocation.h"

#include "comms/common/BitOps.cuh"
#include "comms/prims/core/Checks.h"

#if !defined(__HIP_PLATFORM_AMD__)
#include "comms/prims/platform/CudaDriverLazy.h"
#endif

#include <algorithm>
#include <stdexcept>

namespace comms::prims {

namespace {

bool isPowerOfTwo(std::size_t x) {
  return x != 0 && (x & (x - 1)) == 0;
}

} // namespace

CUmemAllocationProp makeVmmAllocationProp(
    CUdevice cuDev,
    unsigned int requestedHandleTypesMask) {
  CUmemAllocationProp prop = {};
#if !defined(__HIP_PLATFORM_AMD__) && CUDART_VERSION >= 12030
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = cuDev;
  prop.requestedHandleTypes =
      static_cast<CUmemAllocationHandleType>(requestedHandleTypesMask);

  int rdmaSupported = 0;
  pfn_cuDeviceGetAttribute(
      &rdmaSupported, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED, cuDev);
  if (rdmaSupported) {
    prop.allocFlags.gpuDirectRDMACapable = 1;
  }
#else
  (void)cuDev;
  (void)requestedHandleTypesMask;
#endif
  return prop;
}

CuMemAllocation::CuMemAllocation(
    CUmemGenericAllocationHandle handle,
    CUdevice device,
    std::size_t size,
    std::size_t granularity,
    unsigned int supportedHandleTypes) noexcept
    : handle_(handle),
      device_(device),
      size_(size),
      granularity_(granularity),
      supportedHandleTypes_(supportedHandleTypes) {}

std::unique_ptr<CuMemAllocation> CuMemAllocation::create(
    CUdevice cuDev,
    std::size_t size,
    unsigned int requestedHandleTypesMask,
    std::size_t alignFloor) {
#if defined(__HIP_PLATFORM_AMD__) || CUDART_VERSION < 12030
  (void)cuDev;
  (void)size;
  (void)requestedHandleTypesMask;
  (void)alignFloor;
  throw std::runtime_error("CuMemAllocation::create requires CUDA 12.3+");
#else
  if (cuda_driver_lazy_init() != 0) {
    throw std::runtime_error(
        "CuMemAllocation::create: CUDA driver not available");
  }
  if (alignFloor != 0 && !isPowerOfTwo(alignFloor)) {
    throw std::runtime_error(
        "CuMemAllocation::create: alignFloor must be zero or a power of two");
  }

  unsigned int mask = requestedHandleTypesMask;
  auto prop = makeVmmAllocationProp(cuDev, mask);

  std::size_t driverGranularity = 0;
  checkCuError(
      pfn_cuMemGetAllocationGranularity(
          &driverGranularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED),
      "cuMemGetAllocationGranularity failed");
  if (!isPowerOfTwo(driverGranularity)) {
    throw std::runtime_error(
        "CuMemAllocation::create: driver allocation granularity is not a power of two");
  }

  // Both granularities are powers of two, so the effective granularity is a
  // multiple of the driver granularity and cuMemCreate's size requirement is
  // satisfied. alignFloor lets the caller raise the size/alignment floor so a
  // single physical allocation can satisfy a larger granularity requirement
  // (e.g. a multicast object's granularity).
  const std::size_t effGran = std::max(driverGranularity, alignFloor);
  const std::size_t allocatedSize = comms::bitops::roundUp(size, effGran);

  CUmemGenericAllocationHandle handle = 0;
  CUresult createResult = pfn_cuMemCreate(&handle, allocatedSize, &prop, 0);
  if ((createResult == CUDA_ERROR_NOT_PERMITTED ||
       createResult == CUDA_ERROR_NOT_SUPPORTED) &&
      (mask & CU_MEM_HANDLE_TYPE_FABRIC)) {
    // Fabric requested but unavailable (e.g. single host without IMEX). Drop
    // fabric and retry with the remaining handle types (mirrors ncclx
    // allocator.cc).
    mask &= ~static_cast<unsigned int>(CU_MEM_HANDLE_TYPE_FABRIC);
    prop = makeVmmAllocationProp(cuDev, mask);
    createResult = pfn_cuMemCreate(&handle, allocatedSize, &prop, 0);
  }
  checkCuError(createResult, "cuMemCreate failed");

  // From here on the factory owns `handle`. `new` can throw bad_alloc;
  // release the handle so we never leak a raw allocation on storage exhaust.
  try {
    return std::unique_ptr<CuMemAllocation>(
        new CuMemAllocation(handle, cuDev, allocatedSize, effGran, mask));
  } catch (...) {
    pfn_cuMemRelease(handle);
    throw;
  }
#endif
}

std::unique_ptr<CuMemAllocation> CuMemAllocation::retain(void* ptr) {
#if defined(__HIP_PLATFORM_AMD__) || CUDART_VERSION < 12030
  (void)ptr;
  throw std::runtime_error("CuMemAllocation::retain requires CUDA 12.3+");
#else
  if (cuda_driver_lazy_init() != 0) {
    throw std::runtime_error(
        "CuMemAllocation::retain: CUDA driver not available");
  }

  CUmemGenericAllocationHandle handle = 0;
  checkCuError(
      pfn_cuMemRetainAllocationHandle(&handle, ptr),
      "cuMemRetainAllocationHandle failed");

  // After cuMemRetainAllocationHandle succeeds we own one retain reference on
  // `handle`. Any failure from here on must release that reference exactly
  // once — funnel through one try/catch instead of the per-call cleanup that
  // was easy to miss.
  try {
    CUdeviceptr base = 0;
    std::size_t size = 0;
    checkCuError(
        pfn_cuMemGetAddressRange(
            &base, &size, reinterpret_cast<CUdeviceptr>(ptr)),
        "cuMemGetAddressRange failed");

    CUmemAllocationProp prop = {};
    checkCuError(
        pfn_cuMemGetAllocationPropertiesFromHandle(&prop, handle),
        "cuMemGetAllocationPropertiesFromHandle failed");

    std::size_t granularity = 0;
    checkCuError(
        pfn_cuMemGetAllocationGranularity(
            &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
        "cuMemGetAllocationGranularity failed");

    return std::unique_ptr<CuMemAllocation>(new CuMemAllocation(
        handle,
        static_cast<CUdevice>(prop.location.id),
        size,
        granularity,
        static_cast<unsigned int>(prop.requestedHandleTypes)));
  } catch (...) {
    pfn_cuMemRelease(handle);
    throw;
  }
#endif
}

std::unique_ptr<CuMemAllocation> CuMemAllocation::adopt(
    CUmemGenericAllocationHandle handle,
    CUdevice device,
    std::size_t size) {
#if defined(__HIP_PLATFORM_AMD__) || CUDART_VERSION < 12030
  (void)handle;
  (void)device;
  (void)size;
  throw std::runtime_error("CuMemAllocation::adopt requires CUDA 12.3+");
#else
  if (cuda_driver_lazy_init() != 0) {
    pfn_cuMemRelease(handle);
    throw std::runtime_error(
        "CuMemAllocation::adopt: CUDA driver not available");
  }
  // adopt() takes ownership of `handle` on entry. Query the handle's
  // properties + granularity so the caller never has to hold the raw handle
  // past this call; on any internal failure (CUDA error, `new` bad_alloc)
  // release the handle and rethrow so no raw-handle window leaks out.
  try {
    CUmemAllocationProp prop = {};
    checkCuError(
        pfn_cuMemGetAllocationPropertiesFromHandle(&prop, handle),
        "CuMemAllocation::adopt: cuMemGetAllocationPropertiesFromHandle failed");
    std::size_t granularity = 0;
    checkCuError(
        pfn_cuMemGetAllocationGranularity(
            &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
        "CuMemAllocation::adopt: cuMemGetAllocationGranularity failed");
    return std::unique_ptr<CuMemAllocation>(new CuMemAllocation(
        handle,
        device,
        size,
        granularity,
        static_cast<unsigned int>(prop.requestedHandleTypes)));
  } catch (...) {
    pfn_cuMemRelease(handle);
    throw;
  }
#endif
}

CuMemAllocation::~CuMemAllocation() {
  release();
}

CuMemAllocation::CuMemAllocation(CuMemAllocation&& other) noexcept
    : handle_(other.handle_),
      device_(other.device_),
      size_(other.size_),
      granularity_(other.granularity_),
      supportedHandleTypes_(other.supportedHandleTypes_) {
  other.handle_ = 0;
  other.size_ = 0;
}

CuMemAllocation& CuMemAllocation::operator=(CuMemAllocation&& other) noexcept {
  if (this != &other) {
    release();
    handle_ = other.handle_;
    device_ = other.device_;
    size_ = other.size_;
    granularity_ = other.granularity_;
    supportedHandleTypes_ = other.supportedHandleTypes_;
    other.handle_ = 0;
    other.size_ = 0;
  }
  return *this;
}

bool CuMemAllocation::supportsFabric() const {
#if !defined(__HIP_PLATFORM_AMD__) && CUDART_VERSION >= 12030
  return (supportedHandleTypes_ & CU_MEM_HANDLE_TYPE_FABRIC) != 0;
#else
  return false;
#endif
}

void CuMemAllocation::release() noexcept {
#if !defined(__HIP_PLATFORM_AMD__) && CUDART_VERSION >= 12030
  if (handle_ == 0) {
    return;
  }
  if (cuda_driver_lazy_init() != 0) {
    return;
  }
  CUcontext ctx = nullptr;
  if (pfn_cuCtxGetCurrent == nullptr ||
      pfn_cuCtxGetCurrent(&ctx) != CUDA_SUCCESS || ctx == nullptr) {
    return;
  }
  pfn_cuMemRelease(handle_);
  handle_ = 0;
#endif
}

} // namespace comms::prims
