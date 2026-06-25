// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/memory/CuMemMapping.h"

#include "comms/prims/core/Checks.h"
#include "comms/prims/memory/CuMemAllocation.h"

#if !defined(__HIP_PLATFORM_AMD__)
#include "comms/prims/platform/CudaDriverLazy.h"
#endif

#include <utility>

namespace comms::prims {

CuMemMapping::CuMemMapping(
    CUdevice cuDev,
    CUmemGenericAllocationHandle handle,
    std::size_t size,
    std::size_t granularity,
    std::shared_ptr<void> keepAlive)
    : size_(size), keepAlive_(std::move(keepAlive)) {
#if defined(__HIP_PLATFORM_AMD__) || CUDART_VERSION < 12030
  (void)cuDev;
  (void)handle;
  (void)granularity;
  throw std::runtime_error("CuMemMapping requires CUDA 12.3+");
#else
  try {
    checkCuError(
        pfn_cuMemAddressReserve(&devicePtr_, size_, granularity, 0, 0),
        "cuMemAddressReserve failed");

    checkCuError(
        pfn_cuMemMap(devicePtr_, size_, 0, handle, 0), "cuMemMap failed");
    mapped_ = true;

    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = cuDev;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    checkCuError(
        pfn_cuMemSetAccess(devicePtr_, size_, &accessDesc, 1),
        "cuMemSetAccess failed");
  } catch (...) {
    cleanup();
    throw;
  }
#endif
}

CuMemMapping CuMemMapping::overAllocation(
    std::shared_ptr<CuMemAllocation> alloc,
    std::size_t size,
    std::size_t granularity) {
  if (!alloc) {
    throw std::runtime_error(
        "CuMemMapping::overAllocation: alloc must be non-null");
  }
  const CUdevice cuDev = alloc->device();
  const CUmemGenericAllocationHandle handle = alloc->handle();
  return CuMemMapping(cuDev, handle, size, granularity, std::move(alloc));
}

CuMemMapping::~CuMemMapping() {
  cleanup();
}

CuMemMapping::CuMemMapping(CuMemMapping&& other) noexcept
    : devicePtr_(other.devicePtr_),
      size_(other.size_),
      mapped_(other.mapped_),
      keepAlive_(std::move(other.keepAlive_)) {
  other.devicePtr_ = 0;
  other.size_ = 0;
  other.mapped_ = false;
}

CuMemMapping& CuMemMapping::operator=(CuMemMapping&& other) noexcept {
  if (this != &other) {
    cleanup();
    devicePtr_ = other.devicePtr_;
    size_ = other.size_;
    mapped_ = other.mapped_;
    keepAlive_ = std::move(other.keepAlive_);
    other.devicePtr_ = 0;
    other.size_ = 0;
    other.mapped_ = false;
  }
  return *this;
}

void CuMemMapping::cleanup() noexcept {
#if !defined(__HIP_PLATFORM_AMD__) && CUDART_VERSION >= 12030
  if (devicePtr_ == 0) {
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

  if (mapped_) {
    pfn_cuMemUnmap(devicePtr_, size_);
    mapped_ = false;
  }
  pfn_cuMemAddressFree(devicePtr_, size_);
  devicePtr_ = 0;
#endif
}

} // namespace comms::prims
