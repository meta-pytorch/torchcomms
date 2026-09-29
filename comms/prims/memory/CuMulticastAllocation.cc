// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/memory/CuMulticastAllocation.h"

#include "comms/prims/core/Checks.h"

#if !defined(__HIP_PLATFORM_AMD__)
#include "comms/prims/platform/CudaDriverLazy.h"
#endif

#include <glog/logging.h>

#include <stdexcept>

namespace comms::prims {

CuMulticastAllocation::CuMulticastAllocation(
    CUmemGenericAllocationHandle handle,
    std::size_t size)
    : handle_(handle), size_(size) {}

CuMulticastAllocation CuMulticastAllocation::create(
    const CUmulticastObjectProp& prop) {
#if defined(__HIP_PLATFORM_AMD__) || CUDART_VERSION < 12030
  (void)prop;
  throw std::runtime_error("CuMulticastAllocation::create requires CUDA 12.3+");
#else
  if (cuda_driver_lazy_init() != 0) {
    throw std::runtime_error(
        "CuMulticastAllocation::create: CUDA driver not available");
  }
  CUmemGenericAllocationHandle handle = 0;
  checkCuError(
      pfn_cuMulticastCreate(&handle, &prop), "cuMulticastCreate failed");
  return CuMulticastAllocation(handle, prop.size);
#endif
}

CuMulticastAllocation CuMulticastAllocation::adopt(
    CUmemGenericAllocationHandle handle,
    std::size_t size) {
  return CuMulticastAllocation(handle, size);
}

CuMulticastAllocation::~CuMulticastAllocation() {
  release();
}

CuMulticastAllocation::CuMulticastAllocation(
    CuMulticastAllocation&& other) noexcept
    : handle_(other.handle_),
      device_(other.device_),
      size_(other.size_),
      boundSize_(other.boundSize_),
      boundMcOffset_(other.boundMcOffset_),
      deviceAdded_(other.deviceAdded_),
      bound_(other.bound_) {
  other.handle_ = 0;
  other.bound_ = false;
  other.deviceAdded_ = false;
}

CuMulticastAllocation& CuMulticastAllocation::operator=(
    CuMulticastAllocation&& other) noexcept {
  if (this != &other) {
    release();
    handle_ = other.handle_;
    device_ = other.device_;
    size_ = other.size_;
    boundSize_ = other.boundSize_;
    boundMcOffset_ = other.boundMcOffset_;
    deviceAdded_ = other.deviceAdded_;
    bound_ = other.bound_;
    other.handle_ = 0;
    other.bound_ = false;
    other.deviceAdded_ = false;
  }
  return *this;
}

void CuMulticastAllocation::addDevice(CUdevice cuDev) {
#if !defined(__HIP_PLATFORM_AMD__) && CUDART_VERSION >= 12030
  // Single-device: the destructor unbinds exactly one device (`device_`), so
  // a second addDevice() call would leak the earlier device's binding.
  if (deviceAdded_) {
    throw std::runtime_error(
        "CuMulticastAllocation::addDevice: device already added; "
        "single-device per object is the only supported mode");
  }
  checkCuError(
      pfn_cuMulticastAddDevice(handle_, cuDev), "cuMulticastAddDevice failed");
  device_ = cuDev;
  deviceAdded_ = true;
#else
  (void)cuDev;
  throw std::runtime_error("CuMulticastAllocation requires CUDA 12.3+");
#endif
}

void CuMulticastAllocation::bindMem(
    CUmemGenericAllocationHandle physHandle,
    std::size_t mcOffset,
    std::size_t physOffset,
    std::size_t size) {
#if !defined(__HIP_PLATFORM_AMD__) && CUDART_VERSION >= 12030
  // Single-bind: the destructor unbinds exactly one (mcOffset, size) range;
  // a second bindMem() call would leak the earlier binding.
  if (bound_) {
    throw std::runtime_error(
        "CuMulticastAllocation::bindMem: already bound; "
        "single-bind per object is the only supported mode");
  }
  checkCuError(
      pfn_cuMulticastBindMem(
          handle_, mcOffset, physHandle, physOffset, size, 0),
      "cuMulticastBindMem failed");
  bound_ = true;
  boundSize_ = size;
  boundMcOffset_ = mcOffset;
#else
  (void)physHandle;
  (void)mcOffset;
  (void)physOffset;
  (void)size;
  throw std::runtime_error("CuMulticastAllocation requires CUDA 12.3+");
#endif
}

void CuMulticastAllocation::release() noexcept {
#if !defined(__HIP_PLATFORM_AMD__) && CUDART_VERSION >= 12030
  if (handle_ == 0) {
    return;
  }
  // Destructors can't throw, so the unwind-without-context paths below are
  // best-effort: log when we have to skip cleanup so the leaked multicast
  // handle is visible rather than silent.
  if (cuda_driver_lazy_init() != 0) {
    LOG(ERROR)
        << "CuMulticastAllocation::release: CUDA driver lazy init failed; "
           "leaking multicast handle=0x"
        << std::hex << handle_ << std::dec;
    handle_ = 0;
    return;
  }
  CUcontext ctx = nullptr;
  if (pfn_cuCtxGetCurrent == nullptr ||
      pfn_cuCtxGetCurrent(&ctx) != CUDA_SUCCESS || ctx == nullptr) {
    LOG(ERROR) << "CuMulticastAllocation::release: no current CUDA context; "
                  "leaking multicast handle=0x"
               << std::hex << handle_ << std::dec;
    handle_ = 0;
    return;
  }

  // Unbind the local device's binding before releasing the multicast handle.
  // The earlier no-context / no-driver branches log on failure so leaked
  // resources are observable; mirror that here so a driver-level unbind or
  // release failure isn't silently swallowed.
  if (bound_ && deviceAdded_ && pfn_cuMulticastUnbind != nullptr) {
    const CUresult unbindRc =
        pfn_cuMulticastUnbind(handle_, device_, boundMcOffset_, boundSize_);
    if (unbindRc != CUDA_SUCCESS) {
      LOG(ERROR) << "CuMulticastAllocation::release: cuMulticastUnbind failed "
                    "(CUresult="
                 << static_cast<int>(unbindRc) << ", handle=0x" << std::hex
                 << handle_ << std::dec << ")";
    }
    bound_ = false;
  }
  const CUresult releaseRc = pfn_cuMemRelease(handle_);
  if (releaseRc != CUDA_SUCCESS) {
    LOG(ERROR) << "CuMulticastAllocation::release: cuMemRelease failed "
                  "(CUresult="
               << static_cast<int>(releaseRc) << ", handle=0x" << std::hex
               << handle_ << std::dec << "); leaking multicast handle";
  }
  handle_ = 0;
#endif
}

} // namespace comms::prims
