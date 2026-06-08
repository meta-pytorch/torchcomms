// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/drivers/cuda/CudaDeviceAdapter.h"

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#endif

namespace uniflow {

Result<void*> CudaDeviceAdapter::pinnedHostAlloc(size_t size) {
#ifdef __HIP_PLATFORM_AMD__
  void* ptr = nullptr;
  auto err =
      hipHostMalloc(&ptr, size, hipHostMallocMapped | hipHostMallocPortable);
  if (err != hipSuccess) {
    return Err(
        ErrCode::DriverError,
        std::string("hipHostMalloc failed: ") + hipGetErrorString(err));
  }
  return ptr;
#else
  return cudaApi_->hostAlloc(size, cudaHostAllocMapped | cudaHostAllocPortable);
#endif
}

Status CudaDeviceAdapter::pinnedHostFree(void* ptr) {
#ifdef __HIP_PLATFORM_AMD__
  auto err = hipHostFree(ptr);
  if (err != hipSuccess) {
    return Err(
        ErrCode::DriverError,
        std::string("hipHostFree failed: ") + hipGetErrorString(err));
  }
  return Ok();
#else
  return cudaApi_->hostFree(ptr);
#endif
}

Result<void*> CudaDeviceAdapter::hostGetDevicePointer(void* hostPtr) {
#ifdef __HIP_PLATFORM_AMD__
  void* devicePtr = nullptr;
  auto err = hipHostGetDevicePointer(&devicePtr, hostPtr, 0);
  if (err != hipSuccess) {
    return Err(
        ErrCode::DriverError,
        std::string("hipHostGetDevicePointer failed: ") +
            hipGetErrorString(err));
  }
  return devicePtr;
#else
  return cudaApi_->hostGetDevicePointer(hostPtr);
#endif
}

std::shared_ptr<DeviceAdapter> createDeviceAdapter(
    std::shared_ptr<CudaApi> cudaApi) {
#ifdef __HIP_PLATFORM_AMD__
  // On AMD, we don't use CudaApi - return a simple adapter
  // The CudaDeviceAdapter class works for both CUDA and HIP
  return std::make_shared<CudaDeviceAdapter>(nullptr);
#else
  if (!cudaApi) {
    cudaApi = std::make_shared<CudaApi>();
  }
  return std::make_shared<CudaDeviceAdapter>(std::move(cudaApi));
#endif
}

} // namespace uniflow
