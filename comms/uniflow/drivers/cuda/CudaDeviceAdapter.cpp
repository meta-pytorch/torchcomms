// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/drivers/cuda/CudaDeviceAdapter.h"

namespace uniflow {

Result<void*> CudaDeviceAdapter::pinnedHostAlloc(size_t size) {
  return cudaApi_->hostAlloc(size, cudaHostAllocMapped | cudaHostAllocPortable);
}

Status CudaDeviceAdapter::pinnedHostFree(void* ptr) {
  return cudaApi_->hostFree(ptr);
}

Result<void*> CudaDeviceAdapter::hostGetDevicePointer(void* hostPtr) {
  return cudaApi_->hostGetDevicePointer(hostPtr);
}

std::shared_ptr<DeviceAdapter> createDeviceAdapter(
    std::shared_ptr<CudaApi> cudaApi) {
  if (!cudaApi) {
    cudaApi = std::make_shared<CudaApi>();
  }
  return std::make_shared<CudaDeviceAdapter>(std::move(cudaApi));
}

} // namespace uniflow
