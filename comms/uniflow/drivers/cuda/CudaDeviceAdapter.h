// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>

#include "comms/uniflow/drivers/DeviceAdapter.h"

#ifdef __HIP_PLATFORM_AMD__
// On AMD, CudaApi is not used but we keep the interface compatible
#else
#include "comms/uniflow/drivers/cuda/CudaApi.h"
#endif

namespace uniflow {

class CudaDeviceAdapter : public DeviceAdapter {
 public:
#ifdef __HIP_PLATFORM_AMD__
  explicit CudaDeviceAdapter(std::shared_ptr<void> /* unused */) {}
#else
  explicit CudaDeviceAdapter(std::shared_ptr<CudaApi> cudaApi)
      : cudaApi_(std::move(cudaApi)) {}
#endif

  Result<void*> pinnedHostAlloc(size_t size) override;
  Status pinnedHostFree(void* ptr) override;
  Result<void*> hostGetDevicePointer(void* hostPtr) override;

 private:
#ifndef __HIP_PLATFORM_AMD__
  std::shared_ptr<CudaApi> cudaApi_;
#endif
};

} // namespace uniflow
