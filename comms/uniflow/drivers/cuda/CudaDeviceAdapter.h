// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>

#include "comms/uniflow/drivers/DeviceAdapter.h"
#include "comms/uniflow/drivers/cuda/CudaApi.h"

namespace uniflow {

class CudaDeviceAdapter : public DeviceAdapter {
 public:
  explicit CudaDeviceAdapter(std::shared_ptr<CudaApi> cudaApi)
      : cudaApi_(std::move(cudaApi)) {}

  Result<void*> pinnedHostAlloc(size_t size) override;
  Status pinnedHostFree(void* ptr) override;
  Result<void*> hostGetDevicePointer(void* hostPtr) override;

 private:
  std::shared_ptr<CudaApi> cudaApi_;
};

} // namespace uniflow
