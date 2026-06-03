// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>
#include <memory>

#include "comms/uniflow/drivers/DeviceAdapter.h"
#include "comms/uniflow/drivers/cuda/CudaApi.h"
#include "comms/uniflow/drivers/cuda/CudaDriverApi.h"

namespace uniflow {

class CudaDeviceAdapter : public DeviceAdapter {
 public:
  explicit CudaDeviceAdapter(
      std::shared_ptr<CudaApi> cudaApi,
      std::shared_ptr<CudaDriverApi> cudaDriverApi = nullptr);

  Result<void*> pinnedHostAlloc(size_t size) override;
  Status pinnedHostFree(void* ptr) override;
  Result<void*> hostGetDevicePointer(void* hostPtr) override;

  Result<bool> isDmaBuffSupported(int deviceId) override;
  Result<DmaBuff> exportDmaBuff(int deviceId, void* ptr, size_t len) override;
  Status closeDmaBuff(DmaBuff& buff) override;

 private:
  std::shared_ptr<CudaApi> cudaApi_;
  std::shared_ptr<CudaDriverApi> cudaDriverApi_;
  size_t pageSize_{0};
};

} // namespace uniflow
