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

  DeviceCapabilities capabilities() const noexcept override;
  Status copyAsync(
      int deviceId,
      void* dst,
      const void* src,
      size_t count,
      DeviceCopyKind kind,
      DeviceStream stream) override;
  Status signal64(
      int deviceId,
      DeviceStream stream,
      uint64_t* address,
      uint64_t value) override;
  Result<DeviceStream> createNonBlockingStream(int deviceId) override;
  Status synchronizeStream(int deviceId, DeviceStream stream) override;
  Status destroyStream(int deviceId, DeviceStream stream) override;
  Result<std::unique_ptr<DeviceCompletion>> recordCompletion(
      int deviceId,
      DeviceStream stream) override;

  Result<void*> pinnedHostAlloc(size_t size) override;
  Status pinnedHostFree(void* ptr) override;
  Result<void*> hostGetDevicePointer(void* hostPtr) override;
  Result<void*> pinnedHostAlloc(int deviceId, size_t size) override;
  Status pinnedHostFree(int deviceId, void* ptr) override;
  Result<void*> hostGetDevicePointer(int deviceId, void* hostPtr) override;

  Result<bool> isDmaBuffSupported(int deviceId) override;
  Result<DmaBuff> exportDmaBuff(
      int deviceId,
      void* ptr,
      size_t len,
      DmaBufMapping mapping = DmaBufMapping::Default) override;
  Status closeDmaBuff(DmaBuff& buff) override;

 private:
  std::shared_ptr<CudaApi> cudaApi_;
  std::shared_ptr<CudaDriverApi> cudaDriverApi_;
  size_t pageSize_{0};
};

} // namespace uniflow
