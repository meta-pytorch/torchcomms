// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/uniflow/drivers/DeviceAdapter.h"

namespace uniflow {

/// Page-aligned host storage. NIC registration pins it; it is not GPU-mapped.
class HostDeviceAdapter final : public DeviceAdapter {
 public:
  using DeviceAdapter::pinnedHostAlloc;
  using DeviceAdapter::pinnedHostFree;

  Result<void*> pinnedHostAlloc(size_t size) override;
  Status pinnedHostFree(void* ptr) override;

  /// Host identity is for NIC bookkeeping, not accelerator device access.
  Result<void*> hostGetDevicePointer(void* hostPtr) override;
  Result<void*> hostGetDevicePointer(int deviceId, void* hostPtr) override;

  Result<bool> isDmaBuffSupported(int deviceId) override;
  Result<DmaBuff> exportDmaBuff(
      int deviceId,
      void* ptr,
      size_t len,
      DmaBufMapping mapping = DmaBufMapping::Default) override;
  Status closeDmaBuff(DmaBuff& buff) override;
};

std::shared_ptr<DeviceAdapter> createHostDeviceAdapter();

} // namespace uniflow
