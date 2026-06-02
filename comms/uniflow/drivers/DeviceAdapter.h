// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>
#include <memory>

#include "comms/uniflow/Result.h"

namespace uniflow {

class CudaApi;

/// DeviceAdapter abstracts device-specific host pinned memory allocation
/// for use with DMA from a NIC or accelerator.
class DeviceAdapter {
 public:
  virtual ~DeviceAdapter() = default;

  /// Allocate page-aligned, host-pinned memory suitable for DMA from a NIC
  /// or accelerator.
  virtual Result<void*> pinnedHostAlloc(size_t size) = 0;

  /// Release memory previously returned by pinnedHostAlloc().
  virtual Status pinnedHostFree(void* ptr) = 0;

  /// Return a device-accessible pointer for `hostPtr` (which must have been
  /// obtained from pinnedHostAlloc()).
  virtual Result<void*> hostGetDevicePointer(void* hostPtr) = 0;
};

/// Returns a DeviceAdapter for the current build configuration.
///
/// The CUDA implementation accepts an optional CudaApi for dependency
/// injection in tests; if null, a default CudaApi is constructed. The
/// parameter is ignored by non-CUDA implementations.
std::shared_ptr<DeviceAdapter> createDeviceAdapter(
    std::shared_ptr<CudaApi> cudaApi = nullptr);

} // namespace uniflow
