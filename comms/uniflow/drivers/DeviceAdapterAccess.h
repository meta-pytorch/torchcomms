// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include "comms/uniflow/drivers/DeviceAdapter.h"
#include "comms/uniflow/registry/GpuBackendSelector.h"
#include "comms/uniflow/registry/PlatformRegistry.h"

namespace uniflow {

class GpuTopologyCache;

/// Keeps the optional vendor request unresolved until local GPU work begins.
/// Host allocation and passive pointer decoding do not claim a GPU backend.
class DeviceAdapterAccess {
 public:
  explicit DeviceAdapterAccess(
      std::optional<std::string> gpuBackend = std::nullopt);

  /// Isolated registry/selector access with a passive owned topology cache.
  DeviceAdapterAccess(
      std::optional<std::string> gpuBackend,
      PlatformRegistry& platforms,
      GpuBackendSelector& selector);

  /// Explicit compatibility/test injection, with no production selection.
  /// The supplied adapter also owns host allocations on this path.
  explicit DeviceAdapterAccess(std::shared_ptr<DeviceAdapter> injectedAdapter);

  const std::shared_ptr<DeviceAdapter>& hostAdapter() const noexcept {
    return hostAdapter_;
  }

  /// Call only when initializing resources for a local VRAM export or copy.
  /// Validates the ordinal through selected cached GPU topology before any
  /// adapter factory or staging work. Explicit injection skips that discovery.
  /// The adapter is vendor-wide; scoped operations still require deviceId.
  Result<std::shared_ptr<DeviceAdapter>> gpuAdapter(int deviceId);

  /// A retained metadata decoder, not a GPU resource or a device factory.
  Result<DevicePointerResolver> devicePointerResolver() const;

 private:
  DeviceAdapterAccess(
      std::optional<std::string> gpuBackend,
      PlatformRegistry& platforms,
      GpuBackendSelector& selector,
      GpuTopologyCache* topologyCache);

  const std::optional<std::string> gpuBackend_;
  PlatformRegistry* platforms_{nullptr};
  GpuBackendSelector* selector_{nullptr};
  std::shared_ptr<GpuTopologyCache> ownedTopologyCache_;
  GpuTopologyCache* topologyCache_{nullptr};
  std::shared_ptr<DeviceAdapter> hostAdapter_;
  std::shared_ptr<DeviceAdapter> injectedAdapter_;

  std::mutex mutex_;
  std::shared_future<Result<std::shared_ptr<DeviceAdapter>>> gpuAdapter_;
  std::thread::id initializingThread_;
};

} // namespace uniflow
