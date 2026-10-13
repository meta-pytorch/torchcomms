// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/drivers/DeviceAdapterAccess.h"

#include <exception>
#include <stdexcept>
#include <utility>

#include "comms/uniflow/drivers/HostDeviceAdapter.h"
#include "comms/uniflow/drivers/topology/GpuTopologyCache.h"

namespace uniflow {

DeviceAdapterAccess::DeviceAdapterAccess(std::optional<std::string> gpuBackend)
    : DeviceAdapterAccess(
          std::move(gpuBackend),
          platformRegistry(),
          gpuBackendSelector(),
          &gpuTopologyCache()) {}

DeviceAdapterAccess::DeviceAdapterAccess(
    std::optional<std::string> gpuBackend,
    PlatformRegistry& platforms,
    GpuBackendSelector& selector)
    : DeviceAdapterAccess(std::move(gpuBackend), platforms, selector, nullptr) {
}

DeviceAdapterAccess::DeviceAdapterAccess(
    std::optional<std::string> gpuBackend,
    PlatformRegistry& platforms,
    GpuBackendSelector& selector,
    GpuTopologyCache* topologyCache)
    : gpuBackend_(std::move(gpuBackend)),
      platforms_(&platforms),
      selector_(&selector),
      topologyCache_(topologyCache),
      hostAdapter_(createHostDeviceAdapter()) {
  validateGpuBackendRequest(gpuBackend_);
  if (!topologyCache_) {
    ownedTopologyCache_ =
        std::make_shared<GpuTopologyCache>(platforms, selector);
    topologyCache_ = ownedTopologyCache_.get();
  }
}

DeviceAdapterAccess::DeviceAdapterAccess(
    std::shared_ptr<DeviceAdapter> injectedAdapter)
    : hostAdapter_(injectedAdapter),
      injectedAdapter_(std::move(injectedAdapter)) {
  if (!injectedAdapter_) {
    throw std::invalid_argument("injected device adapter must not be null");
  }
}

Result<std::shared_ptr<DeviceAdapter>> DeviceAdapterAccess::gpuAdapter(
    int deviceId) {
  if (deviceId < 0) {
    return Err(
        ErrCode::InvalidArgument, "GPU adapter requires a local device ID");
  }
  if (injectedAdapter_) {
    return injectedAdapter_;
  }
  auto topology = topologyCache_->get(gpuBackend_);
  CHECK_RETURN(topology);
  const auto deviceCount = topology.value()->gpuCount();
  if (static_cast<size_t>(deviceId) >= deviceCount) {
    return Err(
        ErrCode::InvalidArgument,
        "GPU device ID " + std::to_string(deviceId) +
            " out of range (numDevices=" + std::to_string(deviceCount) + ")");
  }
  auto backend = selector_->resolve(gpuBackend_);
  CHECK_RETURN(backend);
  std::shared_future<Result<std::shared_ptr<DeviceAdapter>>> future;
  std::shared_ptr<std::promise<Result<std::shared_ptr<DeviceAdapter>>>> promise;
  {
    std::lock_guard lock(mutex_);
    if (gpuAdapter_.valid()) {
      if (initializingThread_ == std::this_thread::get_id()) {
        return Err(ErrCode::DriverError, "recursive GPU adapter creation");
      }
      future = gpuAdapter_;
    } else {
      promise = std::make_shared<
          std::promise<Result<std::shared_ptr<DeviceAdapter>>>>();
      future = promise->get_future().share();
      gpuAdapter_ = future;
      initializingThread_ = std::this_thread::get_id();
    }
  }
  if (promise) {
    try {
      auto result = platforms_->createDeviceAdapter(backend.value());
      const bool succeeded = result.hasValue();
      promise->set_value(std::move(result));
      std::lock_guard lock(mutex_);
      initializingThread_ = {};
      if (!succeeded) {
        gpuAdapter_ = {};
      }
    } catch (...) {
      promise->set_exception(std::current_exception());
      std::lock_guard lock(mutex_);
      initializingThread_ = {};
      gpuAdapter_ = {};
    }
  }
  return future.get();
}

Result<DevicePointerResolver> DeviceAdapterAccess::devicePointerResolver()
    const {
  if (injectedAdapter_) {
    return DevicePointerResolver(
        [adapter = injectedAdapter_](const void* ptr) noexcept {
          return adapter->resolveDevicePointer(ptr);
        });
  }
  auto backend = selector_->resolve(gpuBackend_);
  CHECK_RETURN(backend);
  return platforms_->devicePointerResolver(backend.value());
}

} // namespace uniflow
