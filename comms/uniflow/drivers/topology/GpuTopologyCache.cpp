// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/drivers/topology/GpuTopologyCache.h"

#include <exception>
#include <utility>

#include "comms/uniflow/logging/Logger.h"

namespace uniflow {

Result<std::string> claimGpuPlatform(
    PlatformRegistry& platforms,
    GpuBackendSelector& selector,
    const std::optional<std::string>& requested) {
  auto candidate = selector.resolve(requested);
  CHECK_RETURN(candidate);
  if (!platforms.contains(candidate.value())) {
    return Err(
        ErrCode::NotImplemented,
        "GPU platform '" + candidate.value() +
            "' is not registered; register its platform extension before GPU use");
  }
  return selector.claim(candidate.value(), requested);
}

GpuTopologyCache::GpuTopologyCache(
    PlatformRegistry& platforms,
    GpuBackendSelector& selector)
    : platforms_(platforms), selector_(selector) {}

Result<std::shared_ptr<Topology>> GpuTopologyCache::discover(
    const std::string& backend) {
  auto discovery = platforms_.createDiscoveryBackend(backend);
  CHECK_RETURN(discovery);
  auto topology = std::make_shared<Topology>();
  auto status = discovery.value()->discover(*topology);
  if (status.hasError()) {
    UNIFLOW_LOG_ERROR(
        "Topology discovery failed: {}", status.error().toString());
  }
  return topology;
}

Result<std::shared_ptr<Topology>> GpuTopologyCache::get(
    const std::optional<std::string>& requested) {
  auto backend = claimGpuPlatform(platforms_, selector_, requested);
  CHECK_RETURN(backend);
  std::shared_future<Result<std::shared_ptr<Topology>>> future;
  std::shared_ptr<std::promise<Result<std::shared_ptr<Topology>>>> promise;
  {
    std::lock_guard lock(mutex_);
    if (topology_.valid()) {
      if (initializingThread_ == std::this_thread::get_id()) {
        return Err(ErrCode::DriverError, "recursive GPU topology discovery");
      }
      future = topology_;
    } else {
      promise =
          std::make_shared<std::promise<Result<std::shared_ptr<Topology>>>>();
      future = promise->get_future().share();
      topology_ = future;
      initializingThread_ = std::this_thread::get_id();
    }
  }
  if (promise) {
    try {
      auto result = discover(backend.value());
      const bool succeeded = result.hasValue();
      promise->set_value(std::move(result));
      std::lock_guard lock(mutex_);
      initializingThread_ = {};
      if (!succeeded) {
        topology_ = {};
      }
    } catch (...) {
      promise->set_exception(std::current_exception());
      std::lock_guard lock(mutex_);
      initializingThread_ = {};
      topology_ = {};
    }
  }
  return future.get();
}

GpuTopologyCache& gpuTopologyCache() {
  static GpuTopologyCache cache;
  return cache;
}

Result<std::shared_ptr<Topology>> sharedGpuTopology(
    const std::optional<std::string>& requested) {
  return gpuTopologyCache().get(requested);
}

} // namespace uniflow
