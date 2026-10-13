// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include "comms/uniflow/registry/GpuBackendSelector.h"
#include "comms/uniflow/registry/PlatformRegistry.h"
#include "comms/uniflow/transport/Topology.h"

namespace uniflow {

/// Validate registered capability before a GPU initialization claim. Consume
/// the returned vendor, which may have been selected concurrently.
Result<std::string> claimGpuPlatform(
    PlatformRegistry& platforms,
    GpuBackendSelector& selector,
    const std::optional<std::string>& requested);

/// Instance-testable selected GPU discovery, separate from the NIC-only graph.
/// Discovery and platform callbacks run outside the cache mutex.
class GpuTopologyCache {
 public:
  explicit GpuTopologyCache(
      PlatformRegistry& platforms = platformRegistry(),
      GpuBackendSelector& selector = gpuBackendSelector());

  Result<std::shared_ptr<Topology>> get(
      const std::optional<std::string>& requested = std::nullopt);

 private:
  Result<std::shared_ptr<Topology>> discover(const std::string& backend);

  PlatformRegistry& platforms_;
  GpuBackendSelector& selector_;
  std::mutex mutex_;
  std::shared_future<Result<std::shared_ptr<Topology>>> topology_;
  std::thread::id initializingThread_;
};

GpuTopologyCache& gpuTopologyCache();

Result<std::shared_ptr<Topology>> sharedGpuTopology(
    const std::optional<std::string>& requested = std::nullopt);

} // namespace uniflow
