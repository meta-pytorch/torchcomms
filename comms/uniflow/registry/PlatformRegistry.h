// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "comms/uniflow/drivers/DeviceAdapter.h"
#include "comms/uniflow/drivers/TopologyDiscovery.h"
#include "comms/uniflow/registry/RuntimeVersion.h"

namespace uniflow {

struct PlatformRegistration {
  /// Canonical vendor key, for example nvidia, amd or mtia.
  std::string name;
  std::function<Result<std::shared_ptr<DeviceAdapter>>()> createDeviceAdapter;
  std::function<Result<std::unique_ptr<TopologyDiscoveryBackend>>()>
      createDiscoveryBackend;
  RuntimeVersion version;
};

/// Stores platform factories without selecting a vendor or probing devices.
/// Calling a factory is explicit and never holds the registry lock.
class PlatformRegistry {
 public:
  Status registerPlatform(PlatformRegistration registration);

  bool contains(std::string_view name) const;
  std::vector<std::string> registeredPlatforms() const;

  Result<std::shared_ptr<DeviceAdapter>> createDeviceAdapter(
      std::string_view name) const;
  Result<std::unique_ptr<TopologyDiscoveryBackend>> createDiscoveryBackend(
      std::string_view name) const;

 private:
  using Registration = std::shared_ptr<const PlatformRegistration>;
  Registration find(std::string_view name) const;

  mutable std::mutex mutex_;
  std::vector<Registration> registrations_;
};

PlatformRegistry& platformRegistry();

/// Checked static registration. Explicit loaders should use registerPlatform()
/// to return a compatibility or duplicate error to their caller.
class PlatformRegistrar {
 public:
  explicit PlatformRegistrar(PlatformRegistration registration);
};

} // namespace uniflow

#define UNIFLOW_REGISTER_PLATFORM(name, ...)                             \
  namespace {                                                            \
  [[maybe_unused]] const ::uniflow::PlatformRegistrar name{__VA_ARGS__}; \
  }
