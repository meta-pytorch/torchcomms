// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/registry/PlatformRegistry.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace uniflow {

Status PlatformRegistry::registerPlatform(PlatformRegistration registration) {
  CHECK_EXPR(validateRuntimeVersion(registration.version));
  if (registration.name.empty() || !registration.createDeviceAdapter ||
      !registration.createDiscoveryBackend) {
    return Err(ErrCode::InvalidArgument, "invalid platform registration");
  }
  auto entry =
      std::make_shared<const PlatformRegistration>(std::move(registration));
  std::lock_guard lock(mutex_);
  for (const auto& current : registrations_) {
    if (current->name == entry->name) {
      return Err(
          ErrCode::InvalidArgument,
          "duplicate platform registration: " + entry->name);
    }
  }
  registrations_.push_back(std::move(entry));
  return Ok();
}

PlatformRegistry::Registration PlatformRegistry::find(
    std::string_view name) const {
  std::lock_guard lock(mutex_);
  for (const auto& entry : registrations_) {
    if (entry->name == name) {
      return entry;
    }
  }
  return nullptr;
}

bool PlatformRegistry::contains(std::string_view name) const {
  return find(name) != nullptr;
}

std::vector<std::string> PlatformRegistry::registeredPlatforms() const {
  std::vector<std::string> names;
  {
    std::lock_guard lock(mutex_);
    for (const auto& entry : registrations_) {
      names.push_back(entry->name);
    }
  }
  std::sort(names.begin(), names.end());
  return names;
}

Result<std::shared_ptr<DeviceAdapter>> PlatformRegistry::createDeviceAdapter(
    std::string_view name) const {
  auto entry = find(name);
  if (!entry) {
    return Err(
        ErrCode::NotImplemented,
        "GPU platform '" + std::string(name) + "' is not registered");
  }
  auto result = entry->createDeviceAdapter();
  CHECK_RETURN(result);
  if (!result.value()) {
    return Err(
        ErrCode::DriverError,
        "GPU platform '" + std::string(name) + "' returned no device adapter");
  }
  return result;
}

Result<std::unique_ptr<TopologyDiscoveryBackend>>
PlatformRegistry::createDiscoveryBackend(std::string_view name) const {
  auto entry = find(name);
  if (!entry) {
    return Err(
        ErrCode::NotImplemented,
        "GPU platform '" + std::string(name) + "' is not registered");
  }
  auto result = entry->createDiscoveryBackend();
  CHECK_RETURN(result);
  if (!result.value()) {
    return Err(
        ErrCode::DriverError,
        "GPU platform '" + std::string(name) +
            "' returned no discovery backend");
  }
  return result;
}

Result<DevicePointerResolver> PlatformRegistry::devicePointerResolver(
    std::string_view name) const {
  auto entry = find(name);
  if (entry && entry->resolveDevicePointer) {
    return entry->resolveDevicePointer;
  }
  if (name == "nvidia" || name == "amd") {
    return DevicePointerResolver([](const void* ptr) noexcept {
      return reinterpret_cast<uint64_t>(ptr);
    });
  }
  return Err(
      ErrCode::NotImplemented,
      "GPU platform '" + std::string(name) +
          "' has no registered device-pointer metadata policy; register its "
          "platform extension before importing remote device memory");
}

PlatformRegistry& platformRegistry() {
  static PlatformRegistry registry;
  return registry;
}

PlatformRegistrar::PlatformRegistrar(PlatformRegistration registration) {
  auto result = platformRegistry().registerPlatform(std::move(registration));
  if (!result) {
    throw std::runtime_error(result.error().toString());
  }
}

} // namespace uniflow
