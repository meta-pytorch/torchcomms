// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/registry/TransportProviderRegistry.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace uniflow {

Status TransportProviderRegistry::registerProvider(
    TransportProviderRegistration registration) {
  CHECK_EXPR(validateRuntimeVersion(registration.version));
  if (registration.name.empty() || !registration.provider ||
      registration.type >= TransportType::NumTransportType) {
    return Err(ErrCode::InvalidArgument, "invalid transport registration");
  }
  auto entry = std::make_shared<const TransportProviderRegistration>(
      std::move(registration));
  std::lock_guard lock(mutex_);
  for (const auto& current : registrations_) {
    if (current->name == entry->name) {
      return Err(
          ErrCode::InvalidArgument,
          "duplicate transport registration: " + entry->name);
    }
    const bool overlapping = current->platform.empty() ||
        entry->platform.empty() || current->platform == entry->platform;
    if (current->type == entry->type && overlapping) {
      return Err(
          ErrCode::InvalidArgument,
          "overlapping transport providers: " + current->name + " and " +
              entry->name);
    }
  }
  registrations_.push_back(std::move(entry));
  return Ok();
}

std::vector<TransportProviderRegistry::Registration>
TransportProviderRegistry::snapshot(std::string_view platform) const {
  std::vector<Registration> entries;
  {
    std::lock_guard lock(mutex_);
    for (const auto& entry : registrations_) {
      if (entry->platform.empty() || entry->platform == platform) {
        entries.push_back(entry);
      }
    }
  }
  std::sort(
      entries.begin(), entries.end(), [](const auto& lhs, const auto& rhs) {
        return lhs->type < rhs->type;
      });
  return entries;
}

std::vector<TransportType> TransportProviderRegistry::registeredTypes(
    std::string_view platform) const {
  std::vector<TransportType> types;
  for (const auto& entry : snapshot(platform)) {
    types.push_back(entry->type);
  }
  return types;
}

Status TransportProviderRegistry::supported(
    TransportType type,
    std::string_view platform) const {
  if (type >= TransportType::NumTransportType) {
    return Err(ErrCode::InvalidArgument, "invalid transport type");
  }
  for (const auto& entry : snapshot(platform)) {
    if (entry->type != type) {
      continue;
    }
    if (entry->provider->transportType() != entry->type) {
      return Err(
          ErrCode::InvalidArgument,
          "transport provider ID does not match registration: " + entry->name);
    }
    return entry->provider->supported();
  }
  return Err(
      ErrCode::NotImplemented,
      std::string(toStringView(type)) +
          " transport is not linked (no registered provider)");
}

Result<std::vector<std::shared_ptr<TransportFactory>>>
TransportProviderRegistry::createFactories(
    const TransportProviderContext& context,
    std::string_view platform) const {
  if (context.deviceId < -1) {
    return Err(ErrCode::InvalidArgument, "deviceId must be -1 or non-negative");
  }
  std::vector<std::shared_ptr<TransportFactory>> factories;
  for (const auto& entry : snapshot(platform)) {
    if (entry->provider->transportType() != entry->type) {
      return Err(
          ErrCode::InvalidArgument,
          "transport provider ID does not match registration: " + entry->name);
    }
    const auto support = entry->provider->supported();
    if (!support) {
      const auto code = support.error().code();
      if (code == ErrCode::NotImplemented ||
          code == ErrCode::ResourceExhausted) {
        continue;
      }
      return support.error();
    }
    auto result = entry->provider->createFactory(context);
    CHECK_RETURN(result);
    auto factory = std::move(result).value();
    if (!factory) {
      continue;
    }
    if (factory->transportType() != entry->type) {
      return Err(
          ErrCode::InvalidArgument,
          "transport factory ID does not match registration: " + entry->name);
    }
    factories.push_back(std::move(factory));
  }
  return factories;
}

TransportProviderRegistry& transportProviderRegistry() {
  static TransportProviderRegistry registry;
  return registry;
}

TransportProviderRegistrar::TransportProviderRegistrar(
    TransportProviderRegistration registration) {
  auto result =
      transportProviderRegistry().registerProvider(std::move(registration));
  if (!result) {
    throw std::runtime_error(result.error().toString());
  }
}

} // namespace uniflow
