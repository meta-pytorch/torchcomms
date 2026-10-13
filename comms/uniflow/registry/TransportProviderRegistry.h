// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "comms/uniflow/registry/RuntimeVersion.h"
#include "comms/uniflow/transport/TransportProvider.h"

namespace uniflow {

struct TransportProviderRegistration {
  std::string name;
  TransportType type{TransportType::NumTransportType};
  /// Empty means vendor-neutral; otherwise a canonical platform key.
  std::string platform;
  std::shared_ptr<const TransportProvider> provider;
  RuntimeVersion version;
};

/// Registration is passive. Capability and creation callbacks run without
/// registry locks against a snapshot taken at the start of each query.
/// Each transport type has at most one provider for an applicable platform.
class TransportProviderRegistry {
 public:
  Status registerProvider(TransportProviderRegistration registration);

  /// Generic registrations plus those matching platform, without probing.
  std::vector<TransportType> registeredTypes(
      std::string_view platform = {}) const;

  Status supported(TransportType type, std::string_view platform = {}) const;

  /// Applicable factories in numeric wire-ID order.
  /// A null factory is inapplicable. A creation error is returned unchanged.
  Result<std::vector<std::shared_ptr<TransportFactory>>> createFactories(
      const TransportProviderContext& context,
      std::string_view platform = {}) const;

 private:
  using Registration = std::shared_ptr<const TransportProviderRegistration>;
  std::vector<Registration> snapshot(std::string_view platform) const;

  mutable std::mutex mutex_;
  std::vector<Registration> registrations_;
};

/// Defined once in the runtime, not in each native component.
TransportProviderRegistry& transportProviderRegistry();

/// Checked static registration. Explicit loaders should use registerProvider()
/// to return a compatibility or duplicate error to their caller.
class TransportProviderRegistrar {
 public:
  explicit TransportProviderRegistrar(
      TransportProviderRegistration registration);
};

} // namespace uniflow

#define UNIFLOW_REGISTER_TRANSPORT_PROVIDER(name, ...)               \
  namespace {                                                        \
  [[maybe_unused]] const ::uniflow::TransportProviderRegistrar name{ \
      __VA_ARGS__};                                                  \
  }
