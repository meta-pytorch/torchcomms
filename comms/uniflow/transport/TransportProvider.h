// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>
#include <vector>

#include "comms/uniflow/transport/Transport.h"

namespace uniflow {

class EventBase;

struct TransportProviderContext {
  int deviceId{-1};
  EventBase* eventBase{nullptr};
};

/// Creates transport factories without selecting platform defaults.
class TransportProvider {
 public:
  virtual ~TransportProvider() = default;

  /// Transport identity must remain constant for the provider's lifetime.
  virtual TransportType transportType() const noexcept = 0;

  /// Capability query, separate from endpoint-specific factory creation.
  /// Used by default backend composition for capability discovery.
  /// Explicit-provider construction does not call this query.
  virtual Status supported() const = 0;

  /// A successful null factory means this provider does not apply here.
  /// Returned factories own their backend dependencies and may borrow
  /// context.eventBase, whose lifetime is shared with live transports.
  virtual Result<std::shared_ptr<TransportFactory>> createFactory(
      const TransportProviderContext& context) const = 0;
};

using TransportProviderList =
    std::vector<std::shared_ptr<const TransportProvider>>;

} // namespace uniflow
