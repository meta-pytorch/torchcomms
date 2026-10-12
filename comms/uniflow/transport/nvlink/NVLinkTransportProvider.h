// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/uniflow/transport/TransportProvider.h"

namespace uniflow {

/// Supplies the NVLink interconnect tier for device endpoints.
class NVLinkTransportProvider : public TransportProvider {
 public:
  TransportType transportType() const noexcept override;
  Status supported() const override;
  Result<std::shared_ptr<TransportFactory>> createFactory(
      const TransportProviderContext& context) const override;
};

} // namespace uniflow
