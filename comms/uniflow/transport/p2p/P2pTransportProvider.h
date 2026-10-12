// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/uniflow/transport/TransportProvider.h"

namespace uniflow {

/// Supplies the P2p interconnect tier for device endpoints.
class P2pTransportProvider : public TransportProvider {
 public:
  TransportType transportType() const noexcept override;
  Status supported() const override;
  Result<std::shared_ptr<TransportFactory>> createFactory(
      const TransportProviderContext& context) const override;
};

} // namespace uniflow
