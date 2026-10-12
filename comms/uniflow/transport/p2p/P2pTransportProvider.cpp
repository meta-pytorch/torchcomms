// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/transport/p2p/P2pTransportProvider.h"

#include "comms/uniflow/transport/p2p/P2pTransport.h"

namespace uniflow {

TransportType P2pTransportProvider::transportType() const noexcept {
  return TransportType::NVLink;
}

Status P2pTransportProvider::supported() const {
  return P2pTransportFactory::supported();
}

Result<std::shared_ptr<TransportFactory>> P2pTransportProvider::createFactory(
    const TransportProviderContext& context) const {
  if (context.deviceId < 0) {
    return std::shared_ptr<TransportFactory>{};
  }
  if (!context.eventBase) {
    return Err(ErrCode::InvalidArgument, "P2p provider requires an EventBase");
  }
  return std::make_shared<P2pTransportFactory>(
      context.deviceId, context.eventBase);
}

} // namespace uniflow
