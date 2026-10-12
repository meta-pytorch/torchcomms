// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/transport/nvlink/NVLinkTransportProvider.h"

#include "comms/uniflow/transport/nvlink/NVLinkTransport.h"

namespace uniflow {

TransportType NVLinkTransportProvider::transportType() const noexcept {
  return TransportType::NVLink;
}

Status NVLinkTransportProvider::supported() const {
  return NVLinkTransportFactory::supported();
}

Result<std::shared_ptr<TransportFactory>>
NVLinkTransportProvider::createFactory(
    const TransportProviderContext& context) const {
  if (context.deviceId < 0) {
    return std::shared_ptr<TransportFactory>{};
  }
  if (!context.eventBase) {
    return Err(
        ErrCode::InvalidArgument, "NVLink provider requires an EventBase");
  }
  return std::make_shared<NVLinkTransportFactory>(
      context.deviceId, context.eventBase);
}

} // namespace uniflow
