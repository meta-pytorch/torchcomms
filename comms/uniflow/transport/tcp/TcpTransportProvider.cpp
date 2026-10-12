// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/transport/tcp/TcpTransportProvider.h"

#include <utility>

namespace uniflow {

TcpTransportProvider::TcpTransportProvider(
    TcpTransportConfig config,
    std::string host,
    std::shared_ptr<CudaApi> cudaApi)
    : config_(std::move(config)),
      host_(std::move(host)),
      cudaApi_(std::move(cudaApi)) {}

TransportType TcpTransportProvider::transportType() const noexcept {
  return TransportType::TCP;
}

Status TcpTransportProvider::supported() const {
  return TcpTransportFactory::supported();
}

Result<std::shared_ptr<TransportFactory>> TcpTransportProvider::createFactory(
    const TransportProviderContext& context) const {
  if (!context.eventBase) {
    return Err(ErrCode::InvalidArgument, "TCP provider requires an EventBase");
  }
  return std::make_shared<TcpTransportFactory>(
      context.deviceId, context.eventBase, config_, host_, cudaApi_);
}

} // namespace uniflow
