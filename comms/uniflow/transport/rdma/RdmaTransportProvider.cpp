// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/transport/rdma/RdmaTransportProvider.h"

#include <utility>

namespace uniflow {

RdmaTransportProvider::RdmaTransportProvider(
    std::vector<std::string> deviceNames,
    RdmaTransportConfig config,
    std::shared_ptr<IbvApi> ibvApi,
    std::shared_ptr<CudaDriverApi> cudaDriverApi,
    std::shared_ptr<CudaApi> cudaApi,
    std::optional<uint8_t> portNum,
    std::shared_ptr<DeviceAdapter> deviceAdapter)
    : deviceNames_(std::move(deviceNames)),
      config_(config),
      ibvApi_(std::move(ibvApi)),
      cudaDriverApi_(std::move(cudaDriverApi)),
      cudaApi_(std::move(cudaApi)),
      portNum_(portNum),
      deviceAdapter_(std::move(deviceAdapter)) {}

TransportType RdmaTransportProvider::transportType() const noexcept {
  return TransportType::RDMA;
}

Status RdmaTransportProvider::supported() const {
  return RdmaTransportFactory::supported(ibvApi_);
}

Result<std::shared_ptr<TransportFactory>> RdmaTransportProvider::createFactory(
    const TransportProviderContext& context) const {
  if (deviceNames_.empty()) {
    return std::shared_ptr<TransportFactory>{};
  }
  if (!context.eventBase) {
    return Err(ErrCode::InvalidArgument, "RDMA provider requires an EventBase");
  }

  auto ibvApi = ibvApi_ ? ibvApi_ : std::make_shared<IbvApi>();
  auto cudaDriverApi =
      cudaDriverApi_ ? cudaDriverApi_ : std::make_shared<CudaDriverApi>();
  auto cudaApi = cudaApi_ ? cudaApi_ : std::make_shared<CudaApi>();
  auto deviceAdapter = deviceAdapter_
      ? deviceAdapter_
      : createDeviceAdapter(cudaApi, cudaDriverApi);

  return std::make_shared<RdmaTransportFactory>(
      deviceNames_,
      context.eventBase,
      config_,
      std::move(ibvApi),
      std::move(cudaDriverApi),
      std::move(cudaApi),
      portNum_,
      std::move(deviceAdapter));
}

} // namespace uniflow
