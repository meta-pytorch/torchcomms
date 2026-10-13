// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/transport/rdma/RdmaTransportProvider.h"

#include <optional>
#include <string>
#include <utility>

#include "comms/uniflow/MultiTransportOptions.h"
#include "comms/uniflow/drivers/DeviceAdapterAccess.h"

namespace uniflow {

RdmaTransportProvider::RdmaTransportProvider(
    std::shared_ptr<DeviceAdapterAccess> deviceAdapters,
    std::vector<std::string> deviceNames,
    RdmaTransportConfig config,
    std::shared_ptr<IbvApi> ibvApi,
    std::optional<uint8_t> portNum,
    std::shared_ptr<const Topology> topology)
    : deviceAdapters_(std::move(deviceAdapters)),
      deviceNames_(std::move(deviceNames)),
      config_(config),
      ibvApi_(std::move(ibvApi)),
      portNum_(portNum),
      topology_(std::move(topology)) {}

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

  auto deviceAdapters = deviceAdapters_;
  if (!deviceAdapters) {
    std::optional<std::string> gpuBackend;
    if (context.options) {
      gpuBackend = context.options->gpuBackend;
    }
    deviceAdapters =
        std::make_shared<DeviceAdapterAccess>(std::move(gpuBackend));
  }
  return std::make_shared<RdmaTransportFactory>(
      std::move(deviceAdapters),
      deviceNames_,
      context.eventBase,
      config_,
      ibvApi_,
      portNum_,
      context.deviceId,
      topology_ ? topology_.get() : context.topology);
}

} // namespace uniflow
