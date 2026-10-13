// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/uniflow/transport/TransportProvider.h"
#include "comms/uniflow/transport/rdma/RdmaTransport.h"

namespace uniflow {

/// Supplies RDMA factories with shared host/GPU access and NIC settings.
class RdmaTransportProvider : public TransportProvider {
 public:
  RdmaTransportProvider(
      std::shared_ptr<DeviceAdapterAccess> deviceAdapters,
      std::vector<std::string> deviceNames,
      RdmaTransportConfig config = {},
      std::shared_ptr<IbvApi> ibvApi = nullptr,
      std::optional<uint8_t> portNum = std::nullopt,
      std::shared_ptr<const Topology> topology = nullptr);

  /// Legacy CUDA-shaped constructor, supplied by NVIDIA compatibility code.
  explicit RdmaTransportProvider(
      std::vector<std::string> deviceNames,
      RdmaTransportConfig config = {},
      std::shared_ptr<IbvApi> ibvApi = nullptr,
      std::shared_ptr<CudaDriverApi> cudaDriverApi = nullptr,
      std::shared_ptr<CudaApi> cudaApi = nullptr,
      std::optional<uint8_t> portNum = std::nullopt,
      std::shared_ptr<DeviceAdapter> deviceAdapter = nullptr);

  TransportType transportType() const noexcept override;
  Status supported() const override;
  Result<std::shared_ptr<TransportFactory>> createFactory(
      const TransportProviderContext& context) const override;

 private:
  std::shared_ptr<DeviceAdapterAccess> deviceAdapters_;
  std::vector<std::string> deviceNames_;
  RdmaTransportConfig config_;
  std::shared_ptr<IbvApi> ibvApi_;
  std::optional<uint8_t> portNum_;
  std::shared_ptr<const Topology> topology_;
};

} // namespace uniflow
