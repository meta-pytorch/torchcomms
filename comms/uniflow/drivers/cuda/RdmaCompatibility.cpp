// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <utility>

#include "comms/uniflow/drivers/DeviceAdapter.h"
#include "comms/uniflow/drivers/DeviceAdapterAccess.h"
#include "comms/uniflow/drivers/TopologyDiscovery.h"
#include "comms/uniflow/drivers/cuda/CudaApi.h"
#include "comms/uniflow/drivers/cuda/CudaDriverApi.h"
#include "comms/uniflow/transport/rdma/RdmaSlabPool.h"
#include "comms/uniflow/transport/rdma/RdmaTransport.h"
#include "comms/uniflow/transport/rdma/RdmaTransportProvider.h"

namespace uniflow {
namespace {

std::shared_ptr<DeviceAdapterAccess> legacyDeviceAdapters(
    std::shared_ptr<CudaApi> cudaApi,
    std::shared_ptr<CudaDriverApi> cudaDriverApi,
    std::shared_ptr<DeviceAdapter> deviceAdapter) {
  if (deviceAdapter) {
    return std::make_shared<DeviceAdapterAccess>(std::move(deviceAdapter));
  }
  if (cudaApi || cudaDriverApi) {
    return std::make_shared<DeviceAdapterAccess>(
        createDeviceAdapter(std::move(cudaApi), std::move(cudaDriverApi)));
  }
  ensureDefaultPlatformRegistration();
  return nullptr;
}

std::shared_ptr<DeviceAdapter> legacyHostAdapter(
    std::shared_ptr<CudaApi> cudaApi,
    std::shared_ptr<DeviceAdapter> deviceAdapter) {
  auto access = legacyDeviceAdapters(
      std::move(cudaApi), nullptr, std::move(deviceAdapter));
  if (!access) {
    access = std::make_shared<DeviceAdapterAccess>();
  }
  return access->hostAdapter();
}

} // namespace

RdmaTransport::RdmaTransport(
    std::shared_ptr<IbvApi> ibvApi,
    std::shared_ptr<CudaApi> cudaApi,
    std::shared_ptr<CudaDriverApi> cudaDriverApi,
    EventBase* evb,
    std::shared_ptr<std::vector<NicResources>> nics,
    uint64_t domainId,
    RdmaTransportConfig config,
    std::shared_ptr<RdmaSlabPool> slabPool,
    std::shared_ptr<DeviceAdapter> deviceAdapter)
    : RdmaTransport(
          legacyDeviceAdapters(
              std::move(cudaApi),
              std::move(cudaDriverApi),
              std::move(deviceAdapter)),
          std::move(ibvApi),
          evb,
          std::move(nics),
          domainId,
          config,
          std::move(slabPool)) {}

RdmaTransportFactory::RdmaTransportFactory(
    const std::vector<std::string>& deviceNames,
    EventBase* evb,
    RdmaTransportConfig config,
    std::shared_ptr<IbvApi> ibvApi,
    std::shared_ptr<CudaDriverApi> cudaDriverApi,
    std::shared_ptr<CudaApi> cudaApi,
    std::optional<uint8_t> portNum,
    std::shared_ptr<DeviceAdapter> deviceAdapter)
    : RdmaTransportFactory(
          legacyDeviceAdapters(
              std::move(cudaApi),
              std::move(cudaDriverApi),
              std::move(deviceAdapter)),
          deviceNames,
          evb,
          config,
          std::move(ibvApi),
          portNum) {}

RdmaSlabPool::RdmaSlabPool(
    const RdmaSlabPoolConfig& config,
    std::shared_ptr<CudaApi> cudaApi,
    std::shared_ptr<IbvApi> ibvApi,
    std::shared_ptr<std::vector<NicResources>> nics,
    std::shared_ptr<DeviceAdapter> deviceAdapter)
    : RdmaSlabPool(
          legacyHostAdapter(std::move(cudaApi), std::move(deviceAdapter)),
          config,
          std::move(ibvApi),
          std::move(nics)) {}

RdmaTransportProvider::RdmaTransportProvider(
    std::vector<std::string> deviceNames,
    RdmaTransportConfig config,
    std::shared_ptr<IbvApi> ibvApi,
    std::shared_ptr<CudaDriverApi> cudaDriverApi,
    std::shared_ptr<CudaApi> cudaApi,
    std::optional<uint8_t> portNum,
    std::shared_ptr<DeviceAdapter> deviceAdapter)
    : RdmaTransportProvider(
          legacyDeviceAdapters(
              std::move(cudaApi),
              std::move(cudaDriverApi),
              std::move(deviceAdapter)),
          std::move(deviceNames),
          config,
          std::move(ibvApi),
          portNum) {}

} // namespace uniflow
