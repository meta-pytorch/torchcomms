// Copyright (c) Meta Platforms, Inc. and affiliates.
// NCCL GIN Device Backend - Static Method Implementations

#include "comms/torchcomms/TorchCommLogging.hpp"
#include "comms/torchcomms/device/DeviceBackendTraits.hpp"
#include "comms/torchcomms/device/TorchCommDeviceComm.hpp"
#include "comms/torchcomms/ncclx/NcclxApi.hpp"

#include <memory>
#include <stdexcept>
#include <string>

namespace torchcomms::device {

std::unique_ptr<TorchCommDeviceWindow<NCCLGinBackend>>
NCCLGinBackend::create_device_window(
    ncclComm_t nccl_comm,
    torch::comms::NcclxApi* nccl_api,
    const DeviceBackendConfig& config,
    Window host_window,
    void* base,
    size_t size) {
  if (nccl_comm == nullptr) {
    throw std::runtime_error(
        "[NCCLGinBackend::create_device_window]: NCCL communicator cannot be null");
  }
  if (nccl_api == nullptr) {
    throw std::runtime_error(
        "[NCCLGinBackend::create_device_window]: NCCL API cannot be null");
  }
  if (base == nullptr && size > 0) {
    throw std::runtime_error(
        "[NCCLGinBackend::create_device_window]: Window base cannot be null with non-zero size");
  }

  // Set up ncclDevCommRequirements with GIN enabled using designated
  // initializers
  ncclDevCommRequirements reqs = {
      .resourceRequirementsList = nullptr,
      .teamRequirementsList = nullptr,
      .lsaMultimem = false,
      .barrierCount = config.barrier_count,
      .lsaBarrierCount = 0,
      .railGinBarrierCount = config.barrier_count,
      .lsaLLA2ABlockCount = 0,
      .lsaLLA2ASlotCount = 0,
      .ginForceEnable = true,
      .ginContextCount = 1,
      .ginSignalCount = config.signal_count,
      .ginCounterCount = config.counter_count,
  };

  // Create NCCL device communicator with GIN state
  ncclDevComm nccl_dev_comm{};
  auto result = nccl_api->devCommCreate(nccl_comm, &reqs, &nccl_dev_comm);
  if (result != ncclSuccess) {
    throw std::runtime_error(
        "[NCCLGinBackend::create_device_window]: Failed to create NCCL device communicator. "
        "Error: " +
        std::string(nccl_api->getErrorString(result)));
  }

  // Construct and return the device window via unique_ptr using constructor
  return std::make_unique<TorchCommDeviceWindow<NCCLGinBackend>>(
      nccl_dev_comm,
      host_window,
      base,
      size,
      config.comm_rank,
      config.comm_size);
}

void NCCLGinBackend::destroy_device_window(
    ncclComm_t nccl_comm,
    torch::comms::NcclxApi* nccl_api,
    std::unique_ptr<TorchCommDeviceWindow<NCCLGinBackend>> device_window) {
  if (nccl_comm == nullptr || nccl_api == nullptr || !device_window) {
    return;
  }

  auto result = nccl_api->devCommDestroy(nccl_comm, &device_window->comm_);
  if (result != ncclSuccess) {
    TC_LOG(ERROR) << "Failed to destroy NCCL device communicator: "
                  << nccl_api->getErrorString(result);
  }
  // unique_ptr automatically destroys the device_window when it goes out of
  // scope
}

} // namespace torchcomms::device
