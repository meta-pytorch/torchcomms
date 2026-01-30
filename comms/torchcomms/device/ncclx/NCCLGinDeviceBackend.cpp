// Copyright (c) Meta Platforms, Inc. and affiliates.
// NCCL GIN Device Backend - Static Method Implementations

#include "comms/torchcomms/TorchCommLogging.hpp"
#include "comms/torchcomms/device/DeviceBackendTraits.hpp"
#include "comms/torchcomms/device/TorchCommDeviceComm.hpp"
#include "comms/torchcomms/ncclx/NcclxApi.hpp"

#include <stdexcept>
#include <string>

namespace torchcomms::device {

TorchCommDeviceWindow<NCCLGinBackend> NCCLGinBackend::create_device_window(
    ncclComm_t nccl_comm,
    torch::comms::NcclxApi* nccl_api,
    const DeviceBackendConfig& config,
    Window orig_window,
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

  // Set up ncclDevCommRequirements with GIN enabled
  ncclDevCommRequirements reqs = {};
  reqs.resourceRequirementsList = nullptr;
  reqs.teamRequirementsList = nullptr;
  reqs.lsaMultimem = false;
  reqs.barrierCount = config.barrier_count;
  reqs.lsaBarrierCount = 0;
  reqs.railGinBarrierCount = config.barrier_count;
  reqs.lsaLLA2ABlockCount = 0;
  reqs.lsaLLA2ASlotCount = 0;
  reqs.ginForceEnable = true;
  reqs.ginContextCount = 1;
  reqs.ginSignalCount = config.signal_count;
  reqs.ginCounterCount = config.counter_count;

  // Create NCCL device communicator with GIN state
  ncclDevComm nccl_dev_comm{};
  auto result = nccl_api->devCommCreate(nccl_comm, &reqs, &nccl_dev_comm);
  if (result != ncclSuccess) {
    throw std::runtime_error(
        "[NCCLGinBackend::create_device_window]: Failed to create NCCL device communicator. "
        "Error: " +
        std::string(nccl_api->getErrorString(result)));
  }

  // Construct and return the device window struct
  TorchCommDeviceWindow<NCCLGinBackend> device_window;
  device_window.comm_ = nccl_dev_comm;
  device_window.window_ = orig_window;
  device_window.base_ = base;
  device_window.size_ = size;
  device_window.rank_ = config.comm_rank;
  device_window.num_ranks_ = config.comm_size;

  return device_window;
}

void NCCLGinBackend::destroy_device_window(
    ncclComm_t nccl_comm,
    torch::comms::NcclxApi* nccl_api,
    TorchCommDeviceWindow<NCCLGinBackend>& device_window) {
  if (nccl_comm == nullptr || nccl_api == nullptr) {
    return;
  }

  auto result = nccl_api->devCommDestroy(nccl_comm, &device_window.comm_);
  if (result != ncclSuccess) {
    TC_LOG(ERROR) << "Failed to destroy NCCL device communicator: "
                  << nccl_api->getErrorString(result);
  }

  // Reset the device window struct
  device_window.comm_ = {};
  device_window.window_ = nullptr;
  device_window.base_ = nullptr;
  device_window.size_ = 0;
  device_window.rank_ = 0;
  device_window.num_ranks_ = 0;
}

} // namespace torchcomms::device
