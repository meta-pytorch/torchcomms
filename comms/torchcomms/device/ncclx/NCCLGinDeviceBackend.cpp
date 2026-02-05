// Copyright (c) Meta Platforms, Inc. and affiliates.
// NCCL GIN Device Backend - Static Method Implementations

#include "comms/torchcomms/TorchCommLogging.hpp"
#include "comms/torchcomms/device/DeviceBackendTraits.hpp"
#include "comms/torchcomms/device/TorchCommDeviceComm.hpp"
#include "comms/torchcomms/ncclx/NcclxApi.hpp"

#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <string>

namespace torchcomms::device {

// =============================================================================
// DeviceWindowDeleter Implementation
// =============================================================================

void NCCLGinBackend::DeviceWindowDeleter::operator()(
    TorchCommDeviceWindow<NCCLGinBackend>* ptr) const {
  // Only free the device memory - caller is responsible for calling
  // ncclDevCommDestroy using the dev_comm stored in this deleter
  if (ptr != nullptr) {
    cudaFree(ptr);
  }
}

// =============================================================================
// create_device_window Implementation
// =============================================================================

NCCLGinBackend::Ptr NCCLGinBackend::create_device_window(
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

  // Create device window on host first (stack allocation)
  TorchCommDeviceWindow<NCCLGinBackend> host_dev_window(
      nccl_dev_comm,
      host_window,
      base,
      size,
      config.comm_rank,
      config.comm_size);

  // Allocate device memory for the window struct
  TorchCommDeviceWindow<NCCLGinBackend>* device_ptr = nullptr;
  cudaError_t cuda_result =
      cudaMalloc(&device_ptr, sizeof(TorchCommDeviceWindow<NCCLGinBackend>));
  if (cuda_result != cudaSuccess) {
    // Cleanup ncclDevComm before throwing
    nccl_api->devCommDestroy(nccl_comm, &nccl_dev_comm);
    throw std::runtime_error(
        "[NCCLGinBackend::create_device_window]: Failed to allocate device memory for window. "
        "CUDA error: " +
        std::string(cudaGetErrorString(cuda_result)));
  }

  // Copy the window struct to device memory
  cuda_result = cudaMemcpy(
      device_ptr,
      &host_dev_window,
      sizeof(TorchCommDeviceWindow<NCCLGinBackend>),
      cudaMemcpyHostToDevice);
  if (cuda_result != cudaSuccess) {
    // Cleanup on error
    cudaFree(device_ptr);
    nccl_api->devCommDestroy(nccl_comm, &nccl_dev_comm);
    throw std::runtime_error(
        "[NCCLGinBackend::create_device_window]: Failed to copy window to device memory. "
        "CUDA error: " +
        std::string(cudaGetErrorString(cuda_result)));
  }

  // Create custom deleter that stores nccl_comm, nccl_api, and dev_comm
  // The caller accesses dev_comm via get_deleter() for ncclDevCommDestroy
  DeviceWindowDeleter deleter(nccl_comm, nccl_api, nccl_dev_comm);

  return Ptr(device_ptr, deleter);
}

} // namespace torchcomms::device
