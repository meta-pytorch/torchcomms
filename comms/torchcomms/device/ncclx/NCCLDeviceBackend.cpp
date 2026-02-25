// Copyright (c) Meta Platforms, Inc. and affiliates.
// NCCL Device Backend - Static Method Implementations

#include "comms/torchcomms/TorchCommLogging.hpp"
#include "comms/torchcomms/device/DeviceBackendTraits.hpp"
#include "comms/torchcomms/device/TorchCommDeviceWindow.hpp"
#include "comms/torchcomms/device/cuda/CudaApi.hpp"
#include "comms/torchcomms/ncclx/NcclxApi.hpp"

#include <memory>
#include <stdexcept>
#include <string>

namespace torchcomms::device {

// =============================================================================
// DeviceWindowDeleter Implementation
// =============================================================================

void NCCLDeviceBackend::DeviceWindowDeleter::operator()(
    TorchCommDeviceWindow<NCCLDeviceBackend>* ptr) const {
  // Only free the device memory - caller is responsible for calling
  // ncclDevCommDestroy using the dev_comm stored in this deleter
  if (ptr != nullptr) {
    CUDA_CHECK_IGNORE(
        cuda_api, cuda_api->free(ptr), "Failed to free device window");
  }
}

// =============================================================================
// create_device_window Implementation
// =============================================================================

NCCLDeviceBackend::Ptr NCCLDeviceBackend::create_device_window(
    ncclComm_t nccl_comm,
    torch::comms::NcclxApi* nccl_api,
    torch::comms::CudaApi* cuda_api,
    const DeviceBackendConfig& config,
    Window host_window,
    void* base,
    size_t size) {
  if (nccl_comm == nullptr) {
    throw std::runtime_error(
        "[NCCLDeviceBackend::create_device_window]: NCCL communicator cannot be null");
  }
  if (nccl_api == nullptr) {
    throw std::runtime_error(
        "[NCCLDeviceBackend::create_device_window]: NCCL API cannot be null");
  }
  if (cuda_api == nullptr) {
    throw std::runtime_error(
        "[NCCLDeviceBackend::create_device_window]: CUDA API cannot be null");
  }
  if (base == nullptr && size > 0) {
    throw std::runtime_error(
        "[NCCLDeviceBackend::create_device_window]: Window base cannot be null with non-zero size");
  }

  // Per-peer signal slot buffer: num_signals * num_ranks * sizeof(uint64_t)
  // Each signal_id has num_ranks slots (one per sender), avoiding the
  // cross-transport atomicity hazard between NVLink and RDMA atomics.
  ncclDevResourceHandle signal_buffer_handle = 0;
  ncclDevResourceRequirements signal_resource_reqs = {};
  const size_t signal_buffer_size = static_cast<size_t>(config.signal_count) *
      config.comm_size * sizeof(uint64_t);

  if (signal_buffer_size > 0) {
    signal_resource_reqs.next = nullptr;
    signal_resource_reqs.bufferSize = signal_buffer_size;
    signal_resource_reqs.bufferAlign = 8;
    signal_resource_reqs.outBufferHandle = &signal_buffer_handle;
    signal_resource_reqs.ginSignalCount = 0;
    signal_resource_reqs.ginCounterCount = 0;
    signal_resource_reqs.outGinSignalStart = nullptr;
    signal_resource_reqs.outGinCounterStart = nullptr;
  }

  // Set up ncclDevCommRequirements with GIN enabled using designated
  // initializers
  ncclDevCommRequirements reqs = {
      .resourceRequirementsList =
          (signal_buffer_size > 0) ? &signal_resource_reqs : nullptr,
      .teamRequirementsList = nullptr,
      .lsaMultimem = false,
      .barrierCount = config.barrier_count,
      .lsaBarrierCount = config.barrier_count,
      .railGinBarrierCount = config.barrier_count,
      .lsaLLA2ABlockCount = 0,
      .lsaLLA2ASlotCount = 0,
      .ginForceEnable = true,
      .ginContextCount = 1,
      .ginSignalCount = 0,
      .ginCounterCount = config.counter_count,
  };

  // Create NCCL device communicator with GIN state
  ncclDevComm nccl_dev_comm{};
  auto result = nccl_api->devCommCreate(nccl_comm, &reqs, &nccl_dev_comm);
  if (result != ncclSuccess) {
    throw std::runtime_error(
        "[NCCLDeviceBackend::create_device_window]: Failed to create NCCL device communicator. "
        "Error: " +
        std::string(nccl_api->getErrorString(result)));
  }

  // Create device window on host first (stack allocation)
  TorchCommDeviceWindow<NCCLDeviceBackend> host_dev_window(
      nccl_dev_comm,
      host_window,
      base,
      size,
      config.comm_rank,
      config.comm_size,
      signal_buffer_handle);

  // Allocate device memory for the window struct
  TorchCommDeviceWindow<NCCLDeviceBackend>* device_ptr = nullptr;
  cudaError_t cuda_result = cuda_api->malloc(
      reinterpret_cast<void**>(&device_ptr),
      sizeof(TorchCommDeviceWindow<NCCLDeviceBackend>));
  if (cuda_result != cudaSuccess) {
    // Cleanup ncclDevComm before throwing
    nccl_api->devCommDestroy(nccl_comm, &nccl_dev_comm);
    throw std::runtime_error(
        "[NCCLDeviceBackend::create_device_window]: Failed to allocate device memory for window. "
        "CUDA error: " +
        std::string(cuda_api->getErrorString(cuda_result)));
  }

  // Copy the window struct to device memory
  // device_ptr is non-null here (cuda_api->malloc succeeded above)
  // NOLINTNEXTLINE(facebook-hte-NullableDereference,facebook-security-vulnerable-memcpy)
  cuda_result = cuda_api->memcpy(
      device_ptr,
      &host_dev_window,
      sizeof(TorchCommDeviceWindow<NCCLDeviceBackend>),
      cudaMemcpyHostToDevice);
  if (cuda_result != cudaSuccess) {
    // Cleanup on error
    // NOLINTNEXTLINE(facebook-hte-NullableDereference)
    CUDA_CHECK_IGNORE(
        cuda_api,
        cuda_api->free(device_ptr),
        "Failed to free device window during error cleanup");
    nccl_api->devCommDestroy(nccl_comm, &nccl_dev_comm);
    throw std::runtime_error(
        "[NCCLDeviceBackend::create_device_window]: Failed to copy window to device memory. "
        "CUDA error: " +
        std::string(cuda_api->getErrorString(cuda_result)));
  }

  // Create custom deleter that stores nccl_comm, nccl_api, cuda_api, and
  // dev_comm. The caller accesses dev_comm via get_deleter() for
  // ncclDevCommDestroy.
  DeviceWindowDeleter deleter(nccl_comm, nccl_api, cuda_api, nccl_dev_comm);

  return Ptr(device_ptr, deleter);
}

} // namespace torchcomms::device
