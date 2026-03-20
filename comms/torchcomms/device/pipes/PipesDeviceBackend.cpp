// Copyright (c) Meta Platforms, Inc. and affiliates.
// PipesDeviceBackend - Static method implementations

#if defined(ENABLE_PIPES)

#include "comms/torchcomms/device/pipes/PipesDeviceBackend.hpp"
#include "comms/pipes/MultiPeerDeviceHandle.cuh"
#include "comms/torchcomms/device/DeviceBackendTraits.hpp"
#include "comms/torchcomms/device/TorchCommDeviceWindow.hpp"
#include "comms/torchcomms/device/cuda/CudaApi.hpp"
#include "comms/torchcomms/ncclx/NcclxApi.hpp"
#include "comms/torchcomms/utils/Logging.hpp"

#include <stdexcept>
#include <string>

namespace torchcomms::device {

// =============================================================================
// DeviceWindowDeleter Implementation
// =============================================================================

void PipesDeviceBackend::DeviceWindowDeleter::operator()(
    TorchCommDeviceWindow<PipesDeviceBackend>* ptr) const {
  if (cuda_api == nullptr) {
    return;
  }
  // Destroy the Pipes DeviceWindow via the ncclx API.
  // This mirrors the error-cleanup paths in create_device_window() and
  // ensures ncclx internal state (CtranWin, HostWindow) is properly torn down.
  if (pipes_device_window != nullptr && nccl_api != nullptr) {
    auto result = nccl_api->winDestroyDeviceWin(pipes_device_window);
    if (result != ncclSuccess) {
      TC_LOG(ERROR) << "[PipesDeviceBackend]: winDestroyDeviceWin failed "
                    << "during cleanup";
    }
  }
  // Free the TorchCommDeviceWindow struct in device memory.
  if (ptr != nullptr) {
    CUDA_CHECK_IGNORE(
        cuda_api, cuda_api->free(ptr), "Failed to free Pipes device window");
  }
}

// =============================================================================
// create_device_window Implementation
// =============================================================================

PipesDeviceBackend::Ptr PipesDeviceBackend::create_device_window(
    ncclComm_t /* nccl_comm */,
    torch::comms::NcclxApi* nccl_api,
    torch::comms::CudaApi* cuda_api,
    const DeviceBackendConfig& config,
    ncclWindow_t nccl_win,
    void* base,
    size_t size) {
  if (nccl_api == nullptr) {
    throw std::runtime_error(
        "[PipesDeviceBackend::create_device_window]: nccl_api cannot be null");
  }
  if (nccl_win == nullptr) {
    throw std::runtime_error(
        "[PipesDeviceBackend::create_device_window]: nccl_win cannot be null");
  }
  if (cuda_api == nullptr) {
    throw std::runtime_error(
        "[PipesDeviceBackend::create_device_window]: cuda_api cannot be null");
  }

  // Step 1: Create the Pipes DeviceWindow in device memory via ncclx.
  // COLLECTIVE on first call — all ranks must call together.
  void* pipes_device_win = nullptr;
  auto nccl_result = nccl_api->winCreateDeviceWin(
      nccl_win,
      config.signal_count,
      config.counter_count,
      config.barrier_count,
      &pipes_device_win);
  if (nccl_result != ncclSuccess) {
    throw std::runtime_error(
        "[PipesDeviceBackend::create_device_window]: "
        "winCreateDeviceWin failed");
  }

  // Step 2: Build TorchCommDeviceWindow<PipesDeviceBackend> on host.
  // comm_ = nullptr (unused for Pipes; no separate communicator handle)
  // window_ = typed device pointer to DeviceWindow
  auto* device_win = static_cast<comms::pipes::DeviceWindow*>(pipes_device_win);
  TorchCommDeviceWindow<PipesDeviceBackend> host_dev_window(
      nullptr, // Comm = void*, unused for Pipes
      device_win, // Window = DeviceWindow*, device pointer
      base,
      size,
      config.comm_rank,
      config.comm_size,
      0 /* signal_buffer_handle, unused for Pipes */);

  // Step 3: Allocate device memory for the TorchCommDeviceWindow struct.
  TorchCommDeviceWindow<PipesDeviceBackend>* device_ptr = nullptr;
  cudaError_t cuda_result = cuda_api->malloc(
      reinterpret_cast<void**>(&device_ptr),
      sizeof(TorchCommDeviceWindow<PipesDeviceBackend>));
  if (cuda_result != cudaSuccess) {
    // Clean up Pipes DeviceWindow on failure.
    auto destroy_result = nccl_api->winDestroyDeviceWin(pipes_device_win);
    if (destroy_result != ncclSuccess) {
      TC_LOG(ERROR) << "[PipesDeviceBackend]: Failed to clean up Pipes device "
                    << "window after malloc failure";
    }
    throw std::runtime_error(
        "[PipesDeviceBackend::create_device_window]: Failed to allocate "
        "device memory for TorchCommDeviceWindow. CUDA error: " +
        std::string(cuda_api->getErrorString(cuda_result)));
  }

  // Step 4: Copy TorchCommDeviceWindow struct to device memory.
  // NOLINTNEXTLINE(facebook-hte-NullableDereference,facebook-security-vulnerable-memcpy)
  cuda_result = cuda_api->memcpy(
      device_ptr,
      &host_dev_window,
      sizeof(TorchCommDeviceWindow<PipesDeviceBackend>),
      cudaMemcpyHostToDevice);
  if (cuda_result != cudaSuccess) {
    // NOLINTNEXTLINE(facebook-hte-NullableDereference)
    CUDA_CHECK_IGNORE(
        cuda_api,
        cuda_api->free(device_ptr),
        "Failed to free device window during error cleanup");
    auto destroy_result = nccl_api->winDestroyDeviceWin(pipes_device_win);
    if (destroy_result != ncclSuccess) {
      TC_LOG(ERROR) << "[PipesDeviceBackend]: Failed to clean up Pipes device "
                    << "window after memcpy failure";
    }
    throw std::runtime_error(
        "[PipesDeviceBackend::create_device_window]: Failed to copy "
        "TorchCommDeviceWindow to device memory. CUDA error: " +
        std::string(cuda_api->getErrorString(cuda_result)));
  }

  DeviceWindowDeleter deleter(nccl_api, cuda_api, pipes_device_win);
  return Ptr(device_ptr, deleter);
}

// =============================================================================
// register_local_buffer / deregister_local_buffer Implementation
// =============================================================================

RegisteredBuffer PipesDeviceBackend::register_local_buffer(
    torch::comms::NcclxApi* nccl_api,
    ncclComm_t nccl_comm,
    void* ptr,
    size_t size) {
  uint32_t lkey = 0;
  auto result = nccl_api->winLocalRegisterBuffer(nccl_comm, ptr, size, &lkey);
  if (result != ncclSuccess) {
    throw std::runtime_error(
        "[PipesDeviceBackend::register_local_buffer]: "
        "winLocalRegisterBuffer failed");
  }

  // Pipes (IBGDA) put uses lkey for WQE construction during RDMA writes.
  // backend_window is unused by Pipes — only the GIN backend needs it.
  RegisteredBuffer buf;
  buf.base_ptr = ptr;
  buf.size = size;
  buf.backend_window = nullptr;
  buf.lkey = lkey;
  return buf;
}

void PipesDeviceBackend::deregister_local_buffer(
    torch::comms::NcclxApi* nccl_api,
    ncclComm_t nccl_comm,
    RegisteredBuffer& buf) {
  if (buf.base_ptr == nullptr) {
    return;
  }
  auto result = nccl_api->winLocalDeregisterBuffer(nccl_comm, buf.base_ptr);
  if (result != ncclSuccess) {
    TC_LOG(ERROR) << "[PipesDeviceBackend]: Failed to deregister local buffer";
  }
  buf.backend_window = nullptr;
  buf.base_ptr = nullptr;
  buf.size = 0;
  buf.lkey = 0;
}

// =============================================================================
// fetch_transport_handle Implementation
// =============================================================================

comms::pipes::MultiPeerDeviceHandle PipesDeviceBackend::fetch_transport_handle(
    ncclComm_t nccl_comm,
    torch::comms::NcclxApi* nccl_api) {
  void* transports_ptr = nullptr;
  int my_rank = -1;
  int n_ranks = 0;
  int num_nvl_peers = 0;
  int num_ib_peers = 0;

  auto result = nccl_api->getMultiPeerDeviceHandle(
      nccl_comm,
      &transports_ptr,
      &my_rank,
      &n_ranks,
      &num_nvl_peers,
      &num_ib_peers);

  if (result != ncclSuccess) {
    throw std::runtime_error(
        "[PipesDeviceBackend::fetch_transport_handle] "
        "Failed to get MultiPeerDeviceHandle. "
        "Ensure NCCL_CTRAN_USE_PIPES=1 is set.");
  }

  return comms::pipes::MultiPeerDeviceHandle{
      my_rank,
      n_ranks,
      {static_cast<comms::pipes::Transport*>(transports_ptr),
       static_cast<
           comms::pipes::DeviceSpan<comms::pipes::Transport>::size_type>(
           n_ranks)},
      num_nvl_peers,
      num_ib_peers};
}

// =============================================================================
// TransportHandleDeleter Implementation
// =============================================================================

void PipesDeviceBackend::TransportHandleDeleter::operator()(void* ptr) const {
  if (ptr != nullptr && cuda_api != nullptr) {
    CUDA_CHECK_IGNORE(
        cuda_api, cuda_api->free(ptr), "Failed to free transport handle");
  }
}

// =============================================================================
// get_device_transport Implementation
// =============================================================================

PipesDeviceBackend::TransportHandleDevPtr
PipesDeviceBackend::get_device_transport(
    ncclComm_t nccl_comm,
    torch::comms::NcclxApi* nccl_api,
    torch::comms::CudaApi* cuda_api) {
  if (nccl_api == nullptr || cuda_api == nullptr) {
    throw std::runtime_error(
        "[PipesDeviceBackend::get_device_transport]: "
        "nccl_api and cuda_api must not be null");
  }

  // Get handle on host (private helper)
  auto handle = fetch_transport_handle(nccl_comm, nccl_api);

  // Allocate device memory
  void* device_ptr = nullptr;
  cudaError_t err = cuda_api->malloc(
      &device_ptr, sizeof(comms::pipes::MultiPeerDeviceHandle));
  if (err != cudaSuccess) {
    throw std::runtime_error(
        "[PipesDeviceBackend::get_device_transport]: "
        "cudaMalloc failed: " +
        std::string(cuda_api->getErrorString(err)));
  }

  // Copy to device
  // NOLINTNEXTLINE(facebook-hte-NullableDereference,facebook-security-vulnerable-memcpy)
  err = cuda_api->memcpy(
      device_ptr,
      &handle,
      sizeof(comms::pipes::MultiPeerDeviceHandle),
      cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    CUDA_CHECK_IGNORE(
        cuda_api,
        cuda_api->free(device_ptr),
        "Failed to free transport handle during error cleanup");
    throw std::runtime_error(
        "[PipesDeviceBackend::get_device_transport]: "
        "cudaMemcpy failed: " +
        std::string(cuda_api->getErrorString(err)));
  }

  return TransportHandleDevPtr(device_ptr, TransportHandleDeleter{cuda_api});
}

} // namespace torchcomms::device

#endif // ENABLE_PIPES
