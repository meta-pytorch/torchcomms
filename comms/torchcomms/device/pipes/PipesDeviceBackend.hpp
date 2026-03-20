// Copyright (c) Meta Platforms, Inc. and affiliates.
// TorchComms Device API - Pipes Backend Traits
//
// Defines the PipesDeviceBackend trait type for compile-time polymorphism
// in TorchCommDeviceWindow. Uses Pipes IBGDA and NVLink transports via the
// ctran window infrastructure (accessed through opaque ncclx APIs).
//
// This backend replaces GIN (GPU Initiated Networking) with Pipes for
// device-side P2P operations:
//   - NVLink peers (same node): direct memcpy with NVLink-mapped pointers
//   - IBGDA peers (cross-node): RDMA writes via DOCA GPUNetIO
//
// The Window type is a typed device pointer (comms::pipes::DeviceWindow*)
// allocated by ncclx. Device-side code uses it directly without casting.

#pragma once

#if defined(ENABLE_PIPES)

#include <memory>

#include <nccl.h> // @manual=//comms/ncclx:nccl

namespace comms::pipes {
class DeviceWindow;
struct MultiPeerDeviceHandle;
} // namespace comms::pipes

namespace torch::comms {
class CudaApi;
class NcclxApi;
class TorchCommNCCLX;
} // namespace torch::comms

namespace torchcomms::device {

struct DeviceBackendConfig;
struct RegisteredBuffer;

template <typename Backend>
class TorchCommDeviceWindow;

// =============================================================================
// PipesDeviceBackend - IBGDA + NVLink backend
// =============================================================================
//
// Types:
//   - Comm:   void* — unused for Pipes (no separate communicator handle).
//   - Window: comms::pipes::DeviceWindow* — typed device pointer to the
//             transport window allocated by ncclx via winCreateDeviceWin().

struct PipesDeviceBackend {
  // Comm is unused for Pipes: there is no separate communicator handle.
  // Set to nullptr in the TorchCommDeviceWindow constructor.
  using Comm = void*;

  // Typed device pointer to the Pipes DeviceWindow. Device-side code accesses
  // transport handles, NVLink-mapped remote pointers, and IBGDA descriptors
  // directly through this pointer without casting.
  using Window = comms::pipes::DeviceWindow*;

  // =========================================================================
  // DeviceWindowDeleter - Custom deleter for device window cleanup
  // =========================================================================
  //
  // Frees both the TorchCommDeviceWindow<PipesDeviceBackend> struct in device
  // memory AND the separate DeviceWindow allocation.
  struct DeviceWindowDeleter {
    torch::comms::NcclxApi* nccl_api{nullptr};
    torch::comms::CudaApi* cuda_api{nullptr};
    // Opaque device pointer to DeviceWindow, allocated by ncclx via
    // winCreateDeviceWin(). Must be freed via winDestroyDeviceWin() since
    // it cannot be reached through ptr from host code (ptr is in device
    // memory).
    void* pipes_device_window{nullptr};

    DeviceWindowDeleter() = default;
    DeviceWindowDeleter(
        torch::comms::NcclxApi* nccl_api,
        torch::comms::CudaApi* cuda_api,
        void* win_dev)
        : nccl_api(nccl_api),
          cuda_api(cuda_api),
          pipes_device_window(win_dev) {}

    void operator()(TorchCommDeviceWindow<PipesDeviceBackend>* ptr) const;
  };

  using Ptr = std::unique_ptr<
      TorchCommDeviceWindow<PipesDeviceBackend>,
      DeviceWindowDeleter>;

  // Create fully initialized device window struct in DEVICE memory.
  //
  // Internally calls nccl_api->winCreateDeviceWin() to create the transport
  // device window, then wraps it in a TorchCommDeviceWindow.
  //
  // COLLECTIVE on first call — all ranks must call simultaneously because
  // winCreateDeviceWin() performs an allGather internally.
  //
  // Signature matches NCCLDeviceBackend::create_device_window for unified
  // call sites. nccl_comm is unused for Pipes (no separate communicator).
  //
  // Parameters:
  //   - nccl_comm:  NCCL communicator (unused, for API consistency with GIN)
  //   - nccl_api:   NcclxApi for creating the transport device window
  //   - cuda_api:   CUDA API abstraction (must not be null)
  //   - config:     Device backend configuration (rank, size, signal/counter
  //                 counts, etc.)
  //   - nccl_win:   NCCL window handle (ncclWindow_t from tensor_register)
  //   - base:       Window base pointer (local data buffer)
  //   - size:       Window size in bytes
  static Ptr create_device_window(
      ncclComm_t nccl_comm,
      torch::comms::NcclxApi* nccl_api,
      torch::comms::CudaApi* cuda_api,
      const DeviceBackendConfig& config,
      ncclWindow_t nccl_win,
      void* base,
      size_t size);

  // =========================================================================
  // Backend-specific hooks called from TorchCommWindowNCCLX
  // =========================================================================

  // No additional window registration needed for Pipes.
  static void register_extra_window(
      torch::comms::NcclxApi*,
      ncclComm_t,
      ncclWindow_t*,
      void*,
      size_t) {}

  // No additional window to deregister for Pipes.
  static void
  deregister_extra_window(torch::comms::NcclxApi*, ncclComm_t, ncclWindow_t*) {}

  // Pipes deleter handles all cleanup via winDestroyDeviceWin.
  static void destroy_device_comm(Ptr&) {}

  // Pipes uses the regular ctran window for device window creation.
  static ncclWindow_t select_device_win(
      ncclWindow_t win,
      ncclWindow_t /* nccl_orig_win */) {
    return win;
  }

  // Register a local buffer for device-side put operations (Pipes/IBGDA path).
  // Uses MultiPeerTransport::localRegisterIbgdaBuffer for non-collective
  // local memory registration. Returns RegisteredBuffer with lkey.
  static torchcomms::device::RegisteredBuffer register_local_buffer(
      torch::comms::NcclxApi* nccl_api,
      ncclComm_t nccl_comm,
      void* ptr,
      size_t size);

  // Deregister a previously registered local buffer (Pipes/IBGDA path).
  static void deregister_local_buffer(
      torch::comms::NcclxApi* nccl_api,
      ncclComm_t nccl_comm,
      torchcomms::device::RegisteredBuffer& buf);

  // =========================================================================
  // Transport device handle (device-allocated MultiPeerDeviceHandle)
  // =========================================================================

  struct TransportHandleDeleter {
    torch::comms::CudaApi* cuda_api{nullptr};
    void operator()(void* ptr) const;
  };
  using TransportHandleDevPtr = std::unique_ptr<void, TransportHandleDeleter>;

  // Get a device-allocated MultiPeerDeviceHandle for Triton and CUDA
  // kernels. Calls fetch_transport_handle() internally to get handle by value,
  // then cudaMalloc + cudaMemcpy to device memory.
  // Returns managed pointer — cudaFree on destruction.
  static TransportHandleDevPtr get_device_transport(
      ncclComm_t nccl_comm,
      torch::comms::NcclxApi* nccl_api,
      torch::comms::CudaApi* cuda_api);

 private:
  // Get the pipes transport device handle from the communicator.
  // NON-COLLECTIVE — reads already-exchanged state.
  //
  // Returns a MultiPeerDeviceHandle by value. The handle contains a
  // device pointer to the Transport[] array (already GPU-allocated by
  // MultiPeerTransport::exchange() during ctran init).
  //
  // Throws std::runtime_error if pipes transport is not initialized.
  static comms::pipes::MultiPeerDeviceHandle fetch_transport_handle(
      ncclComm_t nccl_comm,
      torch::comms::NcclxApi* nccl_api);
};

} // namespace torchcomms::device

#endif // ENABLE_PIPES
