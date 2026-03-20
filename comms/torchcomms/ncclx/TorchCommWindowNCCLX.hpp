// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>
#include <vector>

#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy

#include "comms/torchcomms/TorchCommWindow.hpp"
#include "comms/torchcomms/ncclx/NcclxApi.hpp"
#include "comms/torchcomms/ncclx/TorchWorkNCCLX.hpp"

// Device API support requires NCCLX 2.28+ with nccl_device headers
#ifdef TORCHCOMMS_HAS_NCCL_DEVICE_API
#include <nccl_device/impl/comm__types.h> // @manual=//comms/ncclx:nccl
#include "comms/torchcomms/device/DeviceBackendTraits.hpp"
#include "comms/torchcomms/device/TorchCommDeviceWindow.hpp"
#endif

#if defined(ENABLE_PIPES)
#include "comms/torchcomms/device/pipes/PipesDeviceBackend.hpp"
#include "comms/torchcomms/device/pipes/TorchCommDevicePipesTypes.hpp"
#endif

namespace torch::comms {

// =============================================================================
// Backend Types
// =============================================================================
//
// When TORCHCOMMS_HAS_NCCL_DEVICE_API is NOT defined (NCCLX 2.27):
//   HostOnlyBackend is a dummy backend that allows the template to be
//   instantiated without device API headers. All device API methods are
//   gated with #ifdef, so this backend is never actually used at runtime.

#ifndef TORCHCOMMS_HAS_NCCL_DEVICE_API
struct HostOnlyBackend {};
#endif

class TorchCommNCCLX;

// =============================================================================
// TorchCommWindowNCCLX - Host-side Window with Device API Support
// =============================================================================
//
// When TORCHCOMMS_HAS_NCCL_DEVICE_API is defined (NCCLX 2.28+):
//   Template parameter Backend provides compile-time polymorphism:
//     - NCCLDeviceBackend: Unified NCCL backend (GIN + LSA)
//     - Future: NVSHMEMBackend, etc.
//
// When TORCHCOMMS_HAS_NCCL_DEVICE_API is NOT defined (NCCLX 2.27):
//   Device API functionality is disabled. Only host-side window operations
//   (put, signal, wait_signal) are available.
//
// Implementation is in TorchCommWindowNCCLX.cpp with explicit instantiation.

template <typename Backend>
class TorchCommWindowNCCLX : public TorchCommWindow {
 public:
#ifdef TORCHCOMMS_HAS_NCCL_DEVICE_API
  // Type aliases for device-side types (only available with device API)
  // Backend::Comm is the raw communicator type (e.g., ncclDevComm)
  using DeviceWindow = torchcomms::device::TorchCommDeviceWindow<Backend>;
  using DeviceRegisteredBuffer = torchcomms::device::RegisteredBuffer;
#endif

  TorchCommWindowNCCLX() = delete;
  explicit TorchCommWindowNCCLX(
      ncclComm_t ncclComm,
      std::shared_ptr<TorchCommNCCLX> torchComm);
  ~TorchCommWindowNCCLX() noexcept override;

  TorchCommWindowNCCLX(const TorchCommWindowNCCLX& other) = delete;
  TorchCommWindowNCCLX(TorchCommWindowNCCLX&& other) = delete;
  TorchCommWindowNCCLX& operator=(const TorchCommWindowNCCLX& other) = delete;
  TorchCommWindowNCCLX& operator=(TorchCommWindowNCCLX&& other) noexcept =
      delete;

  void tensor_register(const at::Tensor& tensor) override;
  void tensor_deregister() override;

  std::shared_ptr<TorchCommWindow> clone() override;

  c10::intrusive_ptr<TorchWork> put(
      const at::Tensor& tensor,
      int dstRank,
      size_t targetOffsetNelems,
      bool asyncOp,
      const PutOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> signal(
      int peerRank,
      bool asyncOp,
      const SignalOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> wait_signal(
      int peerRank,
      bool asyncOp,
      const WaitSignalOptions& options = {}) override;
  at::Tensor map_remote_tensor(int rank) override;

  std::shared_ptr<TorchCommWindowAttr> get_attr(int peerRank) override;

#ifdef TORCHCOMMS_HAS_NCCL_DEVICE_API
  // ==========================================================================
  // Device API Support (requires NCCLX 2.28+)
  // ==========================================================================

  // Register a local buffer for use as source in device-side put operations.
  // NON-COLLECTIVE — registration is purely local (lkey only, no rkey
  // exchange). The resulting buffer can only be used as a source for put
  // operations.
  //
  // Backend dispatch:
  //   - NCCLDeviceBackend: NCCL_WIN_DEVICE_API | NCCL_WIN_LOCAL_ONLY
  //   - PipesDeviceBackend: MultiPeerTransport::localRegisterIbgdaBuffer
  //
  // Prerequisites: Must call tensor_register() then get_device_window() first.
  DeviceRegisteredBuffer register_local_buffer(const at::Tensor& tensor);

  // Deregister a previously registered local buffer. NON-COLLECTIVE.
  void deregister_local_buffer(DeviceRegisteredBuffer& buf);

  // Get a device-side window handle for GPU-initiated operations.
  // Returns a pointer to the cached device window. The window is lazily
  // created on first call and cached.
  //
  // Thread-safety: This method is NOT thread-safe. Concurrent calls from
  // multiple threads require external synchronization.
  //
  // Lifetime: The returned pointer is valid until this host window is
  // destroyed.
  //
  // Usage (C++):
  //   auto* dev_win = host_window->get_device_window();
  //   my_kernel<<<grid, block>>>(dev_win, ...);
  //   // In kernel: dev_win->put(...), dev_win->signal(...), etc.
  //
  // Usage (Triton):
  //   The same pointer can be passed to Triton kernels via torchcomms_put(),
  //   torchcomms_signal(), etc. wrappers that take void* handles.
  DeviceWindow* get_device_window(
      int signal_count = -1,
      int counter_count = -1,
      int barrier_count = 1);

  // Get the host-side NCCL window handle (GIN backend only).
  // Useful for creating RegisteredBuffer from host code when the device
  // window's window_ field is in device memory and not directly accessible.
  // TODO: Returns nullptr for PipesDeviceBackend (nccl_orig_win_ is not
  // initialized). Gate or return win_ for Pipes if callers need it.
  ncclWindow_t get_nccl_window() const {
    return nccl_orig_win_;
  }

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)
  // Get the NVLink-mapped address of a peer's window memory.
  // Returns the device pointer that can be used to directly access the peer's
  // window buffer via NVLink. Returns 0 (as int64) if the peer is not
  // NVLink-accessible (e.g., remote node over RDMA).
  //
  // Prerequisites: Must call tensor_register() first so that nccl_orig_win_
  // is initialized.
  //
  // Args:
  //   peer: The world rank of the peer whose address to retrieve.
  //   offset: Byte offset within the peer's window (default 0).
  //
  // Returns: Device pointer as void*, or nullptr if not NVLink-accessible.
  void* get_nvlink_address(int peer, size_t offset = 0);
#endif
#endif

 private:
  // Backend-specific behavior is handled via static methods on the Backend
  // type (e.g., Backend::register_extra_window(), Backend::select_device_win())
  // instead of if constexpr dispatch.

  void checkRequestSizeAndThrow(size_t input_size) const;
  void checkDeviceAndThrow(const at::Tensor& tensor) const;
  void checkCommAndThrow() const;
  void checkWindowAndThrow() const;

  ncclComm_t nccl_comm_{};
  std::shared_ptr<TorchCommNCCLX> torch_comm_;
  NcclxWindow win_{nullptr};

#ifdef TORCHCOMMS_HAS_NCCL_DEVICE_API
  // Device API state (only available with NCCLX 2.28+)
  NcclxWindow nccl_orig_win_{nullptr};

  // Device window is allocated in DEVICE memory (cudaMalloc) for GPU access.
  // The pointer can be passed to CUDA kernels and Triton.
  // The custom deleter handles both cudaFree and ncclDevCommDestroy.
  torchcomms::device::DeviceWindowPtr<Backend> device_window_;

  std::vector<DeviceRegisteredBuffer> registered_local_buffers_;

  // No ctran_win_ member needed — Pipes device windows are created
  // on-demand via nccl_api_->winCreateDeviceWin() in get_device_window().
#endif

  // NCCL API abstraction
  NcclxApi* nccl_api_;
  at::Device comm_device_{at::kCUDA};
};

// Type alias for the common case
// With device API: Uses NCCLDeviceBackend with full device capabilities
// Without device API: Uses HostOnlyBackend for host-side operations only
#ifdef TORCHCOMMS_HAS_NCCL_DEVICE_API
using TorchCommWindowNCCLXGin =
    TorchCommWindowNCCLX<torchcomms::device::NCCLDeviceBackend>;
#else
using TorchCommWindowNCCLXGin = TorchCommWindowNCCLX<HostOnlyBackend>;
#endif

// Type alias for the Pipes backend (IBGDA + NVLink device-side P2P).
// Only available when ENABLE_PIPES is defined (propagated from ctran_lib).
#if defined(ENABLE_PIPES)
using TorchCommWindowNCCLXPipes =
    TorchCommWindowNCCLX<torchcomms::device::PipesDeviceBackend>;
#endif

} // namespace torch::comms
