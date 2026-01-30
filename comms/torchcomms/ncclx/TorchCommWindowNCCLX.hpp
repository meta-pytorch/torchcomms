// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>
#include <vector>

#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy

#include "comms/torchcomms/TorchCommWindow.hpp"
#include "comms/torchcomms/device/DeviceBackendTraits.hpp"
#include "comms/torchcomms/device/TorchCommDeviceComm.hpp"
#include "comms/torchcomms/ncclx/NcclxApi.hpp"
#include "comms/torchcomms/ncclx/TorchWorkNCCLX.hpp"

#include <nccl_device/impl/comm__types.h> // @manual=//comms/ncclx:nccl

namespace torch::comms {

class TorchCommNCCLX;

// =============================================================================
// TorchCommWindowNCCLX - Host-side Window with Device API Support
// =============================================================================
//
// Template parameter Backend provides compile-time polymorphism:
//   - NCCLGinBackend: NCCL GIN for GPU-initiated networking
//   - Future: NVSHMEMBackend, etc.
//
// Implementation is in TorchCommWindowNCCLX.cpp with explicit instantiation.

template <typename Backend>
class TorchCommWindowNCCLX : public TorchCommWindow {
 public:
  // Type aliases for device-side types
  // Backend::Comm is the raw communicator type (e.g., ncclDevComm)
  using DeviceWindow = torchcomms::device::TorchCommDeviceWindow<Backend>;
  using DeviceRegisteredBuffer = torchcomms::device::RegisteredBuffer;

  TorchCommWindowNCCLX() = delete;
  explicit TorchCommWindowNCCLX(
      ncclComm_t ncclComm,
      std::shared_ptr<TorchCommNCCLX> torchComm);
  ~TorchCommWindowNCCLX() noexcept override;

  TorchCommWindowNCCLX(const TorchCommWindowNCCLX& other) = delete;
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

  // ==========================================================================
  // Device API Support
  // ==========================================================================

  // Register a local buffer for use as source in device-side put operations.
  // This is NON-COLLECTIVE because it uses local_comm_ (1-rank communicator).
  DeviceRegisteredBuffer register_local_buffer(const at::Tensor& tensor);

  // Deregister a previously registered local buffer. NON-COLLECTIVE.
  void deregister_local_buffer(DeviceRegisteredBuffer& buf);

  // Get a device-side window handle for GPU-initiated operations.
  // Returns by value - CUDA copies kernel arguments automatically.
  //
  // Usage:
  //   auto dev_win = host_window->get_device_window();
  //   my_kernel<<<grid, block>>>(dev_win, ...);
  DeviceWindow get_device_window(
      int signal_count = -1,
      int counter_count = -1,
      int barrier_count = 1);

  // Get the NCCL orig window handle for device-side operations.
  void* get_nccl_orig_window() const {
    return static_cast<void*>(nccl_orig_win_);
  }

 private:
  void initLocalComm();
  void initNcclOrigWindow(void* ptr, size_t size);

  void checkRequestSizeAndThrow(size_t input_size) const;
  void checkDeviceAndThrow(const at::Tensor& tensor) const;
  void checkCommAndThrow() const;
  void checkWindowAndThrow() const;

  ncclComm_t nccl_comm_{};
  std::shared_ptr<TorchCommNCCLX> torch_comm_;
  NcclxWindow win_{nullptr};

  // Device API state
  ncclComm_t local_comm_{nullptr};
  bool local_comm_initialized_{false};
  NcclxWindow nccl_orig_win_{nullptr};
  DeviceWindow device_window_{};
  bool device_window_initialized_{false};
  std::vector<DeviceRegisteredBuffer> registered_local_buffers_;

  // NCCL API abstraction
  NcclxApi* nccl_api_;
  at::Device comm_device_{at::kCUDA};
};

// Type alias for the common case
using TorchCommWindowNCCLXGin =
    TorchCommWindowNCCLX<torchcomms::device::NCCLGinBackend>;

} // namespace torch::comms
