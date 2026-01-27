// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy
#include <nccl_device/impl/comm__types.h> // @manual=//comms/ncclx:nccl_device_api
#include <memory>
#include <vector>
#include "comms/torchcomms/TorchCommWindow.hpp"
#include "comms/torchcomms/device/TorchCommDeviceComm.hpp"
#include "comms/torchcomms/device/cuda/CudaApi.hpp"
#include "comms/torchcomms/ncclx/NcclxApi.hpp"
#include "comms/torchcomms/ncclx/TorchWorkNCCLX.hpp"

namespace torch::comms {

// Forward declaration
class TorchCommNCCLX;

class TorchCommWindowNCCLX : public TorchCommWindow {
 public:
  TorchCommWindowNCCLX() = delete;
  explicit TorchCommWindowNCCLX(
      ncclComm_t ncclComm,
      std::shared_ptr<TorchCommNCCLX> torchComm);
  ~TorchCommWindowNCCLX() noexcept override;

  // We delete the copy constructor and assignment operator to prevent 2 work
  // objects sharing the underlying collective work events.
  TorchCommWindowNCCLX(const TorchCommWindowNCCLX& other) = delete;
  TorchCommWindowNCCLX& operator=(const TorchCommWindowNCCLX& other) = delete;
  // Delete the move assignment operator to prevent accidentally stomping over
  // events if the work is in progress.
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
  //
  // The device API enables GPU-initiated networking from CUDA kernels.
  // It requires:
  //   1. A local_comm_ (1-rank split comm) for non-collective buffer
  //   registration
  //   2. An nccl_orig_win_ registered via NCCL orig path (has GIN support)
  //
  // Design:
  //   - tensor_register() creates BOTH ctran window (for host API) and
  //     nccl orig window (for device API with GIN support)
  //   - local_comm_ is created once via ncclCommSplit and reused for all
  //     register_local_buffer() calls
  //   - RegisteredBuffer can be used as source for device-side put operations

  // Register a local buffer for use as source in device-side put operations.
  // This is NON-COLLECTIVE because it uses local_comm_ (1-rank communicator).
  //
  // The returned RegisteredBuffer must be used with get_device_window() from
  // the SAME TorchCommWindowNCCLX instance that created it.
  //
  // @param tensor The tensor to register as a local source buffer
  // @return RegisteredBuffer handle for use in device-side put operations
  device::RegisteredBuffer register_local_buffer(const at::Tensor& tensor);

  // Deregister a previously registered local buffer.
  // This is NON-COLLECTIVE.
  //
  // @param buf The RegisteredBuffer to deregister
  void deregister_local_buffer(device::RegisteredBuffer& buf);

  // Get a device-side window handle for GPU-initiated operations.
  // The returned pointer is valid for the lifetime of this
  // TorchCommWindowNCCLX.
  //
  // @param signal_count Number of signals to allocate (default: size())
  // @param counter_count Number of counters to allocate (default: size())
  // @param barrier_count Number of barriers to allocate (default: 1)
  // @return Pointer to device window, valid on GPU
  device::TorchCommDeviceWindow* get_device_window(
      int signal_count = -1,
      int counter_count = -1,
      int barrier_count = 1);

 protected:
  friend class TorchCommNCCLX;

 private:
  // internal util functions
  void checkRequestSizeAndThrow(size_t input_size) const;
  void checkDeviceAndThrow(const at::Tensor& tensor) const;
  void checkCommAndThrow() const;
  void checkWindowAndThrow() const;
  void checkLocalCommAndThrow() const;
  void checkDeviceWindowAndThrow() const;

  // Initialize the local communicator for non-collective buffer registration.
  // Called lazily on first register_local_buffer() or get_device_window().
  // This is COLLECTIVE (all ranks must call).
  void initLocalComm();

  // Initialize the NCCL orig window for device API (GIN support).
  // Called during tensor_register() if device API is enabled.
  // This is COLLECTIVE.
  void initNcclOrigWindow(void* ptr, size_t size);

  ncclComm_t nccl_comm_{};
  std::shared_ptr<TorchCommNCCLX> torch_comm_;

  // CTRAN window for host-side RMA operations (put, signal, wait_signal)
  NcclxWindow win_{nullptr};

  // ==========================================================================
  // Device API state
  // ==========================================================================

  // Local communicator for non-collective buffer registration.
  // Created via ncclCommSplit(nccl_comm_, myRank, 0, &local_comm_).
  // This is a 1-rank comm that shares ginState with parent.
  ncclComm_t local_comm_{nullptr};
  bool local_comm_initialized_{false};

  // NCCL orig window for device-side operations (has GIN support).
  // Registered via the NCCL orig path (not CTRAN).
  NcclxWindow nccl_orig_win_{nullptr};

  // NCCL device communicator (contains GIN state)
  ncclDevComm nccl_dev_comm_{};
  bool nccl_dev_comm_initialized_{false};

  // Device window handle (allocated on GPU, valid for device-side operations)
  device::TorchCommDeviceWindow* device_window_{nullptr};

  // Device comm handle (allocated on GPU)
  device::TorchCommDeviceComm_* device_comm_{nullptr};

  // Device-side ncclDevComm pointer (backend_state_ points here)
  // Contains GIN signals/counters allocated via ncclDevCommCreate
  ncclDevComm* nccl_dev_comm_device_{nullptr};

  // Registered local buffers for device-side put operations
  std::vector<device::RegisteredBuffer> registered_local_buffers_;

  // NCCL API abstraction
  NcclxApi* nccl_api_;
  // CUDA API abstraction
  CudaApi* cuda_api_;
  // Torchcomm device
  at::Device comm_device_{at::kCUDA};
};

} // namespace torch::comms
