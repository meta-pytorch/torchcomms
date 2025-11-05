// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <comms/torchcomms/TorchCommOptions.hpp>
#include <comms/torchcomms/TorchCommTypes.hpp>
namespace torch {
namespace comms {

// Forward declaration
class TorchWork;

class TorchCommWindow {
 public:
  TorchCommWindow() = default;
  virtual ~TorchCommWindow() = default;

  // Disable copy and move semantics
  TorchCommWindow(const TorchCommWindow&) = delete;
  TorchCommWindow& operator=(const TorchCommWindow&) = delete;
  TorchCommWindow(TorchCommWindow&&) = delete;
  TorchCommWindow& operator=(TorchCommWindow&&) = delete;

  virtual void allocate(
      const size_t window_size,
      bool cpu_buf = false,
      const size_t signal_size = 256) = 0;
  virtual c10::intrusive_ptr<TorchWork>
  put(const at::Tensor& data, int dstRank, size_t targetDisp, bool asyncOp) = 0;
  virtual at::Tensor getTensor(
      int rank,
      at::IntArrayRef sizes,
      at::ScalarType dtype,
      int64_t storageOffset) = 0;
  virtual c10::intrusive_ptr<TorchWork>
  signal(size_t signalDisp, uint64_t signalVal, int dstRank, bool asyncOp) = 0;
  virtual c10::intrusive_ptr<TorchWork> waitSignal(
      size_t signalDisp,
      uint64_t cmpVal,
      SignalCmpOp cmpOp,
      bool asyncOp) = 0;

  size_t get_size() const {
    return win_size_;
  }

 protected:
  void* base_ptr_{};
  // device_: The device where the window is allocated.
  //  The device where the window is allocated may differ from the device used
  //  by the communicator. For example, the window could be allocated on the CPU
  //  while the communicator operates on the GPU. However, if both are using the
  //  GPU, they should reside on the same device.
  bool cpuBuf_{false};
  size_t win_size_{0};
};

} // namespace comms
} // namespace torch
