// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy
#include <memory>
#include "comms/torchcomms/TorchCommWindow.hpp"
#include "comms/torchcomms/device/CudaApi.hpp"
#include "comms/torchcomms/ncclx/NcclxApi.hpp"
#include "comms/torchcomms/ncclx/TorchWorkNCCLX.hpp"

namespace torch {
namespace comms {

// Forward declaration
class TorchCommNCCLX;

class TorchCommWindowNCCLX : public TorchCommWindow {
 public:
  TorchCommWindowNCCLX() = delete;
  explicit TorchCommWindowNCCLX(
      ncclComm_t ncclComm,
      std::shared_ptr<TorchCommNCCLX> torchComm,
      at::Device device);
  ~TorchCommWindowNCCLX() noexcept override;

  // We delete the copy constructor and assignment operator to prevent 2 work
  // objects sharing the underlying collective work events.
  TorchCommWindowNCCLX(const TorchCommWindowNCCLX& other) = delete;
  TorchCommWindowNCCLX& operator=(const TorchCommWindowNCCLX& other) = delete;
  // Delete the move assignment operdator to prevent accidentally stomping over
  // events if the work is in progress.
  TorchCommWindowNCCLX& operator=(TorchCommWindowNCCLX&& other) noexcept =
      delete;

  c10::intrusive_ptr<TorchWork> put(
      const at::Tensor& data,
      int dstRank,
      size_t targetDisp,
      bool asyncOp) override;
  c10::intrusive_ptr<TorchWork> signal(
      size_t signalDisp,
      uint64_t signalVal,
      int dstRank,
      bool asyncOp) override;
  c10::intrusive_ptr<TorchWork> waitSignal(
      size_t signalDisp,
      uint64_t cmpVal,
      SignalCmpOp cmpOp,
      bool asyncOp) override;
  at::Tensor getTensor(
      int rank,
      at::IntArrayRef sizes,
      at::ScalarType dtype,
      int64_t storageOffset) override;

 protected:
  void allocate(
      const size_t window_size,
      bool cpu_buf = false,
      const size_t signal_size = 256) override;
  friend class TorchCommNCCLX;

 private:
  // internal util functions
  void checkRequestSizeAndThrow(size_t input_size);
  void checkDeviceAndThrow(const at::Tensor& tensor);
  void checkCommAndThrow();
  void checkWindowAndThrow();
  void checkEventAndThrow();
  void checkOpStreamAndThrow();
  void checkWaitStreamAndThrow();

  ncclComm_t nccl_comm_{};
  std::shared_ptr<TorchCommNCCLX> torch_comm_;
  NcclxWindow win_{nullptr};
  at::Device device_{at::kCUDA};
  cudaStream_t op_stream_{nullptr}, wait_stream_{nullptr};
  cudaEvent_t rma_event_{nullptr};

  // NCCL API abstraction
  NcclxApi* nccl_api_;
  // CUDA API abstraction
  CudaApi* cuda_api_;
  size_t signal_size_{256};
};

} // namespace comms
} // namespace torch
