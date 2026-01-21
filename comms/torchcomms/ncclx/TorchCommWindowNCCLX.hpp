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
      std::shared_ptr<TorchCommNCCLX> torchComm);
  ~TorchCommWindowNCCLX() noexcept override;

  // We delete the copy constructor and assignment operator to prevent 2 work
  // objects sharing the underlying collective work events.
  TorchCommWindowNCCLX(const TorchCommWindowNCCLX& other) = delete;
  TorchCommWindowNCCLX& operator=(const TorchCommWindowNCCLX& other) = delete;
  // Delete the move constructor and assignment operator to prevent accidentally
  // stomping over events if the work is in progress.
  TorchCommWindowNCCLX(TorchCommWindowNCCLX&& other) noexcept = delete;
  TorchCommWindowNCCLX& operator=(TorchCommWindowNCCLX&& other) noexcept =
      delete;

  void tensor_register(const at::Tensor& tensor) override;
  void tensor_deregister() override;

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

 protected:
  friend class TorchCommNCCLX;

 private:
  // internal util functions
  void checkRequestSizeAndThrow(size_t input_size) const;
  void checkDeviceAndThrow(const at::Tensor& tensor) const;
  void checkCommAndThrow() const;
  void checkWindowAndThrow() const;

  ncclComm_t nccl_comm_{};
  std::shared_ptr<TorchCommNCCLX> torch_comm_;
  NcclxWindow win_{nullptr};

  // NCCL API abstraction
  NcclxApi* nccl_api_;
  // CUDA API abstraction
  CudaApi* cuda_api_;
  // Torchcomm device
  at::Device comm_device_{at::kCUDA};
};

} // namespace comms
} // namespace torch
