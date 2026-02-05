// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <chrono>
#include <memory>
#include <optional>
#include <string_view>

#include <ATen/ATen.h>
#include <hip_runtime.h> // @manual=third-party//cuda:cuda-lazy
#include <vector>
#include "comms/torchcomms/TorchCommTracing.hpp" // @manual=//comms/torchcomms:torchcomms-headers-cpp
#include "comms/torchcomms/TorchWork.hpp" // @manual=//comms/torchcomms:torchcomms-headers-cpp

namespace torch::comms {

// Forward declaration
class TorchCommRCCLX;

class TorchWorkRCCLX : public TorchWork {
 public:
  TorchWorkRCCLX(
      std::shared_ptr<TorchCommRCCLX> comm,
      hipStream_t stream,
      std::chrono::milliseconds timeout_ms,
      const std::vector<at::Tensor>& inputTensors);

  TorchWorkRCCLX(
      std::shared_ptr<TorchCommRCCLX> comm,
      hipStream_t stream,
      std::chrono::milliseconds timeout_ms,
      const at::Tensor& inputTensor);
  ~TorchWorkRCCLX() override;

  // We delete the copy constructor and assignment operator to prevent 2 work
  // objects sharing the underlying collective work events.
  TorchWorkRCCLX(const TorchWorkRCCLX& other) = delete;
  TorchWorkRCCLX& operator=(const TorchWorkRCCLX& other) = delete;
  // Delete the move assignment operator to prevent accidentally stomping over
  // events if the work is in progress.
  TorchWorkRCCLX& operator=(TorchWorkRCCLX&& other) noexcept = delete;
  TorchWorkRCCLX(TorchWorkRCCLX&&) = delete;

  // Override virtual functions from TorchWork
  void wait() override;

  // Check the status of the work object
  WorkStatus checkStatus();
  std::chrono::milliseconds getTimeout() const override {
    return timeout_ms_;
  }

  // Test-only accessors to verify tensor storage behavior
  // Returns true if any tensors are stored in this work object
  bool hasTensorsStored() const {
    return !inputTensors_.empty() || inputTensor_.defined();
  }

 protected:
  void recordStart(std::string_view coll_name);
  void recordEnd();

  friend class TorchCommRCCLX;

 private:
  void recordFunctionStart(std::string_view coll_name);

  // Tensors supplied might either be a vector of tensors,
  // or a single tensor. In case it is a single tensor, we
  // can avoid allocating space for a vector of tensors.
  std::vector<at::Tensor> inputTensors_;
  at::Tensor inputTensor_;

  std::shared_ptr<TorchCommRCCLX> comm_;
  hipEvent_t start_event_;
  hipEvent_t end_event_;
  hipStream_t stream_; // stream is not owned by this class

  std::chrono::milliseconds timeout_ms_;

  std::optional<std::chrono::steady_clock::time_point> start_completed_time_;
  std::optional<at::RecordFunction> recordFunction_;
};

} // namespace torch::comms
