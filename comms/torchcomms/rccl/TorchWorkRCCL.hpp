// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <chrono>
#include <memory>
#include <optional>
#include <string_view>

#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include <hip_runtime.h> // @manual=third-party//cuda:cuda-lazy
#include <vector>
#include "comms/torchcomms/TorchWork.hpp" // @manual=//comms/torchcomms:torchcomms-headers-cpp

namespace torch {
namespace comms {

// Forward declaration
class TorchCommRCCL;

class TorchWorkRCCL : public TorchWork {
 public:
  TorchWorkRCCL(
      std::shared_ptr<TorchCommRCCL> comm,
      hipStream_t stream,
      std::chrono::milliseconds timeout_ms,
      const std::vector<at::Tensor>& inputTensors);
  TorchWorkRCCL(
      std::shared_ptr<TorchCommRCCL> comm,
      hipStream_t stream,
      std::chrono::milliseconds timeout_ms,
      const at::Tensor& inputTensor);
  ~TorchWorkRCCL() override;

  // We delete the copy constructor and assignment operator to prevent 2 work
  // objects sharing the underlying collective work events.
  TorchWorkRCCL(const TorchWorkRCCL& other) = delete;
  TorchWorkRCCL& operator=(const TorchWorkRCCL& other) = delete;
  // Delete the move assignment operator to prevent accidentally stomping over
  // events if the work is in progress.
  TorchWorkRCCL& operator=(TorchWorkRCCL&& other) noexcept = delete;
  TorchWorkRCCL(TorchWorkRCCL&&) = delete;

  // Override virtual functions from TorchWork
  void wait() override;

  // Check the status of the work object
  WorkStatus checkStatus();

  std::chrono::milliseconds getTimeout() {
    return timeout_ms_;
  }

 protected:
  void recordStart(std::string_view coll_name);
  void recordEnd();

  friend class TorchCommRCCL;

 private:
  void recordFunctionStart(std::string_view coll_name);

  std::vector<at::Tensor> inputTensors_;
  at::Tensor inputTensor_;

  std::shared_ptr<TorchCommRCCL> comm_;
  hipEvent_t start_event_;
  hipEvent_t end_event_;
  hipStream_t stream_; // stream is not owned by this class

  std::chrono::milliseconds timeout_ms_;

  std::optional<std::chrono::steady_clock::time_point> start_completed_time_;

  std::optional<at::RecordFunction> recordFunction_;
};

} // namespace comms
} // namespace torch
