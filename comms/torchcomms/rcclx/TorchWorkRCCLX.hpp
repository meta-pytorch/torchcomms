// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <chrono>
#include <memory>
#include <optional>

#include <ATen/ATen.h>
#include <hip_runtime.h> // @manual=third-party//cuda:cuda-lazy
#include <vector>
#include "comms/torchcomms/TorchCommTracing.hpp" // @manual=//comms/torchcomms:torchcomms-headers-cpp
#include "comms/torchcomms/TorchWork.hpp" // @manual=//comms/torchcomms:torchcomms-headers-cpp

namespace torch {
namespace comms {

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

  // Move constructor
  TorchWorkRCCLX(TorchWorkRCCLX&& other) noexcept;

  // Override virtual functions from TorchWork
  void wait() override;

  // Check the status of the work object
  WorkStatus checkStatus();

 protected:
  void recordStart(const std::string& coll_name);
  void recordEnd();

  friend class TorchCommRCCLX;

 private:
  void recordFunctionStart(const std::string& coll_name);
  std::chrono::milliseconds getTimeout() {
    return timeout_ms_;
  }
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

} // namespace comms
} // namespace torch
