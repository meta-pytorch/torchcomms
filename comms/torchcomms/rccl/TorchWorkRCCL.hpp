// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <atomic>
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
class TorchCommRCCL;

class TorchWorkRCCL : public TorchWork {
 public:
  // Status of a work object
  enum class WorkStatus {
    NOT_STARTED, // Work has not started yet
    INPROGRESS, // Work is still in progress,
    COMPLETED, // Work has completed successfully
    TIMEDOUT, // Work has timed out
    ERROR // Work has encountered an error
  };

  TorchWorkRCCL(
      std::shared_ptr<TorchCommRCCL> comm,
      hipStream_t stream,
      std::chrono::milliseconds timeout_ms,
      const std::vector<at::Tensor>& inputTensors,
      std::shared_ptr<TorchCommTracing> tracing);
  ~TorchWorkRCCL() override;

  // We delete the copy constructor and assignment operator to prevent 2 work
  // objects sharing the underlying collective work events.
  TorchWorkRCCL(const TorchWorkRCCL& other) = delete;
  TorchWorkRCCL& operator=(const TorchWorkRCCL& other) = delete;
  // Delete the move assignment operator to prevent accidentally stomping over
  // events if the work is in progress.
  TorchWorkRCCL& operator=(TorchWorkRCCL&& other) noexcept = delete;

  // Move constructor
  TorchWorkRCCL(TorchWorkRCCL&& other) noexcept;

  // Override virtual functions from TorchWork
  bool isCompleted() override;
  void wait() override;

  // Check the status of the work object
  WorkStatus checkStatus();

  std::chrono::milliseconds getTimeout() {
    return timeout_ms_;
  }

 protected:
  void recordStart();
  void recordEnd();

  friend class TorchCommRCCL;

 private:
  std::vector<at::Tensor> inputTensors_;

  std::shared_ptr<TorchCommRCCL> comm_;
  hipEvent_t start_event_;
  hipEvent_t end_event_;
  hipStream_t stream_; // stream is not owned by this class

  std::chrono::milliseconds timeout_ms_;

  // state machine variables. TODO: convert to state machine later
  std::atomic<WorkStatus> state_;

  std::optional<std::chrono::steady_clock::time_point> start_completed_time_;
  std::shared_ptr<TorchCommTracing> tracing_;
};

} // namespace comms
} // namespace torch
