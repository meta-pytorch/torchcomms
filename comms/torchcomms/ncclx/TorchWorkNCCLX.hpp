// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <unordered_map>

#include <ATen/ATen.h>
#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy
#include <vector>
#include "comms/torchcomms/TorchCommTracing.hpp"
#include "comms/torchcomms/TorchWork.hpp"

namespace torch {
namespace comms {

// Forward declaration
class TorchCommNCCLX;
class TorchCommWindowNCCLX;

// Forward declaration for test class
namespace test {
class TorchCommNCCLXTest;
}

class TorchWorkNCCLX : public TorchWork {
 public:
  // Status of a work object
  enum class WorkStatus {
    NOT_STARTED, // Work has not started yet
    INPROGRESS, // Work is still in progress,
    COMPLETED, // Work has completed successfully
    TIMEDOUT, // Work has timed out
    ERROR // Work has encountered an error
  };

  TorchWorkNCCLX(
      std::shared_ptr<TorchCommNCCLX> comm,
      cudaStream_t stream,
      std::chrono::milliseconds timeout_ms,
      const std::vector<at::Tensor>& inputTensors);

  TorchWorkNCCLX(
      std::shared_ptr<TorchCommNCCLX> comm,
      cudaStream_t stream,
      std::chrono::milliseconds timeout_ms,
      const at::Tensor& inputTensor);

  ~TorchWorkNCCLX() override;

  // Delete copy and move operations
  TorchWorkNCCLX(const TorchWorkNCCLX&) = delete;
  TorchWorkNCCLX(TorchWorkNCCLX&&) = delete;
  TorchWorkNCCLX& operator=(const TorchWorkNCCLX&) = delete;
  TorchWorkNCCLX& operator=(TorchWorkNCCLX&&) = delete;

  // Override virtual functions from TorchWork
  bool isCompleted() override;
  void wait() override;

 protected:
  void recordStart();
  void recordEnd();

  friend class TorchCommNCCLX;
  friend class TorchCommWindowNCCLX;
  friend class TorchWorkNCCLXQueue;
  friend class torch::comms::test::TorchCommNCCLXTest;

 private:
  // Check the status of the work object
  WorkStatus checkStatus();

  void recordFunctionStart();

  std::chrono::milliseconds getTimeout() {
    return timeout_ms_;
  }

  // Tensors supplied might either be a vector of tensors,
  // or a single tensor. In case it is a single tensor, we
  // can avoid allocating space for a vector of tensors.
  std::vector<at::Tensor> inputTensors_;
  at::Tensor inputTensor_;

  std::shared_ptr<TorchCommNCCLX> comm_;
  cudaEvent_t start_event_;
  cudaEvent_t end_event_;
  cudaStream_t stream_; // stream is not owned by this class

  std::chrono::milliseconds timeout_ms_;

  // state machine variables. TODO: convert to state machine later
  std::atomic<WorkStatus> state_;

  std::optional<std::chrono::steady_clock::time_point> start_completed_time_;

  std::optional<at::RecordFunction> recordFunction_;
};

class TorchWorkNCCLXQueue {
 public:
  TorchWorkNCCLXQueue() = default;
  ~TorchWorkNCCLXQueue() = default;

  TorchWorkNCCLX::WorkStatus garbageCollect();
  // Finalize function can only be called from the main thread
  TorchWorkNCCLX::WorkStatus finalize();
  void enqueueWork(std::shared_ptr<TorchWorkNCCLX> work, cudaStream_t stream);

 private:
  std::unordered_map<cudaStream_t, std::queue<std::shared_ptr<TorchWorkNCCLX>>>
      stream_work_queues_;
  std::recursive_mutex work_queues_mutex_;

  friend class TorchWorkNCCLXQueueCommTest;
};

} // namespace comms
} // namespace torch
