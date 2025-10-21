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
#include "comms/torchcomms/TorchCommTracing.hpp"
#include "comms/torchcomms/TorchWork.hpp"

namespace torch {
namespace comms {

// Forward declaration
class TorchCommNCCL;
class TorchCommWindowNCCL;

// Forward declaration for test class
namespace test {
class TorchCommNCCLTest;
}

class TorchWorkNCCL : public TorchWork {
 public:
  // Status of a work object
  enum class WorkStatus {
    NOT_STARTED, // Work has not started yet
    INPROGRESS, // Work is still in progress,
    COMPLETED, // Work has completed successfully
    TIMEDOUT, // Work has timed out
    ERROR // Work has encountered an error
  };

  TorchWorkNCCL(
      std::shared_ptr<TorchCommNCCL> comm,
      cudaStream_t stream,
      std::chrono::milliseconds timeout_ms,
      const std::vector<at::Tensor>& inputTensors,
      std::shared_ptr<TorchCommTracing> tracing);
  ~TorchWorkNCCL() override;

  // Delete copy and move operations
  TorchWorkNCCL(const TorchWorkNCCL&) = delete;
  TorchWorkNCCL(TorchWorkNCCL&&) = delete;
  TorchWorkNCCL& operator=(const TorchWorkNCCL&) = delete;
  TorchWorkNCCL& operator=(TorchWorkNCCL&&) = delete;

  // Override virtual functions from TorchWork
  bool isCompleted() override;
  void wait() override;

 protected:
  void recordStart();
  void recordEnd();

  friend class TorchCommNCCL;
  friend class TorchWorkNCCLQueue;

 private:
  // Check the status of the work object
  WorkStatus checkStatus();

  std::chrono::milliseconds getTimeout() const {
    return timeout_ms_;
  }
  std::vector<at::Tensor> inputTensors_;

  std::shared_ptr<TorchCommNCCL> comm_;
  cudaEvent_t start_event_;
  cudaEvent_t end_event_;
  cudaStream_t stream_; // stream is not owned by this class

  std::chrono::milliseconds timeout_ms_;

  // state machine variables. TODO: convert to state machine later
  std::atomic<WorkStatus> state_;

  std::optional<std::chrono::steady_clock::time_point> start_completed_time_;
  std::shared_ptr<TorchCommTracing> tracing_;
};

class TorchWorkNCCLQueue {
 public:
  TorchWorkNCCLQueue() = default;
  ~TorchWorkNCCLQueue() = default;

  TorchWorkNCCL::WorkStatus garbageCollect(bool isMainThread);
  // Finalize function can only be called from the main thread
  TorchWorkNCCL::WorkStatus finalize();
  void enqueueWork(std::shared_ptr<TorchWorkNCCL> work, cudaStream_t stream);

 private:
  std::unordered_map<cudaStream_t, std::queue<std::shared_ptr<TorchWorkNCCL>>>
      stream_work_queues_;
  std::vector<std::shared_ptr<TorchWorkNCCL>> completed_work_queue_;
  std::recursive_mutex work_queues_mutex_;
};

} // namespace comms
} // namespace torch
