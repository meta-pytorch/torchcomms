#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <unordered_map>

#include "comms/torchcomms/TorchCommTracing.hpp"
#include "comms/torchcomms/TorchWork.hpp"
#include "comms/torchcomms/device/XpuApi.hpp"
#include <ATen/ATen.h>

namespace torch {
namespace comms {

// Forward declaration
class TorchCommXCCL;

class TorchWorkXCCL : public TorchWork {
public:
  // Status of a work object
  enum class WorkStatus {
    NOT_STARTED, // Work has not started yet
    INPROGRESS,  // Work is still in progress,
    COMPLETED,   // Work has completed successfully
    TIMEDOUT,    // Work has timed out
    ERROR        // Work has encountered an error
  };

  TorchWorkXCCL(std::shared_ptr<TorchCommXCCL> comm, xpuStream_t stream,
                std::chrono::milliseconds timeout_ms,
                const std::vector<at::Tensor> &inputTensors,
                std::shared_ptr<TorchCommTracing> tracing);
  ~TorchWorkXCCL() override;

  // Delete copy and move operations
  TorchWorkXCCL(const TorchWorkXCCL &) = delete;
  TorchWorkXCCL(TorchWorkXCCL &&) = delete;
  TorchWorkXCCL &operator=(const TorchWorkXCCL &) = delete;
  TorchWorkXCCL &operator=(TorchWorkXCCL &&) = delete;

  // Override virtual functions from TorchWork
  bool isCompleted() override;
  void wait() override;

protected:
  void recordStart();
  void recordEnd();

  friend class TorchCommXCCL;
  friend class TorchWorkXCCLQueue;

private:
  // Check the status of the work object
  WorkStatus checkStatus();

  std::chrono::milliseconds getTimeout() const { return timeout_ms_; }
  std::vector<at::Tensor> inputTensors_;

  std::shared_ptr<TorchCommXCCL> comm_;
  xpuEvent_t start_event_;
  xpuEvent_t end_event_;
  xpuStream_t stream_; // stream is not owned by this class

  std::chrono::milliseconds timeout_ms_;

  // state machine variables. TODO: convert to state machine later
  std::atomic<WorkStatus> state_;

  std::optional<std::chrono::steady_clock::time_point> start_completed_time_;
  std::shared_ptr<TorchCommTracing> tracing_;
};

class TorchWorkXCCLQueue {
public:
  TorchWorkXCCLQueue() = default;
  ~TorchWorkXCCLQueue() = default;

  TorchWorkXCCL::WorkStatus garbageCollect(bool isMainThread);
  // Finalize function can only be called from the main thread
  TorchWorkXCCL::WorkStatus finalize();
  void enqueueWork(c10::intrusive_ptr<TorchWorkXCCL> work, xpuStream_t stream);

private:
  std::unordered_map<xpuStream_t, std::queue<c10::intrusive_ptr<TorchWorkXCCL>>>
      stream_work_queues_;
  std::vector<c10::intrusive_ptr<TorchWorkXCCL>> completed_work_queue_;
  std::recursive_mutex work_queues_mutex_;
};

} // namespace comms
} // namespace torch
