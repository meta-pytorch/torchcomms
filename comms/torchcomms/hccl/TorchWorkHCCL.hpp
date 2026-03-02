#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <unordered_map>

#include <ATen/ATen.h>
#include "comms/torchcomms/TorchCommTracing.hpp"
#include "comms/torchcomms/TorchWork.hpp"
#include "comms/torchcomms/device/npu/NpuApi.hpp"

namespace torch::comms {

// Forward declaration
class TorchCommHCCL;

class TorchWorkHCCL : public TorchWork {
 public:
  // Status of a work object
  enum class WorkStatus {
    NOT_STARTED, // Work has not started yet
    INPROGRESS, // Work is still in progress,
    COMPLETED, // Work has completed successfully
    TIMEDOUT, // Work has timed out
    ERROR // Work has encountered an error
  };

  TorchWorkHCCL(
      std::shared_ptr<TorchCommHCCL> comm,
      npuStream_t stream,
      std::chrono::milliseconds timeout_ms,
      const std::vector<at::Tensor>& inputTensors,
      std::shared_ptr<TorchCommTracing> tracing);
  ~TorchWorkHCCL() override;

  // Delete copy and move operations
  TorchWorkHCCL(const TorchWorkHCCL&) = delete;
  TorchWorkHCCL(TorchWorkHCCL&&) = delete;
  TorchWorkHCCL& operator=(const TorchWorkHCCL&) = delete;
  TorchWorkHCCL& operator=(TorchWorkHCCL&&) = delete;

  void wait() override;

 protected:
  void recordStart();
  void recordEnd();

  friend class TorchCommHCCL;
  friend class TorchWorkHCCLQueue;

 private:
  // Check the status of the work object
  WorkStatus checkStatus();

  std::chrono::milliseconds getTimeout() const {
    return timeout_ms_;
  }
  std::vector<at::Tensor> inputTensors_;

  std::shared_ptr<TorchCommHCCL> comm_;
  npuEvent_t start_event_;
  npuEvent_t end_event_;
  npuStream_t stream_; // stream is not owned by this class

  std::chrono::milliseconds timeout_ms_;

  // state machine variables. TODO: convert to state machine later
  std::atomic<WorkStatus> state_;

  std::optional<std::chrono::steady_clock::time_point> start_completed_time_;
  std::shared_ptr<TorchCommTracing> tracing_;
};

class TorchWorkHCCLQueue {
 public:
  TorchWorkHCCLQueue() = default;
  ~TorchWorkHCCLQueue() = default;

  TorchWorkHCCL::WorkStatus garbageCollect(bool isMainThread);
  // Finalize function can only be called from the main thread
  TorchWorkHCCL::WorkStatus finalize();
  void enqueueWork(c10::intrusive_ptr<TorchWorkHCCL> work, npuStream_t stream);

 private:
  std::unordered_map<npuStream_t, std::queue<c10::intrusive_ptr<TorchWorkHCCL>>>
      stream_work_queues_;
  std::vector<c10::intrusive_ptr<TorchWorkHCCL>> completed_work_queue_;
  std::recursive_mutex work_queues_mutex_;
};

} // namespace torch::comms
