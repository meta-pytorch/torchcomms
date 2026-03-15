#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <unordered_map>

#include <ATen/ATen.h>
#include <ATen/record_function.h>
#include "comms/torchcomms/TorchWork.hpp"
#include "comms/torchcomms/device/xpu/XpuApi.hpp"

namespace torch::comms {

// Forward declaration
class TorchCommXCCL;

class TorchWorkXCCL : public TorchWork {
 public:
  TorchWorkXCCL(
      std::shared_ptr<TorchCommXCCL> comm,
      xpuStream_t stream,
      std::chrono::milliseconds timeout_ms,
      const std::vector<at::Tensor>& inputTensors);
  TorchWorkXCCL(
      std::shared_ptr<TorchCommXCCL> comm,
      xpuStream_t stream,
      std::chrono::milliseconds timeout_ms,
      const at::Tensor& inputTensor);
  ~TorchWorkXCCL() override;

  // Delete copy and move operations
  TorchWorkXCCL(const TorchWorkXCCL&) = delete;
  TorchWorkXCCL(TorchWorkXCCL&&) = delete;
  TorchWorkXCCL& operator=(const TorchWorkXCCL&) = delete;
  TorchWorkXCCL& operator=(TorchWorkXCCL&&) = delete;

  // Override virtual functions from TorchWork
  void wait() override;
  std::chrono::milliseconds getTimeout() const override {
    return timeout_ms_;
  }

 protected:
  void recordStart(std::string_view coll_name);
  void recordEnd();

  friend class TorchCommXCCL;
  friend class TorchWorkXCCLQueue;

 private:
  // Check the status of the work object
  WorkStatus checkStatus();

  void recordFunctionStart(std::string_view coll_name);

  std::vector<at::Tensor> inputTensors_;
  at::Tensor inputTensor_;

  std::shared_ptr<TorchCommXCCL> comm_;
  xpuEvent_t start_event_;
  xpuEvent_t end_event_;
  xpuStream_t stream_; // stream is not owned by this class

  std::chrono::milliseconds timeout_ms_;

  std::optional<std::chrono::steady_clock::time_point> start_completed_time_;

  std::optional<at::RecordFunction> recordFunction_;
};

class TorchWorkXCCLQueue {
 public:
  TorchWorkXCCLQueue() = default;
  ~TorchWorkXCCLQueue() = default;

  TorchWork::WorkStatus garbageCollect();
  // Finalize function can only be called from the main thread
  TorchWork::WorkStatus finalize();
  void enqueueWork(c10::intrusive_ptr<TorchWorkXCCL> work, xpuStream_t stream);
  bool empty();

 private:
  TorchWork::WorkStatus garbageCollectLocked();
  std::unordered_map<xpuStream_t, std::queue<c10::intrusive_ptr<TorchWorkXCCL>>>
      stream_work_queues_;
  std::mutex work_queues_mutex_;
};

} // namespace torch::comms
