// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string_view>
#include <unordered_map>

#include <ATen/ATen.h>
#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy
#include <vector>
#include "comms/torchcomms/TorchCommTracing.hpp"
#include "comms/torchcomms/TorchWork.hpp"
#include "comms/torchcomms/ncclx/TorchCommNCCLXPersistentRequest.hpp"

namespace torch::comms {

// Forward declaration
class TorchCommNCCLX;

// TorchCommWindowNCCLX is now a template - forward declare
// Note: The type alias TorchCommWindowNCCLXGin is defined in
// TorchCommWindowNCCLX.hpp
template <typename Backend>
class TorchCommWindowNCCLX;

// Forward declaration for test class
namespace test {
class TorchCommNCCLXTest;
}

class TorchWorkNCCLX : public TorchWork {
 public:
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
  void wait() override;
  std::chrono::milliseconds getTimeout() const override {
    return timeout_ms_;
  }

  // Set persistent request reference to keep it alive until work is freed
  void setPersistentRequest(
      at::intrusive_ptr<TorchCommNCCLXPersistentRequest> request) {
    persistent_request_ = std::move(request);
  }

 protected:
  void recordStart(std::string_view coll_name);
  void recordEnd();

  friend class TorchCommNCCLX;
  friend class GraphEventTracker;
  template <typename B>
  friend class TorchCommWindowNCCLX;
  friend class TorchWorkNCCLXQueue;
  friend class torch::comms::test::TorchCommNCCLXTest;

 private:
  // Check the status of the work object
  WorkStatus checkStatus();

  void recordFunctionStart(std::string_view coll_name);

  // Tensors supplied might either be a vector of tensors,
  // or a single tensor. In case it is a single tensor, we
  // can avoid allocating space for a vector of tensors.
  std::vector<at::Tensor> inputTensors_;
  at::Tensor inputTensor_;

  void initEvents();
  void releaseEvents();

  std::shared_ptr<TorchCommNCCLX> comm_;
  cudaEvent_t start_event_{};
  // Completion detection event. In both eager and graph modes, this event is
  // recorded after the NCCL operation completes. In eager mode, it is also
  // used as the join point for work.wait(). In graph mode, it is recorded
  // with cudaEventRecordExternal (host-queryable for watchdog timeout
  // detection) and ownership is transferred to GraphWorkEntry.
  cudaEvent_t end_event_{};
  // Stream synchronization event for graph mode only. Recorded with regular
  // cudaEventRecord to serve as a valid join point for work.wait()
  // (cudaStreamWaitEvent). nullptr in eager mode.
  //
  // In graph mode, all three events (start, end, sync) are ad-hoc created
  // (NOT from the event pool). start_event_ and end_event_ ownership is
  // transferred to GraphWorkEntry in enqueueWork(), which sets them to
  // nullptr. sync_event_ is destroyed in the work destructor.
  cudaEvent_t sync_event_{};
  cudaStream_t stream_; // stream is not owned by this class

  // Whether this work was created during CUDA graph capture. Controls
  // event lifecycle: in graph mode, all events are ad-hoc created;
  // in non-graph mode, start_event_ and end_event_ are from the pool.
  bool graph_capture_mode_{false};

  std::chrono::milliseconds timeout_ms_;

  std::optional<std::chrono::steady_clock::time_point> start_completed_time_;

  std::optional<at::RecordFunction> recordFunction_;

  // Reference to persistent request to keep it alive until work is freed
  at::intrusive_ptr<TorchCommNCCLXPersistentRequest> persistent_request_;
};

class TorchWorkNCCLXQueue {
 public:
  TorchWorkNCCLXQueue() = default;
  ~TorchWorkNCCLXQueue() = default;

  TorchWorkNCCLX::WorkStatus garbageCollect();
  // Finalize function can only be called from the main thread
  TorchWorkNCCLX::WorkStatus finalize();
  void enqueueWork(
      c10::intrusive_ptr<TorchWorkNCCLX> work,
      cudaStream_t stream);

 private:
  TorchWorkNCCLX::WorkStatus garbageCollectLocked();

  std::unordered_map<
      cudaStream_t,
      std::queue<c10::intrusive_ptr<TorchWorkNCCLX>>>
      stream_work_queues_;
  std::mutex work_queues_mutex_;

  friend class TorchWorkNCCLXQueueCommTest;
};

} // namespace torch::comms
