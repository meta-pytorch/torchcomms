// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <chrono>
#include <memory>
#include <mutex>
#include <queue>
#include <unordered_map>

#include <ATen/ATen.h>
#include "comms/mccl/mccl.h"
#include "comms/torchcomms/TorchWork.hpp"
#include "comms/torchcomms/device/cuda/CudaApi.hpp"
#include "comms/torchcomms/utils/TracingGuard.hpp"

namespace torch::comms {

// Forward declaration
class TorchCommMCCL;

class TorchWorkMCCL : public TorchWork {
 public:
  using mcclWork_t = ::mccl::IWorkHandle;

  TorchWorkMCCL(
      std::shared_ptr<TorchCommMCCL> comm,
      cudaStream_t stream,
      const std::vector<at::Tensor>& inputTensors,
      const std::vector<at::Tensor>& outputTensors,
      std::unique_ptr<mcclWork_t> workHandle,
      std::chrono::milliseconds timeout);

  TorchWorkMCCL(
      std::shared_ptr<TorchCommMCCL> comm,
      cudaStream_t stream,
      const at::Tensor& inputTensor,
      const at::Tensor& outputTensor,
      std::unique_ptr<mcclWork_t> workHandle,
      std::chrono::milliseconds timeout);

  ~TorchWorkMCCL() override = default;

  TorchWorkMCCL(const TorchWorkMCCL& other) = delete;
  TorchWorkMCCL& operator=(const TorchWorkMCCL& other) = delete;
  TorchWorkMCCL& operator=(TorchWorkMCCL&& other) noexcept = delete;

  // Move constructor
  TorchWorkMCCL(TorchWorkMCCL&& other) noexcept;

  // Wait for the operation to complete.
  // For GPU operations, this inserts a stream dependency so that subsequent
  // operations on the caller's CUDA stream will wait, but does NOT block
  // the CPU thread. For CPU operations, this blocks the CPU thread
  // (equivalent to waitCPU()).
  void wait() override;

  // Block the CPU thread until the operation completes.
  // This is a synchronous wait; the calling thread will not return
  // until the MCCL work handle reports completion or error.
  void waitCPU();

  // Fault Tolerance API; blocks CPU thread until completion.
  // Equivalent to waitCPU().
  // Implements the TorchWork::waitBlocking() contract.
  void waitBlocking() override;

  // Get the MCCL result if the operation has completed.
  // Returns std::nullopt if the operation is still in progress.
  std::optional<::mccl::Result> getResult();

  cudaStream_t getCurrentCUDAStream();

  // Poll the operation status without blocking.
  // Returns the current WorkStatus. If the operation has exceeded its
  // timeout, returns TIMEDOUT.
  WorkStatus checkStatus();

  void setCpuWork(bool v) {
    isCpuWork_ = v;
  }
  bool isCpuWork() const {
    return isCpuWork_;
  }

  // Force a still-in-flight work to a terminal TIMEDOUT status so its end hook
  // (clog "E") fires. Used when the comm is torn down with a collective still
  // pending (e.g. PAFT fault recovery), where the op will never complete and
  // would otherwise be destroyed without ever reaching a terminal state.
  // TIMEDOUT (rather than ERROR) matches MCCL's fault model: a torn-down
  // in-flight collective was wedged on a peer that never made progress.
  void closeIncompleteOnTeardown() {
    auto s = status();
    if (s == WorkStatus::NOT_STARTED || s == WorkStatus::INPROGRESS) {
      setStatus(WorkStatus::TIMEDOUT);
    }
  }

 protected:
  friend class TorchCommMCCL;

 private:
  // Copied, not reached through the comm: workq_ is a member of
  // TorchCommMCCL, so holding a reference to it cycles.
  std::shared_ptr<CudaApi> cudaApi_;
  int cudaDeviceId_{-1};
  // Cached at construction to avoid checked accessors that throw
  // post-reconfigure (in the case of a failed reconfigure).
  TracingGuardInfo infoOnCreation_;
  cudaStream_t stream_;
  std::vector<at::Tensor> inputTensors_;
  std::vector<at::Tensor> outputTensors_;
  // Sometimes we only have one input/output tensor
  at::Tensor inputTensor_;
  at::Tensor outputTensor_;
  std::unique_ptr<mcclWork_t> mcclWork_;

  std::chrono::milliseconds timeout_;
  std::optional<std::chrono::steady_clock::time_point> start_time_;
  bool isCpuWork_{false};

  void releaseTensors();

  TorchWorkMCCL(
      std::shared_ptr<TorchCommMCCL> comm,
      cudaStream_t stream,
      std::unique_ptr<mcclWork_t> workHandle,
      std::chrono::milliseconds timeout);
};

class TorchWorkMCCLQueue {
 public:
  TorchWorkMCCLQueue() = default;
  ~TorchWorkMCCLQueue();

  // Non-copyable, non-movable (holds a std::mutex). Declared explicitly because
  // the user-declared destructor otherwise suppresses the implicit moves
  // (rule of 5 / cppcoreguidelines-special-member-functions).
  TorchWorkMCCLQueue(const TorchWorkMCCLQueue&) = delete;
  TorchWorkMCCLQueue& operator=(const TorchWorkMCCLQueue&) = delete;
  TorchWorkMCCLQueue(TorchWorkMCCLQueue&&) = delete;
  TorchWorkMCCLQueue& operator=(TorchWorkMCCLQueue&&) = delete;

  TorchWorkMCCL::WorkStatus garbageCollect();

  // Finalize function can only be called from the main thread
  //
  // This function will block the main thread until all the work is
  // completed (i.e., all work items transition out of the NOT_STARTED
  // and IN_PROGRESS states).
  TorchWorkMCCL::WorkStatus finalize();
  void enqueueWork(c10::intrusive_ptr<TorchWorkMCCL> work, cudaStream_t stream);

 private:
  TorchWorkMCCL::WorkStatus garbageCollectLocked();
  std::
      unordered_map<cudaStream_t, std::queue<c10::intrusive_ptr<TorchWorkMCCL>>>
          stream_work_queues_;
  std::mutex work_queues_mutex_;
};

} // namespace torch::comms
