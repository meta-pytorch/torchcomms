// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/ncclx/TorchWorkNCCLX.hpp"
#include <ATen/ThreadLocalState.h>
#include <ATen/cuda/CUDAContext.h>
#include "TorchCommNCCLX.hpp"
#include "comms/torchcomms/TorchCommLogging.hpp"
#include "comms/torchcomms/TorchCommTracing.hpp"

namespace torch {
namespace comms {

TorchWorkNCCLX::TorchWorkNCCLX(
    std::shared_ptr<TorchCommNCCLX> comm,
    cudaStream_t stream,
    std::chrono::milliseconds timeout_ms,
    const std::vector<at::Tensor>& inputTensors)
    : inputTensors_(inputTensors),
      comm_(std::move(comm)),
      stream_(stream),
      timeout_ms_(timeout_ms),
      state_(WorkStatus::NOT_STARTED) {
  // If not in graph capture mode, create the events for start and end
  // recording
  start_event_ = comm_->getEvent();
  end_event_ = comm_->getEvent();

  // Events will be recorded around the actual NCCL operations
}

TorchWorkNCCLX::~TorchWorkNCCLX() {
  if (!comm_) {
    return;
  }
  // If not in graph capture mode, return the events to the pool
  comm_->returnEvent(start_event_);
  comm_->returnEvent(end_event_);
}

void TorchWorkNCCLX::recordFunctionStart() {
  recordFunction_.emplace(at::RecordScope::USER_SCOPE);
  if (!recordFunction_->isActive()) {
    return;
  }
  // Passing input tensor to recordFunction allows for shape information in
  // profiling output.
  std::vector<c10::IValue> inputs;
  inputs.reserve(inputTensors_.size());
  for (const auto& tensor : inputTensors_) {
    inputs.emplace_back(tensor);
  }
  // TODO: pass the collective name to be added
  recordFunction_->before(
      "torchcomms:work",
      c10::ArrayRef<const c10::IValue>(inputs.data(), inputs.size()));
}

void TorchWorkNCCLX::recordStart() {
  recordFunctionStart();

  CUDA_CHECK(
      comm_->getCudaApi(),
      comm_->getCudaApi()->eventRecord(start_event_, stream_),
      "Failed to record start event");
}

void TorchWorkNCCLX::recordEnd() {
  CUDA_CHECK(
      comm_->getCudaApi(),
      comm_->getCudaApi()->eventRecord(end_event_, stream_),
      "Failed to record end event");

  if (recordFunction_ && recordFunction_->isActive()) {
    recordFunction_->end();
  }
}

bool TorchWorkNCCLX::isCompleted() {
  return state_ == WorkStatus::COMPLETED;
}

TorchWorkNCCLX::WorkStatus TorchWorkNCCLX::checkStatus() {
  // If already marked as completed, return COMPLETED
  if (state_ == WorkStatus::COMPLETED || state_ == WorkStatus::ERROR ||
      state_ == WorkStatus::TIMEDOUT) {
    return state_;
  }

  // Step 1: If start_completed_time_ doesn't have a value yet, query the start
  // event
  if (!start_completed_time_.has_value()) {
    cudaError_t start_status = comm_->getCudaApi()->eventQuery(start_event_);

    if (start_status == cudaSuccess) {
      // Start event has completed, store the current time
      start_completed_time_ = std::chrono::steady_clock::now();
      state_ = WorkStatus::INPROGRESS;
    } else if (
        start_status != cudaErrorNotReady &&
        start_status != cudaErrorStreamCaptureUnsupported) {
      // Some other error occurred with the start event
      TC_LOG(ERROR, comm_.get())
          << "CUDA error during start event query: "
          << comm_->getCudaApi()->getErrorString(start_status) << " ("
          << start_status << ")";
      state_ = WorkStatus::ERROR;
    }
  }
  if (state_ == WorkStatus::NOT_STARTED || state_ == WorkStatus::ERROR) {
    return state_;
  }

  // Step 2: If we get here, start event has completed, so query the end event
  cudaError_t end_status = comm_->getCudaApi()->eventQuery(end_event_);

  if (end_status == cudaSuccess) {
    // End event has completed, mark the work as completed
    state_ = WorkStatus::COMPLETED;
  } else if (end_status == cudaErrorNotReady) {
    // End event has not completed yet, check for timeout
    auto current_time = std::chrono::steady_clock::now();
    auto elapsed_milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - start_completed_time_.value());

    // Check if the operation has timed out
    if (elapsed_milliseconds > timeout_ms_) {
      // Operation has timed out
      state_ = WorkStatus::TIMEDOUT;
    }
  } else if (end_status != cudaErrorStreamCaptureUnsupported) {
    // Some other error occurred with the end event
    TC_LOG(ERROR, comm_.get())
        << "CUDA error during end event query: "
        << comm_->getCudaApi()->getErrorString(end_status) << " (" << end_status
        << ")";
    state_ = WorkStatus::ERROR;
  }
  return state_;
}

void TorchWorkNCCLX::wait() {
  // If already completed, return immediately
  WorkStatus local_state = state_;
  if (local_state == WorkStatus::COMPLETED ||
      local_state == WorkStatus::ERROR || local_state == WorkStatus::TIMEDOUT) {
    return;
  }

  TorchCommTracingGuard g(
      std::string(comm_->getCommName()),
      comm_->getSize(),
      "wait",
      comm_->getRank());

  // Get the current stream using the device from the comm object
  cudaStream_t current_stream =
      comm_->getCudaApi()->getCurrentCUDAStream(comm_->device_.index());

  // Add a dependency from the work's stream to the current stream
  // This makes the current stream wait for the end_event_ recorded on the
  // work's stream
  CUDA_CHECK(
      comm_->getCudaApi(),
      comm_->getCudaApi()->streamWaitEvent(current_stream, end_event_, 0),
      "Failed to make stream wait for event");
}
} // namespace comms
} // namespace torch
