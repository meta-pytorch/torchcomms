// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/rcclx/TorchWorkRCCLX.hpp"
#include <ATen/hip/HIPContext.h> // @manual
#include "comms/torchcomms/rcclx/TorchCommRCCLX.hpp"

namespace torch {
namespace comms {

TorchWorkRCCLX::TorchWorkRCCLX(
    std::shared_ptr<TorchCommRCCLX> comm,
    hipStream_t stream,
    std::chrono::milliseconds timeout_ms,
    const std::vector<at::Tensor>& inputTensors)
    : inputTensors_(inputTensors),
      comm_(std::move(comm)),
      stream_(stream),
      timeout_ms_(timeout_ms),
      state_(WorkStatus::NOT_STARTED) {
  start_event_ = comm_->getEvent();
  end_event_ = comm_->getEvent();

  // Events will be recorded around the actual RCCLX operations
}

TorchWorkRCCLX::TorchWorkRCCLX(TorchWorkRCCLX&& other) noexcept
    : inputTensors_(std::move(other.inputTensors_)),
      comm_(std::move(other.comm_)),
      start_event_(other.start_event_),
      end_event_(other.end_event_),
      stream_(other.stream_),
      timeout_ms_(other.timeout_ms_),
      state_(WorkStatus::NOT_STARTED), // Initialize with default value
      start_completed_time_(std::move(other.start_completed_time_)) {
  // Copy the atomic state using load/store
  state_.store(other.state_.load());

  // Transfer ownership of resources and reset the source object
  other.start_event_ = nullptr;
  other.end_event_ = nullptr;
  other.stream_ = nullptr;
  other.timeout_ms_ = std::chrono::milliseconds(0);
  other.state_.store(WorkStatus::NOT_STARTED);
  other.start_completed_time_.reset();
}

TorchWorkRCCLX::TorchWorkRCCLX(
    std::shared_ptr<TorchCommRCCLX> comm,
    hipStream_t stream,
    std::chrono::milliseconds timeout_ms,
    const at::Tensor& inputTensor)
    : inputTensor_(inputTensor),
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

TorchWorkRCCLX::~TorchWorkRCCLX() {
  if (!comm_) {
    return;
  }
  comm_->returnEvent(start_event_);
  comm_->returnEvent(end_event_);
}

void TorchWorkRCCLX::recordFunctionStart(const std::string& coll_name) {
  recordFunction_.emplace(at::RecordScope::USER_SCOPE);
  if (!recordFunction_->isActive()) {
    return;
  }

  // Passing input tensor to recordFunction allows for shape information in
  // profiling output.
  if (!inputTensors_.empty()) {
    std::vector<c10::IValue> inputs;
    inputs.reserve(inputTensors_.size());
    for (const auto& tensor : inputTensors_) {
      inputs.emplace_back(tensor);
    }
    recordFunction_->before(
        coll_name,
        c10::ArrayRef<const c10::IValue>(inputs.data(), inputs.size()));
  } else if (inputTensor_.defined()) {
    recordFunction_->before(
        coll_name, c10::ArrayRef<const c10::IValue>(inputTensor_));
  } else {
    recordFunction_->before(coll_name, c10::ArrayRef<const c10::IValue>{});
  }
}

void TorchWorkRCCLX::recordStart(const std::string& coll_name) {
  recordFunctionStart(coll_name);
  HIP_CHECK(
      comm_->getHipApi(),
      comm_->getHipApi()->eventRecord(start_event_, stream_),
      "Failed to record start event");
}

void TorchWorkRCCLX::recordEnd() {
  HIP_CHECK(
      comm_->getHipApi(),
      comm_->getHipApi()->eventRecord(end_event_, stream_),
      "Failed to record end event");
}

bool TorchWorkRCCLX::isCompleted() {
  return state_ == WorkStatus::COMPLETED;
}

TorchWorkRCCLX::WorkStatus TorchWorkRCCLX::checkStatus() {
  // If already marked as completed, return COMPLETED
  if (state_ == WorkStatus::COMPLETED || state_ == WorkStatus::ERROR ||
      state_ == WorkStatus::TIMEDOUT) {
    return state_;
  }

  // Step 1: If start_completed_time_ doesn't have a value yet, query the start
  // event
  if (!start_completed_time_.has_value()) {
    hipError_t start_status = comm_->getHipApi()->eventQuery(start_event_);

    if (start_status == hipSuccess) {
      // Start event has completed, store the current time
      start_completed_time_ = std::chrono::steady_clock::now();
      state_ = WorkStatus::INPROGRESS;
    } else if (start_status != hipErrorNotReady) {
      // Some other error occurred with the start event
      state_ = WorkStatus::ERROR;
    }
  }
  if (state_ == WorkStatus::NOT_STARTED || state_ == WorkStatus::ERROR) {
    return state_;
  }

  // Step 2: If we get here, start event has completed, so query the end event
  hipError_t end_status = comm_->getHipApi()->eventQuery(end_event_);

  if (end_status == hipSuccess) {
    // End event has completed, mark the work as completed
    state_ = WorkStatus::COMPLETED;

    // Release the input tensors to keep the lifetime of the tensors short
    inputTensors_.clear();
  } else if (end_status == hipErrorNotReady) {
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
  } else {
    // Some other error occurred with the end event
    state_ = WorkStatus::ERROR;
  }
  return state_;
}

void TorchWorkRCCLX::wait() {
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
  hipStream_t current_stream =
      comm_->getHipApi()->getCurrentHIPStreamMasqueradingAsCUDA(
          comm_->device_.index());

  // Add a dependency from the work's stream to the current stream
  // This makes the current stream wait for the end_event_ recorded on the
  // work's stream
  HIP_CHECK(
      comm_->getHipApi(),
      comm_->getHipApi()->streamWaitEvent(current_stream, end_event_, 0),
      "Failed to make stream wait for event");
}
} // namespace comms
} // namespace torch
