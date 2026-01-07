#include "comms/torchcomms/xccl/TorchWorkXCCL.hpp"
#include "comms/torchcomms/TorchCommLogging.hpp"
#include "comms/torchcomms/xccl/TorchCommXCCL.hpp"
#include <ATen/xpu/XPUContext.h>

namespace torch {
namespace comms {

TorchWorkXCCL::TorchWorkXCCL(std::shared_ptr<TorchCommXCCL> comm,
                             xpuStream_t stream,
                             std::chrono::milliseconds timeout_ms,
                             const std::vector<at::Tensor> &inputTensors,
                             std::shared_ptr<TorchCommTracing> tracing)
    : inputTensors_(inputTensors), comm_(std::move(comm)), stream_(stream),
      timeout_ms_(timeout_ms), state_(WorkStatus::NOT_STARTED),
      tracing_(std::move(tracing)) {
  // If not in graph capture mode, create the events for start and end
  // recording
  start_event_ = comm_->getEvent();
  end_event_ = comm_->getEvent();

  // Events will be recorded around the actual XCCL operations
}

TorchWorkXCCL::~TorchWorkXCCL() {
  if (!comm_) {
    return;
  }
  // If not in graph capture mode, return the events to the pool
  comm_->returnEvent(std::move(start_event_));
  comm_->returnEvent(std::move(end_event_));
}

void TorchWorkXCCL::recordStart() {
  XPU_CHECK(comm_->getXpuApi(),
            comm_->getXpuApi()->eventRecord(start_event_, stream_),
            "Failed to record start event");
}

void TorchWorkXCCL::recordEnd() {
  XPU_CHECK(comm_->getXpuApi(),
            comm_->getXpuApi()->eventRecord(end_event_, stream_),
            "Failed to record end event");
}

bool TorchWorkXCCL::isCompleted() { return state_ == WorkStatus::COMPLETED; }

TorchWorkXCCL::WorkStatus TorchWorkXCCL::checkStatus() {
  // If already marked as completed, return COMPLETED
  if (state_ == WorkStatus::COMPLETED || state_ == WorkStatus::ERROR ||
      state_ == WorkStatus::TIMEDOUT) {
    return state_;
  }

  // Step 1: If start_completed_time_ doesn't have a value yet, query the start
  // event
  if (!start_completed_time_.has_value()) {
    xpu_result_t start_status = comm_->getXpuApi()->eventQuery(start_event_);

    if (start_status == XPU_SUCCESS) {
      // Start event has completed, store the current time
      start_completed_time_ = std::chrono::steady_clock::now();
      state_ = WorkStatus::INPROGRESS;
    } else if (start_status != XPU_ERROR_NOT_READY &&
               start_status != XPU_ERROR_UNSUPPORTED) {
      // Some other error occurred with the start event
      TC_LOG(ERROR) << "XPU error during start event query: "
                    << comm_->getXpuApi()->getErrorString(start_status) << " ("
                    << start_status << ")";
      state_ = WorkStatus::ERROR;
    }
  }
  if (state_ == WorkStatus::NOT_STARTED || state_ == WorkStatus::ERROR) {
    return state_;
  }

  // Step 2: If we get here, start event has completed, so query the end event
  xpu_result_t end_status = comm_->getXpuApi()->eventQuery(end_event_);

  if (end_status == XPU_SUCCESS) {
    // End event has completed, mark the work as completed
    state_ = WorkStatus::COMPLETED;

    // Release the input tensors to keep the lifetime of the tensors short
    inputTensors_.clear();
  } else if (end_status == XPU_ERROR_NOT_READY) {
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
    TC_LOG(ERROR) << "XPU error during end event query: "
                  << comm_->getXpuApi()->getErrorString(end_status) << " ("
                  << end_status << ")";
    state_ = WorkStatus::ERROR;
  }
  return state_;
}

void TorchWorkXCCL::wait() {
  // If already completed, return immediately
  WorkStatus local_state = state_;
  if (local_state == WorkStatus::COMPLETED ||
      local_state == WorkStatus::ERROR || local_state == WorkStatus::TIMEDOUT) {
    return;
  }

  tracing_->recordEvent("wait");

  // Get the current stream using the device from the comm object
  xpuStream_t current_stream =
      comm_->getXpuApi()->getCurrentXPUStream(comm_->device_.index());

  // Add a dependency from the work's stream to the current stream
  // This makes the current stream wait for the end_event_ recorded on the
  // work's stream
  XPU_CHECK(comm_->getXpuApi(),
            comm_->getXpuApi()->streamWaitEvent(current_stream, end_event_, 0),
            "Failed to make stream wait for event");
}
} // namespace comms
} // namespace torch
