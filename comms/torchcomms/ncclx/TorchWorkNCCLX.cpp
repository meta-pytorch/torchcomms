// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/ncclx/TorchWorkNCCLX.hpp"
#include <ATen/ThreadLocalState.h>
#include <ATen/cuda/CUDAContext.h>
#include "TorchCommNCCLX.hpp"
#include "comms/torchcomms/TorchCommLogging.hpp"
#include "comms/torchcomms/TorchCommTracing.hpp"

namespace torch::comms {

void TorchWorkNCCLX::initEvents() {
  if (graph_capture_mode_) {
    // Ad-hoc create all three events — NOT from the event pool.
    // start_event_ and end_event_ ownership will be transferred to
    // GraphWork in enqueueWork(). sync_event_ is destroyed in dtor.
    // TODO: sync_event_ is only needed for async_op=true; skip creation
    // for synchronous operations where the work runs on the current stream.
    CUDA_CHECK(
        comm_->getCudaApi(),
        comm_->getCudaApi()->eventCreateWithFlags(
            &start_event_, cudaEventDisableTiming),
        "Failed to create start event for graph capture");
    CUDA_CHECK(
        comm_->getCudaApi(),
        comm_->getCudaApi()->eventCreateWithFlags(
            &end_event_, cudaEventDisableTiming),
        "Failed to create end event for graph capture");
    CUDA_CHECK(
        comm_->getCudaApi(),
        comm_->getCudaApi()->eventCreateWithFlags(
            &sync_event_, cudaEventDisableTiming),
        "Failed to create sync event for graph capture");
  } else {
    start_event_ = comm_->getEvent();
    end_event_ = comm_->getEvent();
  }
}

void TorchWorkNCCLX::releaseEvents() {
  if (graph_capture_mode_) {
    // In graph mode: start_event_ and end_event_ were ad-hoc created and
    // should have been transferred to the GraphWorkEntry (set to nullptr).
    // If transfer didn't happen (error path), destroy them.
    // sync_event_ is always ad-hoc and always destroyed here.
    if (start_event_) {
      (void)comm_->getCudaApi()->eventDestroy(start_event_);
    }
    if (end_event_) {
      (void)comm_->getCudaApi()->eventDestroy(end_event_);
    }
    if (sync_event_) {
      (void)comm_->getCudaApi()->eventDestroy(sync_event_);
    }
  } else {
    // Non-graph mode: both start and end events are from the pool.
    if (start_event_) {
      comm_->returnEvent(start_event_);
    }
    if (end_event_) {
      comm_->returnEvent(end_event_);
    }
  }
}

TorchWorkNCCLX::TorchWorkNCCLX(
    std::shared_ptr<TorchCommNCCLX> comm,
    cudaStream_t stream,
    std::chrono::milliseconds timeout_ms,
    const std::vector<at::Tensor>& inputTensors)
    : inputTensors_(inputTensors),
      comm_(std::move(comm)),
      stream_(stream),
      timeout_ms_(timeout_ms) {
  graph_capture_mode_ = comm_->getGraphCaptureMode();
  initEvents();
}

TorchWorkNCCLX::TorchWorkNCCLX(
    std::shared_ptr<TorchCommNCCLX> comm,
    cudaStream_t stream,
    std::chrono::milliseconds timeout_ms,
    const at::Tensor& inputTensor)
    : inputTensor_(inputTensor),
      comm_(std::move(comm)),
      stream_(stream),
      timeout_ms_(timeout_ms) {
  graph_capture_mode_ = comm_->getGraphCaptureMode();
  initEvents();
}

TorchWorkNCCLX::~TorchWorkNCCLX() {
  if (!comm_) {
    return;
  }
  releaseEvents();
}

void TorchWorkNCCLX::recordFunctionStart(std::string_view coll_name) {
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

void TorchWorkNCCLX::recordStart(std::string_view coll_name) {
  recordFunctionStart(coll_name);

  if (comm_->getGraphCaptureMode()) {
    // Use cudaEventRecordExternal so start_event_ remains host-queryable
    // during graph replay (for watchdog timeout detection).
    // start_event_ is not used as a graph join point, so this is safe.
    CUDA_CHECK(
        comm_->getCudaApi(),
        comm_->getCudaApi()->eventRecordWithFlags(
            start_event_, stream_, cudaEventRecordExternal),
        "Failed to record start event");
  } else {
    CUDA_CHECK(
        comm_->getCudaApi(),
        comm_->getCudaApi()->eventRecord(start_event_, stream_),
        "Failed to record start event");
  }
}

void TorchWorkNCCLX::recordEnd() {
  // During graph capture, end_event_ is recorded with cudaEventRecordExternal
  // so it remains host-queryable during graph replay for watchdog timeout
  // detection. sync_event_ is recorded with regular cudaEventRecord to serve
  // as a valid join point for work.wait() (cudaStreamWaitEvent).
  //
  // In eager mode, end_event_ is recorded with regular cudaEventRecord and
  // serves as both the completion detection event and the join point.
  // sync_event_ is nullptr.
  if (graph_capture_mode_) {
    CUDA_CHECK(
        comm_->getCudaApi(),
        comm_->getCudaApi()->eventRecordWithFlags(
            end_event_, stream_, cudaEventRecordExternal),
        "Failed to record end event");
    CUDA_CHECK(
        comm_->getCudaApi(),
        comm_->getCudaApi()->eventRecord(sync_event_, stream_),
        "Failed to record sync event");
  } else {
    CUDA_CHECK(
        comm_->getCudaApi(),
        comm_->getCudaApi()->eventRecord(end_event_, stream_),
        "Failed to record end event");
  }

  if (recordFunction_ && recordFunction_->isActive()) {
    recordFunction_->end();
  }
}

TorchWorkNCCLX::WorkStatus TorchWorkNCCLX::checkStatus() {
  // If already marked as completed, return COMPLETED
  if (status() == WorkStatus::COMPLETED || status() == WorkStatus::ERROR ||
      status() == WorkStatus::TIMEDOUT) {
    return status();
  }

  // Step 1: If start_completed_time_ doesn't have a value yet, query the start
  // event
  if (!start_completed_time_.has_value()) {
    cudaError_t start_status = comm_->getCudaApi()->eventQuery(start_event_);

    if (start_status == cudaSuccess) {
      // Start event has completed, store the current time
      start_completed_time_ = std::chrono::steady_clock::now();
      setStatus(WorkStatus::INPROGRESS);
    } else if (start_status != cudaErrorNotReady) {
      // Some other error occurred with the start event
      TC_LOG(ERROR, comm_.get())
          << "CUDA error during start event query: "
          << comm_->getCudaApi()->getErrorString(start_status) << " ("
          << start_status << ")";
      setStatus(WorkStatus::ERROR);
    }
  }
  if (status() == WorkStatus::NOT_STARTED || status() == WorkStatus::ERROR) {
    return status();
  }

  // Step 2: If we get here, start event has completed, so query the end event
  cudaError_t end_status = comm_->getCudaApi()->eventQuery(end_event_);

  if (end_status == cudaSuccess) {
    // End event has completed, mark the work as completed
    setStatus(WorkStatus::COMPLETED);
  } else if (end_status == cudaErrorNotReady) {
    // End event has not completed yet, check for timeout
    auto current_time = std::chrono::steady_clock::now();
    auto elapsed_milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - start_completed_time_.value());

    // Check if the operation has timed out
    if (elapsed_milliseconds > timeout_ms_) {
      // Operation has timed out
      setStatus(WorkStatus::TIMEDOUT);
    }
  } else {
    // Some other error occurred with the end event
    TC_LOG(ERROR, comm_.get())
        << "CUDA error during end event query: "
        << comm_->getCudaApi()->getErrorString(end_status) << " (" << end_status
        << ")";
    setStatus(WorkStatus::ERROR);
  }
  return status();
}

std::optional<float> TorchWorkNCCLX::getDuration() const {
  // Duration is only available after work has completed
  if (status() != WorkStatus::COMPLETED) {
    return std::nullopt;
  }

  float duration_ms = 0.0f;
  cudaError_t err =
      cudaEventElapsedTime(&duration_ms, start_event_, end_event_);
  if (err != cudaSuccess) {
    return std::nullopt;
  }
  return duration_ms;
}

void TorchWorkNCCLX::wait() {
  // If already completed, return immediately
  WorkStatus local_state = status();
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

  // Add a dependency from the work's stream to the current stream.
  // In graph mode, use sync_event_ (the regular-recorded join point).
  // In eager mode, use end_event_ (sync_event_ is nullptr).
  cudaEvent_t wait_event = sync_event_ ? sync_event_ : end_event_;
  CUDA_CHECK(
      comm_->getCudaApi(),
      comm_->getCudaApi()->streamWaitEvent(current_stream, wait_event, 0),
      "Failed to make stream wait for event");

  // Release tensor references. The CUDA caching allocator manages stream
  // semantics and will not reclaim memory until the stream operations
  // complete.
  inputTensors_.clear();
  inputTensor_.reset();
}
} // namespace torch::comms
