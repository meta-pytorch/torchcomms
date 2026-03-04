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
    if (graph_timeout_detection_) {
      // Both start and end events are created for timeout monitoring.
      // Lifetime ownership is transferred to GraphWork in enqueueWork(),
      // although we will still hold a reference to end_event_ for wait().
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
    } else {
      // Only end_event_ is needed for stream join in wait().
      // start_event_ remains nullptr — no timeout monitoring overhead.
      CUDA_CHECK(
          comm_->getCudaApi(),
          comm_->getCudaApi()->eventCreateWithFlags(
              &end_event_, cudaEventDisableTiming),
          "Failed to create end event for graph capture");
    }
  } else {
    start_event_ = comm_->getEvent();
    end_event_ = comm_->getEvent();
  }
}

void TorchWorkNCCLX::releaseEvents() {
  if (graph_capture_mode_) {
    if (graph_timeout_detection_) {
      // start_event_ is nulled by addEntry() when events are transferred
      // to GraphEventTracker. if non-null, transfer of ownership hasn't
      // happened — destroy both events here.
      if (start_event_ && end_event_) {
        (void)comm_->getCudaApi()->eventDestroy(start_event_);
        (void)comm_->getCudaApi()->eventDestroy(end_event_);
      }
    } else {
      // Only end_event_ was created (start_event_ is nullptr).
      if (end_event_) {
        (void)comm_->getCudaApi()->eventDestroy(end_event_);
      }
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
  graph_timeout_detection_ =
      graph_capture_mode_ && comm_->configs_.enable_graph_timeout_detection_;
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
  graph_timeout_detection_ =
      graph_capture_mode_ && comm_->configs_.enable_graph_timeout_detection_;
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

  if (graph_timeout_detection_) {
    // Use cudaEventRecordExternal so start_event_ remains host-queryable
    // during graph replay (for watchdog timeout detection).
    CUDA_CHECK(
        comm_->getCudaApi(),
        comm_->getCudaApi()->eventRecordWithFlags(
            start_event_, stream_, cudaEventRecordExternal),
        "Failed to record start event");
  } else if (!graph_capture_mode_) {
    // Eager mode: regular event recording for work queue timeout detection.
    CUDA_CHECK(
        comm_->getCudaApi(),
        comm_->getCudaApi()->eventRecord(start_event_, stream_),
        "Failed to record start event");
  }
  // In graph mode without timeout detection, start_event_ is nullptr — skip.
}

void TorchWorkNCCLX::recordEnd() {
  // When graph timeout detection is active, record with cudaEventRecordExternal
  // so the event remains host-queryable during graph replay.
  // Otherwise (eager mode or graph mode without timeout detection), use regular
  // cudaEventRecord — sufficient for stream join in wait() and for eager-mode
  // work queue timeout detection.
  if (graph_timeout_detection_) {
    CUDA_CHECK(
        comm_->getCudaApi(),
        comm_->getCudaApi()->eventRecordWithFlags(
            end_event_, stream_, cudaEventRecordExternal),
        "Failed to record end event");
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
      TC_LOG(ERROR, comm_.get()) << "Operation timed out after "
                                 << elapsed_milliseconds.count() << " ms";
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

  cudaStream_t current_stream =
      comm_->getCudaApi()->getCurrentCUDAStream(comm_->device_.index());

  if (graph_timeout_detection_) {
    // Clear stream_'s tracked fork so the fork/join checker
    // at capture_end() doesn't see unjoined explicit EVENT_RECORD_EXT
    // nodes. The streamWaitEvent below creates the actual graph edge.
    CUDA_CHECK(
        comm_->getCudaApi(),
        comm_->getCudaApi()->streamUpdateCaptureDependencies(
            stream_, nullptr, 0, cudaStreamSetCaptureDependencies),
        "Failed to clear stream_'s capture dependencies");
  }

  CUDA_CHECK(
      comm_->getCudaApi(),
      comm_->getCudaApi()->streamWaitEvent(
          current_stream,
          end_event_,
          graph_capture_mode_ ? cudaEventWaitExternal : 0x00),
      "Failed to make stream wait for event");

  // Release tensor references. The CUDA caching allocator manages stream
  // semantics and will not reclaim memory until the stream operations
  // complete.
  inputTensors_.clear();
  inputTensor_.reset();
}
} // namespace torch::comms
