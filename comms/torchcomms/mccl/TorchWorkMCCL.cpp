// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "TorchWorkMCCL.hpp"
#include "TorchCommMCCL.hpp"

#include "comms/torchcomms/utils/Logging.hpp"

#include <torch/csrc/distributed/c10d/ParamCommsUtils.hpp> // @manual

namespace torch::comms {

TorchWorkMCCL::TorchWorkMCCL(
    std::shared_ptr<TorchCommMCCL> comm,
    cudaStream_t stream,
    const std::vector<at::Tensor>& inputTensors,
    const std::vector<at::Tensor>& outputTensors,
    std::unique_ptr<mcclWork_t> workHandle,
    std::chrono::milliseconds timeout)
    : TorchWorkMCCL(std::move(comm), stream, std::move(workHandle), timeout) {
  inputTensors_ = inputTensors;
  outputTensors_ = outputTensors;
}

TorchWorkMCCL::TorchWorkMCCL(
    std::shared_ptr<TorchCommMCCL> comm,
    cudaStream_t stream,
    const at::Tensor& inputTensor,
    const at::Tensor& outputTensor,
    std::unique_ptr<mcclWork_t> workHandle,
    std::chrono::milliseconds timeout)
    : TorchWorkMCCL(std::move(comm), stream, std::move(workHandle), timeout) {
  inputTensor_ = inputTensor;
  outputTensor_ = outputTensor;
}

// Move constructor
TorchWorkMCCL::TorchWorkMCCL(TorchWorkMCCL&& other) noexcept
    : cudaApi_(std::move(other.cudaApi_)),
      cudaDeviceId_(other.cudaDeviceId_),
      infoOnCreation_(std::move(other.infoOnCreation_)),
      stream_(other.stream_),
      inputTensors_(std::move(other.inputTensors_)),
      outputTensors_(std::move(other.outputTensors_)),
      inputTensor_(std::move(other.inputTensor_)),
      outputTensor_(std::move(other.outputTensor_)),
      mcclWork_(std::move(other.mcclWork_)),
      timeout_(other.timeout_),
      start_time_(other.start_time_) {
  setStatus(other.status());
}

TorchWorkMCCL::TorchWorkMCCL(
    std::shared_ptr<TorchCommMCCL> comm,
    cudaStream_t stream,
    std::unique_ptr<mcclWork_t> workHandle,
    std::chrono::milliseconds timeout)
    : cudaApi_(comm ? comm->cuda_api_ : nullptr),
      cudaDeviceId_(comm ? comm->cudaDeviceId_ : -1),
      infoOnCreation_{
          .commName = comm ? std::string(comm->getCommName()) : "",
          .commId = comm ? comm->getUuid() : "",
          .commSize = comm && comm->isInitialized() ? comm->getSize() : 0,
          .rank = comm && comm->isInitialized() ? comm->getRank() : -1},
      stream_(stream),
      mcclWork_(std::move(workHandle)),
      timeout_(timeout),
      start_time_(std::chrono::steady_clock::now()) {
  setStatus(WorkStatus::INPROGRESS);
}

TorchWorkMCCL::WorkStatus TorchWorkMCCL::checkStatus() {
  // If already in a terminal state, return it
  if (status() == WorkStatus::COMPLETED || status() == WorkStatus::ERROR ||
      status() == WorkStatus::TIMEDOUT) {
    return status();
  }

  auto workResult = mcclWork_->getResult();
  // Non-nullopt means mcclWork in terminate state
  if (workResult != std::nullopt) {
    if (workResult.value().code == commSuccess) {
      setStatus(WorkStatus::COMPLETED);
    } else {
      TC_LOG(WARNING) << "Work failed with error code: "
                      << workResult.value().code << ", "
                      << workResult.value().message;
      setStatus(WorkStatus::ERROR);
    }
    // NOTE: Do NOT call releaseTensors() here. checkStatus() is called from
    // the watchdog thread, and releasing tensor references from a background
    // thread can race with the training thread's Python code, corrupting
    // TensorImpl reference counts. Tensors are released in wait() instead,
    // which runs on the training thread.
    return status();
  }

  // Work still in progress — check for timeout
  if (timeout_ == std::chrono::milliseconds(0)) {
    TC_LOG(INFO) << "TorchWorkMCCL: Work has no timeout set";
    throw std::runtime_error("TorchWorkMCCL::checkStatus: timeout is set to 0");
  }
  if (!start_time_.has_value()) {
    throw std::runtime_error(
        "TorchWorkMCCL::checkStatus: start_time_ is not set");
  }
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start_time_.value());
  if (elapsed > timeout_) {
    TC_LOG(INFO) << "TorchWorkMCCL: work timed out after " << elapsed.count()
                 << "ms (timeout: " << timeout_.count() << "ms)";
    setStatus(WorkStatus::TIMEDOUT);
  }
  return status();
}

void TorchWorkMCCL::releaseTensors() {
  inputTensors_.clear();
  outputTensors_.clear();
  inputTensor_.reset();
  outputTensor_.reset();
}

cudaStream_t TorchWorkMCCL::getCurrentCUDAStream() {
  if (!cudaApi_) {
    throw std::runtime_error(
        "TorchWorkMCCL::getCurrentCUDAStream: cuda_api_ is null");
  }
  return cudaApi_->getCurrentCUDAStream(cudaDeviceId_);
}

void TorchWorkMCCL::wait() {
  runWaitPreHooks();

  if (!mcclWork_) {
    throw std::runtime_error("TorchWorkMCCL::wait: mcclWork_ is null");
  }

  if (isCpuWork_) {
    waitCPU();
    releaseTensors();
    runWaitPostHooks();
    return;
  }

  cudaStream_t currentStream = getCurrentCUDAStream();
  TracingGuard tracingGuard(infoOnCreation_, "wait");
  mcclWork_->waitStream(currentStream);

  // Release tensor references. The CUDA caching allocator manages stream
  // semantics and will not reclaim memory until the stream operations complete.
  releaseTensors();

  runWaitPostHooks();
}

void TorchWorkMCCL::waitCPU() {
  if (!mcclWork_) {
    throw std::runtime_error("TorchWorkMCCL::waitCPU: mcclWork_ is null");
  }
  TracingGuard tracingGuard(infoOnCreation_, "waitCPU");
  mcclWork_->waitCpu();
  // After the blocking wait completes, update the internal status so that
  // is_completed() returns true. Without this call, the status remains
  // INPROGRESS even after the operation has successfully completed.
  checkStatus();
}

void TorchWorkMCCL::waitBlocking() {
  waitCPU();
}

std::optional<::mccl::Result> TorchWorkMCCL::getResult() {
  return mcclWork_->getResult();
}

} // namespace torch::comms
