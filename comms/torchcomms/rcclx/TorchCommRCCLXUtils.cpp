// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/rcclx/TorchCommRCCLX.hpp"
#include "comms/torchcomms/rcclx/TorchCommRCCLXCCA.hpp"

#include <stdexcept>
#include <string>
#include "comms/torchcomms/TorchCommLogging.hpp" // @manual=//comms/torchcomms:torchcomms-headers-cpp"
#include "rccl.h" // @manual

namespace torch {
namespace comms {

namespace {

// Helper function to get RCCLX data type from tensor
// This mirrors the logic in TorchCommRCCLX::getNcclDataType
ncclDataType_t getNcclDataTypeInternal(const at::Tensor& tensor) {
  switch (tensor.scalar_type()) {
    case at::ScalarType::Float:
      return ncclFloat32;
    case at::ScalarType::Double:
      return ncclFloat64;
    case at::ScalarType::Half:
      return ncclFloat16;
    case at::ScalarType::BFloat16:
      return ncclBfloat16;
    case at::ScalarType::Int:
      return ncclInt32;
    case at::ScalarType::Long:
      return ncclInt64;
    case at::ScalarType::Char:
      return ncclInt8;
    case at::ScalarType::Byte:
      return ncclUint8;
    default:
      throw std::runtime_error("Unsupported tensor data type for RCCLX");
  }
}

template <typename T, ncclDataType_t dataType>
void createPreMulSum(
    ncclRedOp_t* op,
    const PreMulSumFactorT& factor,
    const ncclComm_t& comm,
    RcclxApi* rcclx_api) {
  const bool is_tensor = std::holds_alternative<at::Tensor>(factor);
  const auto residence = is_tensor ? ncclScalarDevice : ncclScalarHostImmediate;

  at::Tensor tensor = is_tensor ? std::get<at::Tensor>(factor) : at::Tensor();
  T scalar_factor = is_tensor ? T{} : static_cast<T>(std::get<double>(factor));
  void* scalar = is_tensor ? tensor.data_ptr() : &scalar_factor;

  TORCH_INTERNAL_ASSERT(
      is_tensor ? dataType == getNcclDataTypeInternal(tensor)
                : dataType != ncclBfloat16,
      "PreMulSum factor type must match input data type");
  rcclx_api->redOpCreatePreMulSum(op, scalar, dataType, residence, comm);
}

} // namespace

TorchCommRCCLX::RedOpRAII::RedOpRAII(ncclRedOp_t op)
    : ncclRedOp_(op), comm_(nullptr) {}

TorchCommRCCLX::RedOpRAII::RedOpRAII(
    const ReduceOp& op,
    const ncclComm_t comm,
    const ncclDataType_t dataType,
    std::shared_ptr<RcclxApi> rcclx_api)
    : comm_(comm), rcclx_api_(std::move(rcclx_api)) {
  TORCH_INTERNAL_ASSERT(
      op == ReduceOp::RedOpType::PREMUL_SUM,
      "Constructing premul_sum RedOpRAII with non-premul_sum RedOpType");

  if (!op.factor().has_value()) {
    ncclRedOp_ = ncclSum;
    comm_ = nullptr;
    return;
  }

  const auto& factor = op.factor().value();
  switch (dataType) {
    case ncclFloat16:
      createPreMulSum<at::Half, ncclFloat16>(
          &ncclRedOp_, factor, comm, rcclx_api_.get());
      break;
    case ncclFloat32:
      createPreMulSum<float, ncclFloat32>(
          &ncclRedOp_, factor, comm, rcclx_api_.get());
      break;
    case ncclBfloat16:
      createPreMulSum<float, ncclBfloat16>(
          &ncclRedOp_, factor, comm, rcclx_api_.get());
      break;
    case ncclFloat64:
      createPreMulSum<double, ncclFloat64>(
          &ncclRedOp_, factor, comm, rcclx_api_.get());
      break;
    default:
      throw std::runtime_error(
          "PreMulSum Data type must be half, float, bfloat16 or double");
  }
}

TorchCommRCCLX::RedOpRAII::~RedOpRAII() {
  if (comm_) {
    RCCLX_CHECK_IGNORE(
        rcclx_api_,
        rcclx_api_->redOpDestroy(ncclRedOp_, comm_),
        "RCCLX redOpDestroy failed");
  }
}

ncclDataType_t TorchCommRCCLX::getNcclDataType(const at::Tensor& tensor) {
  return getNcclDataTypeInternal(tensor);
}

TorchCommRCCLX::RedOpRAII TorchCommRCCLX::getNcclReduceOp(
    const ReduceOp& op,
    const ncclComm_t comm,
    const ncclDataType_t dataType) {
  switch (op) {
    case ReduceOp::RedOpType::SUM:
      return ncclSum;
    case ReduceOp::RedOpType::PRODUCT:
      return ncclProd;
    case ReduceOp::RedOpType::MIN:
      return ncclMin;
    case ReduceOp::RedOpType::MAX:
      return ncclMax;
    case ReduceOp::RedOpType::BAND:
      return ncclSum; // RCCLX doesn't have bitwise AND, using SUM as fallback
    case ReduceOp::RedOpType::BOR:
      return ncclSum; // RCCLX doesn't have bitwise OR, using SUM as fallback
    case ReduceOp::RedOpType::BXOR:
      return ncclSum; // RCCLX doesn't have bitwise XOR, using SUM as fallback
    case ReduceOp::RedOpType::PREMUL_SUM:
      return RedOpRAII(op, comm, dataType, rcclx_api_);
    case ReduceOp::RedOpType::AVG:
      return ncclAvg;
    default:
      throw std::runtime_error("Unsupported reduce operation in RCCLX");
  }
}

void TorchCommRCCLX::garbageCollectWorkQueues() {
  // Keep popping completed elements until we hit an in-progress element
  // or the queue is empty
  // Use an iterator to safely remove empty queues while iterating
  auto it = stream_work_queues_.begin();
  while (it != stream_work_queues_.end()) {
    auto& work_queue = it->second;

    while (!work_queue.empty()) {
      // Get the first work object in the queue
      auto work = work_queue.front();

      // Use the checkStatus function to determine the work status
      TorchWorkRCCLX::WorkStatus status = work->checkStatus();

      switch (status) {
        case TorchWorkRCCLX::WorkStatus::NOT_STARTED:
        case TorchWorkRCCLX::WorkStatus::INPROGRESS:
          break;
        case TorchWorkRCCLX::WorkStatus::TIMEDOUT: {
          comm_state_ = CommState::TIMEOUT;
          return;
        }
        case TorchWorkRCCLX::WorkStatus::ERROR: {
          comm_state_ = CommState::ERROR;
          return;
        }
        case TorchWorkRCCLX::WorkStatus::COMPLETED: {
          // Work is completed, move it from the work queue to the completed
          // queue
          auto completed_work = work_queue.front();
          work_queue.pop();

          // Add to the single completed works queue
          completed_works_.push(completed_work);

          // Continue to the next element in the queue
          continue;
        }
        default:
          TORCH_INTERNAL_ASSERT(false, "Unexpected WorkStatus enum value");
      }
    }

    // If the queue is now empty, remove it from the map
    if (work_queue.empty()) {
      it = stream_work_queues_.erase(it);
    } else {
      ++it;
    }
  }
}

// The timeout thread cannot make NCCL calls.  The only CUDA call it can make
// it hipEventQuery (done inside checkStatus).
void TorchCommRCCLX::timeoutWatchdog() noexcept {
  TC_LOG(INFO, this) << "Timeout thread starting for rank: " << rank_;

  hipStreamCaptureMode mode = hipStreamCaptureModeThreadLocal;
  HIP_CHECK_IGNORE(
      hip_api_,
      hip_api_->threadExchangeStreamCaptureMode(&mode),
      "Failed to swap capture mode for timeout thread");

  while (!shutdown_) {
    {
      std::unique_lock<std::mutex> lock(timeout_mutex_);
      // Wait for a shorter interval to check work objects periodically
      // Wake up either after 1 second or immediately if shutdown is requested
      timeout_cv_.wait_for(
          lock, std::chrono::seconds(1), [this]() { return shutdown_.load(); });

      // If we're shutting down, exit the loop
      if (shutdown_) {
        break;
      }
    }

    // Check work objects for completion or timeout

    std::lock_guard<std::mutex> lock(work_queues_mutex_);
    garbageCollectWorkQueues();
    if (comm_state_ != CommState::NORMAL &&
        options_.abort_process_on_timeout_or_error) {
      // Log the error and abort the process.  We cannot abort the NCCL
      // communicator as it is not safe to call NCCL operations from
      // multiple threads at the same time.
      if (comm_state_ == CommState::TIMEOUT) {
        TC_LOG(ERROR, this) << "Aborting process due to timeout";
      } else if (comm_state_ == CommState::ERROR) {
        TC_LOG(ERROR, this) << "Aborting process due to error";
      }
      abort();
    }
  }

  TC_LOG(INFO, this) << "Timeout thread exiting for rank: " << rank_;
}

void TorchCommRCCLX::checkInitialized() const {
  if (init_state_ != InitializationState::INITIALIZED) {
    throw std::runtime_error("TorchCommRCCLX not initialized");
  }
}

void TorchCommRCCLX::checkAndAbortIfTimedOutOrError() {
  // First, perform garbage collection
  {
    // Acquire the lock to safely clear the completed works queue
    std::lock_guard<std::mutex> lock(work_queues_mutex_);

    // Create an empty queue and swap with the completed_works_ queue
    // This is more efficient than calling clear() as it deallocates memory
    std::queue<c10::intrusive_ptr<TorchWorkRCCLX>> empty;
    std::swap(completed_works_, empty);
    // The old queue will be destroyed when this scope exits
  }

  if (comm_state_ == CommState::TIMEOUT) {
    abortRcclxComm();
    throw std::runtime_error("NCCL operation timed out");
  } else if (comm_state_ == CommState::ERROR) {
    ncclResult_t asyncErr;
    rcclx_api_->commGetAsyncError(nccl_comm_, &asyncErr);
    RCCLXException RCCLXException(
        *rcclx_api_, "NCCL Async Error", asyncErr, nccl_comm_);
    abortRcclxComm();
    throw RCCLXException;
  }
}

c10::intrusive_ptr<TorchWorkRCCLX> TorchCommRCCLX::createWork(
    hipStream_t stream,
    std::chrono::milliseconds timeout,
    const std::vector<at::Tensor>& inputTensors) {
  // Only create the work object without enqueuing it
  auto work = c10::make_intrusive<TorchWorkRCCLX>(
      shared_from_this(), stream, timeout, inputTensors);
  return work;
}

c10::intrusive_ptr<TorchWorkRCCLX> TorchCommRCCLX::createWork(
    hipStream_t stream,
    std::chrono::milliseconds timeout,
    const at::Tensor& inputTensor) {
  // Only create the work object without enqueuing it
  auto work = c10::make_intrusive<TorchWorkRCCLX>(
      shared_from_this(), stream, timeout, inputTensor);
  return work;
}

void TorchCommRCCLX::enqueueWork(
    const c10::intrusive_ptr<TorchWorkRCCLX>& work,
    hipStream_t stream) {
  // Add work to stream's queue after events have been recorded
  std::lock_guard<std::mutex> lock(work_queues_mutex_);
  stream_work_queues_[stream].push(work);
}

hipStream_t TorchCommRCCLX::getOperationStream(bool async_op) {
  if (async_op) {
    // Get current PyTorch CUDA stream for this device
    hipStream_t current_stream =
        hip_api_->getCurrentHIPStreamMasqueradingAsCUDA(device_.index());

    // Record event on current stream and wait for it on internal stream
    HIP_CHECK(
        hip_api_,
        hip_api_->eventRecord(dependency_event_, current_stream),
        "Failed to record dependency event");

    HIP_CHECK(
        hip_api_,
        hip_api_->streamWaitEvent(internal_stream_, dependency_event_, 0),
        "Failed to make internal stream wait for dependency event");

    return internal_stream_;
  } else {
    // Use the current PyTorch CUDA stream for synchronous operations
    return hip_api_->getCurrentHIPStreamMasqueradingAsCUDA(device_.index());
  }
}

void TorchCommRCCLX::ensureTensorContiguous(const at::Tensor& tensor) {
  if (!tensor.is_contiguous()) {
    throw std::runtime_error("Tensor must be contiguous for NCCL operations");
  }
}

// Protected methods (not in the private section of the header)
hipEvent_t TorchCommRCCLX::getEvent() {
  std::lock_guard<std::mutex> lock(event_pool_mutex_);

  if (!event_pool_.empty()) {
    hipEvent_t event = event_pool_.front();
    event_pool_.pop();
    return event;
  }

  // Create new event if pool is empty
  hipEvent_t event;
  HIP_CHECK(hip_api_, hip_api_->eventCreate(&event), "Failed to create event");
  return event;
}

void TorchCommRCCLX::returnEvent(hipEvent_t event) {
  std::lock_guard<std::mutex> lock(event_pool_mutex_);

  if (event_pool_.size() < max_event_pool_size_) {
    event_pool_.push(event);
  } else {
    // Pool is full, destroy the event
    HIP_CHECK(
        hip_api_, hip_api_->eventDestroy(event), "Failed to destroy event");
  }
}

void TorchCommRCCLX::attachMemoryHook() {
  CachingAllocatorHook::getInstance().registerComm(this);
}

void TorchCommRCCLX::detachMemoryHook() {
  CachingAllocatorHook::getInstance().deregisterComm(this);
}

} // namespace comms
} // namespace torch
