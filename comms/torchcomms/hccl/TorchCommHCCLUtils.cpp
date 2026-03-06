#include <hccl/hccl.h>
#include <hccl/hccl_types.h>
#include <stdexcept>
#include <string>
#include "comms/torchcomms/TorchCommLogging.hpp"
#include "comms/torchcomms/hccl/TorchCommHCCL.hpp"

namespace torch::comms {

namespace {

HcclDataType getHcclDataTypeInternal(const at::Tensor& tensor) {
  switch (tensor.scalar_type()) {
    case at::ScalarType::Float:
      return HCCL_DATA_TYPE_FP32;
    case at::ScalarType::Double:
      return HCCL_DATA_TYPE_FP64;
    case at::ScalarType::Half:
      return HCCL_DATA_TYPE_FP16;
    case at::ScalarType::BFloat16:
      return HCCL_DATA_TYPE_BFP16;
    case at::ScalarType::Int:
      return HCCL_DATA_TYPE_INT32;
    case at::ScalarType::Long:
      return HCCL_DATA_TYPE_INT64;
    case at::ScalarType::Char:
      return HCCL_DATA_TYPE_INT8;
    case at::ScalarType::Byte:
      return HCCL_DATA_TYPE_UINT8;
    default:
      throw std::runtime_error("Unsupported tensor data type for HCCL");
  }
}

template <typename T, HcclDataType dataType>
void createPreMulSum(
    HcclReduceOp* op,
    const PreMulSumFactorT& factor,
    const HcclComm& comm,
    HcclApi* hccl_api) {
  // HCCL doesn't support premul_sum, so we just use sum
  // The premul operation will be handled separately
  *op = HCCL_REDUCE_SUM;
}

} // namespace

TorchCommHCCL::RedOpRAII::RedOpRAII(HcclReduceOp op)
    : hcclRedOp_(op), comm_(nullptr) {}

TorchCommHCCL::RedOpRAII::RedOpRAII(
    const ReduceOp& op,
    const HcclComm comm,
    const HcclDataType dataType,
    std::shared_ptr<HcclApi> hccl_api)
    : comm_(comm), hccl_api_(std::move(hccl_api)) {
  TORCH_INTERNAL_ASSERT(
      op == ReduceOp::RedOpType::PREMUL_SUM,
      "Constructing premul_sum RedOpRAII with non-premul_sum RedOpType");

  if (!op.factor().has_value()) {
    hcclRedOp_ = HCCL_REDUCE_SUM;
    comm_ = nullptr;
    return;
  }

  // HCCL doesn't support premul_sum natively, just use sum
  // The premul operation is handled in the preReduce function
  const auto& factor = op.factor().value();
  switch (dataType) {
    case HCCL_DATA_TYPE_FP16:
      createPreMulSum<at::Half, HCCL_DATA_TYPE_FP16>(
          &hcclRedOp_, factor, comm, hccl_api_.get());
      break;
    case HCCL_DATA_TYPE_FP32:
      createPreMulSum<float, HCCL_DATA_TYPE_FP32>(
          &hcclRedOp_, factor, comm, hccl_api_.get());
      break;
    case HCCL_DATA_TYPE_BFP16:
      createPreMulSum<at::BFloat16, HCCL_DATA_TYPE_BFP16>(
          &hcclRedOp_, factor, comm, hccl_api_.get());
      break;
    case HCCL_DATA_TYPE_FP64:
      createPreMulSum<double, HCCL_DATA_TYPE_FP64>(
          &hcclRedOp_, factor, comm, hccl_api_.get());
      break;
    default:
      throw std::runtime_error(
          "PreMulSum Data type must be half, float, bfloat16 or double");
  }
}

TorchCommHCCL::RedOpRAII::~RedOpRAII() {
  // HCCL doesn't need cleanup for reduce ops
}

size_t TorchCommHCCL::wordSize(HcclDataType type) const {
  switch (type) {
    case HCCL_DATA_TYPE_INT8:
    case HCCL_DATA_TYPE_UINT8:
      return 1;
    case HCCL_DATA_TYPE_FP16:
    case HCCL_DATA_TYPE_BFP16:
      return 2;
    case HCCL_DATA_TYPE_INT32:
    case HCCL_DATA_TYPE_UINT32:
    case HCCL_DATA_TYPE_FP32:
      return 4;
    case HCCL_DATA_TYPE_INT64:
    case HCCL_DATA_TYPE_FP64:
      return 8;
    default:
      return 0;
  }
}

HcclDataType TorchCommHCCL::getHcclDataType(const at::Tensor& tensor) {
  return getHcclDataTypeInternal(tensor);
}

TorchCommHCCL::RedOpRAII TorchCommHCCL::getHcclReduceOp(
    const ReduceOp& op,
    const HcclComm comm,
    const HcclDataType dataType) {
  switch (op) {
    case ReduceOp::RedOpType::SUM:
      return HCCL_REDUCE_SUM;
    case ReduceOp::RedOpType::PRODUCT:
      return HCCL_REDUCE_PROD;
    case ReduceOp::RedOpType::MIN:
      return HCCL_REDUCE_MIN;
    case ReduceOp::RedOpType::MAX:
      return HCCL_REDUCE_MAX;
    case ReduceOp::RedOpType::PREMUL_SUM:
      return RedOpRAII(op, comm, dataType, hccl_api_);
    case ReduceOp::RedOpType::AVG:
      // HCCL doesn't support AVG natively
      throw std::runtime_error("AVG reduce operation not supported in HCCL");
    case ReduceOp::RedOpType::BAND:
      // HCCL doesn't have bitwise AND
      throw std::runtime_error("Unsupported BAND reduce operation");
    case ReduceOp::RedOpType::BOR:
      // HCCL doesn't have bitwise OR
      throw std::runtime_error("Unsupported BOR reduce operation");
    case ReduceOp::RedOpType::BXOR:
      // HCCL doesn't have bitwise XOR
      throw std::runtime_error("Unsupported BXOR reduce operation");
    default:
      throw std::runtime_error("Unsupported reduce operation");
  }
}

void TorchCommHCCL::checkWorkQueue(bool isMainThread) {
  TorchWorkHCCL::WorkStatus status = workq_.garbageCollect(isMainThread);

  switch (status) {
    case TorchWorkHCCL::WorkStatus::TIMEDOUT:
      comm_state_ = CommState::TIMEOUT;
      break;
    case TorchWorkHCCL::WorkStatus::ERROR:
      comm_state_ = CommState::ERROR;
      break;
    default:
      // For COMPLETED, NOT_STARTED, and INPROGRESS, no state change needed
      break;
  }
}

// The timeout thread cannot make HCCL calls.  The only NPU call it can make
// is npuEventQuery.
void TorchCommHCCL::timeoutWatchdog() noexcept {
  TC_LOG(INFO) << "Timeout thread starting for rank: " << rank_;
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
    checkWorkQueue(false);
    if (comm_state_ != CommState::NORMAL &&
        options_.abort_process_on_timeout_or_error) {
      // Log the error and abort the process.  We cannot abort the HCCL
      // communicator as it is not safe to call HCCL operations from
      // multiple threads at the same time.
      if (comm_state_ == CommState::TIMEOUT) {
        TC_LOG(ERROR) << "Aborting process due to timeout on rank " << rank_
                      << " - timeout watchdog detected operation timeout";
      } else if (comm_state_ == CommState::ERROR) {
        TC_LOG(ERROR) << "Aborting process due to error on rank " << rank_
                      << " - timeout watchdog detected operation error. ";
      }
      abort();
    }
  }

  TC_LOG(INFO) << "Timeout thread exiting for rank: " << rank_;
}

void TorchCommHCCL::checkInitialized() const {
  if (init_state_ != InitializationState::INITIALIZED) {
    throw std::runtime_error("TorchCommHCCL not initialized");
  }
}

void TorchCommHCCL::checkAndAbortIfTimedOutOrError() {
  // First, check work queue status
  checkWorkQueue(true);

  if (comm_state_ == CommState::TIMEOUT) {
    abortHcclComm();
    if (options_.abort_process_on_timeout_or_error) {
      TC_LOG(ERROR) << "Aborting process due to timeout";
      abort();
    } else {
      throw std::runtime_error("HCCL operation timed out");
    }
  } else if (comm_state_ == CommState::ERROR) {
    HcclResult asyncErr;
    hccl_api_->commGetAsyncError(hccl_comm_, &asyncErr);
    HCCLException hcclException(*hccl_api_, "HCCL Async Error", asyncErr);
    abortHcclComm();
    if (options_.abort_process_on_timeout_or_error) {
      TC_LOG(ERROR) << "Aborting process due to error: "
                    << hcclException.what();
      abort();
    } else {
      throw hcclException;
    }
  }
}

c10::intrusive_ptr<TorchWorkHCCL> TorchCommHCCL::createWork(
    npuStream_t stream,
    std::chrono::milliseconds timeout,
    const std::vector<at::Tensor>& inputTensors) {
  // Only create the work object without enqueuing it
  auto work = c10::make_intrusive<TorchWorkHCCL>(
      shared_from_this(), stream, timeout, inputTensors, tracing_);
  return work;
}

void TorchCommHCCL::enqueueWork(
    c10::intrusive_ptr<TorchWorkHCCL> work,
    npuStream_t stream) {
  // Add work to stream's queue after events have been recorded
  workq_.enqueueWork(std::move(work), stream);
}

npuStream_t TorchCommHCCL::getOperationStream(bool async_op) {
  if (async_op) {
    // Get current PyTorch NPU stream for this device
    npuStream_t current_stream = npu_api_->getCurrentNPUStream(device_.index());

    // Record event on current stream and wait for it on internal stream
    NPU_CHECK(
        npu_api_,
        npu_api_->eventRecord(dependency_event_.value(), current_stream),
        "Failed to record dependency event");

    NPU_CHECK(
        npu_api_,
        npu_api_->streamWaitEvent(
            internal_stream_.value(), dependency_event_.value(), 0),
        "Failed to make internal stream wait for dependency event");

    return internal_stream_.value();
  } else {
    // Use the current PyTorch NPU stream for synchronous operations
    return npu_api_->getCurrentNPUStream(device_.index());
  }
}

void TorchCommHCCL::ensureTensorContiguous(const at::Tensor& tensor) {
  if (!tensor.is_contiguous()) {
    throw std::runtime_error("Tensor must be contiguous for HCCL operations");
  }
}

// Protected methods (not in the private section of the header)
npuEvent_t TorchCommHCCL::getEvent() {
  std::lock_guard<std::mutex> lock(event_pool_mutex_);

  if (!event_pool_.empty()) {
    npuEvent_t event = std::move(event_pool_.front());
    event_pool_.pop();
    return event;
  }

  // Create new event if pool is empty
  npuEvent_t event;
  NPU_CHECK(
      npu_api_,
      npu_api_->eventCreateWithFlags(event, /*flags=*/0),
      "Failed to create event");
  return event;
}

void TorchCommHCCL::returnEvent(npuEvent_t&& event) {
  std::lock_guard<std::mutex> lock(event_pool_mutex_);

  if (event_pool_.size() < max_event_pool_size_) {
    event_pool_.push(std::move(event));
  } else {
    // Pool is full, destroy the event
    NPU_CHECK(
        npu_api_, npu_api_->eventDestroy(event), "Failed to destroy event");
  }
}
} // namespace torch::comms
