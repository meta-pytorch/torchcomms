// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/ncclx/TorchCommNCCLX.hpp"
#include "comms/torchcomms/ncclx/TorchCommNCCLXCCA.hpp"

#include <stdexcept>
#include <string>
#include "comms/torchcomms/TorchCommLogging.hpp"
#include "nccl.h" // @manual

namespace torch {
namespace comms {

namespace {

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
      throw std::runtime_error("Unsupported tensor data type for NCCL");
  }
}

template <typename T, ncclDataType_t dataType>
void createPreMulSum(
    ncclRedOp_t* op,
    const PreMulSumFactorT& factor,
    const ncclComm_t& comm,
    NcclxApi* nccl_api) {
  const bool is_tensor = std::holds_alternative<at::Tensor>(factor);
  const auto residence = is_tensor ? ncclScalarDevice : ncclScalarHostImmediate;

  at::Tensor tensor = is_tensor ? std::get<at::Tensor>(factor) : at::Tensor();
  T scalar_factor = is_tensor ? T{} : static_cast<T>(std::get<double>(factor));
  void* scalar = is_tensor ? tensor.data_ptr() : &scalar_factor;

  TORCH_INTERNAL_ASSERT(
      is_tensor ? dataType == getNcclDataTypeInternal(tensor)
                : dataType != ncclBfloat16,
      "PreMulSum factor type must match input data type");
  nccl_api->redOpCreatePreMulSum(op, scalar, dataType, residence, comm);
}

} // namespace

TorchCommNCCLX::RedOpRAII::RedOpRAII(ncclRedOp_t op)
    : ncclRedOp_(op), comm_(nullptr) {}

TorchCommNCCLX::RedOpRAII::RedOpRAII(
    const ReduceOp& op,
    const ncclComm_t comm,
    const ncclDataType_t dataType,
    std::shared_ptr<NcclxApi> nccl_api)
    : comm_(comm), nccl_api_(std::move(nccl_api)) {
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
          &ncclRedOp_, factor, comm, nccl_api_.get());
      break;
    case ncclFloat32:
      createPreMulSum<float, ncclFloat32>(
          &ncclRedOp_, factor, comm, nccl_api_.get());
      break;
    case ncclBfloat16:
      createPreMulSum<float, ncclBfloat16>(
          &ncclRedOp_, factor, comm, nccl_api_.get());
      break;
    case ncclFloat64:
      createPreMulSum<double, ncclFloat64>(
          &ncclRedOp_, factor, comm, nccl_api_.get());
      break;
    default:
      throw std::runtime_error(
          "PreMulSum Data type must be half, float, bfloat16 or double");
  }
}

TorchCommNCCLX::RedOpRAII::~RedOpRAII() {
  if (comm_) {
    nccl_api_->redOpDestroy(ncclRedOp_, comm_);
  }
}

ncclDataType_t TorchCommNCCLX::getNcclDataType(const at::Tensor& tensor) {
  return getNcclDataTypeInternal(tensor);
}

TorchCommNCCLX::RedOpRAII TorchCommNCCLX::getNcclReduceOp(
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
      return ncclSum; // NCCL doesn't have bitwise AND, using SUM as fallback
    case ReduceOp::RedOpType::BOR:
      return ncclSum; // NCCL doesn't have bitwise OR, using SUM as fallback
    case ReduceOp::RedOpType::BXOR:
      return ncclSum; // NCCL doesn't have bitwise XOR, using SUM as fallback
    case ReduceOp::RedOpType::PREMUL_SUM:
      return RedOpRAII(op, comm, dataType, nccl_api_);
    case ReduceOp::RedOpType::AVG:
      return ncclAvg;
    default:
      throw std::runtime_error("Unsupported reduce operation");
  }
}

NcclxWindowCmpOp TorchCommNCCLX::getNcclSignalCmpOp(SignalCmpOp op) {
#ifdef NCCL_RMA_SUPPORTED
  switch (op) {
    case SignalCmpOp::EQ:
      return ncclCmpEQ;
    case SignalCmpOp::GE:
      return ncclCmpGE;
    case SignalCmpOp::LE:
      return ncclCmpLE;
    default:
      throw std::runtime_error("Unsupported signal compare operation");
  }
#else
  (void)op;
  throw std::logic_error("NCCL RMA is not supported");
#endif
}

void TorchCommNCCLX::checkWorkQueue() {
  TorchWorkNCCLX::WorkStatus status = workq_.garbageCollect();

  switch (status) {
    case TorchWorkNCCLX::WorkStatus::TIMEDOUT:
      comm_state_ = CommState::TIMEOUT;
      break;
    case TorchWorkNCCLX::WorkStatus::ERROR:
      comm_state_ = CommState::ERROR;
      break;
    default:
      // For COMPLETED, NOT_STARTED, and INPROGRESS, no state change needed
      break;
  }
}

// The timeout thread cannot make NCCL calls.  The only CUDA call it can make
// it cudaEventQuery.
void TorchCommNCCLX::timeoutWatchdog() noexcept {
  TC_LOG(INFO, this) << "Timeout thread starting for rank: " << rank_;
  while (!shutdown_) {
    {
      std::unique_lock<std::mutex> lock(timeout_mutex_);
      // Wait for a shorter interval to check work objects periodically
      // Wake up either after 60 seconds or immediately if shutdown is requested
      timeout_cv_.wait_for(lock, std::chrono::seconds(60), [this]() {
        return shutdown_.load();
      });

      // If we're shutting down, exit the loop
      if (shutdown_) {
        break;
      }
    }

    // Check work objects for completion or timeout
    checkWorkQueue();
    if (comm_state_ != CommState::NORMAL &&
        options_.abort_process_on_timeout_or_error) {
      // Log the error and abort the process.  We cannot abort the NCCL
      // communicator as it is not safe to call NCCL operations from
      // multiple threads at the same time.
      if (comm_state_ == CommState::TIMEOUT) {
        TC_LOG(ERROR, this)
            << "Aborting process due to timeout on rank " << rank_
            << " - timeout watchdog detected operation timeout";
      } else if (comm_state_ == CommState::ERROR) {
        TC_LOG(ERROR, this) << "Aborting process due to error on rank " << rank_
                            << " - timeout watchdog detected operation error. ";
      }
      abort();
    }
  }

  TC_LOG(INFO, this) << "Timeout thread exiting for rank: " << rank_;
}

void TorchCommNCCLX::checkInitialized() const {
  if (init_state_ != InitializationState::INITIALIZED) {
    throw std::runtime_error("TorchCommNCCLX not initialized");
  }
}

void TorchCommNCCLX::checkAndAbortIfTimedOutOrError() {
  // Nothing to check in graph capture mode
  if (getGraphCaptureMode()) {
    return;
  }

  // First, check work queue status
  checkWorkQueue();

  if (comm_state_ == CommState::TIMEOUT) {
    abortNcclComm();
    if (options_.abort_process_on_timeout_or_error) {
      TC_LOG(ERROR, this) << "Aborting process due to timeout";
      abort();
    } else {
      throw std::runtime_error("NCCL operation timed out");
    }
  } else if (comm_state_ == CommState::ERROR) {
    ncclResult_t asyncErr;
    nccl_api_->commGetAsyncError(nccl_comm_, &asyncErr);
    NCCLException ncclException(*nccl_api_, "NCCL Async Error", asyncErr);
    abortNcclComm();
    if (options_.abort_process_on_timeout_or_error) {
      TC_LOG(ERROR, this) << "Aborting process due to error: "
                          << ncclException.what();
      abort();
    } else {
      throw ncclException;
    }
  }
}

bool TorchCommNCCLX::getGraphCaptureMode() {
  cudaStream_t current_stream =
      cuda_api_->getCurrentCUDAStream(device_.index());
  cudaStreamCaptureStatus capture_status;

  cudaError_t err =
      cuda_api_->streamIsCapturing(current_stream, &capture_status);
  if (err == cudaSuccess) {
    return capture_status == cudaStreamCaptureStatusActive;
  }

  throw std::runtime_error(
      "Failed to check CUDA stream capture status: " +
      std::string(cuda_api_->getErrorString(err)));
}

std::shared_ptr<TorchWorkNCCLX> TorchCommNCCLX::createWork(
    cudaStream_t stream,
    std::chrono::milliseconds timeout,
    const std::vector<at::Tensor>& inputTensors) {
  // Only create the work object without enqueuing it
  auto work = std::make_shared<TorchWorkNCCLX>(
      shared_from_this(), stream, timeout, inputTensors);
  return work;
}

void TorchCommNCCLX::enqueueWork(
    std::shared_ptr<TorchWorkNCCLX> work,
    cudaStream_t stream) {
  // In graph capture mode, keep a reference to the work object to prevent
  // premature destruction until the graph gets destroyed, organized per graph
  if (getGraphCaptureMode()) {
    cudaStreamCaptureStatus capture_status;
    unsigned long long graph_id;
    cudaGraph_t graph;

    cudaError_t err = cuda_api_->streamGetCaptureInfo_v2(
        stream, &capture_status, &graph_id, &graph, nullptr, nullptr);
    if (err != cudaSuccess) {
      throw std::runtime_error(
          "Failed to get CUDA stream capture info: " +
          std::string(cuda_api_->getErrorString(err)));
    } else if (capture_status == cudaStreamCaptureStatusActive) {
      std::lock_guard<std::mutex> lock(graph_capture_work_mutex_);

      // Check if this is the first work object for this graph
      bool is_first_work = graph_capture_work_refs_[graph_id].empty();

      // Add work reference to the per-graph container
      graph_capture_work_refs_[graph_id].push_back(work);

      // If this is the first work object for this graph, set up automatic
      // cleanup
      if (is_first_work) {
        // Create cleanup data that will be passed to the callback
        auto* cleanup_data = new GraphCleanupData(this, graph_id);

        // Create a CUDA user object with our cleanup callback
        cudaUserObject_t user_object;
        err = cuda_api_->userObjectCreate(
            &user_object,
            cleanup_data,
            graphCleanupCallback,
            1, // initial reference count
            cudaUserObjectNoDestructorSync);
        if (err != cudaSuccess) {
          // If we failed to create the user object, clean up manually
          delete cleanup_data;
          throw std::runtime_error(
              "Failed to create user object: " +
              std::string(cuda_api_->getErrorString(err)));
        } else {
          // Retain the user object in the graph so it gets cleaned up when the
          // graph is destroyed
          err = cuda_api_->graphRetainUserObject(
              graph,
              user_object,
              1, // reference count
              cudaGraphUserObjectMove);
          if (err != cudaSuccess) {
            // If we failed to retain the user object, clean up manually
            delete cleanup_data;
            throw std::runtime_error(
                "Failed to retain user object: " +
                std::string(cuda_api_->getErrorString(err)));
          }
        }
      }
    }
  } else {
    // Add work to stream's queue after events have been recorded
    workq_.enqueueWork(std::move(work), stream);
  }
}

// Static callback function for CUDA user object cleanup
void CUDART_CB TorchCommNCCLX::graphCleanupCallback(void* userData) {
  auto* cleanup_data = static_cast<GraphCleanupData*>(userData);
  if (cleanup_data == nullptr || cleanup_data->comm == nullptr) {
    throw std::runtime_error("Invalid cleanup data");
  }

  // Clear the work references for this graph
  std::lock_guard<std::mutex> lock(
      cleanup_data->comm->graph_capture_work_mutex_);
  cleanup_data->comm->graph_capture_work_refs_.erase(cleanup_data->graph_id);

  // Clean up the cleanup data itself
  delete cleanup_data;
}

cudaStream_t TorchCommNCCLX::getOperationStream(bool async_op) {
  if (async_op) {
    // Get current PyTorch CUDA stream for this device
    cudaStream_t current_stream =
        cuda_api_->getCurrentCUDAStream(device_.index());

    // Record event on current stream and wait for it on internal stream
    CUDA_CHECK(
        cuda_api_,
        cuda_api_->eventRecord(dependency_event_, current_stream),
        "Failed to record dependency event");

    CUDA_CHECK(
        cuda_api_,
        cuda_api_->streamWaitEvent(internal_stream_, dependency_event_, 0),
        "Failed to make internal stream wait for dependency event");

    return internal_stream_;
  } else {
    // Use the current PyTorch CUDA stream for synchronous operations
    return cuda_api_->getCurrentCUDAStream(device_.index());
  }
}

void TorchCommNCCLX::ensureTensorContiguous(const at::Tensor& tensor) {
  if (!tensor.is_contiguous()) {
    throw std::runtime_error("Tensor must be contiguous for NCCL operations");
  }
}

// Protected methods (not in the private section of the header)
cudaEvent_t TorchCommNCCLX::getEvent() {
  std::lock_guard<std::mutex> lock(event_pool_mutex_);

  if (!event_pool_.empty()) {
    cudaEvent_t event = event_pool_.front();
    event_pool_.pop();
    return event;
  }

  // Create new event if pool is empty
  cudaEvent_t event;
  CUDA_CHECK(
      cuda_api_,
      cuda_api_->eventCreateWithFlags(&event, cudaEventDisableTiming),
      "Failed to create event");
  return event;
}

void TorchCommNCCLX::returnEvent(cudaEvent_t event) {
  std::lock_guard<std::mutex> lock(event_pool_mutex_);

  if (event_pool_.size() < max_event_pool_size_) {
    event_pool_.push(event);
  } else {
    // Pool is full, destroy the event
    CUDA_CHECK(
        cuda_api_, cuda_api_->eventDestroy(event), "Failed to destroy event");
  }
}

void TorchCommNCCLX::attachMemoryHook() {
  TC_LOG(INFO, this) << "Attaching memory hook comm=" << this;
  CachingAllocatorHook::getInstance().registerComm(this);
}

void TorchCommNCCLX::detachMemoryHook() {
  TC_LOG(INFO, this) << "Detaching memory hook comm=" << this;
  CachingAllocatorHook::getInstance().deregisterComm(this);
}

} // namespace comms
} // namespace torch
