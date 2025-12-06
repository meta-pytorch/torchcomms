// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/xccl/TorchCommXCCL.hpp"
// #include "comms/torchcomms/xccl/TorchCommXCCLCCA.hpp"

#include <stdexcept>
#include <string>
#include "comms/torchcomms/TorchCommLogging.hpp"
#include <oneapi/ccl.h>
#include <oneapi/ccl.hpp>

namespace torch {
namespace comms {

namespace {

onecclDataType_t getXcclDataTypeInternal(const at::Tensor& tensor) {
  switch (tensor.scalar_type()) {
    case at::ScalarType::Float:
      return onecclFloat32;
    case at::ScalarType::Double:
      return onecclFloat64;
    case at::ScalarType::Half:
      return onecclFloat16;
    case at::ScalarType::BFloat16:
      return onecclBfloat16;
    case at::ScalarType::Int:
      return onecclInt32;
    case at::ScalarType::Long:
      return onecclInt64;
    case at::ScalarType::Char:
      return onecclInt8;
    case at::ScalarType::Byte:
      return onecclUint8;
    default:
      throw std::runtime_error("Unsupported tensor data type for XCCL");
  }
}

template <typename T, onecclDataType_t dataType>
void createPreMulSum(
    onecclRedOp_t* op,
    const PreMulSumFactorT& factor,
    const onecclComm_t& comm,
    XcclApi* xccl_api) {
  const bool is_tensor = std::holds_alternative<at::Tensor>(factor);
  const auto residence = is_tensor ? onecclScalarDevice : onecclScalarHostImmediate;

  at::Tensor tensor = is_tensor ? std::get<at::Tensor>(factor) : at::Tensor();
  T scalar_factor = is_tensor ? T{} : static_cast<T>(std::get<double>(factor));
  void* scalar = is_tensor ? tensor.data_ptr() : &scalar_factor;

  TORCH_INTERNAL_ASSERT(
      is_tensor ? dataType == getXcclDataTypeInternal(tensor)
                : dataType != onecclBfloat16,
      "PreMulSum factor type must match input data type");
  xccl_api->redOpCreatePreMulSum(op, scalar, dataType, residence, comm);
}

} // namespace

TorchCommXCCL::RedOpRAII::RedOpRAII(onecclRedOp_t op)
    : xcclRedOp_(op), comm_(nullptr) {}

TorchCommXCCL::RedOpRAII::RedOpRAII(
    const ReduceOp& op,
    const onecclComm_t comm,
    const onecclDataType_t dataType,
    std::shared_ptr<XcclApi> xccl_api)
    : comm_(comm), xccl_api_(std::move(xccl_api)) {
  TORCH_INTERNAL_ASSERT(
      op == ReduceOp::RedOpType::PREMUL_SUM,
      "Constructing premul_sum RedOpRAII with non-premul_sum RedOpType");

  if (!op.factor().has_value()) {
    xcclRedOp_ = onecclSum;
    comm_ = nullptr;
    return;
  }

  const auto& factor = op.factor().value();
  switch (dataType) {
    case onecclFloat16:
      createPreMulSum<at::Half, onecclFloat16>(
          &xcclRedOp_, factor, comm, xccl_api_.get());
      break;
    case onecclFloat32:
      createPreMulSum<float, onecclFloat32>(
          &xcclRedOp_, factor, comm, xccl_api_.get());
      break;
    case onecclBfloat16:
      createPreMulSum<float, onecclBfloat16>(
          &xcclRedOp_, factor, comm, xccl_api_.get());
      break;
    case onecclFloat64:
      createPreMulSum<double, onecclFloat64>(
          &xcclRedOp_, factor, comm, xccl_api_.get());
      break;
    default:
      throw std::runtime_error(
          "PreMulSum Data type must be half, float, bfloat16 or double");
  }
}

TorchCommXCCL::RedOpRAII::~RedOpRAII() {
  if (comm_) {
    xccl_api_->redOpDestroy(xcclRedOp_, comm_);
  }
}

size_t TorchCommXCCL::wordSize(onecclDataType_t type) const {
  switch (type) {
    case onecclChar:
#if XCCL_MAJOR >= 2
    // case onecclInt8:
    case onecclUint8:
#endif
// #if HAVE_FP8
//     case onecclFloat8e4m3:
//     case onecclFloat8e5m2:
// #endif
      return 1;
    case onecclHalf:
#if HAVE_BF16
    case onecclBfloat16:
#endif
      // case onecclFloat16:
      return 2;
    case onecclInt:
    case onecclFloat:
#if XCCL_MAJOR >= 2
    // case onecclInt32:
    case onecclUint32:
      // case onecclFloat32:
#endif
      return 4;
    case onecclInt64:
    case onecclUint64:
    case onecclDouble:
      // case onecclFloat64:
      return 8;
    default:
      return 0;
  }
}

onecclDataType_t TorchCommXCCL::getXcclDataType(const at::Tensor& tensor) {
  return getXcclDataTypeInternal(tensor);
}

TorchCommXCCL::RedOpRAII TorchCommXCCL::getXcclReduceOp(
    const ReduceOp& op,
    const onecclComm_t comm,
    const onecclDataType_t dataType) {
  switch (op) {
    case ReduceOp::RedOpType::SUM:
      return onecclSum;
    case ReduceOp::RedOpType::PRODUCT:
      return onecclProd;
    case ReduceOp::RedOpType::MIN:
      return onecclMin;
    case ReduceOp::RedOpType::MAX:
      return onecclMax;
    case ReduceOp::RedOpType::BAND:
      return onecclSum; // XCCL doesn't have bitwise AND, using SUM as fallback
    case ReduceOp::RedOpType::BOR:
      return onecclSum; // XCCL doesn't have bitwise OR, using SUM as fallback
    case ReduceOp::RedOpType::BXOR:
      return onecclSum; // XCCL doesn't have bitwise XOR, using SUM as fallback
    case ReduceOp::RedOpType::PREMUL_SUM:
      return RedOpRAII(op, comm, dataType, xccl_api_);
    case ReduceOp::RedOpType::AVG:
      return onecclAvg;
    default:
      throw std::runtime_error("Unsupported reduce operation");
  }
}

void TorchCommXCCL::checkWorkQueue(bool isMainThread) {
  TorchWorkXCCL::WorkStatus status = workq_.garbageCollect(isMainThread);

  switch (status) {
    case TorchWorkXCCL::WorkStatus::TIMEDOUT:
      comm_state_ = CommState::TIMEOUT;
      break;
    case TorchWorkXCCL::WorkStatus::ERROR:
      comm_state_ = CommState::ERROR;
      break;
    default:
      // For COMPLETED, NOT_STARTED, and INPROGRESS, no state change needed
      break;
  }
}

// The timeout thread cannot make XCCL calls.  The only XPU call it can make
// it xpuEventQuery.
void TorchCommXCCL::timeoutWatchdog() noexcept {
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
      // Log the error and abort the process.  We cannot abort the XCCL
      // communicator as it is not safe to call XCCL operations from
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

void TorchCommXCCL::checkInitialized() const {
  if (init_state_ != InitializationState::INITIALIZED) {
    throw std::runtime_error("TorchCommXCCL not initialized");
  }
}

void TorchCommXCCL::checkAndAbortIfTimedOutOrError() {
  // Nothing to check in graph capture mode
  // XPU doesn't support graph capture yet
  // if (getGraphCaptureMode()) {
  //   return;
  // }

  // First, check work queue status
  checkWorkQueue(true);

  if (comm_state_ == CommState::TIMEOUT) {
    abortXcclComm();
    if (options_.abort_process_on_timeout_or_error) {
      TC_LOG(ERROR) << "Aborting process due to timeout";
      abort();
    } else {
      throw std::runtime_error("XCCL operation timed out");
    }
  } else if (comm_state_ == CommState::ERROR) {
    onecclResult_t asyncErr;
    xccl_api_->commGetAsyncError(xccl_comm_, &asyncErr);
    XCCLException xcclException(*xccl_api_, "XCCL Async Error", asyncErr);
    abortXcclComm();
    if (options_.abort_process_on_timeout_or_error) {
      TC_LOG(ERROR) << "Aborting process due to error: "
                    << xcclException.what();
      abort();
    } else {
      throw xcclException;
    }
  }
}

// bool TorchCommXCCL::getGraphCaptureMode() {
//   xpuStream_t current_stream =
//       xpu_api_->getCurrentXPUStream(device_.index());
//   xpuStreamCaptureStatus capture_status;

//   xpuError_t err =
//       xpu_api_->streamIsCapturing(current_stream, &capture_status);
//   if (err == xpuSuccess) {
//     return capture_status == xpuStreamCaptureStatusActive;
//   }

//   throw std::runtime_error(
//       "Failed to check XPU stream capture status: " +
//       std::string(xpu_api_->getErrorString(err)));
// }

std::shared_ptr<TorchWorkXCCL> TorchCommXCCL::createWork(
    xpuStream_t stream,
    std::chrono::milliseconds timeout,
    const std::vector<at::Tensor>& inputTensors) {
  // Only create the work object without enqueuing it
  auto work = std::make_shared<TorchWorkXCCL>(
      shared_from_this(), stream, timeout, inputTensors, tracing_);
  return work;
}

void TorchCommXCCL::enqueueWork(
    std::shared_ptr<TorchWorkXCCL> work,
    xpuStream_t stream) {
  // In graph capture mode, keep a reference to the work object to prevent
  // premature destruction until the graph gets destroyed, organized per graph
  // if (getGraphCaptureMode()) {
  //   xpuStreamCaptureStatus capture_status;
  //   unsigned long long graph_id;
  //   xpuGraph_t graph;

  //   xpuError_t err = xpu_api_->streamGetCaptureInfo_v2(
  //       stream, &capture_status, &graph_id, &graph, nullptr, nullptr);
  //   if (err != xpuSuccess) {
  //     throw std::runtime_error(
  //         "Failed to get XPU stream capture info: " +
  //         std::string(xpu_api_->getErrorString(err)));
  //   } else if (capture_status == xpuStreamCaptureStatusActive) {
  //     std::lock_guard<std::mutex> lock(graph_capture_work_mutex_);

  //     // Check if this is the first work object for this graph
  //     bool is_first_work = graph_capture_work_refs_[graph_id].empty();

  //     // Add work reference to the per-graph container
  //     graph_capture_work_refs_[graph_id].push_back(work);

  //     // If this is the first work object for this graph, set up automatic
  //     // cleanup
  //     if (is_first_work) {
  //       // Create cleanup data that will be passed to the callback
  //       auto* cleanup_data = new GraphCleanupData(this, graph_id);

  //       // Create a XPU user object with our cleanup callback
  //       xpuUserObject_t user_object;
  //       err = xpu_api_->userObjectCreate(
  //           &user_object,
  //           cleanup_data,
  //           graphCleanupCallback,
  //           1, // initial reference count
  //           xpuUserObjectNoDestructorSync);
  //       if (err != xpuSuccess) {
  //         // If we failed to create the user object, clean up manually
  //         delete cleanup_data;
  //         throw std::runtime_error(
  //             "Failed to create user object: " +
  //             std::string(xpu_api_->getErrorString(err)));
  //       } else {
  //         // Retain the user object in the graph so it gets cleaned up when the
  //         // graph is destroyed
  //         err = xpu_api_->graphRetainUserObject(
  //             graph,
  //             user_object,
  //             1, // reference count
  //             xpuGraphUserObjectMove);
  //         if (err != xpuSuccess) {
  //           // If we failed to retain the user object, clean up manually
  //           delete cleanup_data;
  //           throw std::runtime_error(
  //               "Failed to retain user object: " +
  //               std::string(xpu_api_->getErrorString(err)));
  //         }
  //       }
  //     }
  //   }
  // } else {
  // Add work to stream's queue after events have been recorded
  workq_.enqueueWork(std::move(work), stream);
  // }
}

// // Static callback function for XPU user object cleanup
// void XPURT_CB TorchCommXCCL::graphCleanupCallback(void* userData) {
//   auto* cleanup_data = static_cast<GraphCleanupData*>(userData);
//   if (cleanup_data == nullptr || cleanup_data->comm == nullptr) {
//     throw std::runtime_error("Invalid cleanup data");
//   }

//   // Clear the work references for this graph
//   std::lock_guard<std::mutex> lock(
//       cleanup_data->comm->graph_capture_work_mutex_);
//   cleanup_data->comm->graph_capture_work_refs_.erase(cleanup_data->graph_id);

//   // Clean up the cleanup data itself
//   delete cleanup_data;
// }

xpuStream_t TorchCommXCCL::getOperationStream(bool async_op) {
  if (async_op) {
    // Get current PyTorch XPU stream for this device
    xpuStream_t current_stream =
        xpu_api_->getCurrentXPUStream(device_.index());

    // Record event on current stream and wait for it on internal stream
    XPU_CHECK(
        xpu_api_,
        xpu_api_->eventRecord(dependency_event_.value(), current_stream),
        "Failed to record dependency event");

    XPU_CHECK(
        xpu_api_,
        xpu_api_->streamWaitEvent(internal_stream_.value(), dependency_event_.value(), 0),
        "Failed to make internal stream wait for dependency event");

    return internal_stream_.value();
  } else {
    // Use the current PyTorch XPU stream for synchronous operations
    return xpu_api_->getCurrentXPUStream(device_.index());
  }
}

void TorchCommXCCL::ensureTensorContiguous(const at::Tensor& tensor) {
  if (!tensor.is_contiguous()) {
    throw std::runtime_error("Tensor must be contiguous for XCCL operations");
  }
}

// Protected methods (not in the private section of the header)
xpuEvent_t TorchCommXCCL::getEvent() {
  std::lock_guard<std::mutex> lock(event_pool_mutex_);

  if (!event_pool_.empty()) {
    xpuEvent_t event = std::move(event_pool_.front());
    event_pool_.pop();
    return event;
  }

  // Create new event if pool is empty
  xpuEvent_t event;
  XPU_CHECK(
      xpu_api_,
      xpu_api_->eventCreateWithFlags(event, /*flags=*/0),
      "Failed to create event");
  return event;
}

void TorchCommXCCL::returnEvent(xpuEvent_t&& event) {
  std::lock_guard<std::mutex> lock(event_pool_mutex_);

  if (event_pool_.size() < max_event_pool_size_) {
    event_pool_.push(std::move(event));
  } else {
    // Pool is full, destroy the event
    XPU_CHECK(
        xpu_api_, xpu_api_->eventDestroy(event), "Failed to destroy event");
  }
}

void TorchCommXCCL::attachMemoryHook() {
  // NOTE: CachingAllocatorHook is not implemented for XPU/SYCL yet
  // TODO: Implement XPU caching allocator hook when available
  // CachingAllocatorHook::getInstance().registerComm(this);
}

void TorchCommXCCL::detachMemoryHook() {
  // NOTE: CachingAllocatorHook is not implemented for XPU/SYCL yet
  // TODO: Implement XPU caching allocator hook when available
  // CachingAllocatorHook::getInstance().deregisterComm(this);
}

} // namespace comms
} // namespace torch
