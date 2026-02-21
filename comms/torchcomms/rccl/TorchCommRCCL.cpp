// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/rccl/TorchCommRCCL.hpp"

#include <cstdlib>
#include <set>
#include <stdexcept>
#include <string>

#include <ATen/hip/HIPContext.h> // @manual=//caffe2:ATen-custom-hip
#include <fmt/core.h>

#include "comms/torchcomms/TorchCommFactory.hpp"
#include "comms/torchcomms/TorchCommLogging.hpp"
#include "comms/torchcomms/TorchCommTracing.hpp"
#include "comms/torchcomms/rccl/TorchCommRCCLBootstrap.hpp"
#include "rccl.h" // @manual

namespace torch::comms {

namespace {
// Hint key prefix and names for RCCL backend configuration
constexpr std::string_view kHintPrefix = "torchcomm::rccl::";
constexpr std::string_view kHintHighPriorityStream =
    "torchcomm::rccl::high_priority_stream";
constexpr std::string_view kHintMaxEventPoolSize =
    "torchcomm::rccl::max_event_pool_size";
} // namespace

ncclResult_t RCCLException::getResult() const {
  return result_;
}

TorchCommRCCL::TorchCommRCCL()
    : nccl_comm_{nullptr},
      device_(at::kHIP),
      options_(),
      init_state_(InitializationState::UNINITIALIZED),
      shutdown_(false) {}

TorchCommRCCL::TorchCommRCCL(
    const ncclComm_t nccl_comm,
    at::Device device,
    const CommOptions& options)
    : nccl_comm_(nccl_comm),
      device_(device),
      options_(options),
      init_state_(InitializationState::UNINITIALIZED),
      shutdown_(false) {}

TorchCommRCCL::~TorchCommRCCL() {
  if (init_state_ == InitializationState::INITIALIZED) {
    TC_LOG(WARNING, this)
        << "TorchCommRCCL " << name_
        << " was not finalized before destruction. "
        << "This may indicate a resource leak. Please call finalize() explicitly.";

    // Signal shutdown to timeout watchdog thread to prevent it from accessing
    // this object after destruction
    shutdown_ = true;

    // Wake up the timeout watchdog thread
    {
      std::lock_guard<std::mutex> lock(timeout_mutex_);
      timeout_cv_.notify_all();
    }

    // Wait for timeout thread to finish. If we're being called from within
    // the timeout thread itself (e.g., garbageCollect popped a work item whose
    // destruction released the last shared_ptr to this comm), we must detach
    // instead of join to avoid a deadlock.
    if (timeout_thread_.joinable()) {
      if (std::this_thread::get_id() != timeout_thread_.get_id()) {
        timeout_thread_.join();
      } else {
        timeout_thread_.detach();
      }
    }

    // Abort the RCCL communicator since we can't do a clean finalization
    // Note: We don't call the full abortRcclComm() to avoid potential abort()
    // calls from options_.abort_process_on_timeout_or_error
    if (nccl_comm_) {
      // Best effort to abort the communicator - ignore errors since we're
      // in the destructor
      if (rccl_api_) {
        (void)rccl_api_->commAbort(nccl_comm_);
      }
      nccl_comm_ = nullptr;
    }
  }

  // We need to detach the memory hook in case finalize is not called,
  // so that we don't encounter a memory corruption.
  detachMemoryHook();
}

void TorchCommRCCL::init(
    at::Device device,
    const std::string& name,
    const CommOptions& options) {
  // Initialize private members
  device_ = device;
  name_ = name;
  options_ = options;

  // Only initialize once
  if (init_state_ == InitializationState::INITIALIZED) {
    throw std::runtime_error("TorchCommRCCL already initialized");
  } else if (init_state_ == InitializationState::FINALIZED) {
    throw std::runtime_error("TorchCommRCCL already finalized");
  }
  init_state_ = InitializationState::INITIALIZED;

  // Initialize default NCCL API implementation if not already set
  if (!rccl_api_) {
    rccl_api_ = std::make_unique<DefaultRcclApi>();
  }

  // Initialize default HIP API implementation if not already set
  if (!hip_api_) {
    hip_api_ = std::make_unique<DefaultHipApi>();
  }

  if (device_.index() == -1 || nccl_comm_ == nullptr) {
    auto bootstrap = std::make_unique<TorchCommRCCLBootstrap>(
        options_.store, device_, rccl_api_, hip_api_, options_.timeout);
    device_ = bootstrap->getDevice();

    if (nccl_comm_ == nullptr) {
      nccl_comm_ = bootstrap->createNcclComm(name);
    }
  }

  // Set HIP device and verify it's accessible
  HIP_CHECK(
      hip_api_,
      hip_api_->setDevice(device_.index()),
      fmt::format("Failed to set CUDA device to {}", device_.index()));

  // Verify device properties and memory availability
  hipDeviceProp_t device_prop = {};
  HIP_CHECK(
      hip_api_,
      hip_api_->getDeviceProperties(&device_prop, device_.index()),
      fmt::format(
          "Failed to get device properties for device {}", device_.index()));

  // Check available memory
  size_t free_memory, total_memory;
  HIP_CHECK(
      hip_api_,
      hip_api_->memGetInfo(&free_memory, &total_memory),
      fmt::format("Failed to get memory info for device {}", device_.index()));

  // Read hints and store them
  for (const auto& hint : options_.hints) {
    const std::string& key = hint.first;
    const std::string& val = hint.second;
    if (key.starts_with(kHintPrefix)) {
      if (key == kHintHighPriorityStream) {
        high_priority_stream_ = string_to_bool(val);
      } else {
        throw std::runtime_error("Unrecognized hint " + key);
      }
    } else {
      // Ignore keys that do not start with "torchcomm::rccl::"
    }
  }

  // Create internal stream
  //
  // Default priority is 0 as per NVIDIA docs (https://fburl.com/2xb0iqwl).
  int stream_priority = 0;

  // Check for high priority stream hint
  if (high_priority_stream_) {
    int leastPriority, greatestPriority;
    HIP_CHECK(
        hip_api_,
        hip_api_->getStreamPriorityRange(&leastPriority, &greatestPriority),
        "Failed to get stream priority range");
    stream_priority = greatestPriority;
  }

  HIP_CHECK(
      hip_api_,
      hip_api_->streamCreateWithPriority(
          &internal_stream_, hipStreamNonBlocking, stream_priority),
      fmt::format(
          "Failed to create internal CUDA stream on device {}",
          device_.index()));

  // Create dependency event for stream synchronization
  HIP_CHECK(
      hip_api_,
      hip_api_->eventCreate(&dependency_event_),
      fmt::format(
          "Failed to create dependency event on device {}", device_.index()));

  // Allocate CUDA buffer for barrier operations
  HIP_CHECK(
      hip_api_,
      hip_api_->malloc(&barrier_buffer_, sizeof(float)),
      "Failed to allocate barrier buffer");

  if (options_.hints.find(std::string(kHintMaxEventPoolSize)) !=
      options_.hints.end()) {
    max_event_pool_size_ =
        std::stoull(options_.hints.at(std::string(kHintMaxEventPoolSize)));
  } else {
    max_event_pool_size_ = kMaxEventPoolSize;
  }

  // Give up our internal reference to the store object here.  The caller
  // would still need to keep a reference to the store object till the init
  // call returns, at which point the NCCL communicator would already be
  // created.
  if (options_.store) {
    options_.store.reset();
  }

  RCCL_CHECK(
      rccl_api_,
      nccl_comm_,
      rccl_api_->commUserRank(nccl_comm_, &rank_),
      "RCCL User Rank failed");

  tryTorchCommLoggingInit("torchcomm");

  RCCL_CHECK(
      rccl_api_,
      nccl_comm_,
      rccl_api_->commCount(nccl_comm_, &comm_size_),
      "RCCL Count failed");

  TorchCommTracingGuard tracingGuard(name_, comm_size_, "init", rank_);

  // Start timeout watchdog thread
  timeout_thread_ = std::thread(&TorchCommRCCL::timeoutWatchdog, this);

  // Register comm with CachingAllocator
  attachMemoryHook();
}

void TorchCommRCCL::finalize() {
  if (init_state_ == InitializationState::UNINITIALIZED) {
    throw std::runtime_error("TorchCommRCCL not initialized");
  } else if (init_state_ == InitializationState::FINALIZED) {
    throw std::runtime_error("TorchCommRCCL already finalized");
  }
  init_state_ = InitializationState::FINALIZED;

  // Signal shutdown to timeout watchdog
  shutdown_ = true;

  // Wake up the timeout watchdog thread
  {
    std::lock_guard<std::mutex> lock(timeout_mutex_);
    timeout_cv_.notify_all();
  }

  // Wait for timeout thread to finish
  if (timeout_thread_.joinable()) {
    timeout_thread_.join();
  }

  // No need for locks after timeout thread has joined
  // Wait for all pending work objects to complete
  while (!stream_work_queues_.empty() && comm_state_ == CommState::NORMAL) {
    garbageCollectWorkQueues();
  }

  // Clear all work queues
  stream_work_queues_.clear();

  if (comm_state_ == CommState::TIMEOUT) {
    abortRcclComm();
    throw std::runtime_error("Work timed out during finalize");
  } else if (comm_state_ == CommState::ERROR) {
    ncclResult_t asyncErr;
    RCCL_CHECK(
        rccl_api_,
        nccl_comm_,
        rccl_api_->commGetAsyncError(nccl_comm_, &asyncErr),
        "failed to get async error");
    RCCLException RCCLException(
        *rccl_api_, "NCCL Async Error", asyncErr, nccl_comm_);
    abortRcclComm();
    throw RCCLException;
  }

  // Clear the completed works queue
  std::queue<c10::intrusive_ptr<TorchWorkRCCL>> empty;
  std::swap(completed_works_, empty);

  // Clean up event pool
  {
    std::lock_guard<std::mutex> lock(event_pool_mutex_);
    while (!event_pool_.empty()) {
      hipEvent_t event = event_pool_.front();
      event_pool_.pop();
      HIP_CHECK(
          hip_api_, hip_api_->eventDestroy(event), "Failed to destroy event");
    }
  }

  // Free barrier buffer. TODO: handle errors on cuda free and stream destroy
  if (barrier_buffer_) {
    HIP_CHECK(
        hip_api_,
        hip_api_->free(barrier_buffer_),
        "Failed to free barrier buffer");
    barrier_buffer_ = nullptr;
  }

  // Destroy dependency event
  if (dependency_event_) {
    HIP_CHECK(
        hip_api_,
        hip_api_->eventDestroy(dependency_event_),
        "Failed to destroy dependency event");
    dependency_event_ = nullptr;
  }

  // Destroy internal stream
  if (internal_stream_) {
    HIP_CHECK(
        hip_api_,
        hip_api_->streamDestroy(internal_stream_),
        "Failed to destroy internal stream");
    internal_stream_ = nullptr;
  }

  // Destroy NCCL communicator
  // TODO: should probably not call this after calling abort.
  if (nccl_comm_) {
    detachMemoryHook();
    // Deregister comm from the CachingAllocator
    RCCL_CHECK(
        rccl_api_,
        nccl_comm_,
        rccl_api_->commDestroy(nccl_comm_),
        "RCCL Destroy failed");
    nccl_comm_ = nullptr;
  }
}

void TorchCommRCCL::abortRcclComm() {
  detachMemoryHook();
  if (nccl_comm_) {
    RCCL_CHECK(
        rccl_api_,
        nccl_comm_,
        rccl_api_->commAbort(nccl_comm_),
        "RCCL Abort failed");
    nccl_comm_ = nullptr;
  }
}

int TorchCommRCCL::getRank() const {
  checkInitialized();

  int rank;
  RCCL_CHECK(
      rccl_api_,
      nccl_comm_,
      rccl_api_->commUserRank(nccl_comm_, &rank),
      "NCCL User Rank failed");
  return rank;
}

int TorchCommRCCL::getSize() const {
  checkInitialized();

  int comm_size;
  RCCL_CHECK(
      rccl_api_,
      nccl_comm_,
      rccl_api_->commCount(nccl_comm_, &comm_size),
      "NCCL Count failed");
  return comm_size;
}

static inline std::chrono::milliseconds getOperationTimeout(
    std::chrono::milliseconds timeout,
    std::chrono::milliseconds default_timeout) {
  // If timeout is kNoTimeout (0ms), use the default timeout from options
  if (timeout == kNoTimeout) {
    return default_timeout;
  }
  return timeout;
}

// Point-to-Point Operations
c10::intrusive_ptr<TorchWork> TorchCommRCCL::send(
    const at::Tensor& tensor,
    int dst,
    bool async_op,
    const SendOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);

  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "send", dst, tensor, tensor);

  hipStream_t stream = getOperationStream(async_op);
  auto work = createWork(
      stream, getOperationTimeout(options.timeout, options_.timeout), {tensor});

  // Record start event before NCCL operation
  work->recordStart("send");

  RCCL_CHECK(
      rccl_api_,
      nccl_comm_,
      rccl_api_->send(
          tensor.data_ptr(),
          tensor.numel(),
          getNcclDataType(tensor),
          dst,
          nccl_comm_,
          stream),
      "NCCL Send failed");

  // Record end event after NCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommRCCL::recv(
    at::Tensor& tensor,
    int src,
    bool async_op,
    const RecvOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);

  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "recv", src, tensor, tensor);

  hipStream_t stream = getOperationStream(async_op);
  auto work = createWork(
      stream, getOperationTimeout(options.timeout, options_.timeout));

  // Record start event before NCCL operation
  work->recordStart("recv");

  RCCL_CHECK(
      rccl_api_,
      nccl_comm_,
      rccl_api_->recv(
          tensor.data_ptr(),
          tensor.numel(),
          getNcclDataType(tensor),
          src,
          nccl_comm_,
          stream),
      "NCCL Recv failed");

  // Record end event after NCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

// Batch P2P Operations
c10::intrusive_ptr<TorchWork> TorchCommRCCL::batch_op_issue(
    const std::vector<BatchSendRecv::P2POp>& ops,
    bool async_op,
    const BatchP2POptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();

  if (ops.empty()) {
    throw std::runtime_error("Cannot issue empty batch operation");
  }

  // Collect input and output tensors for work tracking
  std::vector<at::Tensor> input_tensors;
  std::vector<at::Tensor> output_tensors;

  for (const auto& op : ops) {
    if (op.type == BatchSendRecv::P2POp::OpType::SEND) {
      at::Tensor tensor = op.tensor;
      ensureTensorContiguous(tensor);
      input_tensors.push_back(tensor);
    } else if (op.type == BatchSendRecv::P2POp::OpType::RECV) {
      at::Tensor tensor = op.tensor;
      ensureTensorContiguous(tensor);
      output_tensors.push_back(tensor);
    } else {
      throw std::runtime_error("Unknown op type");
    }
  }

  TorchCommTracingGuard tracingGuard(
      name_,
      comm_size_,
      "batch_op_issue",
      rank_,
      input_tensors,
      output_tensors);

  hipStream_t stream = getOperationStream(async_op);
  auto work = createWork(
      stream,
      getOperationTimeout(options.timeout, options_.timeout),
      input_tensors);

  // Record start event before NCCL operations
  work->recordStart("batch_op_issue");

  // Start NCCL group for batched operations
  RCCL_CHECK(
      rccl_api_, nccl_comm_, rccl_api_->groupStart(), "NCCL GroupStart failed");

  // Issue each operation individually
  for (const auto& op : ops) {
    if (op.type == BatchSendRecv::P2POp::OpType::SEND) {
      ncclResult_t result = rccl_api_->send(
          op.tensor.data_ptr(),
          op.tensor.numel(),
          getNcclDataType(op.tensor),
          op.peer,
          nccl_comm_,
          stream);

      if (result != ncclSuccess) {
        throw RCCLException(
            *rccl_api_,
            "NCCL Send failed in batch operation",
            result,
            nccl_comm_);
      }
    } else if (op.type == BatchSendRecv::P2POp::OpType::RECV) {
      ncclResult_t result = rccl_api_->recv(
          op.tensor.data_ptr(),
          op.tensor.numel(),
          getNcclDataType(op.tensor),
          op.peer,
          nccl_comm_,
          stream);

      if (result != ncclSuccess) {
        throw RCCLException(
            *rccl_api_,
            "NCCL Recv failed in batch operation",
            result,
            nccl_comm_);
      }
    }
  }

  // End NCCL group
  RCCL_CHECK(
      rccl_api_, nccl_comm_, rccl_api_->groupEnd(), "NCCL GroupEnd failed");

  // Record end event after NCCL operations
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

// Collective Operations
c10::intrusive_ptr<TorchWork> TorchCommRCCL::broadcast(
    at::Tensor& tensor,
    int root,
    bool async_op,
    const BroadcastOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);

  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "broadcast", root, tensor, tensor);

  hipStream_t stream = getOperationStream(async_op);

  auto work = createWork(
      stream, getOperationTimeout(options.timeout, options_.timeout), {tensor});

  // Record start event before NCCL operation
  work->recordStart("broadcast");

  RCCL_CHECK(
      rccl_api_,
      nccl_comm_,
      rccl_api_->bcast(
          tensor.data_ptr(),
          tensor.numel(),
          getNcclDataType(tensor),
          root,
          nccl_comm_,
          stream),
      "NCCL Broadcast failed");

  // Record end event after NCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommRCCL::all_reduce(
    at::Tensor& tensor,
    const ReduceOp& op,
    bool async_op,
    const AllReduceOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);

  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "all_reduce", rank_, tensor, tensor);

  hipStream_t stream = getOperationStream(async_op);
  auto work = createWork(
      stream, getOperationTimeout(options.timeout, options_.timeout), {tensor});

  // Record start event before NCCL operation
  work->recordStart("all_reduce");

  auto dataType = getNcclDataType(tensor);
  RCCL_CHECK(
      rccl_api_,
      nccl_comm_,
      rccl_api_->allReduce(
          tensor.data_ptr(),
          tensor.data_ptr(), // In-place operation
          tensor.numel(),
          dataType,
          getNcclReduceOp(op, nccl_comm_, dataType),
          nccl_comm_,
          stream),
      "NCCL AllReduce failed");

  // Record end event after NCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommRCCL::reduce(
    const at::Tensor& tensor,
    int root,
    const ReduceOp& op,
    bool async_op,
    const ReduceOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);

  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "reduce", root, tensor, tensor);

  hipStream_t stream = getOperationStream(async_op);
  auto work = createWork(
      stream, getOperationTimeout(options.timeout, options_.timeout), {tensor});

  // Record start event before NCCL operation
  work->recordStart("reduce");

  auto dataType = getNcclDataType(tensor);
  RCCL_CHECK(
      rccl_api_,
      nccl_comm_,
      rccl_api_->reduce(
          tensor.data_ptr(),
          rank_ == root ? tensor.data_ptr() : nullptr,
          tensor.numel(),
          dataType,
          getNcclReduceOp(op, nccl_comm_, dataType),
          root,
          nccl_comm_,
          stream),
      "NCCL Reduce failed");

  // Record end event after NCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommRCCL::all_gather(
    const std::vector<at::Tensor>& tensor_list,
    const at::Tensor& tensor,
    bool async_op,
    const AllGatherOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  if (tensor_list.size() != static_cast<size_t>(comm_size_)) {
    throw std::runtime_error(
        "tensor_list size must equal comm_size for all_gather");
  }

  // Ensure input tensor is contiguous
  ensureTensorContiguous(tensor);

  // Check that all output tensors are contiguous and have correct size
  for (const auto& t : tensor_list) {
    ensureTensorContiguous(t);
    if (t.numel() != tensor.numel()) {
      throw std::runtime_error(
          "All tensors in tensor_list must have same size as input tensor");
    }
  }

  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "all_gather", rank_, tensor_list, {tensor});

  hipStream_t stream = getOperationStream(async_op);
  auto work = createWork(
      stream, getOperationTimeout(options.timeout, options_.timeout), {tensor});

  work->recordStart("all_gather");

  // Use multiple broadcast operations for all_gather
  RCCL_CHECK(
      rccl_api_, nccl_comm_, rccl_api_->groupStart(), "NCCL GroupStart failed");

  for (int i = 0; i < comm_size_; ++i) {
    ncclResult_t opResult = rccl_api_->broadcast(
        tensor.data_ptr(),
        tensor_list[i].data_ptr(),
        tensor.numel(),
        getNcclDataType(tensor_list[i]),
        i,
        nccl_comm_,
        stream);
    if (opResult != ncclSuccess) {
      throw RCCLException(
          *rccl_api_,
          "NCCL Broadcast failed in all_gather",
          opResult,
          nccl_comm_);
    }
  }

  RCCL_CHECK(
      rccl_api_, nccl_comm_, rccl_api_->groupEnd(), "NCCL GroupEnd failed");

  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommRCCL::all_gather_v(
    const std::vector<at::Tensor>& tensor_list,
    const at::Tensor& tensor,
    bool async_op,
    const AllGatherOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  if (tensor_list.size() != static_cast<size_t>(comm_size_)) {
    throw std::runtime_error(
        "tensor_list size must equal comm_size for all_gather_v");
  }

  ensureTensorContiguous(tensor);
  for (const auto& t : tensor_list) {
    ensureTensorContiguous(t);
  }

  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "all_gather_v", rank_, tensor_list, {tensor});

  hipStream_t stream = getOperationStream(async_op);
  auto work = async_op
      ? createWork(
            stream,
            getOperationTimeout(options.timeout, options_.timeout),
            tensor)
      : createWork(
            stream, getOperationTimeout(options.timeout, options_.timeout));

  work->recordStart("all_gather_v");

  // Use multiple broadcast operations for all_gather_v
  RCCL_CHECK(
      rccl_api_, nccl_comm_, rccl_api_->groupStart(), "NCCL GroupStart failed");

  for (int i = 0; i < comm_size_; ++i) {
    // For all_gather_v, each rank broadcasts its input tensor to all others
    auto& output = tensor_list[i];
    auto& input = (i == rank_) ? tensor : output;
    if (input.numel() != output.numel()) {
      throw std::runtime_error(
          "Output tensor size must equal input tensor size for all_gather_v");
    }
    ncclResult_t opResult = rccl_api_->broadcast(
        input.data_ptr(),
        output.data_ptr(),
        input.numel(),
        getNcclDataType(output),
        i,
        nccl_comm_,
        stream);
    if (opResult != ncclSuccess) {
      throw RCCLException(
          *rccl_api_,
          "NCCL Broadcast failed in all_gather_v",
          opResult,
          nccl_comm_);
    }
  }

  RCCL_CHECK(
      rccl_api_, nccl_comm_, rccl_api_->groupEnd(), "NCCL GroupEnd failed");

  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommRCCL::all_gather_single(
    at::Tensor& output,
    const at::Tensor& input,
    bool async_op,
    const AllGatherSingleOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(output);
  ensureTensorContiguous(input);

  if (output.numel() != input.numel() * comm_size_) {
    throw std::runtime_error(
        "Output tensor size must be input_size * comm_size for all_gather_single");
  }

  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "all_gather_single", rank_, input, output);

  hipStream_t stream = getOperationStream(async_op);
  auto work = createWork(
      stream, getOperationTimeout(options.timeout, options_.timeout), {input});

  work->recordStart("all_gather_single");

  RCCL_CHECK(
      rccl_api_,
      nccl_comm_,
      rccl_api_->allGather(
          input.data_ptr(),
          output.data_ptr(),
          input.numel(),
          getNcclDataType(input),
          nccl_comm_,
          stream),
      "NCCL AllGather failed");

  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommRCCL::reduce_scatter(
    at::Tensor& output,
    const std::vector<at::Tensor>& input_list,
    const ReduceOp& op,
    bool async_op,
    const ReduceScatterOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(output);

  if (input_list.size() != static_cast<size_t>(comm_size_)) {
    throw std::runtime_error(
        "input_list size must equal comm_size for reduce_scatter");
  }

  // Check that all input tensors are contiguous and have correct size
  for (const auto& t : input_list) {
    ensureTensorContiguous(t);
    if (t.numel() != output.numel()) {
      throw std::runtime_error(
          "All input tensors must have same size as output tensor");
    }
  }

  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "reduce_scatter", rank_, input_list, {output});

  hipStream_t stream = getOperationStream(async_op);
  auto work = createWork(
      stream,
      getOperationTimeout(options.timeout, options_.timeout),
      input_list);

  work->recordStart("reduce_scatter");

  // Use multiple reduce operations for reduce_scatter
  RCCL_CHECK(
      rccl_api_, nccl_comm_, rccl_api_->groupStart(), "NCCL GroupStart failed");

  for (int i = 0; i < comm_size_; ++i) {
    if (i == rank_) {
      // This rank receives the reduced result
      auto dataType = getNcclDataType(input_list[i]);
      ncclResult_t opResult = rccl_api_->reduce(
          input_list[i].data_ptr(),
          output.data_ptr(),
          output.numel(),
          dataType,
          getNcclReduceOp(op, nccl_comm_, dataType),
          i,
          nccl_comm_,
          stream);
      if (opResult != ncclSuccess) {
        throw RCCLException(
            *rccl_api_,
            "NCCL Reduce failed in reduce_scatter",
            opResult,
            nccl_comm_);
      }
    } else {
      // Other ranks contribute to the reduction
      auto dataType = getNcclDataType(input_list[i]);
      ncclResult_t opResult = rccl_api_->reduce(
          input_list[i].data_ptr(),
          nullptr, // Non-root ranks don't receive
          input_list[i].numel(),
          dataType,
          getNcclReduceOp(op, nccl_comm_, dataType),
          i,
          nccl_comm_,
          stream);
      if (opResult != ncclSuccess) {
        throw RCCLException(
            *rccl_api_,
            "NCCL Reduce failed in reduce_scatter",
            opResult,
            nccl_comm_);
      }
    }
  }

  RCCL_CHECK(
      rccl_api_, nccl_comm_, rccl_api_->groupEnd(), "NCCL GroupEnd failed");

  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommRCCL::reduce_scatter_v(
    at::Tensor& output,
    const std::vector<at::Tensor>& input_list,
    const ReduceOp& op,
    bool async_op,
    const ReduceScatterOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(output);

  if (input_list.size() != static_cast<size_t>(comm_size_)) {
    throw std::runtime_error(
        "input_list size must equal comm_size for reduce_scatter_v");
  }

  for (const auto& t : input_list) {
    ensureTensorContiguous(t);
  }

  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "reduce_scatter_v", rank_, input_list, {output});

  hipStream_t stream = getOperationStream(async_op);
  auto work = async_op
      ? createWork(
            stream,
            getOperationTimeout(options.timeout, options_.timeout),
            input_list)
      : createWork(
            stream, getOperationTimeout(options.timeout, options_.timeout));

  work->recordStart("reduce_scatter_v");

  // Use multiple reduce operations for reduce_scatter_v
  RCCL_CHECK(
      rccl_api_, nccl_comm_, rccl_api_->groupStart(), "NCCL GroupStart failed");

  for (int i = 0; i < comm_size_; ++i) {
    if (i == rank_) {
      // This rank receives the reduced result
      auto& input_tensor = input_list[i];
      auto& output_tensor = output;
      if (input_tensor.numel() != output_tensor.numel()) {
        throw std::runtime_error(
            "Output tensor size must equal input tensor size for reduce_scatter_v");
      }
      auto dataType = getNcclDataType(input_tensor);
      ncclResult_t opResult = rccl_api_->reduce(
          input_tensor.data_ptr(),
          output_tensor.data_ptr(),
          output_tensor.numel(),
          dataType,
          getNcclReduceOp(op, nccl_comm_, dataType),
          i,
          nccl_comm_,
          stream);
      if (opResult != ncclSuccess) {
        throw RCCLException(
            *rccl_api_,
            "NCCL Reduce failed in reduce_scatter_v",
            opResult,
            nccl_comm_);
      }
    } else {
      // Other ranks contribute to the reduction
      auto dataType = getNcclDataType(input_list[i]);
      ncclResult_t opResult = rccl_api_->reduce(
          input_list[i].data_ptr(),
          nullptr, // Non-root ranks don't receive
          input_list[i].numel(),
          dataType,
          getNcclReduceOp(op, nccl_comm_, dataType),
          i,
          nccl_comm_,
          stream);
      if (opResult != ncclSuccess) {
        throw RCCLException(
            *rccl_api_,
            "NCCL Reduce failed in reduce_scatter_v",
            opResult,
            nccl_comm_);
      }
    }
  }

  RCCL_CHECK(
      rccl_api_, nccl_comm_, rccl_api_->groupEnd(), "NCCL GroupEnd failed");

  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommRCCL::reduce_scatter_single(
    at::Tensor& output,
    const at::Tensor& input,
    const ReduceOp& op,
    bool async_op,
    const ReduceScatterSingleOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(output);
  ensureTensorContiguous(input);

  if (input.numel() != output.numel() * comm_size_) {
    throw std::runtime_error(
        "Input tensor size must be output_size * comm_size for reduce_scatter_single");
  }

  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "reduce_scatter_single", rank_, input, output);

  hipStream_t stream = getOperationStream(async_op);
  auto work = createWork(
      stream, getOperationTimeout(options.timeout, options_.timeout), {input});

  // Record start event before NCCL operation
  work->recordStart("reduce_scatter_single");

  auto dataType = getNcclDataType(input);
  RCCL_CHECK(
      rccl_api_,
      nccl_comm_,
      rccl_api_->reduceScatter(
          input.data_ptr(),
          output.data_ptr(),
          output.numel(),
          dataType,
          getNcclReduceOp(op, nccl_comm_, dataType),
          nccl_comm_,
          stream),
      "NCCL ReduceScatter failed");

  // Record end event after NCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommRCCL::all_to_all_single(
    at::Tensor& output,
    const at::Tensor& input,
    bool async_op,
    const AllToAllSingleOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(output);
  ensureTensorContiguous(input);

  if (input.numel() != output.numel()) {
    throw std::runtime_error(
        "Input and output tensors must have same size for all_to_all_single");
  }

  if (input.numel() % comm_size_ != 0) {
    throw std::runtime_error(
        "Tensor size must be divisible by comm_size for all_to_all_single");
  }

  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "all_to_all_single", rank_, input, output);

  hipStream_t stream = getOperationStream(async_op);
  auto work = createWork(
      stream, getOperationTimeout(options.timeout, options_.timeout), {input});

  // Record start event before NCCL operation
  work->recordStart("all_to_all_single");

  size_t chunk_size = input.numel() / comm_size_;

  RCCL_CHECK(
      rccl_api_,
      nccl_comm_,
      rccl_api_->allToAll(
          input.data_ptr(),
          output.data_ptr(),
          chunk_size,
          getNcclDataType(input),
          nccl_comm_,
          stream),
      "NCCL AllToAll failed");

  // Record end event after NCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommRCCL::all_to_all_v_single(
    at::Tensor& output,
    const at::Tensor& input,
    const std::vector<uint64_t>& output_split_sizes,
    const std::vector<uint64_t>& input_split_sizes,
    bool async_op,
    const AllToAllvSingleOptions& options) {
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(output);
  ensureTensorContiguous(input);

  // Validate split sizes vectors
  if (input_split_sizes.size() != static_cast<size_t>(comm_size_)) {
    throw std::runtime_error(
        "input_split_sizes length must equal comm_size for all_to_all_v_single");
  }

  if (output_split_sizes.size() != static_cast<size_t>(comm_size_)) {
    throw std::runtime_error(
        "output_split_sizes length must equal comm_size for all_to_all_v_single");
  }

  // Validate that split sizes sum to tensor sizes
  uint64_t input_total = 0;
  uint64_t output_total = 0;
  for (int i = 0; i < comm_size_; ++i) {
    input_total += input_split_sizes[i];
    output_total += output_split_sizes[i];
  }

  if (input_total > static_cast<uint64_t>(input.size(0))) {
    throw std::runtime_error(
        "Sum of input_split_sizes exceeds input tensor size for all_to_all_v_single");
  }

  if (output_total > static_cast<uint64_t>(output.size(0))) {
    throw std::runtime_error(
        "Sum of output_split_sizes exceeds output tensor size for all_to_all_v_single");
  }

  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "all_to_all_v_single", rank_, input, output);

  hipStream_t stream = getOperationStream(async_op);
  auto work = createWork(
      stream, getOperationTimeout(options.timeout, options_.timeout), {input});

  // Record start event before NCCL operation
  work->recordStart("all_to_all_v_single");

  // Convert split sizes to arrays and calculate displacements
  std::vector<size_t> sendcounts(comm_size_);
  std::vector<size_t> recvcounts(comm_size_);
  std::vector<size_t> senddispls(comm_size_);
  std::vector<size_t> recvdispls(comm_size_);

  // Calculate the number of elements per slice along the first dimension
  // For a tensor with shape [N, D1, D2, ..., Dk], each slice of size S along
  // dim 0 contains S * D1 * D2 * ... * Dk elements
  // Use input tensor for send counts and output tensor for recv counts
  size_t send_elements_per_slice =
      input.numel() ? input.numel() / input.size(0) : 0;
  size_t recv_elements_per_slice =
      output.numel() ? output.numel() / output.size(0) : 0;

  size_t sendoffset = 0;
  size_t recvoffset = 0;
  for (int i = 0; i < comm_size_; ++i) {
    sendcounts[i] = input_split_sizes[i] * send_elements_per_slice;
    recvcounts[i] = output_split_sizes[i] * recv_elements_per_slice;
    senddispls[i] = sendoffset;
    recvdispls[i] = recvoffset;
    sendoffset += sendcounts[i];
    recvoffset += recvcounts[i];
  }

  RCCL_CHECK(
      rccl_api_,
      nccl_comm_,
      rccl_api_->allToAllv(
          input.data_ptr(),
          sendcounts.data(),
          senddispls.data(),
          output.data_ptr(),
          recvcounts.data(),
          recvdispls.data(),
          getNcclDataType(input),
          nccl_comm_,
          stream),
      "NCCL AllToAllv failed");

  // Record end event after NCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommRCCL::all_to_all(
    const std::vector<at::Tensor>& output_tensor_list,
    const std::vector<at::Tensor>& input_tensor_list,
    bool async_op,
    const AllToAllOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  if (output_tensor_list.size() != static_cast<size_t>(comm_size_) ||
      input_tensor_list.size() != static_cast<size_t>(comm_size_)) {
    throw std::runtime_error(
        "Tensor list sizes must equal comm_size for all_to_all");
  }

  // Validate all tensors
  for (int i = 0; i < comm_size_; ++i) {
    ensureTensorContiguous(input_tensor_list[i]);
    ensureTensorContiguous(output_tensor_list[i]);
  }

  TorchCommTracingGuard tracingGuard(
      name_,
      comm_size_,
      "all_to_all",
      rank_,
      input_tensor_list,
      output_tensor_list);

  hipStream_t stream = getOperationStream(async_op);
  auto work = createWork(
      stream,
      getOperationTimeout(options.timeout, options_.timeout),
      input_tensor_list);

  // Record start event before NCCL operations
  work->recordStart("all_to_all");

  RCCL_CHECK(
      rccl_api_, nccl_comm_, rccl_api_->groupStart(), "NCCL GroupStart failed");

  for (int i = 0; i < comm_size_; ++i) {
    // Send to rank i
    ncclResult_t sendResult = rccl_api_->send(
        input_tensor_list[i].data_ptr(),
        input_tensor_list[i].numel(),
        getNcclDataType(input_tensor_list[i]),
        i,
        nccl_comm_,
        stream);
    if (sendResult != ncclSuccess) {
      throw RCCLException(
          *rccl_api_, "NCCL Send failed in all_to_all", sendResult, nccl_comm_);
    }

    // Receive from rank i
    ncclResult_t recvResult = rccl_api_->recv(
        output_tensor_list[i].data_ptr(),
        output_tensor_list[i].numel(),
        getNcclDataType(output_tensor_list[i]),
        i,
        nccl_comm_,
        stream);
    if (recvResult != ncclSuccess) {
      throw RCCLException(
          *rccl_api_, "NCCL Recv failed in all_to_all", recvResult, nccl_comm_);
    }
  }

  RCCL_CHECK(
      rccl_api_, nccl_comm_, rccl_api_->groupEnd(), "NCCL GroupEnd failed");

  // Record end event after NCCL operations
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommRCCL::barrier(
    bool async_op,
    const BarrierOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();

  TorchCommTracingGuard tracingGuard(name_, comm_size_, "barrier", rank_);
  hipStream_t stream = getOperationStream(async_op);
  auto work = createWork(
      stream, getOperationTimeout(options.timeout, options_.timeout));

  // Record start event before NCCL operation
  work->recordStart("barrier");

  // Use pre-allocated CUDA buffer for barrier
  RCCL_CHECK(
      rccl_api_,
      nccl_comm_,
      rccl_api_->allReduce(
          barrier_buffer_,
          barrier_buffer_,
          1,
          ncclFloat32,
          ncclSum,
          nccl_comm_,
          stream),
      "NCCL Barrier failed");

  // Record end event after NCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommRCCL::scatter(
    at::Tensor& output_tensor,
    const std::vector<at::Tensor>& input_tensor_list,
    int root,
    bool async_op,
    const ScatterOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(output_tensor);

  // Only the root rank needs valid input tensors
  if (rank_ == root) {
    if (input_tensor_list.size() != static_cast<size_t>(comm_size_)) {
      throw std::runtime_error(
          "input_tensor_list size must equal comm_size for scatter");
    }

    for (const auto& t : input_tensor_list) {
      ensureTensorContiguous(t);
      if (t.numel() != output_tensor.numel()) {
        throw std::runtime_error(
            "All input tensors must have same size as output tensor");
      }
    }
  }

  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "scatter", root, input_tensor_list, {output_tensor});

  hipStream_t stream = getOperationStream(async_op);
  std::vector<at::Tensor> input_tensors;
  if (rank_ == root) {
    input_tensors = input_tensor_list;
  }
  auto work = createWork(
      stream,
      getOperationTimeout(options.timeout, options_.timeout),
      input_tensors);

  // Record start event before NCCL operations
  work->recordStart("scatter");

  // Implement scatter using point-to-point operations
  if (rank_ == root) {
    // Root sends to all ranks (except itself)
    RCCL_CHECK(
        rccl_api_,
        nccl_comm_,
        rccl_api_->groupStart(),
        "NCCL GroupStart failed");
    for (int i = 0; i < comm_size_; ++i) {
      if (i != root) {
        ncclResult_t sendResult = rccl_api_->send(
            input_tensor_list[i].data_ptr(),
            input_tensor_list[i].numel(),
            getNcclDataType(input_tensor_list[i]),
            i,
            nccl_comm_,
            stream);
        if (sendResult != ncclSuccess) {
          throw RCCLException(
              *rccl_api_,
              "NCCL Send failed in scatter",
              sendResult,
              nccl_comm_);
        }
      }
    }
    RCCL_CHECK(
        rccl_api_, nccl_comm_, rccl_api_->groupEnd(), "NCCL GroupEnd failed");

    // Root copies its own data using hipMemcpyAsync
    HIP_CHECK(
        hip_api_,
        hip_api_->memcpyAsync(
            output_tensor.data_ptr(),
            input_tensor_list[root].data_ptr(),
            input_tensor_list[root].numel() *
                input_tensor_list[root].element_size(),
            hipMemcpyDeviceToDevice,
            stream),
        "memcpyAsync failed");
  } else {
    // Non-root ranks receive from root
    ncclResult_t recvResult = rccl_api_->recv(
        output_tensor.data_ptr(),
        output_tensor.numel(),
        getNcclDataType(output_tensor),
        root,
        nccl_comm_,
        stream);
    if (recvResult != ncclSuccess) {
      throw RCCLException(
          *rccl_api_, "NCCL Recv failed in scatter", recvResult, nccl_comm_);
    }
  }

  // Record end event after NCCL operations
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommRCCL::gather(
    const std::vector<at::Tensor>& output_tensor_list,
    const at::Tensor& input_tensor,
    int root,
    bool async_op,
    const GatherOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(input_tensor);

  // Only the root rank needs valid output tensors
  if (rank_ == root) {
    if (output_tensor_list.size() != static_cast<size_t>(comm_size_)) {
      throw std::runtime_error(
          "output_tensor_list size must equal comm_size for gather");
    }

    for (const auto& t : output_tensor_list) {
      ensureTensorContiguous(t);
      if (t.numel() != input_tensor.numel()) {
        throw std::runtime_error(
            "All output tensors must have same size as input tensor");
      }
    }
  }

  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "gather", root, {input_tensor}, output_tensor_list);

  hipStream_t stream = getOperationStream(async_op);
  std::vector<at::Tensor> output_tensors;
  if (rank_ == root) {
    output_tensors = output_tensor_list;
  }
  auto work = createWork(
      stream,
      getOperationTimeout(options.timeout, options_.timeout),
      {input_tensor});

  // Record start event before NCCL operations
  work->recordStart("gather");

  if (rank_ == root) {
    // Root receives from all ranks (except itself)
    RCCL_CHECK(
        rccl_api_,
        nccl_comm_,
        rccl_api_->groupStart(),
        "NCCL GroupStart failed");
    for (int i = 0; i < comm_size_; ++i) {
      if (i != root) {
        ncclResult_t recvResult = rccl_api_->recv(
            output_tensor_list[i].data_ptr(),
            output_tensor_list[i].numel(),
            getNcclDataType(output_tensor_list[i]),
            i,
            nccl_comm_,
            stream);
        if (recvResult != ncclSuccess) {
          throw RCCLException(
              *rccl_api_, "NCCL Recv failed in gather", recvResult, nccl_comm_);
        }
      }
    }
    RCCL_CHECK(
        rccl_api_, nccl_comm_, rccl_api_->groupEnd(), "NCCL GroupEnd failed");

    // Root copies its own data using hipMemcpyAsync
    HIP_CHECK(
        hip_api_,
        hip_api_->memcpyAsync(
            output_tensor_list[root].data_ptr(),
            input_tensor.data_ptr(),
            input_tensor.numel() * input_tensor.element_size(),
            hipMemcpyDeviceToDevice,
            stream),
        "memcpyAsync failed");
  } else {
    // Non-root ranks send to root
    ncclResult_t sendResult = rccl_api_->send(
        input_tensor.data_ptr(),
        input_tensor.numel(),
        getNcclDataType(input_tensor),
        root,
        nccl_comm_,
        stream);
    if (sendResult != ncclSuccess) {
      throw RCCLException(
          *rccl_api_, "NCCL Send failed in gather", sendResult, nccl_comm_);
    }
  }

  // Record end event after NCCL operations
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

std::shared_ptr<TorchCommBackend> TorchCommRCCL::split(
    const std::vector<int>& ranks,
    const std::string& name,
    const CommOptions& options) {
  checkAndAbortIfTimedOutOrError();

  // Validate that all ranks are valid
  for (int rank : ranks) {
    if (rank < 0 || rank >= comm_size_) {
      throw std::runtime_error(
          fmt::format(
              "Invalid rank {} in ranks. Valid ranks are 0 to {}",
              rank,
              comm_size_ - 1));
    }
  }

  // Check for duplicate ranks
  std::set<int> unique_ranks(ranks.begin(), ranks.end());
  if (unique_ranks.size() != ranks.size()) {
    throw std::runtime_error("Duplicate ranks found in ranks list");
  }

  // Determine the color for this rank
  int color;
  int new_rank;

  if (ranks.empty()) {
    // Empty list means exclude all ranks - use NCCL_SPLIT_NOCOLOR
    // NOLINTNEXTLINE(clang-diagnostic-undef)
    color = NCCL_SPLIT_NOCOLOR;
    new_rank = -1; // Will not participate in new communicator
  } else {
    // Check if current rank is in the non-empty list
    auto it = std::find(ranks.begin(), ranks.end(), rank_);
    if (it == ranks.end()) {
      // Current rank is not in the non-empty list - this is an error
      throw std::runtime_error(
          fmt::format(
              "Current rank {} is not included in the provided ranks list",
              rank_));
    }
    // Set color to the lowest rank in the group and calculate new rank
    color = *std::min_element(ranks.begin(), ranks.end());
    new_rank = static_cast<int>(std::distance(ranks.begin(), it));
  }

  // Create a new NCCL communicator
  ncclComm_t new_comm;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;

  // TODO: nccl says that this is not supposed to be called if any operation
  // is outstanding on the comm. We should check for that.
  // TODO: what happens if one rank fails but the others succeed, need to
  // handle the error case.
  // TODO: is this sharing any resources with the original comm?
  RCCL_CHECK(
      rccl_api_,
      nccl_comm_,
      rccl_api_->commSplit(nccl_comm_, color, new_rank, &new_comm, &config),
      "RCCL split failed");

  if (new_rank == -1) {
    return nullptr; // Rank is not in any group, return nullptr
  }

  auto new_torchcomm = std::shared_ptr<TorchCommRCCL>(
      new TorchCommRCCL(new_comm, device_, options));
  new_torchcomm->rccl_api_ = rccl_api_;
  new_torchcomm->hip_api_ = hip_api_;
  new_torchcomm->init(device_, name, options);

  return new_torchcomm;
}

std::string_view TorchCommRCCL::getBackendName() const {
  return kBackendName;
}

std::string_view TorchCommRCCL::getCommName() const {
  return name_;
}

void TorchCommRCCL::register_address(
    const TorchCommRCCL::AddressWithLen& addr) {
  // We got a register after we got rid of the comm. Is this a fatal error?
  if (nccl_comm_ == nullptr) {
    return;
  }

  if (memoryRegistrationHandles_.find(addr.addr) !=
      memoryRegistrationHandles_.end()) {
    throw std::runtime_error("Memory already registered with NCCL");
  }
  void* handle = nullptr;
  RCCL_CHECK(
      rccl_api_,
      nccl_comm_,
      rccl_api_->commRegister(nccl_comm_, addr.addr, addr.len, &handle),
      "Failed to register memory with RCCL");
  memoryRegistrationHandles_.emplace(addr.addr, RegistrationHandle(handle));
}

void TorchCommRCCL::deregister_address(const TorchCommRCCL::Address& addr) {
  // We got a deregister after we got rid of the comm. Is this a fatal error?
  if (nccl_comm_ == nullptr) {
    return;
  }

  auto it = memoryRegistrationHandles_.find(addr.addr);
  if (it == memoryRegistrationHandles_.end()) {
    // it's possible that the memory was registered for a different comm,
    // however failed registration for this comm.
    return;
  }

  void* handle = it->second.regHandle;
  RCCL_CHECK(
      rccl_api_,
      nccl_comm_,
      rccl_api_->commDeregister(nccl_comm_, handle),
      "Failed to deregister memory with RCCL");

  memoryRegistrationHandles_.erase(it);
}

RCCLException::RCCLException(
    RcclApi& rccl_api,
    const std::string& message,
    ncclResult_t result,
    ncclComm_t comm)
    : message_(
          message + ": " + rccl_api.getErrorString(result) +
          " \nRCCL Last Error: " + rccl_api.getLastError(comm)),
      result_(result) {}

const char* RCCLException::what() const noexcept {
  return message_.c_str();
}

} // namespace torch::comms

namespace {
class RCCLRegistration {
 public:
  RCCLRegistration() {
    torch::comms::TorchCommFactory::get().register_backend("rccl", []() {
      return std::make_shared<torch::comms::TorchCommRCCL>();
    });
  }
};

static const RCCLRegistration registration{};
} // namespace
