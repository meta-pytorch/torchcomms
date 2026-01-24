// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/rcclx/TorchCommRCCLX.hpp"

#include <cstdlib>
#include <set>
#include <stdexcept>
#include <string>

#include <ATen/hip/HIPContext.h> // @manual=//caffe2:ATen-custom-hip

#include "comms/torchcomms/TorchCommFactory.hpp"
#include "comms/torchcomms/TorchCommLogging.hpp"
#include "comms/torchcomms/rcclx/TorchCommRCCLXBootstrap.hpp"
#include "rccl.h" // @manual

namespace torch {
namespace comms {

ncclResult_t RCCLXException::getResult() const {
  return result_;
}

TorchCommRCCLX::TorchCommRCCLX()
    : nccl_comm_{nullptr},
      device_(at::kHIP),
      options_(),
      init_state_(InitializationState::UNINITIALIZED),
      shutdown_(false) {}

TorchCommRCCLX::TorchCommRCCLX(
    const ncclComm_t nccl_comm,
    at::Device device,
    const CommOptions& options)
    : nccl_comm_(nccl_comm),
      device_(device),
      options_(options),
      init_state_(InitializationState::UNINITIALIZED),
      shutdown_(false) {}

TorchCommRCCLX::~TorchCommRCCLX() {
  if (init_state_ == InitializationState::INITIALIZED) {
    TC_LOG(ERROR) << "TorchCommRCCLX was not finalized before destruction";
  }

  // We need to detach the memory hook in case finalize is not called,
  // so that we don't encounter a memory corruption.
  detachMemoryHook();
}

void TorchCommRCCLX::init(
    at::Device device,
    const std::string& name,
    const CommOptions& options) {
  // Initialize private members
  device_ = device;
  name_ = name;
  options_ = options;

  // Only initialize once
  if (init_state_ == InitializationState::INITIALIZED) {
    throw std::runtime_error("TorchCommRCCLX already initialized");
  } else if (init_state_ == InitializationState::FINALIZED) {
    throw std::runtime_error("TorchCommRCCLX already finalized");
  }
  init_state_ = InitializationState::INITIALIZED;

  // Initialize default RCCLX API implementation if not already set
  if (!rcclx_api_) {
    rcclx_api_ = std::make_unique<DefaultRcclxApi>();
  }

  // Initialize default HIP API implementation if not already set
  if (!hip_api_) {
    hip_api_ = std::make_unique<DefaultHipApi>();
  }

  if (device_.index() == -1 || nccl_comm_ == nullptr) {
    auto bootstrap = std::make_unique<TorchCommRCCLXBootstrap>(
        options_.store, device_, rcclx_api_, hip_api_, options_.timeout);
    device_ = bootstrap->getDevice();

    if (nccl_comm_ == nullptr) {
      nccl_comm_ = bootstrap->createNcclComm(name);
    }
  }

  // Set HIP device and verify it's accessible
  HIP_CHECK(
      hip_api_,
      hip_api_->setDevice(device_.index()),
      "Failed to set CUDA device to " + std::to_string(device_.index()));

  // Verify device properties and memory availability
  hipDeviceProp_t device_prop = {};
  HIP_CHECK(
      hip_api_,
      hip_api_->getDeviceProperties(&device_prop, device_.index()),
      "Failed to get device properties for device " +
          std::to_string(device_.index()));

  // Check available memory
  size_t free_memory, total_memory;
  HIP_CHECK(
      hip_api_,
      hip_api_->memGetInfo(&free_memory, &total_memory),
      "Failed to get memory info for device " +
          std::to_string(device_.index()));

  // Read hints and store them
  for (const auto& hint : options_.hints) {
    const std::string& key = hint.first;
    const std::string& val = hint.second;
    if (key.substr(0, 17) == "torchcomm::rcclx::") {
      if (key == "torchcomm::rcclx::high_priority_stream") {
        high_priority_stream_ = string_to_bool(val);
      } else {
        throw std::runtime_error("Unrecognized hint " + key);
      }
    } else {
      // Ignore keys that do not start with "torchcomm::rcclx::"
    }
  }

  // Create internal stream
  //
  // Default priority is 0 as per NVIDIA docs (https://fburl.com/2xb0iqwl).
  int stream_priority = 0;

  // Check for high priority stream hint
  if (high_priority_stream_) {
    int leastPriority, greatestPriority;
    auto ret =
        hip_api_->getStreamPriorityRange(&leastPriority, &greatestPriority);
    if (ret != hipSuccess) {
      throw std::runtime_error(
          "Failed to get stream priority range. Error:" + std::to_string(ret));
    }
    stream_priority = greatestPriority;
  }

  HIP_CHECK(
      hip_api_,
      hip_api_->streamCreateWithPriority(
          &internal_stream_, hipStreamNonBlocking, stream_priority),
      "Failed to create internal CUDA stream on device " +
          std::to_string(device_.index()));

  // Create dependency event for stream synchronization
  HIP_CHECK(
      hip_api_,
      hip_api_->eventCreate(&dependency_event_),
      "Failed to create dependency event on device " +
          std::to_string(device_.index()));

  // Allocate CUDA buffer for barrier operations
  HIP_CHECK(
      hip_api_,
      hip_api_->malloc(&barrier_buffer_, sizeof(float)),
      "Failed to allocate barrier buffer");

  if (options_.hints.find("torchcomm::rcclx::max_event_pool_size") !=
      options_.hints.end()) {
    max_event_pool_size_ =
        std::stoull(options_.hints.at("torchcomm::rcclx::max_event_pool_size"));
  } else {
    max_event_pool_size_ = kMaxEventPoolSize;
  }

  // Give up our internal reference to the store object here.  The caller
  // would still need to keep a reference to the store object till the init
  // call returns, at which point the RCCLX communicator would already be
  // created.
  if (options_.store) {
    options_.store.reset();
  }

  ncclResult_t ncclErr;
  ncclErr = rcclx_api_->commUserRank(nccl_comm_, &rank_);
  if (ncclErr != ncclSuccess) {
    throw std::runtime_error("RCCLX User Rank failed");
  }

  tryTorchCommLoggingInit("torchcomm");

  ncclErr = rcclx_api_->commCount(nccl_comm_, &comm_size_);
  if (ncclErr != ncclSuccess) {
    throw std::runtime_error("RCCLX Count failed");
  }

  TorchCommTracingGuard tracingGuard(name_, comm_size_, "init", rank_);

  // Start timeout watchdog thread
  timeout_thread_ = std::thread(&TorchCommRCCLX::timeoutWatchdog, this);

  // Register comm with CachingAllocator
  attachMemoryHook();
}

void TorchCommRCCLX::finalize() {
  if (init_state_ == InitializationState::UNINITIALIZED) {
    throw std::runtime_error("TorchCommRCCLX not initialized");
  } else if (init_state_ == InitializationState::FINALIZED) {
    throw std::runtime_error("TorchCommRCCLX already finalized");
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
    abortRcclxComm();
    throw std::runtime_error("Work timed out during finalize");
  } else if (comm_state_ == CommState::ERROR) {
    ncclResult_t asyncErr;
    rcclx_api_->commGetAsyncError(nccl_comm_, &asyncErr);
    RCCLXException RCCLXException(*rcclx_api_, "RCCLX Async Error", asyncErr);
    abortRcclxComm();
    throw RCCLXException;
  }

  // Clear the completed works queue
  std::queue<c10::intrusive_ptr<TorchWorkRCCLX>> empty;
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

  // Destroy RCCLX communicator
  // TODO: should probably not call this after calling abort.
  if (nccl_comm_) {
    detachMemoryHook();
    // Deregister comm from the CachingAllocator
    rcclx_api_->commDestroy(nccl_comm_);
    nccl_comm_ = nullptr;
  }
}

void TorchCommRCCLX::abortRcclxComm() {
  detachMemoryHook();
  if (nccl_comm_) {
    rcclx_api_->commAbort(nccl_comm_);
    nccl_comm_ = nullptr;
  }
}

int TorchCommRCCLX::getRank() const {
  checkInitialized();

  int rank;
  ncclResult_t ncclErr = rcclx_api_->commUserRank(nccl_comm_, &rank);
  if (ncclErr != ncclSuccess) {
    throw RCCLXException(*rcclx_api_, "RCCLX User Rank failed", ncclErr);
  }
  return rank;
}

int TorchCommRCCLX::getSize() const {
  checkInitialized();

  int comm_size;
  ncclResult_t ncclErr = rcclx_api_->commCount(nccl_comm_, &comm_size);
  if (ncclErr != ncclSuccess) {
    throw RCCLXException(*rcclx_api_, "RCCLX Count failed", ncclErr);
  }
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
c10::intrusive_ptr<TorchWork> TorchCommRCCLX::send(
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
      stream, getOperationTimeout(options.timeout, options_.timeout), tensor);

  // Record start event before RCCLX operation
  work->recordStart("send");

  ncclResult_t result = rcclx_api_->send(
      tensor.data_ptr(),
      tensor.numel(),
      getNcclDataType(tensor),
      dst,
      nccl_comm_,
      stream);

  if (result != ncclSuccess) {
    throw RCCLXException(*rcclx_api_, "RCCLX Send failed", result);
  }

  // Record end event after RCCLX operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommRCCLX::recv(
    at::Tensor& tensor,
    int src,
    bool async_op,
    const RecvOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);

  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "recv", src, {tensor}, {tensor});

  hipStream_t stream = getOperationStream(async_op);
  auto work = createWork(
      stream, getOperationTimeout(options.timeout, options_.timeout));

  // Record start event before RCCL operation
  work->recordStart("recv");

  ncclResult_t result = rcclx_api_->recv(
      tensor.data_ptr(),
      tensor.numel(),
      getNcclDataType(tensor),
      src,
      nccl_comm_,
      stream);

  if (result != ncclSuccess) {
    throw RCCLXException(*rcclx_api_, "RCCLX Recv failed", result);
  }

  // Record end event after RCCLX operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

// Batch P2P Operations
c10::intrusive_ptr<TorchWork> TorchCommRCCLX::batch_op_issue(
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

  // Record start event before RCCL operations
  work->recordStart("batch_op_issue");

  // Start RCCLX group for batched operations
  ncclResult_t result = rcclx_api_->groupStart();
  if (result != ncclSuccess) {
    throw RCCLXException(*rcclx_api_, "RCCLX GroupStart failed", result);
  }

  // Issue each operation individually
  for (const auto& op : ops) {
    if (op.type == BatchSendRecv::P2POp::OpType::SEND) {
      result = rcclx_api_->send(
          op.tensor.data_ptr(),
          op.tensor.numel(),
          getNcclDataType(op.tensor),
          op.peer,
          nccl_comm_,
          stream);

      if (result != ncclSuccess) {
        rcclx_api_->groupEnd(); // Clean up group on error
        throw RCCLXException(
            *rcclx_api_, "RCCLX Send failed in batch operation", result);
      }
    } else if (op.type == BatchSendRecv::P2POp::OpType::RECV) {
      result = rcclx_api_->recv(
          op.tensor.data_ptr(),
          op.tensor.numel(),
          getNcclDataType(op.tensor),
          op.peer,
          nccl_comm_,
          stream);

      if (result != ncclSuccess) {
        rcclx_api_->groupEnd(); // Clean up group on error
        throw RCCLXException(
            *rcclx_api_, "RCCLX Recv failed in batch operation", result);
      }
    }
  }

  // End RCCLX group
  result = rcclx_api_->groupEnd();
  if (result != ncclSuccess) {
    throw RCCLXException(*rcclx_api_, "RCCLX GroupEnd failed", result);
  }

  // Record end event after RCCLX operations
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

// Collective Operations
c10::intrusive_ptr<TorchWork> TorchCommRCCLX::broadcast(
    at::Tensor& tensor,
    int root,
    bool async_op,
    const BroadcastOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);

  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "broadcast", rank_, {tensor}, {tensor});

  hipStream_t stream = getOperationStream(async_op);

  auto work = createWork(
      stream, getOperationTimeout(options.timeout, options_.timeout), tensor);

  // Record start event before RCCLX operation
  work->recordStart("broadcast");

  ncclResult_t result = rcclx_api_->bcast(
      tensor.data_ptr(),
      tensor.numel(),
      getNcclDataType(tensor),
      root,
      nccl_comm_,
      stream);

  if (result != ncclSuccess) {
    throw RCCLXException(*rcclx_api_, "RCCLX Broadcast failed", result);
  }

  // Record end event after RCCLX operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommRCCLX::all_reduce(
    at::Tensor& tensor,
    const ReduceOp& op,
    bool async_op,
    const AllReduceOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);

  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "all_reduce", rank_, {tensor}, {tensor});
  hipStream_t stream = getOperationStream(async_op);
  auto work = createWork(
      stream, getOperationTimeout(options.timeout, options_.timeout), tensor);

  // Record start event before RCCLX operation
  work->recordStart("all_reduce");

  auto dataType = getNcclDataType(tensor);
  ncclResult_t result = rcclx_api_->allReduce(
      tensor.data_ptr(),
      tensor.data_ptr(), // In-place operation
      tensor.numel(),
      dataType,
      getNcclReduceOp(op, nccl_comm_, dataType),
      nccl_comm_,
      stream);

  if (result != ncclSuccess) {
    throw RCCLXException(*rcclx_api_, "RCCLX AllReduce failed", result);
  }

  // Record end event after RCCLX operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommRCCLX::reduce(
    const at::Tensor& tensor,
    int root,
    const ReduceOp& op,
    bool async_op,
    const ReduceOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);

  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "reduce", rank_, {tensor}, {tensor});

  hipStream_t stream = getOperationStream(async_op);
  auto work = createWork(
      stream, getOperationTimeout(options.timeout, options_.timeout), {tensor});

  // Record start event before RCCLX operation
  work->recordStart("reduce");

  auto dataType = getNcclDataType(tensor);
  ncclResult_t result = rcclx_api_->reduce(
      tensor.data_ptr(),
      rank_ == root ? tensor.data_ptr() : nullptr,
      tensor.numel(),
      dataType,
      getNcclReduceOp(op, nccl_comm_, dataType),
      root,
      nccl_comm_,
      stream);

  if (result != ncclSuccess) {
    throw RCCLXException(*rcclx_api_, "RCCLX Reduce failed", result);
  }

  // Record end event after RCCLX operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommRCCLX::all_gather(
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
  rcclx_api_->groupStart();

  for (int i = 0; i < comm_size_; ++i) {
    rcclx_api_->broadcast(
        tensor.data_ptr(),
        tensor_list[i].data_ptr(),
        tensor.numel(),
        getNcclDataType(tensor_list[i]),
        i,
        nccl_comm_,
        stream);
  }

  rcclx_api_->groupEnd();

  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommRCCLX::all_gather_v(
    const std::vector<at::Tensor>& tensor_list,
    const at::Tensor& tensor,
    bool async_op,
    const AllGatherOptions& options) {
  (void)tensor_list;
  (void)tensor;
  (void)async_op;
  (void)options;
  throw std::runtime_error("all_gather_v is not supported in RCCLX backend");
}

c10::intrusive_ptr<TorchWork> TorchCommRCCLX::all_gather_single(
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
      name_, comm_size_, "all_gather_single", rank_, {input}, {output});

  hipStream_t stream = getOperationStream(async_op);
  auto work = createWork(
      stream, getOperationTimeout(options.timeout, options_.timeout), {input});

  work->recordStart("all_gather_single");

  ncclResult_t result = rcclx_api_->allGather(
      input.data_ptr(),
      output.data_ptr(),
      input.numel(),
      getNcclDataType(input),
      nccl_comm_,
      stream);

  if (result != ncclSuccess) {
    throw RCCLXException(*rcclx_api_, "RCCLX AllGather failed", result);
  }

  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommRCCLX::reduce_scatter(
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
  rcclx_api_->groupStart();

  for (int i = 0; i < comm_size_; ++i) {
    if (i == rank_) {
      // This rank receives the reduced result
      auto dataType = getNcclDataType(input_list[i]);
      rcclx_api_->reduce(
          input_list[i].data_ptr(),
          output.data_ptr(),
          output.numel(),
          dataType,
          getNcclReduceOp(op, nccl_comm_, dataType),
          i,
          nccl_comm_,
          stream);
    } else {
      // Other ranks contribute to the reduction
      auto dataType = getNcclDataType(input_list[i]);
      rcclx_api_->reduce(
          input_list[i].data_ptr(),
          nullptr, // Non-root ranks don't receive
          input_list[i].numel(),
          dataType,
          getNcclReduceOp(op, nccl_comm_, dataType),
          i,
          nccl_comm_,
          stream);
    }
  }

  rcclx_api_->groupEnd();

  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommRCCLX::reduce_scatter_v(
    at::Tensor& output,
    const std::vector<at::Tensor>& input_list,
    const ReduceOp& op,
    bool async_op,
    const ReduceScatterOptions& options) {
  (void)output;
  (void)input_list;
  (void)op;
  (void)async_op;
  (void)options;
  throw std::runtime_error(
      "reduce_scatter_v is not supported in RCCLX backend");
}

c10::intrusive_ptr<TorchWork> TorchCommRCCLX::reduce_scatter_single(
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
      name_, comm_size_, "reduce_scatter_single", rank_, {input}, {output});

  hipStream_t stream = getOperationStream(async_op);
  auto work = createWork(
      stream, getOperationTimeout(options.timeout, options_.timeout), {input});

  // Record start event before RCCLX operation
  work->recordStart("reduce_scatter_single");

  auto dataType = getNcclDataType(input);
  ncclResult_t result = rcclx_api_->reduceScatter(
      input.data_ptr(),
      output.data_ptr(),
      output.numel(),
      dataType,
      getNcclReduceOp(op, nccl_comm_, dataType),
      nccl_comm_,
      stream);

  if (result != ncclSuccess) {
    throw RCCLXException(*rcclx_api_, "RCCLX ReduceScatter failed", result);
  }

  // Record end event after RCCLX operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommRCCLX::all_to_all_single(
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
      name_, comm_size_, "all_to_all_single", rank_, {input}, {output});

  hipStream_t stream = getOperationStream(async_op);
  auto work = createWork(
      stream, getOperationTimeout(options.timeout, options_.timeout), {input});

  // Record start event before RCCLX operation
  work->recordStart("all_to_all_single");

  size_t chunk_size = input.numel() / comm_size_;

  ncclResult_t result = rcclx_api_->allToAll(
      input.data_ptr(),
      output.data_ptr(),
      chunk_size,
      getNcclDataType(input),
      nccl_comm_,
      stream);

  if (result != ncclSuccess) {
    throw RCCLXException(*rcclx_api_, "RCCLX AllToAll failed", result);
  }

  // Record end event after RCCLX operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommRCCLX::all_to_all_v_single(
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

  if (input_total != static_cast<uint64_t>(input.numel())) {
    throw std::runtime_error(
        "Sum of input_split_sizes must equal input tensor size for all_to_all_v_single");
  }

  if (output_total != static_cast<uint64_t>(output.numel())) {
    throw std::runtime_error(
        "Sum of output_split_sizes must equal output tensor size for all_to_all_v_single");
  }

  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "all_to_all_v_single", rank_, {input}, {output});

  hipStream_t stream = getOperationStream(async_op);
  auto work = createWork(
      stream, getOperationTimeout(options.timeout, options_.timeout), {input});

  // Record start event before RCCLX operation
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

  ncclResult_t result = rcclx_api_->allToAllv(
      input.data_ptr(),
      sendcounts.data(),
      senddispls.data(),
      output.data_ptr(),
      recvcounts.data(),
      recvdispls.data(),
      getNcclDataType(input),
      nccl_comm_,
      stream);

  if (result != ncclSuccess) {
    throw RCCLXException(*rcclx_api_, "RCCLX AllToAllv failed", result);
  }

  // Record end event after RCCLX operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommRCCLX::all_to_all(
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

  // Record start event before RCCLX operations
  work->recordStart("all_to_all");

  rcclx_api_->groupStart();

  for (int i = 0; i < comm_size_; ++i) {
    // Send to rank i
    rcclx_api_->send(
        input_tensor_list[i].data_ptr(),
        input_tensor_list[i].numel(),
        getNcclDataType(input_tensor_list[i]),
        i,
        nccl_comm_,
        stream);

    // Receive from rank i
    rcclx_api_->recv(
        output_tensor_list[i].data_ptr(),
        output_tensor_list[i].numel(),
        getNcclDataType(output_tensor_list[i]),
        i,
        nccl_comm_,
        stream);
  }

  rcclx_api_->groupEnd();

  // Record end event after RCCLX operations
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommRCCLX::barrier(
    bool async_op,
    const BarrierOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();

  TorchCommTracingGuard tracingGuard(name_, comm_size_, "barrier", rank_);
  hipStream_t stream = getOperationStream(async_op);
  auto work = createWork(
      stream, getOperationTimeout(options.timeout, options_.timeout));

  // Record start event before RCCLX operation
  work->recordStart("barrier");

  // Use pre-allocated CUDA buffer for barrier
  ncclResult_t result = rcclx_api_->allReduce(
      barrier_buffer_,
      barrier_buffer_,
      1,
      ncclFloat32,
      ncclSum,
      nccl_comm_,
      stream);

  if (result != ncclSuccess) {
    throw RCCLXException(*rcclx_api_, "RCCLX Barrier failed", result);
  }

  // Record end event after RCCLX operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommRCCLX::scatter(
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

  // Record start event before RCCLX operations
  work->recordStart("scatter");

  // Implement scatter using point-to-point operations
  if (rank_ == root) {
    // Root sends to all ranks (except itself)
    rcclx_api_->groupStart();
    for (int i = 0; i < comm_size_; ++i) {
      if (i != root) {
        rcclx_api_->send(
            input_tensor_list[i].data_ptr(),
            input_tensor_list[i].numel(),
            getNcclDataType(input_tensor_list[i]),
            i,
            nccl_comm_,
            stream);
      }
    }
    rcclx_api_->groupEnd();

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
    rcclx_api_->recv(
        output_tensor.data_ptr(),
        output_tensor.numel(),
        getNcclDataType(output_tensor),
        root,
        nccl_comm_,
        stream);
  }

  // Record end event after RCCLX operations
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommRCCLX::gather(
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

  // Record start event before RCCLX operations
  work->recordStart("gather");

  if (rank_ == root) {
    // Root receives from all ranks (except itself)
    rcclx_api_->groupStart();
    for (int i = 0; i < comm_size_; ++i) {
      if (i != root) {
        rcclx_api_->recv(
            output_tensor_list[i].data_ptr(),
            output_tensor_list[i].numel(),
            getNcclDataType(output_tensor_list[i]),
            i,
            nccl_comm_,
            stream);
      }
    }
    rcclx_api_->groupEnd();

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
    rcclx_api_->send(
        input_tensor.data_ptr(),
        input_tensor.numel(),
        getNcclDataType(input_tensor),
        root,
        nccl_comm_,
        stream);
  }

  // Record end event after RCCLX operations
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

std::shared_ptr<TorchCommBackend> TorchCommRCCLX::split(
    const std::vector<int>& ranks,
    const std::string& name,
    const CommOptions& options) {
  checkAndAbortIfTimedOutOrError();

  // Validate that all ranks are valid
  for (int rank : ranks) {
    if (rank < 0 || rank >= comm_size_) {
      throw std::runtime_error(
          "Invalid rank " + std::to_string(rank) +
          " in ranks. Valid ranks are 0 to " + std::to_string(comm_size_ - 1));
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
          "Current rank " + std::to_string(rank_) +
          " is not included in the provided ranks list");
    }
    // Set color to the lowest rank in the group and calculate new rank
    color = *std::min_element(ranks.begin(), ranks.end());
    new_rank = static_cast<int>(std::distance(ranks.begin(), it));
  }

  // Create a new RCCLX communicator
  ncclComm_t new_comm;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;

  // TODO: nccl says that this is not supposed to be called if any operation
  // is outstanding on the comm. We should check for that.
  // TODO: what happens if one rank fails but the others succeed, need to
  // handle the error case.
  // TODO: is this sharing any resources with the original comm?
  ncclResult_t result =
      rcclx_api_->commSplit(nccl_comm_, color, new_rank, &new_comm, &config);
  if (result != ncclSuccess) {
    throw RCCLXException(*rcclx_api_, "RCCLX split failed", result);
  }

  if (new_rank == -1) {
    return nullptr; // Rank is not in any group, return nullptr
  }

  auto new_torchcomm = std::shared_ptr<TorchCommRCCLX>(
      new TorchCommRCCLX(new_comm, device_, options));
  new_torchcomm->rcclx_api_ = rcclx_api_;
  new_torchcomm->hip_api_ = hip_api_;
  new_torchcomm->init(device_, name, options);

  return new_torchcomm;
}

std::string_view TorchCommRCCLX::getBackendName() const {
  return kBackendName;
}

std::string_view TorchCommRCCLX::getCommName() const {
  return name_;
}

void TorchCommRCCLX::register_address(
    const TorchCommRCCLX::AddressWithLen& addr) {
  // We got a register after we got rid of the comm. Is this a fatal error?
  if (nccl_comm_ == nullptr) {
    return;
  }

  if (memoryRegistrationHandles_.find(addr.addr) !=
      memoryRegistrationHandles_.end()) {
    throw std::runtime_error("Memory already registered with RCCLX");
  }
  void* handle = nullptr;
  ncclResult_t result =
      rcclx_api_->commRegister(nccl_comm_, addr.addr, addr.len, &handle);
  if (result != ncclSuccess) {
    throw std::runtime_error(
        "Failed to register memory with RCCLX: " +
        std::string(ncclGetErrorString(result)));
  }
  memoryRegistrationHandles_.emplace(addr.addr, RegistrationHandle(handle));
}

void TorchCommRCCLX::deregister_address(const TorchCommRCCLX::Address& addr) {
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
  ncclResult_t result = rcclx_api_->commDeregister(nccl_comm_, handle);
  if (result != ncclSuccess) {
    throw std::runtime_error(
        "Failed to deregister memory with RCCLX: " +
        std::string(rcclx_api_->getErrorString(result)));
  }

  memoryRegistrationHandles_.erase(it);
}

RCCLXException::RCCLXException(
    RcclxApi& rcclx_api,
    const std::string& message,
    ncclResult_t result)
    : message_(message + ": " + rcclx_api.getErrorString(result)),
      result_(result) {}

const char* RCCLXException::what() const noexcept {
  return message_.c_str();
}

} // namespace comms
} // namespace torch

namespace {
class RCCLXRegistration {
 public:
  RCCLXRegistration() {
    torch::comms::TorchCommFactory::get().register_backend("rcclx", []() {
      return std::make_shared<torch::comms::TorchCommRCCLX>();
    });
  }
};

static const RCCLXRegistration registration{};
} // namespace
