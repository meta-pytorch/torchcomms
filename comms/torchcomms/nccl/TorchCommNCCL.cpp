// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/nccl/TorchCommNCCL.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <fmt/core.h>
#include <nccl.h> // @manual
#include <torch/csrc/cuda/CUDAPluggableAllocator.h> // @manual=//caffe2:torch-cpp-cuda

#include "comms/torchcomms/TorchCommFactory.hpp"
#include "comms/torchcomms/TorchCommLogging.hpp"
#include "comms/torchcomms/TorchCommTracing.hpp"
#include "comms/torchcomms/nccl/TorchCommNCCLBootstrap.hpp"

namespace torch::comms {

namespace {
// Hint key prefix and names for NCCL backend configuration
constexpr std::string_view kHintPrefix = "torchcomm::nccl::";
constexpr std::string_view kHintHighPriorityStream =
    "torchcomm::nccl::high_priority_stream";
constexpr std::string_view kHintMaxEventPoolSize =
    "torchcomm::nccl::max_event_pool_size";
} // namespace

ncclResult_t NCCLException::getResult() const noexcept {
  return result_;
}

TorchCommNCCL::TorchCommNCCL()
    : nccl_comm_{nullptr},
      device_(at::kCUDA),
      init_state_(InitializationState::UNINITIALIZED),
      shutdown_(false) {}

TorchCommNCCL::TorchCommNCCL(const ncclComm_t nccl_comm)
    : nccl_comm_(nccl_comm),
      device_(at::kCUDA),
      init_state_(InitializationState::UNINITIALIZED),
      shutdown_(false) {}

TorchCommNCCL::~TorchCommNCCL() {
  if (init_state_ == InitializationState::INITIALIZED) {
    TC_LOG(ERROR, this) << "TorchCommNCCL was not finalized before destruction";
  }

  // We need to detach the memory hook in case finalize is not called,
  // so that we don't encounter a memory corruption.
  detachMemoryHook();
}

void TorchCommNCCL::init(
    at::Device device,
    const std::string& name,
    const CommOptions& options) {
  // Initialize private members
  device_ = device;
  name_ = name;
  options_ = options;

  // Only initialize once
  if (init_state_ == InitializationState::INITIALIZED) {
    throw std::runtime_error("TorchCommNCCL already initialized");
  } else if (init_state_ == InitializationState::FINALIZED) {
    throw std::runtime_error("TorchCommNCCL already finalized");
  }
  init_state_ = InitializationState::INITIALIZED;

  // Initialize default NCCL API implementation if not already set
  if (!nccl_api_) {
    nccl_api_ = std::make_unique<DefaultNcclApi>();
  }

  // Initialize default CUDA API implementation if not already set
  if (!cuda_api_) {
    cuda_api_ = std::make_unique<DefaultCudaApi>();
  }

  if (device_.index() == -1 || nccl_comm_ == nullptr) {
    auto bootstrap = std::make_unique<TorchCommNCCLBootstrap>(
        options_.store, device_, nccl_api_, cuda_api_, options_.timeout);
    device_ = bootstrap->getDevice();

    if (nccl_comm_ == nullptr) {
      nccl_comm_ = bootstrap->createNcclComm(name, options);
    }
  }

  // Set CUDA device and verify it's accessible
  CUDA_CHECK(
      cuda_api_,
      cuda_api_->setDevice(device_.index()),
      fmt::format("Failed to set CUDA device to {}", device_.index()));

  // Verify device properties and memory availability
  cudaDeviceProp device_prop = {};
  CUDA_CHECK(
      cuda_api_,
      cuda_api_->getDeviceProperties(&device_prop, device_.index()),
      fmt::format(
          "Failed to get device properties for device {}", device_.index()));

  // Check available memory
  size_t free_memory, total_memory;
  CUDA_CHECK(
      cuda_api_,
      cuda_api_->memGetInfo(&free_memory, &total_memory),
      fmt::format("Failed to get memory info for device {}", device_.index()));

  // Read hints and store them
  for (auto const& [key, val] : options_.hints) {
    if (key.starts_with(kHintPrefix)) {
      if (key == kHintHighPriorityStream) {
        high_priority_stream_ = string_to_bool(val);
      } else {
        throw std::runtime_error("Unrecognized hint " + key);
      }
    } else {
      // Ignore keys that do not start with "torchcomm::nccl::"
    }
  }

  // Create internal stream
  //
  // Default priority is 0 as per NVIDIA docs (https://fburl.com/2xb0iqwl).
  int stream_priority = 0;

  // Check for high priority stream hint
  if (high_priority_stream_) {
    int leastPriority, greatestPriority;
    cuda_api_->getStreamPriorityRange(&leastPriority, &greatestPriority);
    stream_priority = greatestPriority;
  }

  CUDA_CHECK(
      cuda_api_,
      cuda_api_->streamCreateWithPriority(
          &internal_stream_, cudaStreamNonBlocking, stream_priority),
      fmt::format(
          "Failed to create internal CUDA stream on device {}",
          device_.index()));

  // Create dependency event for stream synchronization
  CUDA_CHECK(
      cuda_api_,
      cuda_api_->eventCreateWithFlags(
          &dependency_event_, cudaEventDisableTiming),
      fmt::format(
          "Failed to create dependency event on device {}", device_.index()));

  // Allocate CUDA buffer for barrier operations
  CUDA_CHECK(
      cuda_api_,
      cuda_api_->malloc(&barrier_buffer_, sizeof(float)),
      "Failed to allocate barrier buffer");

  const auto kHintMaxEventPoolSizeKey = std::string(kHintMaxEventPoolSize);
  if (options_.hints.contains(kHintMaxEventPoolSizeKey)) {
    max_event_pool_size_ =
        std::stoull(options_.hints.at(kHintMaxEventPoolSizeKey));
  } else {
    max_event_pool_size_ = kDefaultMaxEventPoolSize;
  }

  // Give up our internal reference to the store object here.  The caller
  // would still need to keep a reference to the store object till the init
  // call returns, at which point the NCCL communicator would already be
  // created.
  if (options_.store) {
    options_.store.reset();
  }

  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->commUserRank(nccl_comm_, &rank_),
      "NCCL User Rank failed");

  tryTorchCommLoggingInit("torchcomm");

  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->commCount(nccl_comm_, &comm_size_),
      "NCCL Count failed");

  TorchCommTracingGuard tracingGuard(name_, comm_size_, "init", rank_);

  // Start timeout watchdog thread
  timeout_thread_ = std::thread(&TorchCommNCCL::timeoutWatchdog, this);

  // Register comm with CachingAllocator
  attachMemoryHook();
}

void TorchCommNCCL::finalize() {
  if (init_state_ == InitializationState::UNINITIALIZED) {
    throw std::runtime_error("TorchCommNCCL not initialized");
  } else if (init_state_ == InitializationState::FINALIZED) {
    throw std::runtime_error("TorchCommNCCL already finalized");
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

  // Wait for all pending work objects to complete and get final status
  auto work_status = workq_.finalize();

  if (work_status == TorchWorkNCCL::WorkStatus::NOT_STARTED ||
      work_status == TorchWorkNCCL::WorkStatus::INPROGRESS) {
    throw std::runtime_error(
        "WorkQ finalize returned in progress or not started state");
  }

  // Update comm_state_ based on the work status
  if (work_status == TorchWorkNCCL::WorkStatus::TIMEDOUT) {
    comm_state_ = CommState::TIMEOUT;
    abortNcclComm();
    throw std::runtime_error("Work timed out during finalize");
  } else if (work_status == TorchWorkNCCL::WorkStatus::ERROR) {
    comm_state_ = CommState::ERROR;
    ncclResult_t asyncErr;
    nccl_api_->commGetAsyncError(nccl_comm_, &asyncErr);
    NCCLException ncclException(
        *nccl_api_, "NCCL Async Error", asyncErr, nccl_comm_);
    abortNcclComm();
    throw ncclException;
  }

  // Clean up event pool
  {
    std::lock_guard<std::mutex> lock(event_pool_mutex_);
    while (!event_pool_.empty()) {
      cudaEvent_t event = event_pool_.front();
      event_pool_.pop();
      CUDA_CHECK(
          cuda_api_, cuda_api_->eventDestroy(event), "Failed to destroy event");
    }
  }

  // Free barrier buffer (errors handled by CUDA_CHECK)
  if (barrier_buffer_) {
    CUDA_CHECK(
        cuda_api_,
        cuda_api_->free(barrier_buffer_),
        "Failed to free barrier buffer");
    barrier_buffer_ = nullptr;
  }

  // Destroy dependency event
  if (dependency_event_) {
    CUDA_CHECK(
        cuda_api_,
        cuda_api_->eventDestroy(dependency_event_),
        "Failed to destroy dependency event");
    dependency_event_ = nullptr;
  }

  // Destroy internal stream
  if (internal_stream_) {
    CUDA_CHECK(
        cuda_api_,
        cuda_api_->streamDestroy(internal_stream_),
        "Failed to destroy internal stream");
    internal_stream_ = nullptr;
  }

  // Destroy NCCL communicator
  // Note: If abortNcclComm() was called, nccl_comm_ is already nullptr and this
  // is skipped. We must not call commDestroy after commAbort per NCCL docs.
  if (nccl_comm_) {
    detachMemoryHook();
    // Deregister comm from the CachingAllocator
    nccl_api_->commDestroy(nccl_comm_);
    nccl_comm_ = nullptr;
  }
}

void TorchCommNCCL::abortNcclComm() {
  detachMemoryHook();
  if (nccl_comm_) {
    nccl_api_->commAbort(nccl_comm_);
    nccl_comm_ = nullptr;
  }
  if (options_.abort_process_on_timeout_or_error) {
    TC_LOG(ERROR, this) << "Aborting process due to timeout";
    abort();
  }
}

int TorchCommNCCL::getRank() const {
  checkInitialized();

  int rank;
  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->commUserRank(nccl_comm_, &rank),
      "NCCL User Rank failed");
  return rank;
}

int TorchCommNCCL::getSize() const {
  checkInitialized();

  int comm_size;
  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->commCount(nccl_comm_, &comm_size),
      "NCCL Count failed");
  return comm_size;
}

std::string_view TorchCommNCCL::getBackendName() const {
  return kBackendName;
}

std::string_view TorchCommNCCL::getCommName() const {
  return name_;
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
c10::intrusive_ptr<TorchWork> TorchCommNCCL::send(
    const at::Tensor& tensor,
    int dst,
    bool async_op,
    const SendOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);

  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "send", dst, tensor, tensor);

  CoalescingContext ctx(
      this,
      async_op,
      getOperationTimeout(options.timeout, options_.timeout),
      {std::cref(tensor)});

  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->send(
          tensor.data_ptr(),
          tensor.numel(),
          getNcclDataType(tensor),
          dst,
          nccl_comm_,
          ctx.get_stream()),
      "NCCL Send failed");

  return ctx.get_work();
}

c10::intrusive_ptr<TorchWork> TorchCommNCCL::recv(
    at::Tensor& tensor,
    int src,
    bool async_op,
    const RecvOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);

  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "recv", src, tensor, tensor);

  CoalescingContext ctx(
      this,
      async_op,
      getOperationTimeout(options.timeout, options_.timeout),
      {std::cref(tensor)});

  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->recv(
          tensor.data_ptr(),
          tensor.numel(),
          getNcclDataType(tensor),
          src,
          nccl_comm_,
          ctx.get_stream()),
      "NCCL Recv failed");

  return ctx.get_work();
}

// Batch P2P Operations
c10::intrusive_ptr<TorchWork> TorchCommNCCL::batch_op_issue(
    const std::vector<BatchSendRecv::P2POp>& ops,
    bool async_op,
    const BatchP2POptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  TORCH_CHECK(
      !is_coalescing_active(), "batch_op_issue does not support coalescing");

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

  cudaStream_t stream = getOperationStream(async_op);
  auto work = createWork(
      stream,
      getOperationTimeout(options.timeout, options_.timeout),
      input_tensors);

  // Record start event before NCCL operations
  work->recordStart("batch_op_issue");

  // Start NCCL group for batched operations
  NCCL_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupStart(), "NCCL GroupStart failed");

  // Issue each operation individually
  for (const auto& op : ops) {
    if (op.type == BatchSendRecv::P2POp::OpType::SEND) {
      ncclResult_t result = nccl_api_->send(
          op.tensor.data_ptr(),
          op.tensor.numel(),
          getNcclDataType(op.tensor),
          op.peer,
          nccl_comm_,
          stream);

      if (result != ncclSuccess) {
        throw NCCLException(
            *nccl_api_,
            "NCCL Send failed in batch operation",
            result,
            nccl_comm_);
      }
    } else if (op.type == BatchSendRecv::P2POp::OpType::RECV) {
      ncclResult_t result = nccl_api_->recv(
          op.tensor.data_ptr(),
          op.tensor.numel(),
          getNcclDataType(op.tensor),
          op.peer,
          nccl_comm_,
          stream);

      if (result != ncclSuccess) {
        throw NCCLException(
            *nccl_api_,
            "NCCL Recv failed in batch operation",
            result,
            nccl_comm_);
      }
    }
  }

  // End NCCL group
  NCCL_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupEnd(), "NCCL GroupEnd failed");

  // Record end event after NCCL operations
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

// Collective Operations
c10::intrusive_ptr<TorchWork> TorchCommNCCL::broadcast(
    at::Tensor& tensor,
    int root,
    bool async_op,
    const BroadcastOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);

  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "broadcast", root, tensor, tensor);

  CoalescingContext ctx(
      this,
      async_op,
      getOperationTimeout(options.timeout, options_.timeout),
      {std::cref(tensor)});

  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->bcast(
          tensor.data_ptr(),
          tensor.numel(),
          getNcclDataType(tensor),
          root,
          nccl_comm_,
          ctx.get_stream()),
      "NCCL Broadcast failed");

  return ctx.get_work();
}

c10::intrusive_ptr<TorchWork> TorchCommNCCL::all_reduce(
    at::Tensor& tensor,
    const ReduceOp& op,
    bool async_op,
    const AllReduceOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);

  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "all_reduce", rank_, tensor, tensor);

  CoalescingContext ctx(
      this,
      async_op,
      getOperationTimeout(options.timeout, options_.timeout),
      {std::cref(tensor)});

  const auto dataType = getNcclDataType(tensor);
  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->allReduce(
          tensor.data_ptr(),
          tensor.data_ptr(), // In-place operation
          tensor.numel(),
          dataType,
          getNcclReduceOp(op, nccl_comm_, dataType),
          nccl_comm_,
          ctx.get_stream()),
      "NCCL AllReduce failed");

  return ctx.get_work();
}

c10::intrusive_ptr<TorchWork> TorchCommNCCL::reduce(
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

  CoalescingContext ctx(
      this,
      async_op,
      getOperationTimeout(options.timeout, options_.timeout),
      {std::cref(tensor)});

  const auto dataType = getNcclDataType(tensor);
  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->reduce(
          tensor.data_ptr(),
          rank_ == root ? tensor.data_ptr() : nullptr,
          tensor.numel(),
          dataType,
          getNcclReduceOp(op, nccl_comm_, dataType),
          root,
          nccl_comm_,
          ctx.get_stream()),
      "NCCL Reduce failed");

  return ctx.get_work();
}

c10::intrusive_ptr<TorchWork> TorchCommNCCL::all_gather(
    const std::vector<at::Tensor>& tensor_list,
    const at::Tensor& tensor,
    bool async_op,
    const AllGatherOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  TORCH_CHECK(
      !is_coalescing_active(), "all_gather does not support coalescing");
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

  cudaStream_t stream = getOperationStream(async_op);
  auto work = async_op
      ? createWork(
            stream,
            getOperationTimeout(options.timeout, options_.timeout),
            tensor)
      : createWork(
            stream, getOperationTimeout(options.timeout, options_.timeout));

  work->recordStart("all_gather");

  // Use multiple broadcast operations for all_gather
  NCCL_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupStart(), "NCCL GroupStart failed");

  for (int i = 0; i < comm_size_; ++i) {
    ncclResult_t opResult = nccl_api_->broadcast(
        tensor.data_ptr(),
        tensor_list[i].data_ptr(),
        tensor.numel(),
        getNcclDataType(tensor_list[i]),
        i,
        nccl_comm_,
        stream);
    if (opResult != ncclSuccess) {
      throw NCCLException(
          *nccl_api_,
          "NCCL Broadcast failed in all_gather",
          opResult,
          nccl_comm_);
    }
  }

  NCCL_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupEnd(), "NCCL GroupEnd failed");

  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommNCCL::all_gather_v(
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

  cudaStream_t stream = getOperationStream(async_op);
  auto work = async_op
      ? createWork(
            stream,
            getOperationTimeout(options.timeout, options_.timeout),
            tensor)
      : createWork(
            stream, getOperationTimeout(options.timeout, options_.timeout));

  work->recordStart("all_gather_v");

  // Use multiple broadcast operations for all_gather_v
  NCCL_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupStart(), "NCCL GroupStart failed");

  for (int i = 0; i < comm_size_; ++i) {
    // For all_gather_v, each rank broadcasts its input tensor to all others
    auto& output = tensor_list[i];
    auto& input = (i == rank_) ? tensor : output;
    if (input.numel() != output.numel()) {
      throw std::runtime_error(
          "Output tensor size must equal input tensor size for all_gather_v");
    }
    ncclResult_t opResult = nccl_api_->broadcast(
        input.data_ptr(),
        output.data_ptr(),
        input.numel(),
        getNcclDataType(output),
        i,
        nccl_comm_,
        stream);
    if (opResult != ncclSuccess) {
      throw NCCLException(
          *nccl_api_,
          "NCCL Broadcast failed in all_gather_v",
          opResult,
          nccl_comm_);
    }
  }

  NCCL_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupEnd(), "NCCL GroupEnd failed");

  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommNCCL::all_gather_single(
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

  CoalescingContext ctx(
      this,
      async_op,
      getOperationTimeout(options.timeout, options_.timeout),
      {std::cref(input)});

  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->allGather(
          input.data_ptr(),
          output.data_ptr(),
          input.numel(),
          getNcclDataType(input),
          nccl_comm_,
          ctx.get_stream()),
      "NCCL AllGather failed");

  return ctx.get_work();
}

c10::intrusive_ptr<TorchWork> TorchCommNCCL::reduce_scatter(
    at::Tensor& output,
    const std::vector<at::Tensor>& input_list,
    const ReduceOp& op,
    bool async_op,
    const ReduceScatterOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  TORCH_CHECK(
      !is_coalescing_active(), "reduce_scatter does not support coalescing");
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

  cudaStream_t stream = getOperationStream(async_op);
  auto work = async_op
      ? createWork(
            stream,
            getOperationTimeout(options.timeout, options_.timeout),
            input_list)
      : createWork(
            stream, getOperationTimeout(options.timeout, options_.timeout));

  work->recordStart("reduce_scatter");

  // Use multiple reduce operations for reduce_scatter
  NCCL_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupStart(), "NCCL GroupStart failed");

  for (int i = 0; i < comm_size_; ++i) {
    const auto dataType = getNcclDataType(input_list[i]);
    ncclResult_t opResult;
    if (i == rank_) {
      // This rank receives the reduced result
      opResult = nccl_api_->reduce(
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
      opResult = nccl_api_->reduce(
          input_list[i].data_ptr(),
          nullptr, // Non-root ranks don't receive
          input_list[i].numel(),
          dataType,
          getNcclReduceOp(op, nccl_comm_, dataType),
          i,
          nccl_comm_,
          stream);
    }
    if (opResult != ncclSuccess) {
      throw NCCLException(
          *nccl_api_,
          "NCCL Reduce failed in reduce_scatter",
          opResult,
          nccl_comm_);
    }
  }

  NCCL_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupEnd(), "NCCL GroupEnd failed");

  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommNCCL::reduce_scatter_v(
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

  cudaStream_t stream = getOperationStream(async_op);
  auto work = async_op
      ? createWork(
            stream,
            getOperationTimeout(options.timeout, options_.timeout),
            input_list)
      : createWork(
            stream, getOperationTimeout(options.timeout, options_.timeout));

  work->recordStart("reduce_scatter_v");

  // Use multiple reduce operations for reduce_scatter_v
  NCCL_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupStart(), "NCCL GroupStart failed");

  for (int i = 0; i < comm_size_; ++i) {
    const auto dataType = getNcclDataType(input_list[i]);
    ncclResult_t opResult;
    if (i == rank_) {
      // This rank receives the reduced result
      auto& input_tensor = input_list[i];
      auto& output_tensor = output;
      if (input_tensor.numel() != output_tensor.numel()) {
        throw std::runtime_error(
            "Output tensor size must equal input tensor size for reduce_scatter_v");
      }
      opResult = nccl_api_->reduce(
          input_tensor.data_ptr(),
          output_tensor.data_ptr(),
          output_tensor.numel(),
          dataType,
          getNcclReduceOp(op, nccl_comm_, dataType),
          i,
          nccl_comm_,
          stream);
    } else {
      // Other ranks contribute to the reduction
      opResult = nccl_api_->reduce(
          input_list[i].data_ptr(),
          nullptr, // Non-root ranks don't receive
          input_list[i].numel(),
          dataType,
          getNcclReduceOp(op, nccl_comm_, dataType),
          i,
          nccl_comm_,
          stream);
    }
    if (opResult != ncclSuccess) {
      throw NCCLException(
          *nccl_api_,
          "NCCL Reduce failed in reduce_scatter_v",
          opResult,
          nccl_comm_);
    }
  }

  NCCL_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupEnd(), "NCCL GroupEnd failed");

  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommNCCL::reduce_scatter_single(
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

  CoalescingContext ctx(
      this,
      async_op,
      getOperationTimeout(options.timeout, options_.timeout),
      {std::cref(input)});

  const auto dataType = getNcclDataType(input);
  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->reduceScatter(
          input.data_ptr(),
          output.data_ptr(),
          output.numel(),
          dataType,
          getNcclReduceOp(op, nccl_comm_, dataType),
          nccl_comm_,
          ctx.get_stream()),
      "NCCL ReduceScatter failed");

  return ctx.get_work();
}

c10::intrusive_ptr<TorchWork> TorchCommNCCL::all_to_all_single(
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

  size_t chunk_size = input.numel() / comm_size_;
  const auto data_type = getNcclDataType(input);

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
  CoalescingContext ctx(
      this,
      async_op,
      getOperationTimeout(options.timeout, options_.timeout),
      {std::cref(input)});

  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->allToAll(
          input.data_ptr(),
          output.data_ptr(),
          chunk_size,
          data_type,
          nccl_comm_,
          ctx.get_stream()),
      "NCCL AllToAll failed");

  return ctx.get_work();
#else
  cudaStream_t stream = getOperationStream(async_op);
  auto work = createWork(
      stream, getOperationTimeout(options.timeout, options_.timeout), {input});

  // Record start event before NCCL operation
  work->recordStart("");

  size_t offset = chunk_size * wordSize(data_type);
  char* sptr = static_cast<char*>(input.data_ptr());
  char* rptr = static_cast<char*>(output.data_ptr());
  NCCL_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupStart(), "NCCL GroupStart failed");

  for (int i = 0; i < comm_size_; ++i) {
    // Send to rank i
    ncclResult_t opResult = nccl_api_->send(
        sptr + i * offset, chunk_size, data_type, i, nccl_comm_, stream);
    if (opResult != ncclSuccess) {
      throw NCCLException(
          *nccl_api_,
          "NCCL Send failed in all_to_all_single",
          opResult,
          nccl_comm_);
    }

    // Receive from rank i
    opResult = nccl_api_->recv(
        rptr + i * offset, chunk_size, data_type, i, nccl_comm_, stream);
    if (opResult != ncclSuccess) {
      throw NCCLException(
          *nccl_api_,
          "NCCL Recv failed in all_to_all_single",
          opResult,
          nccl_comm_);
    }
  }

  NCCL_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupEnd(), "NCCL GroupEnd failed");

  // Record end event after NCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
#endif
}

c10::intrusive_ptr<TorchWork> TorchCommNCCL::all_to_all_v_single(
    at::Tensor& output,
    const at::Tensor& input,
    const std::vector<uint64_t>& output_split_sizes,
    const std::vector<uint64_t>& input_split_sizes,
    bool async_op,
    const AllToAllvSingleOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  TORCH_CHECK(
      !is_coalescing_active(),
      "all_to_all_v_single does not support coalescing");
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

  // Validate that split sizes sum does not exceed tensor dimensions
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

  cudaStream_t stream = getOperationStream(async_op);
  auto work = async_op
      ? createWork(
            stream,
            getOperationTimeout(options.timeout, options_.timeout),
            input)
      : createWork(
            stream, getOperationTimeout(options.timeout, options_.timeout));

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
  const auto data_type = getNcclDataType(input);
  const size_t type_size = wordSize(data_type);

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

  char* sptr = static_cast<char*>(input.data_ptr());
  char* rptr = static_cast<char*>(output.data_ptr());

  NCCL_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupStart(), "NCCL GroupStart failed");

  for (int i = 0; i < comm_size_; ++i) {
    ncclResult_t opResult = nccl_api_->send(
        sptr + senddispls[i] * type_size,
        sendcounts[i],
        data_type,
        i,
        nccl_comm_,
        stream);
    if (opResult != ncclSuccess) {
      throw NCCLException(
          *nccl_api_,
          "NCCL Send failed in all_to_all_v_single",
          opResult,
          nccl_comm_);
    }
    opResult = nccl_api_->recv(
        rptr + recvdispls[i] * type_size,
        recvcounts[i],
        data_type,
        i,
        nccl_comm_,
        stream);
    if (opResult != ncclSuccess) {
      throw NCCLException(
          *nccl_api_,
          "NCCL Recv failed in all_to_all_v_single",
          opResult,
          nccl_comm_);
    }
  }

  NCCL_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupEnd(), "NCCL GroupEnd failed");

  // Record end event after NCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommNCCL::all_to_all(
    const std::vector<at::Tensor>& output_tensor_list,
    const std::vector<at::Tensor>& input_tensor_list,
    bool async_op,
    const AllToAllOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  TORCH_CHECK(
      !is_coalescing_active(), "all_to_all does not support coalescing");
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

  cudaStream_t stream = getOperationStream(async_op);
  auto work = async_op
      ? createWork(
            stream,
            getOperationTimeout(options.timeout, options_.timeout),
            input_tensor_list)
      : createWork(
            stream, getOperationTimeout(options.timeout, options_.timeout));

  // Record start event before NCCL operations
  work->recordStart("all_to_all");

  NCCL_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupStart(), "NCCL GroupStart failed");

  for (int i = 0; i < comm_size_; ++i) {
    // Send to rank i
    ncclResult_t opResult = nccl_api_->send(
        input_tensor_list[i].data_ptr(),
        input_tensor_list[i].numel(),
        getNcclDataType(input_tensor_list[i]),
        i,
        nccl_comm_,
        stream);
    if (opResult != ncclSuccess) {
      throw NCCLException(
          *nccl_api_, "NCCL Send failed in all_to_all", opResult, nccl_comm_);
    }

    // Receive from rank i
    opResult = nccl_api_->recv(
        output_tensor_list[i].data_ptr(),
        output_tensor_list[i].numel(),
        getNcclDataType(output_tensor_list[i]),
        i,
        nccl_comm_,
        stream);
    if (opResult != ncclSuccess) {
      throw NCCLException(
          *nccl_api_, "NCCL Recv failed in all_to_all", opResult, nccl_comm_);
    }
  }

  NCCL_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupEnd(), "NCCL GroupEnd failed");

  // Record end event after NCCL operations
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommNCCL::barrier(
    bool async_op,
    const BarrierOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  TORCH_CHECK(!is_coalescing_active(), "barrier does not support coalescing");

  TorchCommTracingGuard tracingGuard(name_, comm_size_, "barrier", rank_);
  cudaStream_t stream = getOperationStream(async_op);
  auto work = createWork(
      stream, getOperationTimeout(options.timeout, options_.timeout));

  // Record start event before NCCL operation
  work->recordStart("barrier");

  // Use pre-allocated CUDA buffer for barrier
  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->allReduce(
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

c10::intrusive_ptr<TorchWork> TorchCommNCCL::scatter(
    at::Tensor& output_tensor,
    const std::vector<at::Tensor>& input_tensor_list,
    int root,
    bool async_op,
    const ScatterOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  TORCH_CHECK(!is_coalescing_active(), "scatter does not support coalescing");
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

  cudaStream_t stream = getOperationStream(async_op);
  std::vector<at::Tensor> input_tensors;
  if (async_op && rank_ == root) {
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
    NCCL_CHECK(
        nccl_api_,
        nccl_comm_,
        nccl_api_->groupStart(),
        "NCCL GroupStart failed");
    for (int i = 0; i < comm_size_; ++i) {
      if (i != root) {
        ncclResult_t opResult = nccl_api_->send(
            input_tensor_list[i].data_ptr(),
            input_tensor_list[i].numel(),
            getNcclDataType(input_tensor_list[i]),
            i,
            nccl_comm_,
            stream);
        if (opResult != ncclSuccess) {
          throw NCCLException(
              *nccl_api_, "NCCL Send failed in scatter", opResult, nccl_comm_);
        }
      }
    }
    NCCL_CHECK(
        nccl_api_, nccl_comm_, nccl_api_->groupEnd(), "NCCL GroupEnd failed");

    // Root copies its own data using cudaMemcpyAsync
    CUDA_CHECK(
        cuda_api_,
        cuda_api_->memcpyAsync(
            output_tensor.data_ptr(),
            input_tensor_list[root].data_ptr(),
            input_tensor_list[root].numel() *
                input_tensor_list[root].element_size(),
            cudaMemcpyDeviceToDevice,
            stream),
        "memcpyAsync failed");
  } else {
    // Non-root ranks receive from root
    NCCL_CHECK(
        nccl_api_,
        nccl_comm_,
        nccl_api_->recv(
            output_tensor.data_ptr(),
            output_tensor.numel(),
            getNcclDataType(output_tensor),
            root,
            nccl_comm_,
            stream),
        "NCCL Recv failed in scatter");
  }

  // Record end event after NCCL operations
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommNCCL::gather(
    const std::vector<at::Tensor>& output_tensor_list,
    const at::Tensor& input_tensor,
    int root,
    bool async_op,
    const GatherOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  TORCH_CHECK(!is_coalescing_active(), "gather does not support coalescing");
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

  cudaStream_t stream = getOperationStream(async_op);
  std::vector<at::Tensor> output_tensors;
  if (rank_ == root) {
    output_tensors = output_tensor_list;
  }
  auto work = async_op
      ? createWork(
            stream,
            getOperationTimeout(options.timeout, options_.timeout),
            input_tensor)
      : createWork(
            stream, getOperationTimeout(options.timeout, options_.timeout));

  // Record start event before NCCL operations
  work->recordStart("gather");

  if (rank_ == root) {
    // Root receives from all ranks (except itself)
    NCCL_CHECK(
        nccl_api_,
        nccl_comm_,
        nccl_api_->groupStart(),
        "NCCL GroupStart failed");
    for (int i = 0; i < comm_size_; ++i) {
      if (i != root) {
        ncclResult_t opResult = nccl_api_->recv(
            output_tensor_list[i].data_ptr(),
            output_tensor_list[i].numel(),
            getNcclDataType(output_tensor_list[i]),
            i,
            nccl_comm_,
            stream);
        if (opResult != ncclSuccess) {
          throw NCCLException(
              *nccl_api_, "NCCL Recv failed in gather", opResult, nccl_comm_);
        }
      }
    }
    NCCL_CHECK(
        nccl_api_, nccl_comm_, nccl_api_->groupEnd(), "NCCL GroupEnd failed");

    // Root copies its own data using cudaMemcpyAsync
    CUDA_CHECK(
        cuda_api_,
        cuda_api_->memcpyAsync(
            output_tensor_list[root].data_ptr(),
            input_tensor.data_ptr(),
            input_tensor.numel() * input_tensor.element_size(),
            cudaMemcpyDeviceToDevice,
            stream),
        "memcpyAsync failed");
  } else {
    // Non-root ranks send to root
    NCCL_CHECK(
        nccl_api_,
        nccl_comm_,
        nccl_api_->send(
            input_tensor.data_ptr(),
            input_tensor.numel(),
            getNcclDataType(input_tensor),
            root,
            nccl_comm_,
            stream),
        "NCCL Send failed in gather");
  }

  // Record end event after NCCL operations
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

std::shared_ptr<TorchCommBackend> TorchCommNCCL::split(
    const std::vector<int>& ranks,
    const std::string& name,
    const CommOptions& options) {
  // Validate the ranks list
  checkAndAbortIfTimedOutOrError();

  std::unordered_set<int> rank_seen;
  for (int rank : ranks) {
    if (rank < 0 || rank >= comm_size_) {
      throw std::runtime_error(
          fmt::format(
              "Invalid rank {} in ranks. Valid ranks are 0 to {}",
              rank,
              comm_size_ - 1));
    }
    if (rank_seen.find(rank) != rank_seen.end()) {
      throw std::runtime_error(
          fmt::format("Rank {} appears multiple times in ranks", rank));
    }
    rank_seen.insert(rank);
  }

  // Determine the color for this rank
  int color;
  int new_rank; // Rank within the new communicator

  if (ranks.empty()) {
    // Empty list means exclude all ranks - use NCCL_SPLIT_NOCOLOR
#ifdef NCCL_SPLIT_NOCOLOR
    color = NCCL_SPLIT_NOCOLOR;
#else
    throw std::runtime_error("NCCL_SPLIT_NOCOLOR is not defined");
#endif
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
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 27, 0)
  config.commName = name.c_str();
#endif

  // Populate NCCL config from user-provided hints
  populateNcclConfigFromHints(config, options, name);

  // Note: NCCL documentation states that commSplit should not be called while
  // operations are outstanding on the parent communicator. Callers are
  // responsible for ensuring all operations complete before calling split().
  // Error handling for partial failures (some ranks succeed, others fail) is
  // left to NCCL's internal mechanisms.
  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->commSplit(nccl_comm_, color, new_rank, &new_comm, &config),
      "NCCL split failed");
  if (new_rank == -1) {
    return nullptr; // Rank is not in the group, return nullptr
  }

  auto new_torchcomm =
      std::shared_ptr<TorchCommNCCL>(new TorchCommNCCL(new_comm));
  new_torchcomm->nccl_api_ = nccl_api_;
  new_torchcomm->cuda_api_ = cuda_api_;
  new_torchcomm->init(device_, name, options);

  return new_torchcomm;
}

void TorchCommNCCL::register_address(
    const TorchCommNCCL::AddressWithLen& addr) {
  // We got a register after we got rid of the comm. Is this a fatal error?
  if (nccl_comm_ == nullptr) {
    return;
  }

  if (memoryRegistrationHandles_.contains(addr.addr)) {
    throw std::runtime_error("Memory already registered with NCCL");
  }
  void* handle = nullptr;
  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->commRegister(nccl_comm_, addr.addr, addr.len, &handle),
      "Failed to register memory with NCCL");
  memoryRegistrationHandles_.emplace(addr.addr, RegistrationHandle(handle));
}

void TorchCommNCCL::deregister_address(const TorchCommNCCL::Address& addr) {
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
  NCCL_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->commDeregister(nccl_comm_, handle),
      "Failed to deregister memory with NCCL");

  memoryRegistrationHandles_.erase(it);
}

NCCLException::NCCLException(
    NcclApi& nccl_api,
    const std::string& message,
    ncclResult_t result,
    ncclComm_t comm)
    : message_(
          message + ": " + nccl_api.getErrorString(result) +
          " \nNCCL Last Error: " + nccl_api.getLastError(comm)),
      result_(result) {}

const char* NCCLException::what() const noexcept {
  return message_.c_str();
}

void TorchCommNCCL::onCoalescingStart() {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  coalesced_stream_ = nullptr;
  nccl_api_->groupStart();
}

void TorchCommNCCL::onCoalescingEnd() {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  nccl_api_->groupEnd();
}

c10::intrusive_ptr<TorchWork> TorchCommNCCL::createCoalescedWork(
    const std::vector<at::Tensor>& tensors) {
  if (coalesced_stream_ == nullptr) {
    return nullptr;
  }

  auto ncclWork = createWork(
      coalesced_stream_,
      getOperationTimeout(std::chrono::milliseconds::zero(), options_.timeout),
      tensors);
  ncclWork->recordEnd();
  enqueueWork(ncclWork, coalesced_stream_);

  coalesced_stream_ = nullptr;

  return ncclWork;
}

TorchCommNCCL::CoalescingContext::CoalescingContext(
    TorchCommNCCL* comm,
    bool async_op,
    std::chrono::milliseconds timeout,
    std::initializer_list<std::reference_wrapper<const at::Tensor>> tensors) {
  comm_ = comm;
  is_coalescing_ = comm_->is_coalescing_active();

  if (is_coalescing_) {
    if (comm_->coalesced_stream_ == nullptr) {
      comm_->coalesced_stream_ = comm_->getOperationStream(async_op);
    }
    stream_ = comm_->coalesced_stream_;
    comm_->register_coalesced_tensors(tensors);
  } else {
    stream_ = comm_->getOperationStream(async_op);
    work_ = async_op
        ? comm_->createWork(
              stream_,
              timeout,
              std::vector<at::Tensor>(tensors.begin(), tensors.end()))
        : comm_->createWork(stream_, timeout, std::vector<at::Tensor>{});
    work_->recordStart("");
  }
}

TorchCommNCCL::CoalescingContext::~CoalescingContext() {
  if (!is_coalescing_) {
    work_->recordEnd();
    comm_->enqueueWork(work_, stream_);
  }
}

} // namespace torch::comms

namespace {
class NCCLRegistration {
 public:
  NCCLRegistration() {
    torch::comms::TorchCommFactory::get().register_backend("nccl", []() {
      return std::make_shared<torch::comms::TorchCommNCCL>();
    });

    // Register allocator factory with its own nccl_api instance
    torch::comms::TorchCommFactory::get().register_allocator_factory(
        "nccl", []() {
          // Create nccl_api for this allocator (captured in lambdas below)
          auto nccl_api = std::make_shared<torch::comms::DefaultNcclApi>();

          static std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>
              nccl_allocator =
                  torch::cuda::CUDAPluggableAllocator::createCustomAllocator(
                      // alloc_fn
                      [nccl_api](size_t size, int device, cudaStream_t stream) {
                        at::cuda::OptionalCUDAGuard gpuGuard(device);
                        void* ptr = nullptr;
                        ncclResult_t result = nccl_api->memAlloc(&ptr, size);
                        TORCH_CHECK(
                            result == ncclSuccess,
                            "ncclMemAlloc failed: ",
                            nccl_api->getErrorString(result));
                        LOG(INFO) << "NCCL mem allocator: allocated " << ptr
                                  << " with " << size << " bytes in stream "
                                  << stream;
                        return ptr;
                      },
                      // free_fn
                      [nccl_api](
                          void* ptr,
                          size_t size,
                          int device,
                          cudaStream_t stream) {
                        LOG(INFO)
                            << "NCCL mem allocator: freeing " << ptr << " with "
                            << size << " bytes in stream " << stream;
                        at::cuda::OptionalCUDAGuard gpuGuard(device);
                        ncclResult_t result = nccl_api->memFree(ptr);
                        TORCH_CHECK(
                            result == ncclSuccess,
                            "ncclMemFree failed: ",
                            nccl_api->getErrorString(result));
                      });
          return nccl_allocator;
        });
  }
};

static const NCCLRegistration registration{};
} // namespace
