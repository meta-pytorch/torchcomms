// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/ncclx/TorchCommNCCLX.hpp"

#include <cstdlib>
#include <cstring>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <fmt/core.h>
#include <nccl.h> // @manual
#include <torch/csrc/cuda/CUDAPluggableAllocator.h> // @manual=//caffe2:torch-cpp-cuda

#include "comms/torchcomms/TorchCommFactory.hpp"
#include "comms/torchcomms/TorchCommTracing.hpp"
#include "comms/torchcomms/ncclx/TorchCommNCCLXBootstrap.hpp"
#include "comms/torchcomms/utils/Logging.hpp"
#include "comms/torchcomms/utils/Utils.hpp"

namespace torch::comms {

namespace {
// Hint key prefix and names for NCCLX backend configuration
constexpr std::string_view kHintPrefix = "torchcomm::ncclx::";
constexpr std::string_view kHintHighPriorityStream =
    "torchcomm::ncclx::high_priority_stream";
constexpr std::string_view kHintMaxEventPoolSize =
    "torchcomm::ncclx::max_event_pool_size";
constexpr std::string_view kHintGarbageCollectIntervalMs =
    "torchcomm::ncclx::garbage_collect_interval_ms";
constexpr std::string_view kHintEnableCudaGraphSupport =
    "torchcomm::ncclx::enable_cuda_graph_support";

// Helper function to validate that metadata tensors are int64_t (torch.int64)
void validateInt64Dtype(const at::Tensor& tensor, std::string_view name) {
  if (tensor.scalar_type() != at::kLong) {
    throw std::runtime_error(
        fmt::format(
            "Tensor '{}' must be of type int64 (torch.int64), but has type {}",
            name,
            c10::toString(tensor.scalar_type())));
  }
}

// Helper function to validate that metadata tensors are int (torch.int)
void validateIntDtype(const at::Tensor& tensor, std::string_view name) {
  if (tensor.scalar_type() != at::kInt) {
    throw std::runtime_error(
        fmt::format(
            "Tensor '{}' must be of type int (torch.int32 or torch.int), but has type {}",
            name,
            c10::toString(tensor.scalar_type())));
  }
}

} // namespace

TorchCommNCCLX::TorchCommNCCLX()
    : nccl_comm_(nullptr),
      device_(at::kCUDA),
      split_counter_(0),
      init_state_(InitializationState::UNINITIALIZED),
      shutdown_(false),
      graph_event_tracker_(this) {}

TorchCommNCCLX::TorchCommNCCLX(const ncclComm_t nccl_comm)
    : nccl_comm_(nccl_comm),
      device_(at::kCUDA),
      split_counter_(0),
      init_state_(InitializationState::UNINITIALIZED),
      shutdown_(false),
      graph_event_tracker_(this) {}

TorchCommNCCLX::~TorchCommNCCLX() {
  if (init_state_ == InitializationState::INITIALIZED) {
    TC_LOG(WARNING, this)
        << "TorchCommNCCLX " << name_
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
        timeout_thread_.detach(); // NOLINT(facebook-hte-BadCall-detach)
      }
    }

    // Abort the NCCL communicator since we can't do a clean finalization
    // Note: We don't call the full abortNcclComm() to avoid potential abort()
    // calls from options_.abort_process_on_timeout_or_error
    if (nccl_comm_) {
      // Best effort to abort the communicator - ignore errors since we're
      // in the destructor
      if (nccl_api_) {
        (void)nccl_api_->commAbort(nccl_comm_);
      }
      nccl_comm_ = nullptr;
    }
  }

  // We need to detach the memory hook in case finalize is not called,
  // so that we don't encounter a memory corruption.
  detachMemoryHook();
}

void TorchCommNCCLX::init(
    at::Device device,
    const std::string& name,
    const CommOptions& options) {
  // Initialize private members
  device_ = device;
  name_ = name;
  options_ = options;

  // Only initialize once
  if (init_state_ == InitializationState::INITIALIZED) {
    throw std::runtime_error("TorchCommNCCLX already initialized");
  } else if (init_state_ == InitializationState::FINALIZED) {
    throw std::runtime_error("TorchCommNCCLX already finalized");
  }

  // Initialize default NCCL API implementation if not already set
  if (!nccl_api_) {
    nccl_api_ = std::make_unique<DefaultNcclxApi>();
  }

  // Initialize default CUDA API implementation if not already set
  if (!cuda_api_) {
    cuda_api_ = std::make_unique<DefaultCudaApi>();
  }

  if (device_.index() == -1 || nccl_comm_ == nullptr) {
    auto bootstrap = std::make_unique<TorchCommNCCLXBootstrap>(
        options_.store, device_, nccl_api_, cuda_api_, options_.timeout);
    device_ = bootstrap->getDevice();

    if (nccl_comm_ == nullptr) {
      nccl_comm_ = bootstrap->createNcclComm(name_, options);
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
      // Ignore keys that do not start with "torchcomm::ncclx::"
    }
  }

  // Create internal stream
  //
  // Default priority is 0 as per NVIDIA docs (https://fburl.com/2xb0iqwl).
  int stream_priority = 0;

  // Check for high priority stream hint
  if (high_priority_stream_) {
    int leastPriority, greatestPriority;
    CUDA_CHECK(
        cuda_api_,
        cuda_api_->getStreamPriorityRange(&leastPriority, &greatestPriority),
        "Failed to get stream");
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
    configs_.max_event_pool_size_ =
        std::stoull(options_.hints.at(kHintMaxEventPoolSizeKey));
  }

  const auto kHintGarbageCollectIntervalMsKey =
      std::string(kHintGarbageCollectIntervalMs);
  if (options_.hints.contains(kHintGarbageCollectIntervalMsKey)) {
    configs_.garbage_collect_interval_ms_ =
        std::stoull(options_.hints.at(kHintGarbageCollectIntervalMsKey));
  }

  const auto kHintEnableCudaGraphSupportKey =
      std::string(kHintEnableCudaGraphSupport);
  if (options_.hints.contains(kHintEnableCudaGraphSupportKey)) {
    configs_.enable_cuda_graph_support_ =
        string_to_bool(options_.hints.at(kHintEnableCudaGraphSupportKey));
  }

  // Give up our internal reference to the store object here.  The caller
  // would still need to keep a reference to the store object till the init
  // call returns, at which point the NCCL communicator would already be
  // created.
  if (options_.store) {
    options_.store.reset();
  }

  NCCLX_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->commUserRank(nccl_comm_, &rank_),
      "NCCLX User Rank failed");

  tryTorchCommLoggingInit("torchcomm");

  NCCLX_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->commCount(nccl_comm_, &comm_size_),
      "NCCLX Count failed");

  TorchCommTracingGuard tracingGuard(name_, comm_size_, "init", rank_);

  // Start timeout watchdog thread
  timeout_thread_ = std::thread(&TorchCommNCCLX::timeoutWatchdog, this);

  // Register comm with CachingAllocator
  attachMemoryHook();

  // Mark initialization as complete only after all steps succeed
  init_state_ = InitializationState::INITIALIZED;
}

void TorchCommNCCLX::finalize() {
  // If initialized and in normal state, nccl_comm_ must be valid.
  // However, if comm was aborted (ERROR or TIMEOUT state), nccl_comm_ will be
  // null.
  TORCH_INTERNAL_ASSERT(
      init_state_ != InitializationState::INITIALIZED ||
          comm_state_ != CommState::NORMAL || nccl_comm_ != nullptr,
      "nccl_comm_ is null but state indicates we are initialized and not aborted");

  if (init_state_ == InitializationState::UNINITIALIZED) {
    throw std::runtime_error("TorchCommNCCLX not initialized");
  } else if (init_state_ == InitializationState::FINALIZED) {
    throw std::runtime_error("TorchCommNCCLX already finalized");
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

  TC_LOG(INFO, this) << "Joined timeout thread";

  // Clear graph work entries after timeout thread has joined.
  // Destroy owned ad-hoc events that were transferred from work objects.
  graph_event_tracker_.destroyAll();

  // Wait for all pending work objects to complete and get final status
  auto work_status = workq_.finalize();

  TC_LOG(INFO, this) << "Finalized work queue";

  if (work_status == TorchWorkNCCLX::WorkStatus::NOT_STARTED ||
      work_status == TorchWorkNCCLX::WorkStatus::INPROGRESS) {
    throw std::runtime_error(
        "WorkQ finalize returned in progress or not started state");
  }

  // Update comm_state_ based on the work status
  if (work_status == TorchWorkNCCLX::WorkStatus::TIMEDOUT) {
    TC_LOG(INFO, this) << "Aborting NCCL comm due to timeout";
    comm_state_ = CommState::TIMEOUT;
    abortNcclComm();
    throw std::runtime_error("Work timed out during finalize");
  } else if (work_status == TorchWorkNCCLX::WorkStatus::ERROR) {
    TC_LOG(INFO, this) << "Aborting NCCL comm due to error";
    comm_state_ = CommState::ERROR;
    ncclResult_t asyncErr;
    NCCLX_CHECK(
        nccl_api_,
        nccl_comm_,
        nccl_api_->commGetAsyncError(nccl_comm_, &asyncErr),
        "failed to get async error");
    NCCLXException ncclException(
        *nccl_api_, "NCCLX Async Error", asyncErr, nccl_comm_);
    abortNcclComm();
    throw ncclException;
  }

  // Clean up event pool
  {
    TC_LOG(INFO, this) << "Cleanup event pool";
    std::lock_guard<std::mutex> lock(event_pool_mutex_);
    while (!event_pool_.empty()) {
      cudaEvent_t event = event_pool_.front();
      event_pool_.pop();
      CUDA_CHECK(
          cuda_api_, cuda_api_->eventDestroy(event), "Failed to destroy event");
    }
  }

  // Free barrier buffer (errors handled by CUDA_CHECK)
  TC_LOG(INFO, this) << "Freeing barrier buffer " << barrier_buffer_;
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
    NCCLX_CHECK(
        nccl_api_,
        nccl_comm_,
        nccl_api_->commDestroy(nccl_comm_),
        "NCCLX Destroy failed");
    nccl_comm_ = nullptr;
  }
}

void TorchCommNCCLX::abortNcclComm() {
  // Both runAbortHooks and detachMemoryHook must run before commAbort:
  // - Abort hooks may need to inspect the live NCCL comm for debug info.
  // - detachMemoryHook deregisters this comm from CachingAllocator so that
  //   subsequent alloc/free callbacks do not reference a destroyed comm.
  TC_LOG(INFO, this) << "Calling abort hooks before commAbort.";
  runAbortHooks();
  detachMemoryHook();
  if (nccl_comm_) {
    NCCLX_CHECK(
        nccl_api_,
        nccl_comm_,
        nccl_api_->commAbort(nccl_comm_),
        "NCCLX Abort failed");
    nccl_comm_ = nullptr;
  }
}

int TorchCommNCCLX::getRank() const {
  checkInitialized();

  int rank;
  NCCLX_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->commUserRank(nccl_comm_, &rank),
      "NCCLX User Rank failed");
  return rank;
}

int TorchCommNCCLX::getSize() const {
  checkInitialized();

  int comm_size;
  NCCLX_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->commCount(nccl_comm_, &comm_size),
      "NCCLX Count failed");
  return comm_size;
}

std::string_view TorchCommNCCLX::getBackendName() const {
  return kBackendName;
}

std::string_view TorchCommNCCLX::getCommName() const {
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
c10::intrusive_ptr<TorchWork> TorchCommNCCLX::send(
    const at::Tensor& tensor,
    int dst,
    bool async_op,
    const SendOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);

  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "send", dst, tensor, tensor);

  cudaStream_t stream = getOperationStream(async_op);
  graph_event_tracker_.initOnGraphStart(stream);
  auto work = async_op
      ? createWork(
            stream,
            getOperationTimeout(options.timeout, options_.timeout),
            tensor)
      : createWork(
            stream, getOperationTimeout(options.timeout, options_.timeout));

  // Record start event before NCCL operation
  work->recordStart("send");

  NCCLX_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->send(
          tensor.data_ptr(),
          tensor.numel(),
          getNcclDataType(tensor),
          dst,
          nccl_comm_,
          stream),
      "NCCLX Send failed");

  // Record end event after NCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommNCCLX::recv(
    at::Tensor& tensor,
    int src,
    bool async_op,
    const RecvOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);

  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "recv", src, tensor, tensor);

  cudaStream_t stream = getOperationStream(async_op);
  graph_event_tracker_.initOnGraphStart(stream);
  auto work = createWork(
      stream, getOperationTimeout(options.timeout, options_.timeout));

  // Record start event before NCCL operation
  work->recordStart("recv");

  NCCLX_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->recv(
          tensor.data_ptr(),
          tensor.numel(),
          getNcclDataType(tensor),
          src,
          nccl_comm_,
          stream),
      "NCCLX Recv failed");

  // Record end event after NCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

// Batch P2P Operations
c10::intrusive_ptr<TorchWork> TorchCommNCCLX::batch_op_issue(
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

  cudaStream_t stream = getOperationStream(async_op);
  graph_event_tracker_.initOnGraphStart(stream);
  auto work = createWork(
      stream,
      getOperationTimeout(options.timeout, options_.timeout),
      // NOLINTNEXTLINE(facebook-conditional-operator-argument-copy)
      async_op ? input_tensors : std::vector<at::Tensor>{});

  // Record start event before NCCL operations
  work->recordStart("batch_op_issue");

  // Start NCCL group for batched operations
  NCCLX_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->groupStart(),
      "NCCLX GroupStart failed");

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
        throw NCCLXException(
            *nccl_api_,
            "NCCLX Send failed in batch operation",
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
        throw NCCLXException(
            *nccl_api_,
            "NCCLX Recv failed in batch operation",
            result,
            nccl_comm_);
      }
    }
  }

  // End NCCL group
  NCCLX_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupEnd(), "NCCLX GroupEnd failed");

  // Record end event after NCCL operations
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

// Collective Operations
c10::intrusive_ptr<TorchWork> TorchCommNCCLX::broadcast(
    at::Tensor& tensor,
    int root,
    bool async_op,
    const BroadcastOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);

  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "broadcast", rank_, tensor, tensor);

  cudaStream_t stream = getOperationStream(async_op);

  graph_event_tracker_.initOnGraphStart(stream);
  auto work = async_op
      ? createWork(
            stream,
            getOperationTimeout(options.timeout, options_.timeout),
            tensor)
      : createWork(
            stream, getOperationTimeout(options.timeout, options_.timeout));

  // Record start event before NCCL operation
  work->recordStart("broadcast");

  NCCLX_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->bcast(
          tensor.data_ptr(),
          tensor.numel(),
          getNcclDataType(tensor),
          root,
          nccl_comm_,
          stream),
      "NCCLX Broadcast failed");

  // Record end event after NCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommNCCLX::all_reduce(
    at::Tensor& tensor,
    const ReduceOp& op,
    bool async_op,
    const AllReduceOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);

  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "all_reduce", rank_, tensor, tensor);

  cudaStream_t stream = getOperationStream(async_op);
  graph_event_tracker_.initOnGraphStart(stream);
  auto work = async_op
      ? createWork(
            stream,
            getOperationTimeout(options.timeout, options_.timeout),
            tensor)
      : createWork(
            stream, getOperationTimeout(options.timeout, options_.timeout));

  // Record start event before NCCL operation
  work->recordStart("all_reduce");

  const auto dataType = getNcclDataType(tensor);
  NCCLX_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->allReduce(
          tensor.data_ptr(),
          tensor.data_ptr(), // In-place operation
          tensor.numel(),
          dataType,
          getNcclReduceOp(op, nccl_comm_, dataType),
          nccl_comm_,
          stream),
      "NCCLX AllReduce failed");

  // Record end event after NCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommNCCLX::reduce(
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

  cudaStream_t stream = getOperationStream(async_op);
  graph_event_tracker_.initOnGraphStart(stream);
  auto work = async_op
      ? createWork(
            stream,
            getOperationTimeout(options.timeout, options_.timeout),
            tensor)
      : createWork(
            stream, getOperationTimeout(options.timeout, options_.timeout));

  // Record start event before NCCL operation
  work->recordStart("reduce");

  const auto dataType = getNcclDataType(tensor);
  NCCLX_CHECK(
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
          stream),
      "NCCLX Reduce failed");

  // Record end event after NCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommNCCLX::all_gather(
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

  cudaStream_t stream = getOperationStream(async_op);

  // Allocate temporary contiguous tensor to receive all gathered data
  // PyTorch's caching allocator will manage and reuse this buffer
  const int64_t total_elements = tensor.numel() * comm_size_;
  at::Tensor temp_tensor = at::empty(
      {total_elements},
      at::TensorOptions()
          .dtype(tensor.dtype())
          .device(tensor.device())
          .requires_grad(false));

  // Pass both input tensor and temp_tensor to createWork for refcounting
  // when async_op is true
  graph_event_tracker_.initOnGraphStart(stream);
  auto work = createWork(
      stream,
      getOperationTimeout(options.timeout, options_.timeout),
      async_op ? std::vector<at::Tensor>{tensor, temp_tensor}
               : std::vector<at::Tensor>{});

  work->recordStart("all_gather");

  // Use NCCL allGather to receive data into temporary tensor
  NCCLX_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->allGather(
          tensor.data_ptr(),
          temp_tensor.data_ptr(),
          tensor.numel(),
          getNcclDataType(tensor),
          nccl_comm_,
          stream),
      "NCCLX AllGather failed");

  // Copy data from temporary tensor to individual output tensors
  const size_t element_size = tensor.element_size();
  const size_t per_rank_bytes = tensor.numel() * element_size;
  for (int i = 0; i < comm_size_; ++i) {
    void* src_ptr =
        static_cast<char*>(temp_tensor.data_ptr()) + (i * per_rank_bytes);
    CUDA_CHECK(
        cuda_api_,
        cuda_api_->memcpyAsync(
            tensor_list[i].data_ptr(),
            src_ptr,
            per_rank_bytes,
            cudaMemcpyDeviceToDevice,
            stream),
        "Failed to copy from temporary tensor to output tensor");
  }

  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommNCCLX::all_gather_v(
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

  // Ensure input tensor is contiguous
  ensureTensorContiguous(tensor);

  for (const auto& t : tensor_list) {
    ensureTensorContiguous(t);
  }
  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "all_gather_v", rank_, tensor_list, {tensor});

  cudaStream_t stream = getOperationStream(async_op);
  graph_event_tracker_.initOnGraphStart(stream);
  auto work = async_op
      ? createWork(
            stream,
            getOperationTimeout(options.timeout, options_.timeout),
            tensor)
      : createWork(
            stream, getOperationTimeout(options.timeout, options_.timeout));

  work->recordStart("all_gather_v");

  // Use multiple broadcast operations for all_gather
  NCCLX_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->groupStart(),
      "NCCLX GroupStart failed");

  for (int i = 0; i < comm_size_; ++i) {
    // assign input/output tensors to support vector all_gather (all_gather_v)
    // where unevenly sized inputs are gathered among participating ranks
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
      throw NCCLXException(
          *nccl_api_,
          "NCCLX Broadcast failed in all_gather",
          opResult,
          nccl_comm_);
    }
  }

  NCCLX_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupEnd(), "NCCLX GroupEnd failed");

  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommNCCLX::all_gather_single(
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

  cudaStream_t stream = getOperationStream(async_op);
  graph_event_tracker_.initOnGraphStart(stream);
  auto work = async_op
      ? createWork(
            stream,
            getOperationTimeout(options.timeout, options_.timeout),
            input)
      : createWork(
            stream, getOperationTimeout(options.timeout, options_.timeout));

  work->recordStart("all_gather_single");

  NCCLX_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->allGather(
          input.data_ptr(),
          output.data_ptr(),
          input.numel(),
          getNcclDataType(input),
          nccl_comm_,
          stream),
      "NCCLX AllGather failed");

  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommNCCLX::reduce_scatter(
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

  cudaStream_t stream = getOperationStream(async_op);
  graph_event_tracker_.initOnGraphStart(stream);
  auto work = createWork(
      stream,
      getOperationTimeout(options.timeout, options_.timeout),
      // NOLINTNEXTLINE(facebook-conditional-operator-argument-copy)
      async_op ? input_list : std::vector<at::Tensor>{});

  work->recordStart("reduce_scatter");

  // Use multiple reduce operations for reduce_scatter
  NCCLX_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->groupStart(),
      "NCCLX GroupStart failed");

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
      throw NCCLXException(
          *nccl_api_,
          "NCCLX Reduce failed in reduce_scatter",
          opResult,
          nccl_comm_);
    }
  }

  NCCLX_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupEnd(), "NCCLX GroupEnd failed");

  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommNCCLX::reduce_scatter_v(
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

  // Check that all input tensors are contiguous and have correct size
  for (const auto& t : input_list) {
    ensureTensorContiguous(t);
  }

  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "reduce_scatter_v", rank_, input_list, {output});

  cudaStream_t stream = getOperationStream(async_op);
  graph_event_tracker_.initOnGraphStart(stream);
  auto work = createWork(
      stream,
      getOperationTimeout(options.timeout, options_.timeout),
      // NOLINTNEXTLINE(facebook-conditional-operator-argument-copy)
      async_op ? input_list : std::vector<at::Tensor>{});

  work->recordStart("reduce_scatter_v");

  // Use multiple reduce operations for reduce_scatter
  NCCLX_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->groupStart(),
      "NCCLX GroupStart failed");

  for (int i = 0; i < comm_size_; ++i) {
    const auto dataType = getNcclDataType(input_list[i]);
    ncclResult_t opResult;
    if (i == rank_) {
      // This rank receives the reduced result
      // assign input/output tensor to support vector reduce_scatter
      // (reduce_scatter_v) where inputs are reduced and scattered unevenly
      // among participating ranks
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
      throw NCCLXException(
          *nccl_api_,
          "NCCLX Reduce failed in reduce_scatter_v",
          opResult,
          nccl_comm_);
    }
  }

  NCCLX_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupEnd(), "NCCLX GroupEnd failed");

  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommNCCLX::reduce_scatter_single(
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

  cudaStream_t stream = getOperationStream(async_op);
  graph_event_tracker_.initOnGraphStart(stream);
  auto work = async_op
      ? createWork(
            stream,
            getOperationTimeout(options.timeout, options_.timeout),
            input)
      : createWork(
            stream, getOperationTimeout(options.timeout, options_.timeout));

  // Record start event before NCCL operation
  work->recordStart("reduce_scatter_single");

  const auto dataType = getNcclDataType(input);
  NCCLX_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->reduceScatter(
          input.data_ptr(),
          output.data_ptr(),
          output.numel(),
          dataType,
          getNcclReduceOp(op, nccl_comm_, dataType),
          nccl_comm_,
          stream),
      "NCCLX ReduceScatter failed");

  // Record end event after NCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommNCCLX::all_to_all_single(
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

  cudaStream_t stream = getOperationStream(async_op);
  graph_event_tracker_.initOnGraphStart(stream);
  auto work = async_op
      ? createWork(
            stream,
            getOperationTimeout(options.timeout, options_.timeout),
            input)
      : createWork(
            stream, getOperationTimeout(options.timeout, options_.timeout));

  // Record start event before NCCL operation
  work->recordStart("all_to_all_single");

  size_t chunk_size = input.numel() / comm_size_;

  NCCLX_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->allToAll(
          input.data_ptr(),
          output.data_ptr(),
          chunk_size,
          getNcclDataType(input),
          nccl_comm_,
          stream),
      "NCCLX AllToAll failed");

  // Record end event after NCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommNCCLX::all_to_all_v_single(
    at::Tensor& output,
    const at::Tensor& input,
    const std::vector<uint64_t>& output_split_sizes,
    const std::vector<uint64_t>& input_split_sizes,
    bool async_op,
    const AllToAllvSingleOptions& options) {
  checkInitialized();
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
  graph_event_tracker_.initOnGraphStart(stream);
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

  NCCLX_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->allToAllv(
          input.data_ptr(),
          sendcounts.data(),
          senddispls.data(),
          output.data_ptr(),
          recvcounts.data(),
          recvdispls.data(),
          getNcclDataType(input),
          nccl_comm_,
          stream),
      "NCCLX AllToAllv failed");

  // Record end event after NCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommNCCLX::all_to_all(
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

  cudaStream_t stream = getOperationStream(async_op);
  graph_event_tracker_.initOnGraphStart(stream);
  auto work = createWork(
      stream,
      getOperationTimeout(options.timeout, options_.timeout),
      // NOLINTNEXTLINE(facebook-conditional-operator-argument-copy)
      async_op ? input_tensor_list : std::vector<at::Tensor>{});

  // Record start event before NCCL operations
  work->recordStart("all_to_all");

  NCCLX_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->groupStart(),
      "NCCLX GroupStart failed");

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
      throw NCCLXException(
          *nccl_api_, "NCCLX Send failed in all_to_all", opResult, nccl_comm_);
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
      throw NCCLXException(
          *nccl_api_, "NCCLX Recv failed in all_to_all", opResult, nccl_comm_);
    }
  }

  NCCLX_CHECK(
      nccl_api_, nccl_comm_, nccl_api_->groupEnd(), "NCCLX GroupEnd failed");

  // Record end event after NCCL operations
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommNCCLX::alltoallv_dynamic_dispatch(
    const std::vector<at::Tensor>& output_tensor_list,
    at::Tensor& output_chunk_sizes_per_rank,
    const at::Tensor& input_tensor,
    const at::Tensor& input_chunk_sizes,
    const at::Tensor& input_chunk_indices,
    const at::Tensor& input_chunk_count_per_rank,
    bool async_op) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(input_tensor);
  ensureTensorContiguous(input_chunk_sizes);
  ensureTensorContiguous(input_chunk_indices);
  ensureTensorContiguous(input_chunk_count_per_rank);
  ensureTensorContiguous(output_chunk_sizes_per_rank);

  for (const auto& t : output_tensor_list) {
    ensureTensorContiguous(t);
  }

  // Validate metadata tensor types - all must be int64_t (torch.int64)
  validateInt64Dtype(input_chunk_sizes, "input_chunk_sizes");
  validateInt64Dtype(input_chunk_indices, "input_chunk_indices");
  validateInt64Dtype(input_chunk_count_per_rank, "input_chunk_count_per_rank");
  validateInt64Dtype(
      output_chunk_sizes_per_rank, "output_chunk_sizes_per_rank");

  TorchCommTracingGuard tracingGuard(
      name_,
      comm_size_,
      "alltoallv_dynamic_dispatch",
      rank_,
      {input_tensor},
      output_tensor_list);

  // Convert vector of tensors to a CPU tensor holding pointers, which will be
  // held by torchComm.
  //
  // Note: PyTorch does not provide tensors of void* or uintptr_t, so we
  // workaround it by using tensors of int64_t, which should work on almost
  // every platform that we care about. We use memcpy for type punning instead
  // of reinterpret_cast to avoid undefined behavior due to strict aliasing
  // rules.
  static_assert(
      sizeof(void*) == sizeof(int64_t),
      "void* and int64_t must have the same size for pointer storage");
  at::Tensor output_tensor_ptrs = at::zeros(
      {static_cast<int64_t>(output_tensor_list.size())},
      at::TensorOptions().dtype(at::kLong).device(at::kCPU));
  int64_t* ptr_storage = output_tensor_ptrs.data_ptr<int64_t>();
  for (size_t i = 0; i < output_tensor_list.size(); ++i) {
    void* data_ptr = output_tensor_list[i].data_ptr();
    std::memcpy(&ptr_storage[i], &data_ptr, sizeof(void*));
  }

  cudaStream_t stream = getOperationStream(async_op);
  graph_event_tracker_.initOnGraphStart(stream);
  auto work = createWork(
      stream,
      options_.timeout,
      async_op
          ? std::vector<
                at::Tensor>{input_tensor, input_chunk_sizes, input_chunk_indices, input_chunk_count_per_rank}
          : std::vector<at::Tensor>{});

  // Save the CPU pointer tensor to keep it alive for the lifetime of the work
  // object. output_tensor_ptrs is a CPU tensor holding raw pointers to the
  // output tensors and must remain valid during async operations and graph
  // replay. The output_tensor_list (GPU tensors) is kept alive by the caller.
  work->setCPUTensors({output_tensor_ptrs});

  // Record start event before NCCL operation
  work->recordStart("alltoallv_dynamic_dispatch");

  // Ensure int64_t and size_t are compatible for safe casting
  static_assert(
      sizeof(int64_t) == sizeof(size_t),
      "int64_t and size_t must have the same size for metadata tensors");

  ncclResult_t result = nccl_api_->alltoallvDynamicDispatch(
      input_tensor.data_ptr(),
      reinterpret_cast<size_t*>(input_chunk_sizes.data_ptr()),
      input_chunk_sizes.numel(),
      reinterpret_cast<size_t*>(input_chunk_indices.data_ptr()),
      reinterpret_cast<size_t*>(input_chunk_count_per_rank.data_ptr()),
      reinterpret_cast<void* const*>(output_tensor_ptrs.data_ptr()),
      reinterpret_cast<size_t*>(output_chunk_sizes_per_rank.data_ptr()),
      input_tensor.numel(),
      output_tensor_list[0].numel(),
      getNcclDataType(input_tensor),
      nccl_comm_,
      stream);

  NCCLX_CHECK(
      nccl_api_, nccl_comm_, result, "NCCLX alltoallvDynamicDispatch failed");

  // Record end event after NCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommNCCLX::alltoallv_dynamic_combine(
    at::Tensor& output_tensor,
    const at::Tensor& input_tensor,
    const at::Tensor& input_chunk_sizes,
    const at::Tensor& input_chunk_indices,
    const at::Tensor& input_chunk_count_per_rank,
    bool async_op) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(output_tensor);
  ensureTensorContiguous(input_tensor);
  ensureTensorContiguous(input_chunk_sizes);
  ensureTensorContiguous(input_chunk_indices);
  ensureTensorContiguous(input_chunk_count_per_rank);

  // Validate metadata tensor types - all must be int64_t (torch.int64)
  validateInt64Dtype(input_chunk_sizes, "input_chunk_sizes");
  validateInt64Dtype(input_chunk_indices, "input_chunk_indices");
  validateInt64Dtype(input_chunk_count_per_rank, "input_chunk_count_per_rank");

  TorchCommTracingGuard tracingGuard(
      name_,
      comm_size_,
      "alltoallv_dynamic_combine",
      rank_,
      input_tensor,
      output_tensor);

  cudaStream_t stream = getOperationStream(async_op);
  graph_event_tracker_.initOnGraphStart(stream);
  auto work = createWork(
      stream,
      options_.timeout,
      async_op
          ? std::vector<
                at::Tensor>{input_tensor, input_chunk_sizes, input_chunk_indices, input_chunk_count_per_rank}
          : std::vector<at::Tensor>{});

  // Record start event before NCCL operation
  work->recordStart("alltoallv_dynamic_combine");

  // Ensure int64_t and size_t are compatible for safe casting
  static_assert(
      sizeof(int64_t) == sizeof(size_t),
      "int64_t and size_t must have the same size for metadata tensors");

  // Cast int64_t* to size_t* for NCCL API (safe on 64-bit systems)
  ncclResult_t result = nccl_api_->alltoallvDynamicCombine(
      input_tensor.data_ptr(),
      reinterpret_cast<size_t*>(input_chunk_sizes.data_ptr()),
      input_chunk_sizes.numel(),
      reinterpret_cast<size_t*>(input_chunk_indices.data_ptr()),
      reinterpret_cast<size_t*>(input_chunk_count_per_rank.data_ptr()),
      output_tensor.data_ptr(),
      input_tensor.numel(),
      output_tensor.numel(),
      getNcclDataType(input_tensor),
      nccl_comm_,
      stream);

  NCCLX_CHECK(
      nccl_api_, nccl_comm_, result, "NCCLX alltoallvDynamicCombine failed");

  // Record end event after NCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchCommNCCLXPersistentRequest>
TorchCommNCCLX::alltoallv_dedup_init(
    const int num_send_blocks,
    const int block_count,
    const int block_num_recv_buckets,
    const int num_recv_buckets,
    at::ScalarType dtype,
    // async_op decides the stream, thus we need to specify at init time as
    // required by NCCLX API
    bool async_op) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();

  cudaStream_t stream = getOperationStream(async_op);

  void* pReq = nullptr;
  ncclResult_t result = nccl_api_->alltoallvDedupInit(
      num_send_blocks,
      block_count,
      block_num_recv_buckets,
      num_recv_buckets,
      getNcclDataType(dtype),
      nccl_comm_,
      stream,
      &pReq);

  NCCLX_CHECK(nccl_api_, nccl_comm_, result, "NCCLX alltoallvDedupInit failed");
  return at::make_intrusive<TorchCommNCCLXPersistentRequest>(
      shared_from_this(), pReq, stream);
}

c10::intrusive_ptr<TorchWork> TorchCommNCCLX::alltoallv_dedup_exec(
    at::Tensor& output_tensor,
    at::Tensor& recv_block_ids,
    const at::Tensor& input_tensor,
    const at::Tensor& send_indices,
    const at::Tensor& forward_indices,
    const at::Tensor& recv_indices,
    at::intrusive_ptr<TorchCommNCCLXPersistentRequest> pReq) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();

  ensureTensorContiguous(output_tensor);
  ensureTensorContiguous(recv_block_ids);
  ensureTensorContiguous(input_tensor);
  ensureTensorContiguous(send_indices);
  ensureTensorContiguous(forward_indices);
  ensureTensorContiguous(recv_indices);

  validateIntDtype(send_indices, "send_indices");
  validateIntDtype(forward_indices, "forward_indices");
  validateIntDtype(recv_indices, "recv_indices");

  TorchCommTracingGuard tracingGuard(
      name_,
      comm_size_,
      "alltoallv_dedup_exec",
      rank_,
      input_tensor,
      output_tensor);

  TORCH_CHECK(
      pReq->getStream() != std::nullopt,
      "cuda stream is not recorded at alltoallv_dedup_init before calling alltoallv_dedup_exec");
  auto stream = pReq->getStream().value();
  graph_event_tracker_.initOnGraphStart(stream);
  auto work = createWork(stream, options_.timeout, input_tensor);
  // Keep the persistent request alive until last dedup work has completed and
  // cleaned up by CPU, because work->wait() doesn't let CPU wait for kernel
  // to complete.
  work->setPersistentRequest(pReq);

  // Record start event before NCCL operation
  work->recordStart("alltoallv_dedup_exec");

  ncclResult_t result = nccl_api_->alltoallvDedupExec(
      input_tensor.data_ptr(),
      send_indices.data_ptr<int>(),
      forward_indices.data_ptr<int>(),
      recv_indices.data_ptr<int>(),
      output_tensor.data_ptr(),
      recv_block_ids.data_ptr<int>(),
      pReq->getRequestPtr());

  NCCLX_CHECK(nccl_api_, nccl_comm_, result, "NCCLX alltoallvDedupExec failed");

  // Record end event after NCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommNCCLX::alltoallv_dedup_combine(
    at::Tensor& output_tensor,
    const at::Tensor& input_tensor,
    const at::Tensor& send_indices,
    const at::Tensor& forward_indices,
    const at::Tensor& recv_indices,
    at::intrusive_ptr<TorchCommNCCLXPersistentRequest> pReq) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();

  ensureTensorContiguous(output_tensor);
  ensureTensorContiguous(input_tensor);
  ensureTensorContiguous(send_indices);
  ensureTensorContiguous(forward_indices);
  ensureTensorContiguous(recv_indices);

  validateIntDtype(send_indices, "send_indices");
  validateIntDtype(forward_indices, "forward_indices");
  validateIntDtype(recv_indices, "recv_indices");

  TorchCommTracingGuard tracingGuard(
      name_,
      comm_size_,
      "alltoallv_dedup_combine",
      rank_,
      input_tensor,
      output_tensor);

  TORCH_CHECK(
      pReq->getStream() != std::nullopt,
      "cuda stream is not recorded at alltoallv_dedup_init before calling alltoallv_dedup_combine");
  auto stream = pReq->getStream().value();
  graph_event_tracker_.initOnGraphStart(stream);
  auto work = createWork(stream, options_.timeout, input_tensor);

  // Record start event before NCCL operation
  work->recordStart("alltoallv_dedup_combine");

  ncclResult_t result = nccl_api_->alltoallvDedupCombine(
      input_tensor.data_ptr(),
      send_indices.data_ptr<int>(),
      forward_indices.data_ptr<int>(),
      recv_indices.data_ptr<int>(),
      output_tensor.data_ptr(),
      pReq->getRequestPtr());

  NCCLX_CHECK(
      nccl_api_, nccl_comm_, result, "NCCLX alltoallvDedupCombine failed");

  // Record end event after NCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommNCCLX::barrier(
    bool async_op,
    const BarrierOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();

  TorchCommTracingGuard tracingGuard(name_, comm_size_, "barrier", rank_);

  cudaStream_t stream = getOperationStream(async_op);
  graph_event_tracker_.initOnGraphStart(stream);
  auto work = createWork(
      stream, getOperationTimeout(options.timeout, options_.timeout));

  // Record start event before NCCL operation
  work->recordStart("barrier");

  // Use pre-allocated CUDA buffer for barrier
  NCCLX_CHECK(
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
      "NCCLX Barrier failed");

  // Record end event after NCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommNCCLX::scatter(
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

  cudaStream_t stream = getOperationStream(async_op);
  std::vector<at::Tensor> input_tensors;
  if (async_op && rank_ == root) {
    input_tensors = input_tensor_list;
  }
  graph_event_tracker_.initOnGraphStart(stream);
  auto work = createWork(
      stream,
      getOperationTimeout(options.timeout, options_.timeout),
      input_tensors);

  // Record start event before NCCL operations
  work->recordStart("scatter");

  // Implement scatter using point-to-point operations
  if (rank_ == root) {
    // Root sends to all ranks (except itself)
    NCCLX_CHECK(
        nccl_api_,
        nccl_comm_,
        nccl_api_->groupStart(),
        "NCCLX GroupStart failed");
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
          throw NCCLXException(
              *nccl_api_, "NCCLX Send failed in scatter", opResult, nccl_comm_);
        }
      }
    }
    NCCLX_CHECK(
        nccl_api_, nccl_comm_, nccl_api_->groupEnd(), "NCCLX GroupEnd failed");

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
    NCCLX_CHECK(
        nccl_api_,
        nccl_comm_,
        nccl_api_->recv(
            output_tensor.data_ptr(),
            output_tensor.numel(),
            getNcclDataType(output_tensor),
            root,
            nccl_comm_,
            stream),
        "NCCLX Recv failed in scatter");
  }

  // Record end event after NCCL operations
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommNCCLX::gather(
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

  cudaStream_t stream = getOperationStream(async_op);
  std::vector<at::Tensor> output_tensors;
  if (rank_ == root) {
    output_tensors = output_tensor_list;
  }
  graph_event_tracker_.initOnGraphStart(stream);
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
    NCCLX_CHECK(
        nccl_api_,
        nccl_comm_,
        nccl_api_->groupStart(),
        "NCCLX GroupStart failed");
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
          throw NCCLXException(
              *nccl_api_, "NCCLX Recv failed in gather", opResult, nccl_comm_);
        }
      }
    }
    NCCLX_CHECK(
        nccl_api_, nccl_comm_, nccl_api_->groupEnd(), "NCCLX GroupEnd failed");

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
    NCCLX_CHECK(
        nccl_api_,
        nccl_comm_,
        nccl_api_->send(
            input_tensor.data_ptr(),
            input_tensor.numel(),
            getNcclDataType(input_tensor),
            root,
            nccl_comm_,
            stream),
        "NCCLX Send failed in gather");
  }

  // Record end event after NCCL operations
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

// Window & One-sided Operations
std::shared_ptr<TorchCommWindow> TorchCommNCCLX::new_window(
    const std::optional<at::Tensor>& tensor) {
  auto win =
      std::make_shared<TorchCommWindowNCCLXGin>(nccl_comm_, shared_from_this());
  if (tensor.has_value()) {
    win->tensor_register(tensor.value());
  }
  return win;
}

std::shared_ptr<TorchCommBackend> TorchCommNCCLX::split(
    const std::vector<int>& ranks,
    const std::string& split_name,
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

  // Determine the color and new rank for this rank
  int color;
  int new_rank;

  if (ranks.empty()) {
    // Empty list means exclude all ranks - use NCCL_SPLIT_NOCOLOR
    color = -1; // Use -1 as equivalent to NCCL_SPLIT_NOCOLOR
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
  std::string commDesc = fmt::format(
      "{}::split::{}_{}_{}", name_, color, split_name, split_counter_++);
  config.commDesc = commDesc.c_str();

  // Set splitGroupRanks and splitGroupSize hints automatically based on ranks
  // parameter
  if (!ranks.empty()) {
    config.splitGroupRanks = const_cast<int*>(ranks.data());
    config.splitGroupSize = static_cast<int>(ranks.size());
  }

  // Populate NCCL config from user-provided hints
  populateNcclConfigFromHints(config, options, commDesc);

  // Verify the correct CUDA device is set before calling ncclCommSplit.
  // NCCL expects the caller to have set the device matching the communicator.
  {
    int currentDevice = -1;
    CUDA_CHECK(
        cuda_api_,
        cuda_api_->getDevice(&currentDevice),
        "Failed to get current CUDA device in split");
    if (currentDevice != device_.index()) {
      TC_LOG(WARNING, this) << "CUDA device mismatch in split: expected "
                            << device_.index() << " but current device is "
                            << currentDevice << ". Setting to correct device.";
      CUDA_CHECK(
          cuda_api_,
          cuda_api_->setDevice(device_.index()),
          fmt::format(
              "Failed to set CUDA device to {} in split", device_.index()));
    }
  }

  // Note: NCCL documentation states that commSplit should not be called while
  // operations are outstanding on the parent communicator. Callers are
  // responsible for ensuring all operations complete before calling split().
  // Error handling for partial failures (some ranks succeed, others fail) is
  // left to NCCL's internal mechanisms.
  NCCLX_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->commSplit(nccl_comm_, color, new_rank, &new_comm, &config),
      "NCCLX split failed");

  if (new_rank == -1) {
    return nullptr; // Rank is not in any group, return nullptr
  }

  auto new_torchcomm =
      std::shared_ptr<TorchCommNCCLX>(new TorchCommNCCLX(new_comm));
  new_torchcomm->nccl_api_ = nccl_api_;
  new_torchcomm->cuda_api_ = cuda_api_;
  new_torchcomm->init(device_, commDesc, options);

  return new_torchcomm;
}

void TorchCommNCCLX::register_address(
    const TorchCommNCCLX::AddressWithLen& addr) {
  // We got a register after we got rid of the comm. Is this a fatal error?
  if (nccl_comm_ == nullptr) {
    return;
  }

  if (memoryRegistrationHandles_.contains(addr.addr)) {
    throw std::runtime_error("Memory already registered with NCCLX");
  }
  void* handle = nullptr;
  NCCLX_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->commRegister(nccl_comm_, addr.addr, addr.len, &handle),
      "Failed to register memory with NCCLX");
  // ncclCommRegister may return a NULL handle when registration is a no-op.
  // Skip storing it so that deregister_address() does not later call
  // ncclCommDeregister with a NULL handle.
  if (handle != nullptr) {
    memoryRegistrationHandles_.emplace(addr.addr, RegistrationHandle(handle));
  }
}

void TorchCommNCCLX::deregister_address(const TorchCommNCCLX::Address& addr) {
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
  NCCLX_CHECK(
      nccl_api_,
      nccl_comm_,
      nccl_api_->commDeregister(nccl_comm_, handle),
      "Failed to deregister memory with NCCLX");

  memoryRegistrationHandles_.erase(it);
}

namespace {
class NCCLXRegistration {
 public:
  NCCLXRegistration() {
    TorchCommFactory::get().register_backend(
        "ncclx", []() { return std::make_shared<TorchCommNCCLX>(); });

    // Register allocator factory with its own nccl_api instance
    TorchCommFactory::get().register_allocator_factory("ncclx", []() {
      // Create nccl_api for this allocator (captured in lambdas below)
      auto nccl_api = std::make_shared<DefaultNcclxApi>();

      static std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>
          ncclx_allocator =
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
                    LOG(INFO)
                        << "NCCL mem allocator: allocated " << ptr << " with "
                        << size << " bytes in stream " << stream;
                    return ptr;
                  },
                  // free_fn
                  [nccl_api](
                      void* ptr, size_t size, int device, cudaStream_t stream) {
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
      return ncclx_allocator;
    });
  }
};

static const NCCLXRegistration registration{};
} // namespace

} // namespace torch::comms
