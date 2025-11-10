#include "comms/torchcomms/xccl/TorchCommXCCL.hpp"

#include "comms/torchcomms/TorchCommFactory.hpp"
#include "comms/torchcomms/TorchCommLogging.hpp"
#include "comms/torchcomms/xccl/TorchCommXCCLBootstrap.hpp"
#include <ATen/xpu/XPUContext.h>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace torch {
namespace comms {

onecclResult_t XCCLException::getResult() const { return result_; }

TorchCommXCCL::TorchCommXCCL()
    : xccl_comm_{nullptr}, device_(at::kXPU),
      init_state_(InitializationState::UNINITIALIZED), shutdown_(false) {}

TorchCommXCCL::TorchCommXCCL(const onecclComm_t xccl_comm)
    : xccl_comm_(xccl_comm), device_(at::kXPU),
      init_state_(InitializationState::UNINITIALIZED), shutdown_(false) {}

TorchCommXCCL::~TorchCommXCCL() {
  if (init_state_ == InitializationState::INITIALIZED) {
    TC_LOG(ERROR) << "TorchCommXCCL was not finalized before destruction";

    // If finalize was not called, we need to clean up the timeout thread
    if (timeout_thread_.joinable()) {
      shutdown_.store(true);
      timeout_thread_.join();
    }
  }
}

void TorchCommXCCL::init(at::Device device, const std::string &name,
                         const CommOptions &options) {
  // Initialize private members
  device_ = device;
  name_ = name;
  options_ = options;

  // Only initialize once
  if (init_state_ == InitializationState::INITIALIZED) {
    throw std::runtime_error("TorchCommXCCL already initialized");
  } else if (init_state_ == InitializationState::FINALIZED) {
    throw std::runtime_error("TorchCommXCCL already finalized");
  }
  init_state_ = InitializationState::INITIALIZED;

  // Initialize default XCCL API implementation if not already set
  if (!xccl_api_) {
    xccl_api_ = std::make_unique<DefaultXcclApi>();
  }

  // Initialize default XPU API implementation if not already set
  if (!xpu_api_) {
    xpu_api_ = std::make_unique<DefaultXpuApi>();
  }

  if (device_.index() == -1 || xccl_comm_ == nullptr) {
    auto bootstrap = new TorchCommXCCLBootstrap(
        options_.store, device_, xccl_api_, xpu_api_, options_.timeout);
    device_ = bootstrap->getDevice();

    if (xccl_comm_ == nullptr) {
      xccl_comm_ = bootstrap->createXcclComm(name, options);
    }

    delete bootstrap;
  }

  // Set XPU device and verify it' accessible
  XPU_CHECK(xpu_api_, xpu_api_->setDevice(device_.index()),
            "Failed to set XPU device to " + std::to_string(device_.index()));

  // Read hints and store them
  for (auto const &[key, val] : options_.hints) {
    if (key.starts_with("torchcomm::xccl::")) {
      if (key == "torchcomm::xccl::high_priority_stream") {
        high_priority_stream_ = string_to_bool(val);
      } else {
        throw std::runtime_error("Unrecognized hint " + key);
      }
    } else {
      // Ignore keys that do not start with "torchcomm::xccl::"
    }
  }

  // Create internal stream
  int stream_priority = 0;

  // Check for high priority stream hint
  if (high_priority_stream_) {
    stream_priority = -1;
  }

  // Initialize internal stream
  xpuStream_t temp_stream = xpu_api_->getCurrentXPUStream(device_.index());
  XPU_CHECK(xpu_api_,
            xpu_api_->streamCreateWithPriority(temp_stream, /*flags=*/0,
                                               stream_priority),
            "Failed to create internal XPU stream on device " +
                std::to_string(device_.index()));
  internal_stream_ = std::move(temp_stream);

  // Create dependency event for stream synchronization
  xpuEvent_t temp_event(/*enable_timing=*/false);
  XPU_CHECK(xpu_api_, xpu_api_->eventCreateWithFlags(temp_event, /*flags=*/0),
            "Failed to create dependency event on device " +
                std::to_string(device_.index()));
  dependency_event_ = std::move(temp_event);

  // Allocate XPU buffer for barrier operations
  XPU_CHECK(xpu_api_, xpu_api_->malloc(&barrier_buffer_, sizeof(float)),
            "Failed to allocate barrier buffer");

  if (options_.hints.contains("torchcomm::xccl::max_event_pool_size")) {
    max_event_pool_size_ =
        std::stoull(options_.hints.at("torchcomm::xccl::max_event_pool_size"));
  } else {
    max_event_pool_size_ = kMaxEventPoolSize;
  }

  // Give up our internal reference to the store object here.  The caller
  // would still need to keep a reference to the store object till the init
  // call returns, at which point the XCCL communicator would already be
  // created.
  if (options_.store) {
    options_.store.reset();
  }

  onecclResult_t xcclErr;
  xcclErr = xccl_api_->commUserRank(xccl_comm_, &rank_);
  if (xcclErr != onecclSuccess) {
    throw std::runtime_error("XCCL User Rank failed");
  }

  tryTorchCommLoggingInit("torchcomm");

  xcclErr = xccl_api_->commCount(xccl_comm_, &comm_size_);
  if (xcclErr != onecclSuccess) {
    throw std::runtime_error("XCCL Count failed");
  }

  tracing_ = std::make_shared<TorchCommTracing>(name, comm_size_, rank_);
  tracing_->recordEvent("init");

  // Start timeout watchdog thread
  timeout_thread_ = std::thread(&TorchCommXCCL::timeoutWatchdog, this);
}

void TorchCommXCCL::finalize() {
  if (init_state_ == InitializationState::UNINITIALIZED) {
    throw std::runtime_error("TorchCommXCCL not initialized");
  } else if (init_state_ == InitializationState::FINALIZED) {
    throw std::runtime_error("TorchCommXCCL already finalized");
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

  if (work_status == TorchWorkXCCL::WorkStatus::NOT_STARTED ||
      work_status == TorchWorkXCCL::WorkStatus::INPROGRESS) {
    throw std::runtime_error(
        "WorkQ finalize returned in progress or not started state");
  }

  // Update comm_state_ based on the work status
  if (work_status == TorchWorkXCCL::WorkStatus::TIMEDOUT) {
    comm_state_ = CommState::TIMEOUT;
    abortXcclComm();
    throw std::runtime_error("Work timed out during finalize");
  } else if (work_status == TorchWorkXCCL::WorkStatus::ERROR) {
    comm_state_ = CommState::ERROR;
    onecclResult_t asyncErr;
    xccl_api_->commGetAsyncError(xccl_comm_, &asyncErr);
    XCCLException xcclException(*xccl_api_, "XCCL Async Error", asyncErr);
    abortXcclComm();
    throw xcclException;
  }

  // Clean up event pool
  {
    std::lock_guard<std::mutex> lock(event_pool_mutex_);
    while (!event_pool_.empty()) {
      xpuEvent_t event = std::move(event_pool_.front());
      event_pool_.pop();
      XPU_CHECK(xpu_api_, xpu_api_->eventDestroy(event),
                "Failed to destroy event");
    }
  }

  // Free barrier buffer. TODO: handle errors on xpu free and stream destroy
  if (barrier_buffer_) {
    XPU_CHECK(xpu_api_, xpu_api_->free(barrier_buffer_),
              "Failed to free barrier buffer");
    barrier_buffer_ = nullptr;
  }

  // Destroy dependency event
  if (dependency_event_.has_value()) {
    XPU_CHECK(xpu_api_, xpu_api_->eventDestroy(dependency_event_.value()),
              "Failed to destroy dependency event");
    dependency_event_.reset();
  }

  // Destroy internal stream
  if (internal_stream_.has_value()) {
    XPU_CHECK(xpu_api_, xpu_api_->streamDestroy(internal_stream_.value()),
              "Failed to destroy internal stream");
    internal_stream_.reset();
  }

  // Destroy XCCL communicator
  // TODO: should probably not call this after calling abort.
  if (xccl_comm_) {
    xccl_api_->commDestroy(xccl_comm_);
    xccl_comm_ = nullptr;
  }
}

void TorchCommXCCL::abortXcclComm() {
  if (xccl_comm_) {
    xccl_api_->commAbort(xccl_comm_);
    xccl_comm_ = nullptr;
  }
  if (options_.abort_process_on_timeout_or_error) {
    TC_LOG(ERROR) << "Aborting process due to timeout";
    abort();
  }
}

int TorchCommXCCL::getRank() const {
  checkInitialized();

  int rank;
  onecclResult_t xcclErr = xccl_api_->commUserRank(xccl_comm_, &rank);
  if (xcclErr != onecclSuccess) {
    throw XCCLException(*xccl_api_, "XCCL User Rank failed", xcclErr);
  }
  return rank;
}

int TorchCommXCCL::getSize() const {
  checkInitialized();

  int comm_size;
  onecclResult_t xcclErr = xccl_api_->commCount(xccl_comm_, &comm_size);
  if (xcclErr != onecclSuccess) {
    throw XCCLException(*xccl_api_, "XCCL Count failed", xcclErr);
  }
  return comm_size;
}

std::string_view TorchCommXCCL::getBackendName() const { return kBackendName; }

std::string_view TorchCommXCCL::getCommName() const { return name_; }

static inline std::chrono::milliseconds
getOperationTimeout(std::chrono::milliseconds timeout,
                    std::chrono::milliseconds default_timeout) {
  // If timeout is kNoTimeout (0ms), use the default timeout from options
  if (timeout == kNoTimeout) {
    return default_timeout;
  }
  return timeout;
}

// Point-to-Point Operations
c10::intrusive_ptr<TorchWork> TorchCommXCCL::send(const at::Tensor &tensor,
                                               int dst, bool async_op,
                                               const SendOptions &options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);

  tracing_->recordEventWithInputOutput("send", dst, {tensor}, {tensor});

  xpuStream_t stream = getOperationStream(async_op);
  auto work = createWork(
      stream, getOperationTimeout(options.timeout, options_.timeout), {tensor});

  work->recordStart();

  onecclResult_t result =
      xccl_api_->send(tensor.data_ptr(), tensor.numel(),
                      getXcclDataType(tensor), dst, xccl_comm_, stream);

  if (result != onecclSuccess) {
    throw XCCLException(*xccl_api_, "XCCL Send failed", result);
  }

  work->recordEnd();

  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommXCCL::recv(at::Tensor &tensor, int src,
                                               bool async_op,
                                               const RecvOptions &options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);

  tracing_->recordEventWithInputOutput("recv", src, {tensor}, {tensor});

  xpuStream_t stream = getOperationStream(async_op);
  auto work = createWork(
      stream, getOperationTimeout(options.timeout, options_.timeout), {});

  work->recordStart();

  onecclResult_t result =
      xccl_api_->recv(tensor.data_ptr(), tensor.numel(),
                      getXcclDataType(tensor), src, xccl_comm_, stream);

  if (result != onecclSuccess) {
    throw XCCLException(*xccl_api_, "XCCL Recv failed", result);
  }

  work->recordEnd();

  enqueueWork(work, stream);

  return work;
}

// Batch P2P Operations
c10::intrusive_ptr<TorchWork>
TorchCommXCCL::batch_op_issue(const std::vector<BatchSendRecv::P2POp> &ops,
                              bool async_op, const BatchP2POptions &options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();

  if (ops.empty()) {
    throw std::runtime_error("Cannot issue empty batch operation");
  }

  // Collect input and output tensors for work tracking
  std::vector<at::Tensor> input_tensors;
  std::vector<at::Tensor> output_tensors;

  for (const auto &op : ops) {
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

  tracing_->recordEventWithInputOutput("batch_op_issue", rank_, input_tensors,
                                       output_tensors);

  xpuStream_t stream = getOperationStream(async_op);
  auto work =
      createWork(stream, getOperationTimeout(options.timeout, options_.timeout),
                 input_tensors);

  work->recordStart();

  // Start XCCL group for batched operations
  onecclResult_t result = xccl_api_->groupStart();
  if (result != onecclSuccess) {
    throw XCCLException(*xccl_api_, "XCCL GroupStart failed", result);
  }

  // Issue each operation individually
  for (const auto &op : ops) {
    if (op.type == BatchSendRecv::P2POp::OpType::SEND) {
      result = xccl_api_->send(op.tensor.data_ptr(), op.tensor.numel(),
                               getXcclDataType(op.tensor), op.peer, xccl_comm_,
                               stream);

      if (result != onecclSuccess) {
        xccl_api_->groupEnd(); // Clean up group on error
        throw XCCLException(*xccl_api_, "XCCL Send failed in batch operation",
                            result);
      }
    } else if (op.type == BatchSendRecv::P2POp::OpType::RECV) {
      result = xccl_api_->recv(op.tensor.data_ptr(), op.tensor.numel(),
                               getXcclDataType(op.tensor), op.peer, xccl_comm_,
                               stream);

      if (result != onecclSuccess) {
        xccl_api_->groupEnd(); // Clean up group on error
        throw XCCLException(*xccl_api_, "XCCL Recv failed in batch operation",
                            result);
      }
    }
  }

  result = xccl_api_->groupEnd();
  if (result != onecclSuccess) {
    throw XCCLException(*xccl_api_, "XCCL GroupEnd failed", result);
  }

  work->recordEnd();

  enqueueWork(work, stream);

  return work;
}

// Collective Operations
c10::intrusive_ptr<TorchWork>
TorchCommXCCL::broadcast(at::Tensor &tensor, int root, bool async_op,
                         const BroadcastOptions &options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);

  tracing_->recordEventWithInputOutput("broadcast", root, {tensor}, {tensor});

  xpuStream_t stream = getOperationStream(async_op);

  auto work = createWork(
      stream, getOperationTimeout(options.timeout, options_.timeout), {tensor});

  work->recordStart();

  onecclResult_t result =
      xccl_api_->bcast(tensor.data_ptr(), tensor.numel(),
                       getXcclDataType(tensor), root, xccl_comm_, stream);

  if (result != onecclSuccess) {
    throw XCCLException(*xccl_api_, "XCCL Broadcast failed", result);
  }

  work->recordEnd();

  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork>
TorchCommXCCL::all_reduce(at::Tensor &tensor, const ReduceOp &op, bool async_op,
                          const AllReduceOptions &options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);

  tracing_->recordEventWithInputOutput("all_reduce", rank_, {tensor}, {tensor});

  xpuStream_t stream = getOperationStream(async_op);
  auto work = createWork(
      stream, getOperationTimeout(options.timeout, options_.timeout), {tensor});

  work->recordStart();

  const auto dataType = getXcclDataType(tensor);
  onecclResult_t result = xccl_api_->allReduce(
      tensor.data_ptr(),
      tensor.data_ptr(), // In-place operation
      tensor.numel(), dataType, getXcclReduceOp(op, xccl_comm_, dataType),
      xccl_comm_, stream);

  if (result != onecclSuccess) {
    throw XCCLException(*xccl_api_, "XCCL AllReduce failed", result);
  }

  work->recordEnd();

  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommXCCL::reduce(const at::Tensor &tensor,
                                                 int root, const ReduceOp &op,
                                                 bool async_op,
                                                 const ReduceOptions &options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);

  tracing_->recordEventWithInputOutput("reduce", root, {tensor}, {tensor});

  xpuStream_t stream = getOperationStream(async_op);
  std::vector<at::Tensor> output_tensors;
  if (rank_ == root) {
    output_tensors.push_back(tensor);
  }
  auto work = createWork(
      stream, getOperationTimeout(options.timeout, options_.timeout), {tensor});

  work->recordStart();

  const auto dataType = getXcclDataType(tensor);
  onecclResult_t result = xccl_api_->reduce(
      tensor.data_ptr(),
      tensor.data_ptr(), // Use same buffer for all ranks
      tensor.numel(), dataType, getXcclReduceOp(op, xccl_comm_, dataType), root,
      xccl_comm_, stream);

  if (result != onecclSuccess) {
    throw XCCLException(*xccl_api_, "XCCL Reduce failed", result);
  }

  work->recordEnd();

  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork>
TorchCommXCCL::all_gather(const std::vector<at::Tensor> &tensor_list,
                          const at::Tensor &tensor, bool async_op,
                          const AllGatherOptions &options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  if (tensor_list.size() != static_cast<size_t>(comm_size_)) {
    throw std::runtime_error(
        "tensor_list size must equal comm_size for all_gather");
  }

  ensureTensorContiguous(tensor);

  for (const auto &t : tensor_list) {
    ensureTensorContiguous(t);
    if (t.numel() != tensor.numel()) {
      throw std::runtime_error(
          "All tensors in tensor_list must have same size as input tensor");
    }
  }

  tracing_->recordEventWithInputOutput("all_gather", rank_, tensor_list,
                                       {tensor});

  xpuStream_t stream = getOperationStream(async_op);
  auto work = createWork(
      stream, getOperationTimeout(options.timeout, options_.timeout), {tensor});

  work->recordStart();

  xccl_api_->groupStart();

  for (int i = 0; i < comm_size_; ++i) {
    xccl_api_->broadcast(tensor.data_ptr(), tensor_list[i].data_ptr(),
                         tensor.numel(), getXcclDataType(tensor_list[i]), i,
                         xccl_comm_, stream);
  }

  xccl_api_->groupEnd();

  work->recordEnd();

  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork>
TorchCommXCCL::all_gather_v(const std::vector<at::Tensor> &tensor_list,
                            const at::Tensor &tensor, bool async_op,
                            const AllGatherOptions &options) {
  throw std::runtime_error("all_gather_v is not supported in XCCL backend");
}

c10::intrusive_ptr<TorchWork>
TorchCommXCCL::all_gather_single(at::Tensor &output, const at::Tensor &input,
                                 bool async_op,
                                 const AllGatherSingleOptions &options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(output);
  ensureTensorContiguous(input);

  if (output.numel() != input.numel() * comm_size_) {
    throw std::runtime_error("Output tensor size must be input_size * "
                             "comm_size for all_gather_single");
  }

  tracing_->recordEventWithInputOutput("all_gather_single", rank_, {input},
                                       {output});

  xpuStream_t stream = getOperationStream(async_op);
  auto work = createWork(
      stream, getOperationTimeout(options.timeout, options_.timeout), {input});

  work->recordStart();

  onecclResult_t result =
      xccl_api_->allGather(input.data_ptr(), output.data_ptr(), input.numel(),
                           getXcclDataType(input), xccl_comm_, stream);

  if (result != onecclSuccess) {
    throw XCCLException(*xccl_api_, "XCCL AllGather failed", result);
  }

  work->recordEnd();

  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommXCCL::reduce_scatter(
    at::Tensor &output, const std::vector<at::Tensor> &input_list, const ReduceOp &op,
    bool async_op, const ReduceScatterOptions &options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(output);

  if (input_list.size() != static_cast<size_t>(comm_size_)) {
    throw std::runtime_error(
        "input_list size must equal comm_size for reduce_scatter");
  }

  // Check that all input tensors are contiguous and have correct size
  for (const auto &t : input_list) {
    ensureTensorContiguous(t);
    if (t.numel() != output.numel()) {
      throw std::runtime_error(
          "All input tensors must have same size as output tensor");
    }
  }

  tracing_->recordEventWithInputOutput("reduce_scatter", rank_, input_list,
                                       {output});

  xpuStream_t stream = getOperationStream(async_op);
  auto work =
      createWork(stream, getOperationTimeout(options.timeout, options_.timeout),
                 input_list);

  work->recordStart();

  // Use multiple reduce operations for reduce_scatter
  xccl_api_->groupStart();

  for (int i = 0; i < comm_size_; ++i) {
    const auto dataType = getXcclDataType(input_list[i]);
    xccl_api_->reduce(input_list[i].data_ptr(),
                      i == rank_ ? output.data_ptr() : input_list[i].data_ptr(),
                      i == rank_ ? output.numel() : input_list[i].numel(),
                      dataType, getXcclReduceOp(op, xccl_comm_, dataType), i,
                      xccl_comm_, stream);
  }

  xccl_api_->groupEnd();

  work->recordEnd();

  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommXCCL::reduce_scatter_v(
    at::Tensor &output, const std::vector<at::Tensor> &input_list,
    const ReduceOp &op, bool async_op, const ReduceScatterOptions &options) {
  throw std::runtime_error("reduce_scatter_v is not supported in XCCL backend");
}

c10::intrusive_ptr<TorchWork> TorchCommXCCL::reduce_scatter_single(
    at::Tensor &output, const at::Tensor &input, const ReduceOp &op, bool async_op,
    const ReduceScatterSingleOptions &options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(output);
  ensureTensorContiguous(input);

  if (input.numel() != output.numel() * comm_size_) {
    throw std::runtime_error("Input tensor size must be output_size * "
                             "comm_size for reduce_scatter_single");
  }

  tracing_->recordEventWithInputOutput("reduce_scatter_single", rank_, {input},
                                       {output});

  xpuStream_t stream = getOperationStream(async_op);
  auto work = createWork(
      stream, getOperationTimeout(options.timeout, options_.timeout), {input});

  work->recordStart();

  const auto dataType = getXcclDataType(input);
  onecclResult_t result = xccl_api_->reduceScatter(
      input.data_ptr(), output.data_ptr(), output.numel(), dataType,
      getXcclReduceOp(op, xccl_comm_, dataType), xccl_comm_, stream);

  if (result != onecclSuccess) {
    throw XCCLException(*xccl_api_, "XCCL ReduceScatter failed", result);
  }

  work->recordEnd();

  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork>
TorchCommXCCL::all_to_all_single(at::Tensor &output, const at::Tensor &input,
                                 bool async_op,
                                 const AllToAllSingleOptions &options) {
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

  tracing_->recordEventWithInputOutput("all_to_all_single", rank_, {input},
                                       {output});

  xpuStream_t stream = getOperationStream(async_op);
  auto work = createWork(
      stream, getOperationTimeout(options.timeout, options_.timeout), {input});

  work->recordStart();

  size_t chunk_size = input.numel() / comm_size_;
  const auto data_type = getXcclDataType(input);

  onecclResult_t result =
      xccl_api_->allToAll(input.data_ptr(), output.data_ptr(), chunk_size,
                          data_type, xccl_comm_, stream);
  if (result != onecclSuccess) {
    throw XCCLException(*xccl_api_, "XCCL AllToAll failed", result);
  }

  work->recordEnd();

  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommXCCL::all_to_all_v_single(
    at::Tensor &output, const at::Tensor &input,
    const std::vector<uint64_t> &output_split_sizes,
    const std::vector<uint64_t> &input_split_sizes, bool async_op,
    const AllToAllvSingleOptions &options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(output);
  ensureTensorContiguous(input);

  // Validate split sizes vectors
  if (input_split_sizes.size() != static_cast<size_t>(comm_size_)) {
    throw std::runtime_error("input_split_sizes length must equal comm_size "
                             "for all_to_all_v_single");
  }

  if (output_split_sizes.size() != static_cast<size_t>(comm_size_)) {
    throw std::runtime_error("output_split_sizes length must equal comm_size "
                             "for all_to_all_v_single");
  }

  tracing_->recordEventWithInputOutput("all_to_all_v_single", rank_, {input},
                                       {output});

  xpuStream_t stream = getOperationStream(async_op);
  auto work = createWork(
      stream, getOperationTimeout(options.timeout, options_.timeout), {input});

  work->recordStart();

  // Convert split sizes to arrays and calculate displacements
  std::vector<size_t> sendcounts(comm_size_);
  std::vector<size_t> recvcounts(comm_size_);
  std::vector<size_t> senddispls(comm_size_);
  std::vector<size_t> recvdispls(comm_size_);

  // Calculate the number of elements per slice along the first dimension
  // For a tensor with shape [N, D1, D2, ..., Dk], each slice of size S along
  // dim 0 contains S * D1 * D2 * ... * Dk elements
  size_t elements_per_slice = input.numel() ? input.numel() / input.size(0) : 0;
  const auto data_type = getXcclDataType(input);
  const size_t type_size = wordSize(data_type);

  size_t sendoffset = 0;
  size_t recvoffset = 0;
  for (int i = 0; i < comm_size_; ++i) {
    sendcounts[i] = input_split_sizes[i] * elements_per_slice;
    recvcounts[i] = output_split_sizes[i] * elements_per_slice;
    senddispls[i] = sendoffset;
    recvdispls[i] = recvoffset;
    sendoffset += sendcounts[i];
    recvoffset += recvcounts[i];
  }

  char *sptr = static_cast<char *>(input.data_ptr());
  char *rptr = static_cast<char *>(output.data_ptr());

  xccl_api_->groupStart();

  for (int i = 0; i < comm_size_; ++i) {
    xccl_api_->send(sptr + senddispls[i] * type_size, sendcounts[i], data_type,
                    i, xccl_comm_, stream);
    xccl_api_->recv(rptr + recvdispls[i] * type_size, recvcounts[i], data_type,
                    i, xccl_comm_, stream);
  }

  xccl_api_->groupEnd();

  work->recordEnd();

  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork>
TorchCommXCCL::all_to_all(const std::vector<at::Tensor> &output_tensor_list,
                          const std::vector<at::Tensor> &input_tensor_list,
                          bool async_op, const AllToAllOptions &options) {
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

  tracing_->recordEventWithInputOutput("all_to_all", rank_, input_tensor_list,
                                       output_tensor_list);

  xpuStream_t stream = getOperationStream(async_op);
  auto work =
      createWork(stream, getOperationTimeout(options.timeout, options_.timeout),
                 input_tensor_list);

  work->recordStart();

  xccl_api_->groupStart();

  for (int i = 0; i < comm_size_; ++i) {
    // Send to rank i
    xccl_api_->send(
        input_tensor_list[i].data_ptr(), input_tensor_list[i].numel(),
        getXcclDataType(input_tensor_list[i]), i, xccl_comm_, stream);

    // Receive from rank i
    xccl_api_->recv(
        output_tensor_list[i].data_ptr(), output_tensor_list[i].numel(),
        getXcclDataType(output_tensor_list[i]), i, xccl_comm_, stream);
  }

  xccl_api_->groupEnd();

  work->recordEnd();

  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork>
TorchCommXCCL::barrier(bool async_op, const BarrierOptions &options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();

  tracing_->recordEvent("barrier");
  xpuStream_t stream = getOperationStream(async_op);
  auto work = createWork(
      stream, getOperationTimeout(options.timeout, options_.timeout), {});

  work->recordStart();

  // Use pre-allocated XPU buffer for barrier
  onecclResult_t result =
      xccl_api_->allReduce(barrier_buffer_, barrier_buffer_, 1, onecclFloat32,
                           onecclSum, xccl_comm_, stream);

  if (result != onecclSuccess) {
    throw XCCLException(*xccl_api_, "XCCL Barrier failed", result);
  }

  work->recordEnd();

  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork>
TorchCommXCCL::scatter(at::Tensor &output_tensor,
                       const std::vector<at::Tensor> &input_tensor_list,
                       int root, bool async_op, const ScatterOptions &options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(output_tensor);

  // Only the root rank needs valid input tensors
  if (rank_ == root) {
    if (input_tensor_list.size() != static_cast<size_t>(comm_size_)) {
      throw std::runtime_error(
          "input_tensor_list size must equal comm_size for scatter");
    }

    for (const auto &t : input_tensor_list) {
      ensureTensorContiguous(t);
      if (t.numel() != output_tensor.numel()) {
        throw std::runtime_error(
            "All input tensors must have same size as output tensor");
      }
    }
  }

  tracing_->recordEventWithInputOutput("scatter", root, input_tensor_list,
                                       {output_tensor});

  xpuStream_t stream = getOperationStream(async_op);
  std::vector<at::Tensor> input_tensors;
  if (rank_ == root) {
    input_tensors = input_tensor_list;
  }
  auto work =
      createWork(stream, getOperationTimeout(options.timeout, options_.timeout),
                 input_tensors);

  work->recordStart();

  // Implement scatter using point-to-point operations
  if (rank_ == root) {
    // Root sends to all ranks (except itself)
    xccl_api_->groupStart();
    for (int i = 0; i < comm_size_; ++i) {
      if (i != root) {
        xccl_api_->send(
            input_tensor_list[i].data_ptr(), input_tensor_list[i].numel(),
            getXcclDataType(input_tensor_list[i]), i, xccl_comm_, stream);
      }
    }
    xccl_api_->groupEnd();

    // Root copies its own data using xpuMemcpyAsync
    XPU_CHECK(xpu_api_,
              xpu_api_->memcpyAsync(output_tensor.data_ptr(),
                                    input_tensor_list[root].data_ptr(),
                                    input_tensor_list[root].numel() *
                                        input_tensor_list[root].element_size(),
                                    stream),
              "memcpyAsync failed");
  } else {
    // Non-root ranks receive from root
    xccl_api_->recv(output_tensor.data_ptr(), output_tensor.numel(),
                    getXcclDataType(output_tensor), root, xccl_comm_, stream);
  }

  work->recordEnd();

  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork>
TorchCommXCCL::gather(const std::vector<at::Tensor> &output_tensor_list,
                      const at::Tensor &input_tensor, int root, bool async_op,
                      const GatherOptions &options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(input_tensor);

  // Only the root rank needs valid output tensors
  if (rank_ == root) {
    if (output_tensor_list.size() != static_cast<size_t>(comm_size_)) {
      throw std::runtime_error(
          "output_tensor_list size must equal comm_size for gather");
    }

    for (const auto &t : output_tensor_list) {
      ensureTensorContiguous(t);
      if (t.numel() != input_tensor.numel()) {
        throw std::runtime_error(
            "All output tensors must have same size as input tensor");
      }
    }
  }

  tracing_->recordEventWithInputOutput("gather", root, {input_tensor},
                                       output_tensor_list);

  xpuStream_t stream = getOperationStream(async_op);
  std::vector<at::Tensor> output_tensors;
  if (rank_ == root) {
    output_tensors = output_tensor_list;
  }
  auto work =
      createWork(stream, getOperationTimeout(options.timeout, options_.timeout),
                 {input_tensor});

  work->recordStart();

  if (rank_ == root) {
    // Root receives from all ranks (except itself)
    xccl_api_->groupStart();
    for (int i = 0; i < comm_size_; ++i) {
      if (i != root) {
        xccl_api_->recv(
            output_tensor_list[i].data_ptr(), output_tensor_list[i].numel(),
            getXcclDataType(output_tensor_list[i]), i, xccl_comm_, stream);
      }
    }
    xccl_api_->groupEnd();

    // Root copies its own data using xpuMemcpyAsync
    XPU_CHECK(xpu_api_,
              xpu_api_->memcpyAsync(
                  output_tensor_list[root].data_ptr(), input_tensor.data_ptr(),
                  input_tensor.numel() * input_tensor.element_size(), stream),
              "memcpyAsync failed");
  } else {
    // Non-root ranks send to root
    xccl_api_->send(input_tensor.data_ptr(), input_tensor.numel(),
                    getXcclDataType(input_tensor), root, xccl_comm_, stream);
  }

  work->recordEnd();

  enqueueWork(work, stream);

  return work;
}

std::shared_ptr<TorchCommBackend>
TorchCommXCCL::split(const std::vector<int> &ranks, const std::string &name,
                     const CommOptions &options) {
  throw std::runtime_error("Split is not supported now in XCCL");
}

XCCLException::XCCLException(XcclApi &xccl_api, const std::string &message,
                             onecclResult_t result)
    : message_(message + ": " + xccl_api.getErrorString(result)),
      result_(result) {}

const char *XCCLException::what() const noexcept { return message_.c_str(); }

} // namespace comms
} // namespace torch

namespace {
class XCCLRegistration {
public:
  XCCLRegistration() {
    torch::comms::TorchCommFactory::get().register_backend("xccl", []() {
      return std::make_shared<torch::comms::TorchCommXCCL>();
    });
  }
};

static XCCLRegistration registration{};
} // namespace
