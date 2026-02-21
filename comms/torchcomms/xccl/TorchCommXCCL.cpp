#include "comms/torchcomms/xccl/TorchCommXCCL.hpp"

#include <ATen/xpu/XPUContext.h>
#include <c10/xpu/XPUStream.h>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include "comms/torchcomms/TorchCommFactory.hpp"
#include "comms/torchcomms/TorchCommLogging.hpp"
#include "comms/torchcomms/xccl/TorchCommXCCLBootstrap.hpp"

namespace torch::comms {

static c10::DeviceType checkAndReturnCommonDeviceType(
    const std::vector<at::Tensor>& tensors,
    const std::vector<at::Tensor>& tensors2 = {}) {
  if (tensors.empty() && tensors2.empty()) [[unlikely]] {
    throw std::runtime_error("No tensors provided for device check");
  }

  // Determine common device type
  c10::DeviceType common_device_type = !tensors.empty()
      ? tensors[0].device().type()
      : tensors2[0].device().type();

  // Check all tensors are on the same device
  for (size_t i = 1; i < tensors.size(); ++i) {
    if (common_device_type != tensors[i].device().type()) [[unlikely]] {
      throw std::runtime_error(
          "All tensors must be on the same device. Found on " +
          c10::DeviceTypeName(common_device_type) + " and " +
          c10::DeviceTypeName(tensors[i].device().type()));
    }
  }

  // Check second set of tensors if provided
  // If `tensors` is empty, `common_device` was found from `tensors2[0]`,
  // so we can start checking from index 1 to avoid a redundant self-check.
  size_t start_idx = tensors.empty() ? 1 : 0;
  for (size_t i = start_idx; i < tensors2.size(); ++i) {
    if (common_device_type != tensors2[i].device().type()) [[unlikely]] {
      throw std::runtime_error(
          "All tensors must be on the same device. Found on " +
          c10::DeviceTypeName(common_device_type) + " and " +
          c10::DeviceTypeName(tensors2[i].device().type()));
    }
  }

  return common_device_type;
}

static void checkAllTensorsOnXPUorCPU(
    const std::vector<at::Tensor>& tensors,
    const std::vector<at::Tensor>& tensors2 = {}) {
  const auto xpu_or_cpu_device =
      checkAndReturnCommonDeviceType(tensors, tensors2);
  if (xpu_or_cpu_device != at::kXPU && xpu_or_cpu_device != at::kCPU)
      [[unlikely]] {
    throw std::runtime_error("All tensors must be on XPU or CPU devices");
  }
}

static ReduceOp applyPremulSumWorkaround(
    at::Tensor& tensor,
    const ReduceOp& r) {
  TORCH_CHECK(r.factor().has_value(), "PREMUL_SUM requires a scaling factor");
  std::visit([&tensor](auto&& arg) { tensor.mul_(arg); }, *r.factor());
  return ReduceOp::RedOpType::SUM;
}

template <typename T>
static std::pair<T, ReduceOp> applyPremulSumWorkaround(
    const T& input,
    const ReduceOp& r) {
  TORCH_CHECK(r.factor().has_value(), "PREMUL_SUM requires a scaling factor");
  auto scale_fn = [&](const at::Tensor& t) {
    return std::visit([&t](auto&& arg) { return t.mul(arg); }, *r.factor());
  };

  if constexpr (std::is_same_v<T, at::Tensor>) {
    return {scale_fn(input), ReduceOp::RedOpType::SUM};
  } else if constexpr (std::is_same_v<T, std::vector<at::Tensor>>) {
    std::vector<at::Tensor> scaled_tensors;
    scaled_tensors.reserve(input.size());
    for (const auto& tensor : input) {
      scaled_tensors.push_back(scale_fn(tensor));
    }
    return {scaled_tensors, ReduceOp::RedOpType::SUM};
  } else {
    throw std::runtime_error("Unsupported type for PREMUL_SUM workaround");
  }
}

onecclResult_t XCCLException::getResult() const {
  return result_;
}

TorchCommXCCL::TorchCommXCCL()
    : xccl_comm_{nullptr},
      device_(at::kXPU),
      init_state_(InitializationState::UNINITIALIZED),
      shutdown_(false) {}

TorchCommXCCL::TorchCommXCCL(const onecclComm_t xccl_comm)
    : xccl_comm_(xccl_comm),
      device_(at::kXPU),
      init_state_(InitializationState::UNINITIALIZED),
      shutdown_(false) {}

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

void TorchCommXCCL::init(
    at::Device device,
    const std::string& name,
    const CommOptions& options) {
  // Initialize private members
  device_ = device;
  name_ = name;
  options_ = options;

  // Only initialize once
  if (init_state_ == InitializationState::INITIALIZED) [[unlikely]] {
    throw std::runtime_error("TorchCommXCCL already initialized");
  } else if (init_state_ == InitializationState::FINALIZED) [[unlikely]] {
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
    TorchCommXCCLBootstrap bootstrap(
        options_.store, device_, xccl_api_, xpu_api_, options_.timeout);
    device_ = bootstrap.getDevice();

    if (xccl_comm_ == nullptr) {
      xccl_comm_ = bootstrap.createXcclComm(name, options);
    }
  }

  // Set XPU device and verify it' accessible
  XPU_CHECK(
      xpu_api_,
      xpu_api_->setDevice(device_.index()),
      "Failed to set XPU device to " + std::to_string(device_.index()));

  // Verify device properties and memory availability
  [[maybe_unused]] xpuDeviceProp device_prop = {};
  XPU_CHECK(
      xpu_api_,
      xpu_api_->getDeviceProperties(&device_prop, device_.index()),
      "Failed to get device properties for device " +
          std::to_string(device_.index()));

  // Check available memory
  [[maybe_unused]] size_t free_memory, total_memory;
  XPU_CHECK(
      xpu_api_,
      xpu_api_->memGetInfo(&free_memory, &total_memory),
      "Failed to get memory info for device " +
          std::to_string(device_.index()));

  // Read hints and store them
  for (auto const& [key, val] : options_.hints) {
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
  XPU_CHECK(
      xpu_api_,
      xpu_api_->streamCreateWithPriority(
          temp_stream, /*flags=*/0, stream_priority),
      "Failed to create internal XPU stream on device " +
          std::to_string(device_.index()));
  internal_stream_ = std::move(temp_stream);

  // Create dependency event for stream synchronization
  xpuEvent_t temp_event(/*enable_timing=*/false);
  XPU_CHECK(
      xpu_api_,
      xpu_api_->eventCreateWithFlags(temp_event, /*flags=*/0),
      "Failed to create dependency event on device " +
          std::to_string(device_.index()));
  dependency_event_ = std::move(temp_event);

  // Allocate XPU buffer for barrier operations
  XPU_CHECK(
      xpu_api_,
      xpu_api_->malloc(&barrier_buffer_, sizeof(float)),
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

  onecclResult_t result;
  result = xccl_api_->commUserRank(xccl_comm_, &rank_);
  if (result != onecclSuccess) [[unlikely]] {
    throw std::runtime_error("XCCL commUserRank failed");
  }

  tryTorchCommLoggingInit("torchcomm");

  xccl_api_->setVersionInfo();

  result = xccl_api_->commCount(xccl_comm_, &comm_size_);
  if (result != onecclSuccess) [[unlikely]] {
    throw std::runtime_error("XCCL commCount failed");
  }

  TorchCommTracingGuard tracingGuard(name_, comm_size_, "init", rank_);

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
      XPU_CHECK(
          xpu_api_, xpu_api_->eventDestroy(event), "Failed to destroy event");
    }
  }

  // Free barrier buffer. TODO: handle errors on xpu free and stream destroy
  if (barrier_buffer_) {
    XPU_CHECK(
        xpu_api_,
        xpu_api_->free(barrier_buffer_),
        "Failed to free barrier buffer");
    barrier_buffer_ = nullptr;
  }

  // Destroy dependency event
  if (dependency_event_.has_value()) {
    XPU_CHECK(
        xpu_api_,
        xpu_api_->eventDestroy(dependency_event_.value()),
        "Failed to destroy dependency event");
    dependency_event_.reset();
  }

  // Destroy internal stream
  if (internal_stream_.has_value()) {
    XPU_CHECK(
        xpu_api_,
        xpu_api_->streamDestroy(internal_stream_.value()),
        "Failed to destroy internal stream");
    internal_stream_.reset();
  }

  // Destroy XCCL communicator
  // TODO: should probably not call this after calling abort.
  if (xccl_comm_) {
    onecclResult_t result = xccl_api_->commDestroy(xccl_comm_);
    if (result != onecclSuccess) [[unlikely]] {
      TC_LOG(ERROR) << "XCCL commDestroy failed: "
                    << xccl_api_->getErrorString(result);
    }
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
  onecclResult_t result = xccl_api_->commUserRank(xccl_comm_, &rank);
  if (result != onecclSuccess) [[unlikely]] {
    throw XCCLException(*xccl_api_, "XCCL commUserRank failed", result);
  }
  return rank;
}

int TorchCommXCCL::getSize() const {
  checkInitialized();

  int comm_size;
  onecclResult_t result = xccl_api_->commCount(xccl_comm_, &comm_size);
  if (result != onecclSuccess) [[unlikely]] {
    throw XCCLException(*xccl_api_, "XCCL commCount failed", result);
  }
  return comm_size;
}

std::string_view TorchCommXCCL::getBackendName() const {
  return kBackendName;
}

std::string_view TorchCommXCCL::getCommName() const {
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
c10::intrusive_ptr<TorchWork> TorchCommXCCL::send(
    const at::Tensor& tensor,
    int dst,
    bool async_op,
    const SendOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);
  checkAllTensorsOnXPUorCPU({tensor});

  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "send", dst, tensor, tensor);

  xpuStream_t stream = getOperationStream(async_op);
  auto work = async_op
      ? createWork(
            stream,
            getOperationTimeout(options.timeout, options_.timeout),
            tensor)
      : createWork(
            stream, getOperationTimeout(options.timeout, options_.timeout));

  work->recordStart("send");

  if (dst == rank_) [[unlikely]] {
    throw XCCLException(
        *xccl_api_,
        "XCCL send called with destination rank equal to current rank " +
            std::to_string(rank_) + "; operation would hang",
        onecclInvalidUsage);
  }

  onecclResult_t result = xccl_api_->send(
      tensor.data_ptr(),
      tensor.numel(),
      getXcclDataType(tensor),
      dst,
      xccl_comm_,
      stream);

  if (result != onecclSuccess) [[unlikely]] {
    throw XCCLException(*xccl_api_, "XCCL send failed", result);
  }

  work->recordEnd();

  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommXCCL::recv(
    at::Tensor& tensor,
    int src,
    bool async_op,
    const RecvOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);
  checkAllTensorsOnXPUorCPU({tensor});

  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "recv", src, tensor, tensor);

  xpuStream_t stream = getOperationStream(async_op);
  auto work = createWork(
      stream, getOperationTimeout(options.timeout, options_.timeout));

  work->recordStart("recv");

  if (src == rank_) [[unlikely]] {
    throw XCCLException(
        *xccl_api_,
        "XCCL recv called with source rank equal to current rank " +
            std::to_string(rank_) + "; operation would hang",
        onecclInvalidUsage);
  }

  onecclResult_t result = xccl_api_->recv(
      tensor.data_ptr(),
      tensor.numel(),
      getXcclDataType(tensor),
      src,
      xccl_comm_,
      stream);

  if (result != onecclSuccess) [[unlikely]] {
    throw XCCLException(*xccl_api_, "XCCL recv failed", result);
  }

  work->recordEnd();

  enqueueWork(work, stream);

  return work;
}

// Batch P2P Operations
c10::intrusive_ptr<TorchWork> TorchCommXCCL::batch_op_issue(
    const std::vector<BatchSendRecv::P2POp>& ops,
    bool async_op,
    const BatchP2POptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();

  if (ops.empty()) [[unlikely]] {
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
      throw std::runtime_error("Unknown op type in batch_op_issue");
    }
  }

  checkAllTensorsOnXPUorCPU(input_tensors, output_tensors);

  TorchCommTracingGuard tracingGuard(
      name_,
      comm_size_,
      "batch_op_issue",
      rank_,
      input_tensors,
      output_tensors);

  xpuStream_t stream = getOperationStream(async_op);
  auto work = createWork(
      stream,
      getOperationTimeout(options.timeout, options_.timeout),
      input_tensors);

  work->recordStart("batch_op_issue");

  onecclResult_t result = xccl_api_->groupStart();
  if (result != onecclSuccess) [[unlikely]] {
    throw XCCLException(
        *xccl_api_, "XCCL groupStart failed in batch_op_issue", result);
  }

  // Issue each operation individually
  for (const auto& op : ops) {
    if (op.type == BatchSendRecv::P2POp::OpType::SEND) {
      result = xccl_api_->send(
          op.tensor.data_ptr(),
          op.tensor.numel(),
          getXcclDataType(op.tensor),
          op.peer,
          xccl_comm_,
          stream);

      if (result != onecclSuccess) [[unlikely]] {
        onecclResult_t result_cleanup =
            xccl_api_->groupEnd(); // Clean up group on error
        if (result_cleanup != onecclSuccess) {
          TC_LOG(ERROR)
              << "XCCL groupEnd failed during error cleanup after send failure in batch_op_issue: "
              << xccl_api_->getErrorString(result_cleanup);
        }
        throw XCCLException(
            *xccl_api_, "XCCL send failed in batch_op_issue", result);
      }
    } else if (op.type == BatchSendRecv::P2POp::OpType::RECV) {
      result = xccl_api_->recv(
          op.tensor.data_ptr(),
          op.tensor.numel(),
          getXcclDataType(op.tensor),
          op.peer,
          xccl_comm_,
          stream);

      if (result != onecclSuccess) [[unlikely]] {
        onecclResult_t result_cleanup =
            xccl_api_->groupEnd(); // Clean up group on error
        if (result_cleanup != onecclSuccess) {
          TC_LOG(ERROR)
              << "XCCL groupEnd failed during error cleanup after recv failure in batch_op_issue: "
              << xccl_api_->getErrorString(result_cleanup);
        }
        throw XCCLException(
            *xccl_api_, "XCCL recv failed in batch_op_issue", result);
      }
    }
  }

  result = xccl_api_->groupEnd();
  if (result != onecclSuccess) [[unlikely]] {
    throw XCCLException(
        *xccl_api_, "XCCL groupEnd failed in batch_op_issue", result);
  }

  work->recordEnd();

  enqueueWork(work, stream);

  return work;
}

// Collective Operations
c10::intrusive_ptr<TorchWork> TorchCommXCCL::broadcast(
    at::Tensor& tensor,
    int root,
    bool async_op,
    const BroadcastOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);
  checkAllTensorsOnXPUorCPU({tensor});
  checkRankRange(root);

  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "broadcast", root, tensor, tensor);

  xpuStream_t stream = getOperationStream(async_op);

  auto work = async_op
      ? createWork(
            stream,
            getOperationTimeout(options.timeout, options_.timeout),
            tensor)
      : createWork(
            stream, getOperationTimeout(options.timeout, options_.timeout));

  work->recordStart("broadcast");

  // No-op for empty tensor
  // TODO: Consider removing this check once oneCCL supports zero-sized tensors
  // in broadcast operation.
  if (tensor.numel() == 0) [[unlikely]] {
    TC_LOG(WARNING) << "XCCL broadcast called with empty tensor on rank "
                    << rank_;
    work->recordEnd();
    enqueueWork(work, stream);
    return work;
  }

  const auto dataType = getXcclDataType(tensor);
  onecclResult_t result = xccl_api_->broadcast(
      tensor.data_ptr(),
      tensor.data_ptr(),
      tensor.numel(),
      dataType,
      root,
      xccl_comm_,
      stream);

  if (result != onecclSuccess) [[unlikely]] {
    throw XCCLException(*xccl_api_, "XCCL broadcast failed", result);
  }
  work->recordEnd();
  enqueueWork(work, stream);
  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommXCCL::all_reduce(
    at::Tensor& tensor,
    const ReduceOp& op,
    bool async_op,
    const AllReduceOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);
  checkAllTensorsOnXPUorCPU({tensor});

  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "all_reduce", rank_, tensor, tensor);

  xpuStream_t stream = getOperationStream(async_op);
  auto work = async_op
      ? createWork(
            stream,
            getOperationTimeout(options.timeout, options_.timeout),
            tensor)
      : createWork(
            stream, getOperationTimeout(options.timeout, options_.timeout));

  work->recordStart("all_reduce");

  // No-op for empty input tensor
  // TODO: Consider removing this check once oneCCL supports zero-sized tensors
  // for all_reduce operation.
  if (tensor.numel() == 0) [[unlikely]] {
    TC_LOG(WARNING) << "XCCL all_reduce called with empty input tensor on rank "
                    << rank_;
    work->recordEnd();
    enqueueWork(work, stream);
    return work;
  }

  // PreMulSum/AVG are not fully supported yet so we convert all PreMulSum/AVG
  // ops to SUM and apply workarounds manually.
  //
  // PREMUL_SUM issue: https://github.com/uxlfoundation/oneCCL/issues/196
  // AVG issue: https://github.com/uxlfoundation/oneCCL/issues/195
  //
  // TODO: remove this workaround when oneCCL fully supports PREMUL_SUM/AVG
  // reductions natively.
  ReduceOp maybe_new_op = (op == ReduceOp::RedOpType::PREMUL_SUM)
      ? applyPremulSumWorkaround(tensor, op)
      : op;

  ReduceOp final_op = (op == ReduceOp::RedOpType::AVG)
      ? ReduceOp(ReduceOp::RedOpType::SUM)
      : maybe_new_op;

  const auto dataType = getXcclDataType(tensor);
  onecclResult_t result = xccl_api_->allReduce(
      tensor.data_ptr(),
      tensor.data_ptr(), // In-place operation
      tensor.numel(),
      dataType,
      getXcclReduceOp(final_op, xccl_comm_, dataType),
      xccl_comm_,
      stream);

  if (result != onecclSuccess) [[unlikely]] {
    throw XCCLException(*xccl_api_, "XCCL allReduce failed", result);
  }

  if (op == ReduceOp::RedOpType::AVG) {
    // For scale-out all_reduce with AVG, oneCCL does not support AVG
    // reduction natively on this path. We therefore perform a SUM across all
    // ranks and then divide the result in-place by comm_size on every rank to
    // obtain the correct average value on all ranks.
    c10::optional<c10::string_view> rounding_mode = c10::nullopt;
    if (c10::isIntegralType(tensor.scalar_type(), /*includeBool*/ false)) {
      // For integer tensors, we use truncation-based division to keep an
      // integer result while matching typical integer division semantics for
      // negative values (round toward zero).
      rounding_mode = "trunc";
    }
    {
      c10::StreamGuard guard(stream);
      tensor.div_(comm_size_, rounding_mode);
    }
  }

  work->recordEnd();

  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommXCCL::reduce(
    const at::Tensor& tensor,
    int root,
    const ReduceOp& op,
    bool async_op,
    const ReduceOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);
  checkAllTensorsOnXPUorCPU({tensor});
  checkRankRange(root);

  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "reduce", root, tensor, tensor);

  xpuStream_t stream = getOperationStream(async_op);

  // PreMulSum/AVG are not fully supported yet so we convert all PreMulSum/AVG
  // ops to SUM and apply workarounds manually.
  //
  // PREMUL_SUM issue: https://github.com/uxlfoundation/oneCCL/issues/196
  // AVG issue: https://github.com/uxlfoundation/oneCCL/issues/195
  //
  // TODO: remove this workaround when oneCCL fully supports PREMUL_SUM/AVG
  // reductions natively.
  const auto [maybe_scaled_tensor, maybe_new_op] =
      (op == ReduceOp::RedOpType::PREMUL_SUM)
      ? applyPremulSumWorkaround(tensor, op)
      : std::make_pair(tensor, op);

  ReduceOp final_op = (op == ReduceOp::RedOpType::AVG)
      ? ReduceOp(ReduceOp::RedOpType::SUM)
      : maybe_new_op;

  auto work = async_op
      ? createWork(
            stream,
            getOperationTimeout(options.timeout, options_.timeout),
            maybe_scaled_tensor)
      : createWork(
            stream, getOperationTimeout(options.timeout, options_.timeout));

  // Record start event before XCCL operation
  work->recordStart("reduce");

  // No-op for empty input tensor
  // TODO: Consider removing this check once oneCCL supports zero-sized tensors
  // in reduce operations.
  if (tensor.numel() == 0) [[unlikely]] {
    TC_LOG(WARNING) << "XCCL reduce called with empty input tensor on rank "
                    << rank_;
    work->recordEnd();
    enqueueWork(work, stream);
    return work;
  }

  const auto dataType = getXcclDataType(tensor);
  onecclResult_t result = xccl_api_->reduce(
      maybe_scaled_tensor.data_ptr(),
      tensor.data_ptr(),
      tensor.numel(),
      dataType,
      getXcclReduceOp(final_op, xccl_comm_, dataType),
      root,
      xccl_comm_,
      stream);

  if (result != onecclSuccess) [[unlikely]] {
    throw XCCLException(*xccl_api_, "XCCL reduce failed", result);
  }

  if (rank_ == root && op == ReduceOp::RedOpType::AVG) {
    // If this is a reduce with AVG, we need to divide the result by
    // comm_size to get the correct average value for root. oneCCL does not
    // support AVG reduction natively, so we have to do this manually after the
    // reduce.
    c10::optional<c10::string_view> rounding_mode = c10::nullopt;
    if (c10::isIntegralType(
            maybe_scaled_tensor.scalar_type(), /*includeBool*/ false)) {
      // For integer tensors, we use truncation-based division to keep an
      // integer result while matching typical integer division semantics for
      // negative values (round toward zero).
      rounding_mode = "trunc";
    }
    {
      c10::StreamGuard guard(stream);
      maybe_scaled_tensor.div_(comm_size_, rounding_mode);
    }
  }

  // Record end event after XCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommXCCL::all_gather(
    const std::vector<at::Tensor>& tensor_list,
    const at::Tensor& tensor,
    bool async_op,
    const AllGatherOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);
  checkAllTensorsOnXPUorCPU({tensor});

  if (tensor_list.size() != static_cast<size_t>(comm_size_)) [[unlikely]] {
    throw std::runtime_error(
        "tensor_list size must equal comm_size for all_gather");
  }

  // Check that all output tensors are contiguous and have correct size
  for (const auto& t : tensor_list) {
    ensureTensorContiguous(t);
    if (t.numel() != tensor.numel()) [[unlikely]] {
      throw std::runtime_error(
          "All tensors in tensor_list must have same size as input tensor");
    }
  }

  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "all_gather", rank_, tensor_list, {tensor});

  xpuStream_t stream = getOperationStream(async_op);

  auto work = async_op
      ? createWork(
            stream,
            getOperationTimeout(options.timeout, options_.timeout),
            tensor)
      : createWork(
            stream, getOperationTimeout(options.timeout, options_.timeout));

  work->recordStart("all_gather");

  // No-op for empty input tensor
  // TODO: Consider removing this check once oneCCL supports zero-sized tensors
  // in broadcast operation.
  if (tensor.numel() == 0) [[unlikely]] {
    TC_LOG(WARNING) << "XCCL all_gather called with empty input tensor on rank "
                    << rank_;
    work->recordEnd();
    enqueueWork(work, stream);
    return work;
  }

  onecclResult_t result = xccl_api_->groupStart();
  if (result != onecclSuccess) [[unlikely]] {
    throw XCCLException(
        *xccl_api_, "XCCL groupStart failed in all_gather", result);
  }

  // Use multiple broadcast operations for all_gather
  for (int i = 0; i < comm_size_; ++i) {
    result = xccl_api_->broadcast(
        tensor.data_ptr(),
        tensor_list[i].data_ptr(),
        tensor.numel(),
        getXcclDataType(tensor_list[i]),
        i,
        xccl_comm_,
        stream);
    if (result != onecclSuccess) [[unlikely]] {
      onecclResult_t result_cleanup =
          xccl_api_->groupEnd(); // clean up group before throwing
      if (result_cleanup != onecclSuccess) {
        TC_LOG(ERROR)
            << "XCCL groupEnd failed during error cleanup after broadcast failure in all_gather: "
            << xccl_api_->getErrorString(result_cleanup);
      }
      throw XCCLException(
          *xccl_api_, "XCCL broadcast failed in all_gather", result);
    }
  }

  result = xccl_api_->groupEnd();
  if (result != onecclSuccess) [[unlikely]] {
    throw XCCLException(
        *xccl_api_, "XCCL groupEnd failed in all_gather", result);
  }

  work->recordEnd();

  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommXCCL::all_gather_v(
    const std::vector<at::Tensor>& tensor_list,
    const at::Tensor& tensor,
    bool async_op,
    const AllGatherOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(tensor);
  checkAllTensorsOnXPUorCPU({tensor});

  if (tensor_list.size() != static_cast<size_t>(comm_size_)) [[unlikely]] {
    throw std::runtime_error(
        "tensor_list size must equal comm_size for all_gather_v");
  }

  for (const auto& t : tensor_list) {
    ensureTensorContiguous(t);
  }

  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "all_gather_v", rank_, tensor_list, {tensor});

  xpuStream_t stream = getOperationStream(async_op);

  auto work = async_op
      ? createWork(
            stream,
            getOperationTimeout(options.timeout, options_.timeout),
            tensor)
      : createWork(
            stream, getOperationTimeout(options.timeout, options_.timeout));

  work->recordStart("all_gather_v");

  // Use multiple broadcast operations for all_gather_v
  onecclResult_t result = xccl_api_->groupStart();
  if (result != onecclSuccess) [[unlikely]] {
    throw XCCLException(
        *xccl_api_, "XCCL groupStart failed in all_gather_v", result);
  }

  for (int i = 0; i < comm_size_; ++i) {
    // assign input/output tensors to support vector all_gather (all_gather_v)
    // where unevenly sized inputs are gathered among participating ranks
    auto& output = tensor_list[i];
    auto& input = (i == rank_) ? tensor : output;
    if (input.numel() != output.numel()) [[unlikely]] {
      throw std::runtime_error(
          "Output tensor size must equal input tensor size for all_gather_v");
    }

    // No-op for empty tensor
    // TODO: Consider removing this check once oneCCL supports zero-sized
    // tensors in broadcast operation.
    if (input.numel() == 0) [[unlikely]] {
      continue;
    }

    result = xccl_api_->broadcast(
        input.data_ptr(),
        output.data_ptr(),
        input.numel(),
        getXcclDataType(output),
        i,
        xccl_comm_,
        stream);

    if (result != onecclSuccess) [[unlikely]] {
      onecclResult_t result_cleanup =
          xccl_api_->groupEnd(); // clean up group before throwing
      if (result_cleanup != onecclSuccess) {
        TC_LOG(ERROR)
            << "XCCL groupEnd failed during error cleanup after broadcast failure in all_gather_v: "
            << xccl_api_->getErrorString(result_cleanup);
      }
      throw XCCLException(
          *xccl_api_, "XCCL broadcast failed in all_gather_v", result);
    }
  }

  result = xccl_api_->groupEnd();
  if (result != onecclSuccess) [[unlikely]] {
    throw XCCLException(
        *xccl_api_, "XCCL groupEnd failed in all_gather_v", result);
  }

  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommXCCL::all_gather_single(
    at::Tensor& output,
    const at::Tensor& input,
    bool async_op,
    const AllGatherSingleOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(output);
  ensureTensorContiguous(input);
  checkAllTensorsOnXPUorCPU({input, output});

  if (output.numel() != input.numel() * comm_size_) [[unlikely]] {
    throw std::runtime_error(
        "Output tensor size must be input_size * comm_size for all_gather_single");
  }

  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "all_gather_single", rank_, input, output);

  xpuStream_t stream = getOperationStream(async_op);

  auto work = async_op
      ? createWork(
            stream,
            getOperationTimeout(options.timeout, options_.timeout),
            input)
      : createWork(
            stream, getOperationTimeout(options.timeout, options_.timeout));

  work->recordStart("all_gather_single");

  // No-op for empty input tensor
  // TODO: Consider removing this check once oneCCL supports zero-sized tensors
  // in all_gather operations.
  if (input.numel() == 0) [[unlikely]] {
    TC_LOG(WARNING)
        << "XCCL all_gather_single called with empty input tensor on rank "
        << rank_;
    work->recordEnd();
    enqueueWork(work, stream);
    return work;
  }

  onecclResult_t result = xccl_api_->allGather(
      input.data_ptr(),
      output.data_ptr(),
      input.numel(),
      getXcclDataType(input),
      xccl_comm_,
      stream);

  if (result != onecclSuccess) [[unlikely]] {
    throw XCCLException(
        *xccl_api_, "XCCL allGather failed in all_gather_single", result);
  }

  work->recordEnd();

  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommXCCL::reduce_scatter(
    at::Tensor& output,
    const std::vector<at::Tensor>& input_list,
    const ReduceOp& op,
    bool async_op,
    const ReduceScatterOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(output);
  checkAllTensorsOnXPUorCPU(input_list, {output});

  if (input_list.size() != static_cast<size_t>(comm_size_)) [[unlikely]] {
    throw std::runtime_error(
        "reduce_scatter: input_list size must equal comm_size");
  }

  for (const auto& t : input_list) {
    ensureTensorContiguous(t);
    if (t.numel() != output.numel()) [[unlikely]] {
      throw std::runtime_error(
          "reduce_scatter: input tensor sizes must match output tensor size");
    }
  }

  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "reduce_scatter", rank_, input_list, {output});

  // PreMulSum/AVG are not fully supported yet so we convert all PreMulSum/AVG
  // ops to SUM and apply workarounds manually.
  //
  // PREMUL_SUM issue: https://github.com/uxlfoundation/oneCCL/issues/196
  // AVG issue: https://github.com/uxlfoundation/oneCCL/issues/195
  //
  // TODO: remove this workaround when oneCCL fully supports PREMUL_SUM/AVG
  // reductions natively.
  const auto [maybe_scaled_input_list, maybe_new_op] =
      (op == ReduceOp::RedOpType::PREMUL_SUM)
      ? applyPremulSumWorkaround(input_list, op)
      : std::make_pair(input_list, op);

  ReduceOp final_op = (op == ReduceOp::RedOpType::AVG)
      ? ReduceOp(ReduceOp::RedOpType::SUM)
      : maybe_new_op;

  xpuStream_t stream = getOperationStream(async_op);

  auto work = async_op
      ? createWork(
            stream,
            getOperationTimeout(options.timeout, options_.timeout),
            maybe_scaled_input_list)
      : createWork(
            stream, getOperationTimeout(options.timeout, options_.timeout));

  work->recordStart("reduce_scatter");

  if (output.numel() == 0) [[unlikely]] {
    work->recordEnd();
    enqueueWork(work, stream);
    return work;
  }

  onecclResult_t result = xccl_api_->groupStart();
  if (result != onecclSuccess) [[unlikely]] {
    throw XCCLException(
        *xccl_api_, "XCCL groupStart failed in reduce_scatter", result);
  }

  for (int peer = 0; peer < comm_size_; ++peer) {
    const auto dtype = getXcclDataType(maybe_scaled_input_list[peer]);
    // We should only need to use output.data_ptr() but we get a hang in oneCCL
    // when output is empty (nullptr). So we use input_tensor.data_ptr() for all
    // non-root ranks.
    void* out_ptr = (peer == rank_) ? output.data_ptr()
                                    : maybe_scaled_input_list[peer].data_ptr();

    result = xccl_api_->reduce(
        maybe_scaled_input_list[peer].data_ptr(),
        out_ptr,
        maybe_scaled_input_list[peer].numel(),
        dtype,
        getXcclReduceOp(final_op, xccl_comm_, dtype),
        peer,
        xccl_comm_,
        stream);

    if (result != onecclSuccess) [[unlikely]] {
      onecclResult_t result_cleanup =
          xccl_api_->groupEnd(); // clean up group before throwing
      if (result_cleanup != onecclSuccess) {
        TC_LOG(ERROR)
            << "XCCL groupEnd failed during error cleanup after reduce failure: "
            << xccl_api_->getErrorString(result_cleanup);
      }
      throw XCCLException(
          *xccl_api_, "XCCL reduce failed in reduce_scatter", result);
    }
  }

  result = xccl_api_->groupEnd();
  if (result != onecclSuccess) [[unlikely]] {
    throw XCCLException(
        *xccl_api_, "XCCL groupEnd failed in reduce_scatter", result);
  }

  if (op == ReduceOp::RedOpType::AVG) {
    // If this is a reduce_scatter with AVG, we need to divide the result by
    // comm_size to get the correct average value. oneCCL does not support
    // AVG reduction natively, so we have to do this manually after the reduce.
    c10::optional<c10::string_view> rounding_mode = c10::nullopt;
    if (c10::isIntegralType(output.scalar_type(), /*includeBool*/ false)) {
      // For integer tensors, we use truncation-based division to keep an
      // integer result while matching typical integer division semantics for
      // negative values (round toward zero).
      rounding_mode = "trunc";
    }
    {
      c10::StreamGuard guard(stream);
      output.div_(comm_size_, rounding_mode);
    }
  }

  work->recordEnd();
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommXCCL::reduce_scatter_v(
    at::Tensor& output,
    const std::vector<at::Tensor>& input_list,
    const ReduceOp& op,
    bool async_op,
    const ReduceScatterOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(output);
  checkAllTensorsOnXPUorCPU(input_list, {output});

  if (input_list.size() != static_cast<size_t>(comm_size_)) [[unlikely]] {
    throw std::runtime_error(
        "input_list size must equal comm_size for reduce_scatter_v");
  }

  // Check that all input tensors are contiguous and have correct size
  for (int i = 0; i < comm_size_; ++i) {
    auto& t = input_list[i];
    ensureTensorContiguous(t);
    if (i == rank_ && t.numel() != output.numel()) [[unlikely]] {
      throw std::runtime_error(
          "Output tensor size must equal input tensor size at rank position for reduce_scatter_v");
    }
  }

  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "reduce_scatter_v", rank_, input_list, {output});

  // PreMulSum/AVG are not fully supported yet so we convert all PreMulSum/AVG
  // ops to SUM and apply workarounds manually.
  //
  // PREMUL_SUM issue: https://github.com/uxlfoundation/oneCCL/issues/196
  // AVG issue: https://github.com/uxlfoundation/oneCCL/issues/195
  //
  // TODO: remove this workaround when oneCCL fully supports PREMUL_SUM/AVG
  // reductions natively.
  const auto [maybe_scaled_input_list, maybe_new_op] =
      (op == ReduceOp::RedOpType::PREMUL_SUM)
      ? applyPremulSumWorkaround(input_list, op)
      : std::make_pair(input_list, op);

  ReduceOp final_op = (op == ReduceOp::RedOpType::AVG)
      ? ReduceOp(ReduceOp::RedOpType::SUM)
      : maybe_new_op;

  xpuStream_t stream = getOperationStream(async_op);

  auto work = async_op
      ? createWork(
            stream,
            getOperationTimeout(options.timeout, options_.timeout),
            maybe_scaled_input_list)
      : createWork(
            stream, getOperationTimeout(options.timeout, options_.timeout));

  work->recordStart("reduce_scatter_v");

  // Use multiple reduce operations for reduce_scatter
  onecclResult_t result = xccl_api_->groupStart();
  if (result != onecclSuccess) [[unlikely]] {
    throw XCCLException(
        *xccl_api_, "XCCL groupStart failed in reduce_scatter_v", result);
  }

  for (int i = 0; i < comm_size_; ++i) {
    auto& input_tensor = maybe_scaled_input_list[i];
    if (input_tensor.numel() == 0) [[unlikely]] {
      TC_LOG(WARNING) << "XCCL skipping empty input tensor for rank " << i
                      << " in reduce_scatter_v";
      continue; // skip empty tensors
    }
    const auto dataType = getXcclDataType(input_tensor);
    // We should only need to use output.data_ptr() but we get a hang in oneCCL
    // when output is empty (nullptr). So we use input_tensor.data_ptr() for all
    // non-root ranks.
    auto out_ptr = (i == rank_) ? output.data_ptr() : input_tensor.data_ptr();
    result = xccl_api_->reduce(
        input_tensor.data_ptr(),
        out_ptr,
        input_tensor.numel(),
        dataType,
        getXcclReduceOp(final_op, xccl_comm_, dataType),
        i,
        xccl_comm_,
        stream);
    if (result != onecclSuccess) [[unlikely]] {
      onecclResult_t result_cleanup =
          xccl_api_->groupEnd(); // clean up group before throwing
      if (result_cleanup != onecclSuccess) {
        TC_LOG(ERROR)
            << "XCCL groupEnd failed during error cleanup after reduce failure in reduce_scatter_v: "
            << xccl_api_->getErrorString(result_cleanup);
      }
      throw XCCLException(
          *xccl_api_, "XCCL reduce failed in reduce_scatter_v", result);
    }
  }

  result = xccl_api_->groupEnd();
  if (result != onecclSuccess) [[unlikely]] {
    throw XCCLException(
        *xccl_api_, "XCCL groupEnd failed in reduce_scatter_v", result);
  }

  if (op == ReduceOp::RedOpType::AVG) {
    // If this is a reduce_scatter with AVG, we need to divide the result by
    // comm_size to get the correct average value. oneCCL does not support
    // AVG reduction natively, so we have to do this manually after the reduce.
    c10::optional<c10::string_view> rounding_mode = c10::nullopt;
    if (c10::isIntegralType(output.scalar_type(), /*includeBool*/ false)) {
      // For integer tensors, we use truncation-based division to keep an
      // integer result while matching typical integer division semantics for
      // negative values (round toward zero).
      rounding_mode = "trunc";
    }
    {
      c10::StreamGuard guard(stream);
      output.div_(comm_size_, rounding_mode);
    }
  }

  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommXCCL::reduce_scatter_single(
    at::Tensor& output,
    const at::Tensor& input,
    const ReduceOp& op,
    bool async_op,
    const ReduceScatterSingleOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(output);
  ensureTensorContiguous(input);
  checkAllTensorsOnXPUorCPU({input, output});

  if (input.numel() != output.numel() * comm_size_) [[unlikely]] {
    throw std::runtime_error(
        "Input tensor size must be output_size * comm_size for reduce_scatter_single");
  }

  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "reduce_scatter_single", rank_, input, output);

  // PreMulSum/AVG are not fully supported yet so we convert all PreMulSum/AVG
  // ops to SUM and apply workarounds manually.
  //
  // PREMUL_SUM issue: https://github.com/uxlfoundation/oneCCL/issues/196
  // AVG issue: https://github.com/uxlfoundation/oneCCL/issues/195
  //
  // TODO: remove this workaround when oneCCL fully supports PREMUL_SUM/AVG
  // reductions natively.
  const auto [maybe_scaled_input, maybe_new_op] =
      (op == ReduceOp::RedOpType::PREMUL_SUM)
      ? applyPremulSumWorkaround(input, op)
      : std::make_pair(input, op);

  ReduceOp final_op = (op == ReduceOp::RedOpType::AVG)
      ? ReduceOp(ReduceOp::RedOpType::SUM)
      : maybe_new_op;

  xpuStream_t stream = getOperationStream(async_op);

  auto work = async_op
      ? createWork(
            stream,
            getOperationTimeout(options.timeout, options_.timeout),
            maybe_scaled_input)
      : createWork(
            stream, getOperationTimeout(options.timeout, options_.timeout));

  // Record start event before XCCL operation
  work->recordStart("reduce_scatter_single");

  // No-op for empty input tensor
  // TODO: Consider removing this check once oneCCL supports zero-sized tensors
  // in reduce operations.
  if (input.numel() == 0) [[unlikely]] {
    TC_LOG(WARNING)
        << "XCCL reduce_scatter_single called with empty input tensor on rank "
        << rank_;
    work->recordEnd();
    enqueueWork(work, stream);
    return work;
  }

  const auto dataType = getXcclDataType(maybe_scaled_input);
  onecclResult_t result = xccl_api_->reduceScatter(
      maybe_scaled_input.data_ptr(),
      output.data_ptr(),
      output.numel(),
      dataType,
      getXcclReduceOp(final_op, xccl_comm_, dataType),
      xccl_comm_,
      stream);

  if (result != onecclSuccess) [[unlikely]] {
    throw XCCLException(
        *xccl_api_,
        "XCCL reduceScatter failed in reduce_scatter_single",
        result);
  }

  if (op == ReduceOp::RedOpType::AVG) {
    // If this is a reduce_scatter with AVG, we need to divide the result by
    // comm_size to get the correct average value. oneCCL does not support
    // AVG reduction natively, so we have to do this manually after the reduce.
    c10::optional<c10::string_view> rounding_mode = c10::nullopt;
    if (c10::isIntegralType(output.scalar_type(), /*includeBool*/ false)) {
      // For integer tensors, we use truncation-based division to keep an
      // integer result while matching typical integer division semantics for
      // negative values (round toward zero).
      rounding_mode = "trunc";
    }
    {
      c10::StreamGuard guard(stream);
      output.div_(comm_size_, rounding_mode);
    }
  }

  // Record end event after XCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommXCCL::all_to_all_single(
    at::Tensor& output,
    const at::Tensor& input,
    bool async_op,
    const AllToAllSingleOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(output);
  ensureTensorContiguous(input);
  checkAllTensorsOnXPUorCPU({input, output});

  if (input.numel() != output.numel()) [[unlikely]] {
    throw std::runtime_error(
        "Input and output tensors must have same size for all_to_all_single");
  }

  if (input.numel() % comm_size_ != 0) [[unlikely]] {
    throw std::runtime_error(
        "Tensor size must be divisible by comm_size for all_to_all_single");
  }

  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "all_to_all_single", rank_, input, output);

  xpuStream_t stream = getOperationStream(async_op);

  auto work = async_op
      ? createWork(
            stream,
            getOperationTimeout(options.timeout, options_.timeout),
            input)
      : createWork(
            stream, getOperationTimeout(options.timeout, options_.timeout));

  // Record start event before XCCL operation
  work->recordStart("all_to_all_single");

  // No-op for empty input tensor
  // TODO: Consider removing this check once oneCCL supports zero-sized tensors
  // in allToAll operations.
  if (input.numel() == 0) [[unlikely]] {
    work->recordEnd();
    enqueueWork(work, stream);
    return work;
  }

  size_t chunk_size = input.numel() / comm_size_;
  const auto data_type = getXcclDataType(input);

  onecclResult_t result = xccl_api_->allToAll(
      input.data_ptr(),
      output.data_ptr(),
      chunk_size,
      data_type,
      xccl_comm_,
      stream);

  if (result != onecclSuccess) [[unlikely]] {
    throw XCCLException(
        *xccl_api_, "XCCL allToAll failed in all_to_all_single", result);
  }

  // Record end event after XCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommXCCL::all_to_all_v_single(
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
  checkAllTensorsOnXPUorCPU({input, output});

  // Validate split sizes vectors
  if (input_split_sizes.size() != static_cast<size_t>(comm_size_))
      [[unlikely]] {
    throw std::runtime_error(
        "input_split_sizes length must equal comm_size for all_to_all_v_single");
  }

  if (output_split_sizes.size() != static_cast<size_t>(comm_size_))
      [[unlikely]] {
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

  if (input_total > static_cast<uint64_t>(input.size(0))) [[unlikely]] {
    throw std::runtime_error(
        "Sum of input_split_sizes exceeds input tensor size for all_to_all_v_single");
  }

  if (output_total > static_cast<uint64_t>(output.size(0))) [[unlikely]] {
    throw std::runtime_error(
        "Sum of output_split_sizes exceeds output tensor size for all_to_all_v_single");
  }

  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "all_to_all_v_single", rank_, input, output);

  xpuStream_t stream = getOperationStream(async_op);
  auto work = async_op
      ? createWork(
            stream,
            getOperationTimeout(options.timeout, options_.timeout),
            input)
      : createWork(
            stream, getOperationTimeout(options.timeout, options_.timeout));

  // Record start event before XCCL operation
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
  const auto data_type = getXcclDataType(input);
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

  onecclResult_t result = xccl_api_->groupStart();
  if (result != onecclSuccess) [[unlikely]] {
    throw XCCLException(
        *xccl_api_, "XCCL groupStart failed in all_to_all_v_single", result);
  }

  for (int i = 0; i < comm_size_; ++i) {
    if (sendcounts[i] == 0 && recvcounts[i] == 0) [[unlikely]] {
      // No-op for empty send/recv
      continue;
    }
    if (sendcounts[i] > 0) {
      if (i == rank_) {
        // For own rank, use memcpyAsync
        XPU_CHECK(
            xpu_api_,
            xpu_api_->memcpyAsync(
                rptr + recvdispls[i] * type_size,
                sptr + senddispls[i] * type_size,
                sendcounts[i] * type_size,
                stream),
            "memcpyAsync failed in all_to_all_v_single");
        continue;
      } else {
        // Send to rank i
        result = xccl_api_->send(
            sptr + senddispls[i] * type_size,
            sendcounts[i],
            data_type,
            i,
            xccl_comm_,
            stream);
        if (result != onecclSuccess) [[unlikely]] {
          onecclResult_t result_cleanup =
              xccl_api_->groupEnd(); // clean up group before throwing
          if (result_cleanup != onecclSuccess) {
            TC_LOG(ERROR)
                << "XCCL groupEnd failed during error cleanup after send failure in all_to_all_v_single: "
                << xccl_api_->getErrorString(result_cleanup);
          }
          throw XCCLException(
              *xccl_api_, "XCCL send failed in all_to_all_v_single", result);
        }
      }
    }

    if (recvcounts[i] > 0) {
      result = xccl_api_->recv(
          rptr + recvdispls[i] * type_size,
          recvcounts[i],
          data_type,
          i,
          xccl_comm_,
          stream);
      if (result != onecclSuccess) [[unlikely]] {
        onecclResult_t result_cleanup =
            xccl_api_->groupEnd(); // clean up group before throwing
        if (result_cleanup != onecclSuccess) {
          TC_LOG(ERROR)
              << "XCCL groupEnd failed during error cleanup after recv failure in all_to_all_v_single: "
              << xccl_api_->getErrorString(result_cleanup);
        }
        throw XCCLException(
            *xccl_api_, "XCCL recv failed in all_to_all_v_single", result);
      }
    }
  }

  result = xccl_api_->groupEnd();
  if (result != onecclSuccess) [[unlikely]] {
    throw XCCLException(
        *xccl_api_, "XCCL groupEnd failed in all_to_all_v_single", result);
  }

  // Record end event after XCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommXCCL::all_to_all(
    const std::vector<at::Tensor>& output_tensor_list,
    const std::vector<at::Tensor>& input_tensor_list,
    bool async_op,
    const AllToAllOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  checkAllTensorsOnXPUorCPU(input_tensor_list, output_tensor_list);

  if (output_tensor_list.size() != static_cast<size_t>(comm_size_) ||
      input_tensor_list.size() != static_cast<size_t>(comm_size_))
      [[unlikely]] {
    throw std::runtime_error(
        "Tensor list sizes must equal comm_size for all_to_all");
  }

  // Validate all tensors
  for (int i = 0; i < comm_size_; ++i) {
    if (input_tensor_list[i].numel() != output_tensor_list[i].numel())
        [[unlikely]] {
      throw std::runtime_error(
          "Input and output tensor sizes must match for all_to_all");
    }
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

  xpuStream_t stream = getOperationStream(async_op);
  auto work = async_op
      ? createWork(
            stream,
            getOperationTimeout(options.timeout, options_.timeout),
            input_tensor_list)
      : createWork(
            stream, getOperationTimeout(options.timeout, options_.timeout));

  // Record start event before XCCL operations
  work->recordStart("all_to_all");

  onecclResult_t result = xccl_api_->groupStart();
  if (result != onecclSuccess) [[unlikely]] {
    throw XCCLException(
        *xccl_api_, "XCCL groupStart failed in all_to_all", result);
  }

  for (int i = 0; i < comm_size_; ++i) {
    if (input_tensor_list[i].numel() == 0) [[unlikely]] {
      // No-op for empty tensor
      continue;
    }
    if (i == rank_) {
      // For own rank, use memcpyAsync
      XPU_CHECK(
          xpu_api_,
          xpu_api_->memcpyAsync(
              output_tensor_list[i].data_ptr(),
              input_tensor_list[i].data_ptr(),
              input_tensor_list[i].numel() *
                  input_tensor_list[i].element_size(),
              stream),
          "memcpyAsync failed in all_to_all");
      continue;
    } else {
      // Send to rank i
      result = xccl_api_->send(
          input_tensor_list[i].data_ptr(),
          input_tensor_list[i].numel(),
          getXcclDataType(input_tensor_list[i]),
          i,
          xccl_comm_,
          stream);
      if (result != onecclSuccess) [[unlikely]] {
        onecclResult_t result_cleanup =
            xccl_api_->groupEnd(); // clean up group before throwing
        if (result_cleanup != onecclSuccess) {
          TC_LOG(ERROR)
              << "XCCL groupEnd failed during error cleanup after send failure in all_to_all: "
              << xccl_api_->getErrorString(result_cleanup);
        }
        throw XCCLException(
            *xccl_api_, "XCCL send failed in all_to_all", result);
      }
    }

    // Receive from rank i
    result = xccl_api_->recv(
        output_tensor_list[i].data_ptr(),
        output_tensor_list[i].numel(),
        getXcclDataType(output_tensor_list[i]),
        i,
        xccl_comm_,
        stream);
    if (result != onecclSuccess) [[unlikely]] {
      onecclResult_t result_cleanup =
          xccl_api_->groupEnd(); // clean up group before throwing
      if (result_cleanup != onecclSuccess) {
        TC_LOG(ERROR)
            << "XCCL groupEnd failed during error cleanup after recv failure in all_to_all: "
            << xccl_api_->getErrorString(result_cleanup);
      }
      throw XCCLException(*xccl_api_, "XCCL recv failed in all_to_all", result);
    }
  }

  result = xccl_api_->groupEnd();
  if (result != onecclSuccess) [[unlikely]] {
    throw XCCLException(
        *xccl_api_, "XCCL groupEnd failed in all_to_all", result);
  }

  // Record end event after XCCL operations
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommXCCL::barrier(
    bool async_op,
    const BarrierOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();

  TorchCommTracingGuard tracingGuard(name_, comm_size_, "barrier", rank_);
  xpuStream_t stream = getOperationStream(async_op);
  auto work = createWork(
      stream, getOperationTimeout(options.timeout, options_.timeout));

  // Record start event before XCCL operation
  work->recordStart("barrier");

  // Use pre-allocated XPU buffer for barrier
  onecclResult_t result = xccl_api_->allReduce(
      barrier_buffer_,
      barrier_buffer_,
      1,
      onecclFloat32,
      onecclSum,
      xccl_comm_,
      stream);

  if (result != onecclSuccess) [[unlikely]] {
    throw XCCLException(*xccl_api_, "XCCL barrier failed", result);
  }

  // Record end event after XCCL operation
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommXCCL::scatter(
    at::Tensor& output_tensor,
    const std::vector<at::Tensor>& input_tensor_list,
    int root,
    bool async_op,
    const ScatterOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(output_tensor);
  checkAllTensorsOnXPUorCPU(input_tensor_list, {output_tensor});
  checkRankRange(root);

  // Only the root rank needs valid tensors
  if (rank_ == root) {
    if (input_tensor_list.size() != static_cast<size_t>(comm_size_))
        [[unlikely]] {
      throw std::runtime_error(
          "input_tensor_list must equal comm_size for scatter");
    }

    for (const auto& t : input_tensor_list) {
      ensureTensorContiguous(t);
      if (t.numel() != output_tensor.numel()) [[unlikely]] {
        throw std::runtime_error(
            "All input tensors must have same size as output tensor");
      }
    }
  }

  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "scatter", root, input_tensor_list, {output_tensor});

  xpuStream_t stream = getOperationStream(async_op);
  std::vector<at::Tensor> input_tensors;
  if (async_op && rank_ == root) {
    input_tensors = input_tensor_list;
  }
  auto work = createWork(
      stream,
      getOperationTimeout(options.timeout, options_.timeout),
      input_tensors);

  // Record start event before XCCL operations
  work->recordStart("scatter");

  // No-op for empty input tensor
  // TODO: Remove this check once oneCCL supports zero-sized tensors
  // in scatter operations.
  if (output_tensor.numel() == 0) [[unlikely]] {
    TC_LOG(WARNING) << "XCCL scatter called with empty tensor on rank "
                    << rank_;
    work->recordEnd();
    enqueueWork(work, stream);
    return work;
  }

  // Unlike the NCCL implementation which groups only the send operations,
  // we group both send and receive operations to avoid a hang in oneCCL.
  // See https://github.com/uxlfoundation/oneCCL/issues/193 for details.

  // Implement Scatter using point-to-point operations
  onecclResult_t result = xccl_api_->groupStart();
  if (result != onecclSuccess) [[unlikely]] {
    throw XCCLException(
        *xccl_api_, "XCCL groupStart failed in scatter", result);
  }

  if (rank_ == root) {
    // Root sends to all ranks (except itself)
    for (int i = 0; i < comm_size_; ++i) {
      if (i != root) {
        result = xccl_api_->send(
            input_tensor_list[i].data_ptr(),
            input_tensor_list[i].numel(),
            getXcclDataType(input_tensor_list[i]),
            i,
            xccl_comm_,
            stream);
        if (result != onecclSuccess) [[unlikely]] {
          onecclResult_t result_cleanup =
              xccl_api_->groupEnd(); // Clean up group on error
          if (result_cleanup != onecclSuccess) {
            TC_LOG(ERROR)
                << "XCCL groupEnd failed during error cleanup after send failure in scatter: "
                << xccl_api_->getErrorString(result_cleanup);
          }
          throw XCCLException(
              *xccl_api_, "XCCL send failed in scatter", result);
        }
      }
    }

    // Root copies its own data using memcpyAsync
    XPU_CHECK(
        xpu_api_,
        xpu_api_->memcpyAsync(
            output_tensor.data_ptr(),
            input_tensor_list[root].data_ptr(),
            input_tensor_list[root].numel() *
                input_tensor_list[root].element_size(),
            stream),
        "memcpyAsync failed in scatter");
  } else {
    // Non-root ranks receive from root
    result = xccl_api_->recv(
        output_tensor.data_ptr(),
        output_tensor.numel(),
        getXcclDataType(output_tensor),
        root,
        xccl_comm_,
        stream);
    if (result != onecclSuccess) [[unlikely]] {
      onecclResult_t result_cleanup =
          xccl_api_->groupEnd(); // Clean up group on error
      if (result_cleanup != onecclSuccess) {
        TC_LOG(ERROR)
            << "XCCL groupEnd failed during error cleanup after recv failure in scatter: "
            << xccl_api_->getErrorString(result_cleanup);
      }
      throw XCCLException(*xccl_api_, "XCCL recv failed in scatter", result);
    }
  }
  result = xccl_api_->groupEnd();
  if (result != onecclSuccess) [[unlikely]] {
    throw XCCLException(*xccl_api_, "XCCL groupEnd failed in scatter", result);
  }

  // Record end event after XCCL operations
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommXCCL::gather(
    const std::vector<at::Tensor>& output_tensor_list,
    const at::Tensor& input_tensor,
    int root,
    bool async_op,
    const GatherOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  ensureTensorContiguous(input_tensor);
  checkAllTensorsOnXPUorCPU({input_tensor}, output_tensor_list);
  checkRankRange(root);

  // Only the root rank needs valid output tensors
  if (rank_ == root) {
    if (output_tensor_list.size() != static_cast<size_t>(comm_size_))
        [[unlikely]] {
      throw std::runtime_error(
          "output_tensor_list size (" +
          std::to_string(output_tensor_list.size()) +
          ") must equal comm_size (" + std::to_string(comm_size_) +
          ") for gather operation");
    }

    for (size_t i = 0; i < output_tensor_list.size(); ++i) {
      const auto& t = output_tensor_list[i];
      ensureTensorContiguous(t);
      if (t.numel() != input_tensor.numel()) [[unlikely]] {
        throw std::runtime_error(
            "Output tensor at index " + std::to_string(i) + " has size " +
            std::to_string(t.numel()) + " but expected " +
            std::to_string(input_tensor.numel()) + " to match input tensor");
      }
    }
  }

  TorchCommTracingGuard tracingGuard(
      name_, comm_size_, "gather", root, {input_tensor}, output_tensor_list);

  xpuStream_t stream = getOperationStream(async_op);

  auto work = async_op
      ? createWork(
            stream,
            getOperationTimeout(options.timeout, options_.timeout),
            input_tensor)
      : createWork(
            stream, getOperationTimeout(options.timeout, options_.timeout));

  // Record start event before XCCL operations
  work->recordStart("gather");

  // No-op for empty input tensor
  // TODO: Consider removing this check once oneCCL supports zero-sized tensors
  // in send/recv operations.
  if (input_tensor.numel() == 0) [[unlikely]] {
    TC_LOG(WARNING) << "XCCL gather called with empty input tensor on rank "
                    << rank_;
    work->recordEnd();
    enqueueWork(work, stream);
    return work;
  }

  // Unlike the NCCL implementation which groups only the send operations,
  // we group both send and receive operations to avoid a hang in oneCCL.
  // See https://github.com/uxlfoundation/oneCCL/issues/193 for details.
  onecclResult_t result = xccl_api_->groupStart();
  if (result != onecclSuccess) [[unlikely]] {
    throw XCCLException(*xccl_api_, "XCCL groupStart failed in gather", result);
  }

  if (rank_ == root) {
    // Root receives from all ranks (except itself)
    for (int peer_rank = 0; peer_rank < comm_size_; ++peer_rank) {
      if (peer_rank != root) {
        auto& peer_tensor = output_tensor_list[peer_rank];
        onecclResult_t result = xccl_api_->recv(
            peer_tensor.data_ptr(),
            peer_tensor.numel(),
            getXcclDataType(peer_tensor),
            peer_rank,
            xccl_comm_,
            stream);
        if (result != onecclSuccess) [[unlikely]] {
          onecclResult_t result_cleanup =
              xccl_api_->groupEnd(); // Clean up group on error
          if (result_cleanup != onecclSuccess) {
            TC_LOG(ERROR)
                << "XCCL groupEnd failed during error cleanup after recv failure in gather: "
                << xccl_api_->getErrorString(result_cleanup);
          }
          throw XCCLException(*xccl_api_, "XCCL recv failed in gather", result);
        }
      }
    }

    // Root copies its own data using memcpyAsync
    XPU_CHECK(
        xpu_api_,
        xpu_api_->memcpyAsync(
            output_tensor_list[root].data_ptr(),
            input_tensor.data_ptr(),
            input_tensor.numel() * input_tensor.element_size(),
            stream),
        "memcpyAsync failed in gather");
  } else {
    // Non-root ranks send to root
    onecclResult_t result = xccl_api_->send(
        input_tensor.data_ptr(),
        input_tensor.numel(),
        getXcclDataType(input_tensor),
        root,
        xccl_comm_,
        stream);
    if (result != onecclSuccess) [[unlikely]] {
      onecclResult_t result_cleanup =
          xccl_api_->groupEnd(); // Clean up group on error
      if (result_cleanup != onecclSuccess) {
        TC_LOG(ERROR)
            << "XCCL groupEnd failed during error cleanup after send failure in gather: "
            << xccl_api_->getErrorString(result_cleanup);
      }
      throw XCCLException(*xccl_api_, "XCCL send failed in gather", result);
    }
  }
  result = xccl_api_->groupEnd();
  if (result != onecclSuccess) [[unlikely]] {
    throw XCCLException(*xccl_api_, "XCCL groupEnd failed in gather", result);
  }

  // Record end event after XCCL operations
  work->recordEnd();

  // Enqueue the work after events have been recorded
  enqueueWork(work, stream);

  return work;
}

// Remove [[maybe_unused]] when split is implemented
std::shared_ptr<TorchCommBackend> TorchCommXCCL::split(
    [[maybe_unused]] const std::vector<int>& ranks,
    [[maybe_unused]] const std::string& name,
    [[maybe_unused]] const CommOptions& options) {
  throw std::runtime_error(
      "XCCL split is not supported now and will be added later");
}

XCCLException::XCCLException(
    XcclApi& xccl_api,
    const std::string& message,
    onecclResult_t result)
    : message_(message + ": " + xccl_api.getErrorString(result)),
      result_(result) {}

const char* XCCLException::what() const noexcept {
  return message_.c_str();
}

} // namespace torch::comms

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
