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

static void preReduce(at::Tensor& tensor, const ReduceOp& r) {
  if (r.type() == ReduceOp::RedOpType::PREMUL_SUM) {
    std::visit([&tensor](auto&& arg) { tensor.mul_(arg); }, *r.factor());
  }
}

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
  throw std::runtime_error("XCCL send is not supported now and will be added later");
}

c10::intrusive_ptr<TorchWork> TorchCommXCCL::recv(at::Tensor &tensor, int src,
                                               bool async_op,
                                               const RecvOptions &options) {
  throw std::runtime_error("XCCL recv is not supported now and will be added later");
}

// Batch P2P Operations
c10::intrusive_ptr<TorchWork>
TorchCommXCCL::batch_op_issue(const std::vector<BatchSendRecv::P2POp> &ops,
                              bool async_op, const BatchP2POptions &options) {
  throw std::runtime_error("XCCL batch_op_issue is not supported now and will be added later");
}

// Collective Operations
c10::intrusive_ptr<TorchWork>
TorchCommXCCL::broadcast(at::Tensor &tensor, int root, bool async_op,
                         const BroadcastOptions &options) {
  throw std::runtime_error("XCCL broadcast is not supported now and will be added later");
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

  // No-op for empty input tensor
  // TODO: Consider removing this check once oneCCL supports zero-sized tensors
  // for all_reduce operation.
  if (tensor.numel() == 0) [[unlikely]] {
    TC_LOG(WARNING) << "all_reduce called with empty input tensor";
    work->recordEnd();
    enqueueWork(work, stream);
    return work;
  }

  // oneCCL bug skips premul sum if comm_size is 1, so handle it here
  // TODO: remove this workaround when oneCCL bug is fixed
  if (comm_size_ == 1) {
    preReduce(tensor, op);
  }

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
    throw std::runtime_error("XCCL reduce is not supported now and will be added later");
}

c10::intrusive_ptr<TorchWork>
TorchCommXCCL::all_gather(const std::vector<at::Tensor> &tensor_list,
                          const at::Tensor &tensor, bool async_op,
                          const AllGatherOptions &options) {
    throw std::runtime_error("XCCL all_gather is not supported now and will be added later");
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
    throw std::runtime_error("XCCL all_gather_single is not supported now and will be added later");
}

c10::intrusive_ptr<TorchWork> TorchCommXCCL::reduce_scatter(
    at::Tensor &output, const std::vector<at::Tensor> &input_list, const ReduceOp &op,
    bool async_op, const ReduceScatterOptions &options) {
    throw std::runtime_error("XCCL reduce_scatter is not supported now and will be added later");
}

c10::intrusive_ptr<TorchWork> TorchCommXCCL::reduce_scatter_v(
    at::Tensor &output, const std::vector<at::Tensor> &input_list,
    const ReduceOp &op, bool async_op, const ReduceScatterOptions &options) {
  throw std::runtime_error("reduce_scatter_v is not supported in XCCL backend");
}

c10::intrusive_ptr<TorchWork> TorchCommXCCL::reduce_scatter_single(
    at::Tensor &output, const at::Tensor &input, const ReduceOp &op, bool async_op,
    const ReduceScatterSingleOptions &options) {
    throw std::runtime_error("XCCL reduce_scatter_single is not supported now and will be added later");
}

c10::intrusive_ptr<TorchWork>
TorchCommXCCL::all_to_all_single(at::Tensor &output, const at::Tensor &input,
                                 bool async_op,
                                 const AllToAllSingleOptions &options) {
    throw std::runtime_error("XCCL all_to_all_single is not supported now and will be added later");
}

c10::intrusive_ptr<TorchWork> TorchCommXCCL::all_to_all_v_single(
    at::Tensor &output, const at::Tensor &input,
    const std::vector<uint64_t> &output_split_sizes,
    const std::vector<uint64_t> &input_split_sizes, bool async_op,
    const AllToAllvSingleOptions &options) {
    throw std::runtime_error("XCCL all_to_all_v_single is not supported now and will be added later");
}

c10::intrusive_ptr<TorchWork>
TorchCommXCCL::all_to_all(const std::vector<at::Tensor> &output_tensor_list,
                          const std::vector<at::Tensor> &input_tensor_list,
                          bool async_op, const AllToAllOptions &options) {
    throw std::runtime_error("XCCL all_to_all is not supported now and will be added later");
}

c10::intrusive_ptr<TorchWork>
TorchCommXCCL::barrier(bool async_op, const BarrierOptions &options) {
    throw std::runtime_error("XCCL barrier is not supported now and will be added later");
}

c10::intrusive_ptr<TorchWork>
TorchCommXCCL::scatter(at::Tensor &output_tensor,
                       const std::vector<at::Tensor> &input_tensor_list,
                       int root, bool async_op, const ScatterOptions &options) {
    throw std::runtime_error("XCCL scatter is not supported now and will be added later");
}

c10::intrusive_ptr<TorchWork>
TorchCommXCCL::gather(const std::vector<at::Tensor> &output_tensor_list,
                      const at::Tensor &input_tensor, int root, bool async_op,
                      const GatherOptions &options) {
    throw std::runtime_error("XCCL gather is not supported now and will be added later");
}

std::shared_ptr<TorchCommBackend>
TorchCommXCCL::split(const std::vector<int> &ranks, const std::string &name,
                     const CommOptions &options) {
  throw std::runtime_error("XCCL split is not supported now and will be added later");
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
