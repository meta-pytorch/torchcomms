#include "comms/torchcomms/hccl/TorchCommHCCL.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include "comms/torchcomms/TorchCommFactory.hpp"
#include "comms/torchcomms/TorchCommLogging.hpp"
#include "comms/torchcomms/hccl/TorchCommHCCLBootstrap.hpp"

namespace torch::comms {

HcclResult HCCLException::getResult() const {
  return result_;
}

static void preReduce(at::Tensor& tensor, const ReduceOp& r) {
  if (r.type() == ReduceOp::RedOpType::PREMUL_SUM) {
    std::visit([&tensor](auto&& arg) { tensor.mul_(arg); }, *r.factor());
  }
}

TorchCommHCCL::TorchCommHCCL()
    : hccl_comm_{nullptr},
      device_(at::kPrivateUse1),
      init_state_(InitializationState::UNINITIALIZED),
      shutdown_(false) {}

TorchCommHCCL::TorchCommHCCL(const HcclComm hccl_comm)
    : hccl_comm_(hccl_comm),
      device_(at::kPrivateUse1),
      init_state_(InitializationState::UNINITIALIZED),
      shutdown_(false) {}

TorchCommHCCL::~TorchCommHCCL() {
  if (init_state_ == InitializationState::INITIALIZED) {
    TC_LOG(ERROR) << "TorchCommHCCL was not finalized before destruction";

    // If finalize was not called, we need to clean up the timeout thread
    if (timeout_thread_.joinable()) {
      shutdown_.store(true);
      timeout_thread_.join();
    }
  }
}

void TorchCommHCCL::init(
    at::Device device,
    const std::string& name,
    const CommOptions& options) {
  // Initialize private members
  device_ = device;
  name_ = name;
  options_ = options;

  // Only initialize once
  if (init_state_ == InitializationState::INITIALIZED) {
    throw std::runtime_error("TorchCommHCCL already initialized");
  } else if (init_state_ == InitializationState::FINALIZED) {
    throw std::runtime_error("TorchCommHCCL already finalized");
  }
  init_state_ = InitializationState::INITIALIZED;

  // Initialize default HCCL API implementation if not already set
  if (!hccl_api_) {
    hccl_api_ = std::make_shared<DefaultHcclApi>();
  }

  // Initialize default NPU API implementation if not already set
  if (!npu_api_) {
    npu_api_ = std::make_shared<DefaultNpuApi>();
  }

  if (device_.index() == -1 || hccl_comm_ == nullptr) {
    auto bootstrap = new TorchCommHCCLBootstrap(
        options_.store, device_, hccl_api_, npu_api_, options_.timeout);
    device_ = bootstrap->getDevice();

    if (hccl_comm_ == nullptr) {
      hccl_comm_ = bootstrap->createHcclComm(name, options);
    }

    delete bootstrap;
  }

  // Set NPU device and verify it's accessible
  NPU_CHECK(
      npu_api_,
      npu_api_->setDevice(device_.index()),
      "Failed to set NPU device to " + std::to_string(device_.index()));

  // Verify device properties and memory availability
  [[maybe_unused]] npuDeviceProp device_prop = {};
  NPU_CHECK(
      npu_api_,
      npu_api_->getDeviceProperties(&device_prop, device_.index()),
      "Failed to get device properties for device " +
          std::to_string(device_.index()));

  // Check available memory
  [[maybe_unused]] size_t free_memory, total_memory;
  NPU_CHECK(
      npu_api_,
      npu_api_->memGetInfo(&free_memory, &total_memory),
      "Failed to get memory info for device " +
          std::to_string(device_.index()));

  // Read hints and store them
  for (auto const& [key, val] : options_.hints) {
    if (key.starts_with("torchcomm::hccl::")) {
      if (key == "torchcomm::hccl::high_priority_stream") {
        high_priority_stream_ = string_to_bool(val);
      } else {
        throw std::runtime_error("Unrecognized hint " + key);
      }
    } else {
      // Ignore keys that do not start with "torchcomm::hccl::"
    }
  }

  // Create internal stream
  int stream_priority = 0;

  // Check for high priority stream hint
  if (high_priority_stream_) {
    stream_priority = -1;
  }

  // Initialize internal stream
  npuStream_t temp_stream = npu_api_->getCurrentNPUStream(device_.index());
  NPU_CHECK(
      npu_api_,
      npu_api_->streamCreateWithPriority(
          temp_stream, /*flags=*/0, stream_priority),
      "Failed to create internal NPU stream on device " +
          std::to_string(device_.index()));
  internal_stream_ = std::move(temp_stream);

  // Create dependency event for stream synchronization
  npuEvent_t temp_event;
  NPU_CHECK(
      npu_api_,
      npu_api_->eventCreateWithFlags(temp_event, /*flags=*/0),
      "Failed to create dependency event on device " +
          std::to_string(device_.index()));
  dependency_event_ = std::move(temp_event);

  // Allocate NPU buffer for barrier operations
  NPU_CHECK(
      npu_api_,
      npu_api_->malloc(&barrier_buffer_, sizeof(float)),
      "Failed to allocate barrier buffer");

  if (options_.hints.contains("torchcomm::hccl::max_event_pool_size")) {
    max_event_pool_size_ =
        std::stoull(options_.hints.at("torchcomm::hccl::max_event_pool_size"));
  } else {
    max_event_pool_size_ = kMaxEventPoolSize;
  }

  // Give up our internal reference to the store object here.  The caller
  // would still need to keep a reference to the store object till the init
  // call returns, at which point the HCCL communicator would already be
  // created.
  if (options_.store) {
    options_.store.reset();
  }

  uint32_t rank_u32;
  HcclResult hcclErr = hccl_api_->getRankId(hccl_comm_, &rank_u32);
  if (hcclErr != HCCL_SUCCESS) {
    throw std::runtime_error("HCCL getRankId failed");
  }
  rank_ = static_cast<int>(rank_u32);

  tryTorchCommLoggingInit("torchcomm");

  uint32_t comm_size_u32;
  hcclErr = hccl_api_->getRankSize(hccl_comm_, &comm_size_u32);
  if (hcclErr != HCCL_SUCCESS) {
    throw std::runtime_error("HCCL getRankSize failed");
  }
  comm_size_ = static_cast<int>(comm_size_u32);

  tracing_ = std::make_shared<TorchCommTracing>(name, comm_size_, rank_);
  tracing_->recordEvent("init");

  // Start timeout watchdog thread
  timeout_thread_ = std::thread(&TorchCommHCCL::timeoutWatchdog, this);
}

void TorchCommHCCL::finalize() {
  if (init_state_ == InitializationState::UNINITIALIZED) {
    throw std::runtime_error("TorchCommHCCL not initialized");
  } else if (init_state_ == InitializationState::FINALIZED) {
    throw std::runtime_error("TorchCommHCCL already finalized");
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

  if (work_status == TorchWorkHCCL::WorkStatus::NOT_STARTED ||
      work_status == TorchWorkHCCL::WorkStatus::INPROGRESS) {
    throw std::runtime_error(
        "WorkQ finalize returned in progress or not started state");
  }

  // Update comm_state_ based on the work status
  if (work_status == TorchWorkHCCL::WorkStatus::TIMEDOUT) {
    comm_state_ = CommState::TIMEOUT;
    abortHcclComm();
    throw std::runtime_error("Work timed out during finalize");
  } else if (work_status == TorchWorkHCCL::WorkStatus::ERROR) {
    comm_state_ = CommState::ERROR;
    HcclResult asyncErr;
    hccl_api_->commGetAsyncError(hccl_comm_, &asyncErr);
    HCCLException hcclException(*hccl_api_, "HCCL Async Error", asyncErr);
    abortHcclComm();
    throw hcclException;
  }

  // Clean up event pool
  {
    std::lock_guard<std::mutex> lock(event_pool_mutex_);
    while (!event_pool_.empty()) {
      npuEvent_t event = std::move(event_pool_.front());
      event_pool_.pop();
      NPU_CHECK(
          npu_api_, npu_api_->eventDestroy(event), "Failed to destroy event");
    }
  }

  // Free barrier buffer
  if (barrier_buffer_) {
    NPU_CHECK(
        npu_api_,
        npu_api_->free(barrier_buffer_),
        "Failed to free barrier buffer");
    barrier_buffer_ = nullptr;
  }

  // Destroy dependency event
  if (dependency_event_.has_value()) {
    NPU_CHECK(
        npu_api_,
        npu_api_->eventDestroy(dependency_event_.value()),
        "Failed to destroy dependency event");
    dependency_event_.reset();
  }

  // Destroy internal stream
  if (internal_stream_.has_value()) {
    NPU_CHECK(
        npu_api_,
        npu_api_->streamDestroy(internal_stream_.value()),
        "Failed to destroy internal stream");
    internal_stream_.reset();
  }

  // Destroy HCCL communicator
  if (hccl_comm_) {
    hccl_api_->commDestroy(hccl_comm_);
    hccl_comm_ = nullptr;
  }
}

void TorchCommHCCL::abortHcclComm() {
  if (hccl_comm_) {
    hccl_api_->commAbort(hccl_comm_);
    hccl_comm_ = nullptr;
  }
  if (options_.abort_process_on_timeout_or_error) {
    TC_LOG(ERROR) << "Aborting process due to timeout";
    abort();
  }
}

int TorchCommHCCL::getRank() const {
  checkInitialized();

  uint32_t rank;
  HcclResult hcclErr = hccl_api_->getRankId(hccl_comm_, &rank);
  if (hcclErr != HCCL_SUCCESS) {
    throw HCCLException(*hccl_api_, "HCCL getRankId failed", hcclErr);
  }
  return static_cast<int>(rank);
}

int TorchCommHCCL::getSize() const {
  checkInitialized();

  uint32_t comm_size;
  HcclResult hcclErr = hccl_api_->getRankSize(hccl_comm_, &comm_size);
  if (hcclErr != HCCL_SUCCESS) {
    throw HCCLException(*hccl_api_, "HCCL getRankSize failed", hcclErr);
  }
  return static_cast<int>(comm_size);
}

std::string_view TorchCommHCCL::getBackendName() const {
  return kBackendName;
}

std::string_view TorchCommHCCL::getCommName() const {
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
c10::intrusive_ptr<TorchWork> TorchCommHCCL::send(
    const at::Tensor& tensor,
    int dst,
    bool async_op,
    const SendOptions& options) {
  throw std::runtime_error(
      "HCCL send is not supported now and will be added later");
}

c10::intrusive_ptr<TorchWork> TorchCommHCCL::recv(
    at::Tensor& tensor,
    int src,
    bool async_op,
    const RecvOptions& options) {
  throw std::runtime_error(
      "HCCL recv is not supported now and will be added later");
}

// Batch P2P Operations
c10::intrusive_ptr<TorchWork> TorchCommHCCL::batch_op_issue(
    const std::vector<BatchSendRecv::P2POp>& ops,
    bool async_op,
    const BatchP2POptions& options) {
  throw std::runtime_error(
      "HCCL batch_op_issue is not supported now and will be added later");
}

// Collective Operations
c10::intrusive_ptr<TorchWork> TorchCommHCCL::broadcast(
    at::Tensor& tensor,
    int root,
    bool async_op,
    const BroadcastOptions& options) {
  throw std::runtime_error(
      "HCCL broadcast is not supported now and will be added later");
}

c10::intrusive_ptr<TorchWork> TorchCommHCCL::all_reduce(
    at::Tensor& tensor,
    const ReduceOp& op,
    bool async_op,
    const AllReduceOptions& options) {
  checkInitialized();
  checkAndAbortIfTimedOutOrError();
  // Ensure correct device is set before HCCL calls
  NPU_CHECK(
      npu_api_,
      npu_api_->setDevice(device_.index()),
      "Failed to set NPU device to " + std::to_string(device_.index()));
  ensureTensorContiguous(tensor);

  tracing_->recordEventWithInputOutput("all_reduce", rank_, {tensor}, {tensor});

  npuStream_t stream = getOperationStream(async_op);
  auto work = createWork(
      stream, getOperationTimeout(options.timeout, options_.timeout), {tensor});

  work->recordStart();

  // No-op for empty input tensor
  if (tensor.numel() == 0) [[unlikely]] {
    TC_LOG(WARNING) << "all_reduce called with empty input tensor";
    work->recordEnd();
    enqueueWork(work, stream);
    return work;
  }

  // HCCL handles premul sum differently, apply locally if comm_size is 1
  if (comm_size_ == 1) {
    preReduce(tensor, op);
  }

  const auto dataType = getHcclDataType(tensor);
  HcclResult result = hccl_api_->allReduce(
      tensor.data_ptr(),
      tensor.data_ptr(), // In-place operation
      tensor.numel(),
      dataType,
      getHcclReduceOp(op, hccl_comm_, dataType),
      hccl_comm_,
      stream);

  if (result != HCCL_SUCCESS) {
    throw HCCLException(*hccl_api_, "HCCL AllReduce failed", result);
  }

  work->recordEnd();

  enqueueWork(work, stream);

  return work;
}

c10::intrusive_ptr<TorchWork> TorchCommHCCL::reduce(
    const at::Tensor& tensor,
    int root,
    const ReduceOp& op,
    bool async_op,
    const ReduceOptions& options) {
  throw std::runtime_error(
      "HCCL reduce is not supported now and will be added later");
}

c10::intrusive_ptr<TorchWork> TorchCommHCCL::all_gather(
    const std::vector<at::Tensor>& tensor_list,
    const at::Tensor& tensor,
    bool async_op,
    const AllGatherOptions& options) {
  throw std::runtime_error(
      "HCCL all_gather is not supported now and will be added later");
}

c10::intrusive_ptr<TorchWork> TorchCommHCCL::all_gather_v(
    const std::vector<at::Tensor>& tensor_list,
    const at::Tensor& tensor,
    bool async_op,
    const AllGatherOptions& options) {
  throw std::runtime_error("all_gather_v is not supported in HCCL backend");
}

c10::intrusive_ptr<TorchWork> TorchCommHCCL::all_gather_single(
    at::Tensor& output,
    const at::Tensor& input,
    bool async_op,
    const AllGatherSingleOptions& options) {
  throw std::runtime_error(
      "HCCL all_gather_single is not supported now and will be added later");
}

c10::intrusive_ptr<TorchWork> TorchCommHCCL::reduce_scatter(
    at::Tensor& output,
    const std::vector<at::Tensor>& input_list,
    const ReduceOp& op,
    bool async_op,
    const ReduceScatterOptions& options) {
  throw std::runtime_error(
      "HCCL reduce_scatter is not supported now and will be added later");
}

c10::intrusive_ptr<TorchWork> TorchCommHCCL::reduce_scatter_v(
    at::Tensor& output,
    const std::vector<at::Tensor>& input_list,
    const ReduceOp& op,
    bool async_op,
    const ReduceScatterOptions& options) {
  throw std::runtime_error("reduce_scatter_v is not supported in HCCL backend");
}

c10::intrusive_ptr<TorchWork> TorchCommHCCL::reduce_scatter_single(
    at::Tensor& output,
    const at::Tensor& input,
    const ReduceOp& op,
    bool async_op,
    const ReduceScatterSingleOptions& options) {
  throw std::runtime_error(
      "HCCL reduce_scatter_single is not supported now and will be added later");
}

c10::intrusive_ptr<TorchWork> TorchCommHCCL::all_to_all_single(
    at::Tensor& output,
    const at::Tensor& input,
    bool async_op,
    const AllToAllSingleOptions& options) {
  throw std::runtime_error(
      "HCCL all_to_all_single is not supported now and will be added later");
}

c10::intrusive_ptr<TorchWork> TorchCommHCCL::all_to_all_v_single(
    at::Tensor& output,
    const at::Tensor& input,
    const std::vector<uint64_t>& output_split_sizes,
    const std::vector<uint64_t>& input_split_sizes,
    bool async_op,
    const AllToAllvSingleOptions& options) {
  throw std::runtime_error(
      "HCCL all_to_all_v_single is not supported now and will be added later");
}

c10::intrusive_ptr<TorchWork> TorchCommHCCL::all_to_all(
    const std::vector<at::Tensor>& output_tensor_list,
    const std::vector<at::Tensor>& input_tensor_list,
    bool async_op,
    const AllToAllOptions& options) {
  throw std::runtime_error(
      "HCCL all_to_all is not supported now and will be added later");
}

c10::intrusive_ptr<TorchWork> TorchCommHCCL::barrier(
    bool async_op,
    const BarrierOptions& options) {
  throw std::runtime_error(
      "HCCL barrier is not supported now and will be added later");
}

c10::intrusive_ptr<TorchWork> TorchCommHCCL::scatter(
    at::Tensor& output_tensor,
    const std::vector<at::Tensor>& input_tensor_list,
    int root,
    bool async_op,
    const ScatterOptions& options) {
  throw std::runtime_error(
      "HCCL scatter is not supported now and will be added later");
}

c10::intrusive_ptr<TorchWork> TorchCommHCCL::gather(
    const std::vector<at::Tensor>& output_tensor_list,
    const at::Tensor& input_tensor,
    int root,
    bool async_op,
    const GatherOptions& options) {
  throw std::runtime_error(
      "HCCL gather is not supported now and will be added later");
}

std::shared_ptr<TorchCommBackend> TorchCommHCCL::split(
    const std::vector<int>& ranks,
    const std::string& name,
    const CommOptions& options) {
  throw std::runtime_error(
      "HCCL split is not supported now and will be added later");
}

std::shared_ptr<c10::Allocator> TorchCommHCCL::getMemAllocator() {
  throw std::runtime_error(
      "HCCL getMemAllocator is not supported now and will be added later");
}

HCCLException::HCCLException(
    HcclApi& hccl_api,
    const std::string& message,
    HcclResult result)
    : message_(message + ": " + hccl_api.getErrorString(result)),
      result_(result) {}

const char* HCCLException::what() const noexcept {
  return message_.c_str();
}

} // namespace torch::comms

namespace {
class HCCLRegistration {
 public:
  HCCLRegistration() {
    torch::comms::TorchCommFactory::get().register_backend("hccl", []() {
      return std::make_shared<torch::comms::TorchCommHCCL>();
    });
  }
};

static HCCLRegistration registration{};
} // namespace
