// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "TorchCommMCCL.hpp"

#include <c10/cuda/CUDAGuard.h>
#include <fmt/core.h>
#include <torch/csrc/cuda/CUDAPluggableAllocator.h> // @manual=//caffe2:torch-cpp-cuda

#include "comms/ctran/utils/Alloc.h"
#include "comms/ctran/utils/CudaWrap.h"
#include "comms/torchcomms/TorchCommFactory.hpp"
#include "comms/torchcomms/utils/TracingGuard.hpp"
#include "comms/torchcomms/utils/Utils.hpp"
#include "comms/utils/CudaRAII.h"

#include "comms/torchcomms/mccl/TorchCommMCCLCCA.hpp"
#include "comms/torchcomms/mccl/TorchCommMCCLUtils.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <string_view>

namespace torch::comms {

using namespace torch::comms::mccl;

namespace {
// Stable token for fleet log analysis; the surrounding text may be reworded.
constexpr std::string_view kWatchdogCommAbortToken =
    "[MCCL_WATCHDOG_COMM_ABORT]";
} // namespace

TorchCommMCCL::TorchCommMCCL(std::shared_ptr<CudaApi> cuda_api)
    : device_(at::kCUDA),
      initState_(InitializationState::UNINITIALIZED),
      cuda_api_(std::move(cuda_api)) {}

TorchCommMCCL::TorchCommMCCL(
    std::unique_ptr<::mccl::IComm> mcclComm,
    int rank,
    int size,
    std::shared_ptr<CudaApi> cuda_api)
    : device_(at::kCUDA),
      commSize_(size),
      rank_(rank),
      initState_(InitializationState::UNINITIALIZED),
      cuda_api_(std::move(cuda_api)),
      mccl_comm_(std::move(mcclComm)) {}

TorchCommMCCL::~TorchCommMCCL() {
  if (initState_ == InitializationState::INITIALIZED) {
    TC_LOG(ERROR) << "TorchCommMCCL " << commName_
                  << " was not finalized before destruction";
  }
  // Stop the watchdog even if finalize() was never called (PAFT destroys the
  // comm directly); ~std::thread on a joinable thread calls std::terminate().
  signalWatchdogShutdown();
  if (timeout_thread_.joinable()) {
    // The per-iteration comm.reset() can run this destructor on the watchdog
    // thread, where join() would self-join. Detaching is safe: the thread owns
    // its control block and never touches the freed comm.
    if (std::this_thread::get_id() != timeout_thread_.get_id()) {
      timeout_thread_.join();
    } else {
      timeout_thread_.detach(); // NOLINT(facebook-hte-BadCall-detach)
    }
  }

  // Return any orphaned bracketing events (a collective recorded a Start but
  // never reached maybeRecordGraphCollectiveEnd) to the tracker's pool, so they
  // are destroyed with the rest at ~McclGraphEventTracker. Safe now that the
  // watchdog has stopped.
  for (auto& [_, events] : pending_graph_events_) {
    graph_event_tracker_.releaseEvent(events.first);
    graph_event_tracker_.releaseEvent(events.second);
  }
  pending_graph_events_.clear();
}

void TorchCommMCCL::init(
    at::Device device,
    const std::string& name,
    const CommOptions& options) {
  device_ = device;
  options_ = options;
  commName_ = name;

  // TorchCommMCCL only supports CUDA devices
  if (device_.type() != at::kCUDA) {
    throw std::runtime_error(
        fmt::format(
            "TorchCommMCCL only supports CUDA devices, but got device type: {}",
            c10::DeviceTypeName(device_.type())));
  }

  // Only initialize once
  CHECK_THROW(
      initState_ == InitializationState::UNINITIALIZED, std::runtime_error);

  if (!cuda_api_) {
    cuda_api_ = std::make_unique<DefaultCudaApi>();
  }

  // Get rank and size candidates from environment variables. In static regime
  // communicator is initialized such that it guarantees that these candidates
  // correspond to rank and size of communicator. In dynamic regime communicator
  // is not initialized yet and rank and size are not known.
  int rankCandidate, commSizeCandidate;
  std::tie(rankCandidate, commSizeCandidate) = query_ranksize();

  if (device_.index() == -1) {
    int device_count{0};
    CUDA_CHECK(
        cuda_api_,
        cuda_api_->getDeviceCount(&device_count),
        "Failed to get CUDA device count");

    device_ = c10::Device(c10::kCUDA, rankCandidate % device_count);
    TC_LOG(INFO) << "User did not provide device ID; using device cuda:"
                 << static_cast<int>(device_.index());
  }
  CUDA_CHECK(
      cuda_api_,
      cuda_api_->setDevice(device_.index()),
      fmt::format("Failed to set device to {}", device_.index()));
  cudaDeviceId_ = device_.index();

  // Create stream with default priority
  int streamPriority = 0;
  CUDA_CHECK(
      cuda_api_,
      cuda_api_->streamCreateWithPriority(
          &internalStream_, cudaStreamNonBlocking, streamPriority),
      fmt::format(
          "Failed to create CUDA stream on device {}", device_.index()));

  // Create dependency event for stream synchronization
  CUDA_CHECK(
      cuda_api_,
      cuda_api_->eventCreate(&dependencyEvent_),
      fmt::format(
          "Failed to create dependency event on device {}", device_.index()));

  // Side stream used by the graph-monitor to host its captured work (external
  // clog EVENT_RECORD nodes + replay-counter increment) off the main stream's
  // critical path, with fork/rejoin so capture stays joined. Created here
  // (outside any graph capture); only when monitoring is enabled.
  if (!graph_monitor_side_stream_ && isMcclGraphTimeoutMonitoringEnabled()) {
    graph_monitor_side_stream_ =
        std::make_unique<::meta::comms::GraphSideStream>(streamPriority);
  }

  // Allocate CUDA buffer for barrier operations
  CUDA_CHECK(
      cuda_api_,
      cuda_api_->malloc(
          &barrierBuffer_, kMCCLBarrierBufferSize * sizeof(float)),
      "Failed to allocate barrier buffer");

  const auto defaultTimeout =
      options_.timeout == kNoTimeout ? kMcclCommTimeout : options_.timeout;

  if (mccl_comm_ == nullptr) {
    mccl_comm_ = ::mccl::commCreate(
        ::mccl::CommCreateOpts{.cudaDeviceId = device_.index()});
    // Seed the comm-level default timeout at creation so collectives that omit
    // a per-op timeout fall back to the configured default even when
    // setTimeout() is never called. Preserved across reconfigure(), so this
    // also covers the dynamic regime's first reconfigure.
    if (!mccl_comm_->getTimeout().has_value()) {
      mccl_comm_->setTimeout(defaultTimeout);
    }

    auto initUrl = mccl_comm_->getInitURL();
    TC_LOG(INFO) << "TorchCommMCCL created"
                 << " initURL: " << initUrl;

    configs_ = parseOptions(options);

    // Normalize enable_reconfigure so TorchComm-level guards work
    // regardless of whether the legacy hint or first-class option was used.
    if (configs_.initDynamicRegime_) {
      options_.enable_reconfigure = true;
    }
    if (!configs_.initDynamicRegime_) {
      rank_ = rankCandidate;
      commSize_ = commSizeCandidate;
      TorchCommMCCL::setTCPStore(options.store);
      initStaticRegime(initUrl, defaultTimeout);
      if (createdInternalStore_) {
        cleanInternalStore();
      }
      if (options_.store != nullptr) {
        options_.store.reset();
      }
      initState_ = InitializationState::INITIALIZED;
      TracingGuard tracingGuard(name, commSize_, "init", rank_);
      attachMemoryHook();
      TC_LOG(INFO) << "TorchCommMCCL Initialized rank " << rank_
                   << " initURL: " << initUrl;
      // Start timeout watchdog thread
      startTimeoutWatchdog();
    }
  } else {
    // If mccl_comm_ is not null, it means the mccl_comm_ is already initialized
    initState_ = InitializationState::INITIALIZED;
    TracingGuard tracingGuard(name, commSize_, "init", rank_);
    attachMemoryHook();
    if (!mccl_comm_->getTimeout().has_value()) {
      mccl_comm_->setTimeout(defaultTimeout);
    }
    TC_LOG(INFO) << "TorchCommMCCL Initialized rank " << rank_;
    startTimeoutWatchdog();
  }
  // In dynamic regime, don't set initState_ to INITIALIZED or create tracing_
  // until reconfigure() is called
}

c10::intrusive_ptr<TorchWork> TorchCommMCCL::reconfigure(
    const torch::comms::ReconfigureOptions& opts) {
  if (!mccl_comm_) {
    throw std::runtime_error("Communicator not created");
  }

  // Copy hints and add commName as commDesc for MCCL logging/debugging
  auto kvPairs = opts.hints;
  if (!commName_.empty() && kvPairs.find("commDesc") == kvPairs.end()) {
    kvPairs["commDesc"] = commName_;
  }

  ::mccl::InitOpts mcclOpts;
  mcclOpts.uuid = std::to_string(opts.uuid);
  std::visit([&](const auto& v) { mcclOpts.urls = v; }, opts.handles);
  mcclOpts.timeout = opts.timeout;
  mcclOpts.kvPairs = std::move(kvPairs);

  // Reset to sentinel values before reconfigure so that on failure or
  // exception the communicator is left in a clean uninitialized state.
  initState_ = InitializationState::UNINITIALIZED;

  rank_ = -1;
  commSize_ = 0;
  uuid_.clear();

  auto mcclWorkHandle = mccl_comm_->reconfigure(mcclOpts);

  // MCCL reconfigure is currently blocking, so calling waitCpu() does not
  // impact performance. In the future, when we make reconfigure asynchronous,
  // we'll need to revisit the code below. The challenge is that we cannot
  // determine rank_ and commSize_ until reconfigure completes, so this code
  // must be executed together with MCCL reconfigure in its thread. One possible
  // approach is to define a callback and pass it to MCCL reconfigure. In this
  // case, the work handler would only be marked as done when both MCCL
  // reconfigure and the code below have finished.
  // TODO: revisit code below after mccl reconfigure become async
  mcclWorkHandle->waitCpu();

  if (mcclWorkHandle->getResult()->code == commSuccess) {
    auto ra = mccl_comm_->getRankAssignment();
    rank_ = ra.urlToRank[mccl_comm_->getInitURL()];
    commSize_ = static_cast<int>(ra.rankToUrl.size());
    initState_ = InitializationState::INITIALIZED;
    uuid_ = mcclOpts.uuid;
    attachMemoryHook();
    // Dynamic-regime comms are not INITIALIZED in init(), so the timeout
    // watchdog (which drives McclGraphEventTracker::checkAll() and the graph-
    // replay clog) is not started there. Start it on the first successful
    // reconfigure(). Idempotent: reconfigure() runs on every quorum change, so
    // guard on joinable() to start the thread exactly once. joinable() is also
    // false after finalize() join()s the thread, but finalize() is terminal
    // (it nulls mccl_comm_, and reconfigure() throws on a null comm before
    // reaching here), so a restart-after-finalize cannot occur.
    if (!timeout_thread_.joinable()) {
      startTimeoutWatchdog();
    }
  }

  TracingGuard tracingGuard(commName_, commSize_, "reconfigure", rank_);

  auto torchWorkHandle = createWork(internalStream_, std::move(mcclWorkHandle));

  return torchWorkHandle;
}

void TorchCommMCCL::finalize() {
  TC_MCCL_CHECK_INITIALIZED();

  // Signal shutdown to timeout watchdog
  signalWatchdogShutdown();

  // Wait for timeout thread to finish. finalize() is called by the owner, never
  // from the watchdog itself, so an unconditional join is safe here.
  if (timeout_thread_.joinable()) {
    timeout_thread_.join();
  }

  TC_LOG(INFO, this) << "Joined timeout thread";

  // Deliver any final graph-replay clog events and release tracked CUDA events.
  // Safe now that the watchdog (the only checkAll() caller) has stopped.
  graph_event_tracker_.destroyAll();

  auto work_status = workq_.finalize();
  if (work_status == TorchWorkMCCL::WorkStatus::NOT_STARTED ||
      work_status == TorchWorkMCCL::WorkStatus::INPROGRESS ||
      work_status == TorchWorkMCCL::WorkStatus::TIMEDOUT) {
    throw std::runtime_error(
        "WorkQ finalize returned in progress or not started state");
  }

  if (commState_ != CommState::NORMAL) {
    throw std::runtime_error(
        "TorchCommMCCL::finalize: communicator not closed");
  }

  // Synchronize and destroy internal CUDA stream to ensure all pending
  // operations complete before releasing other resources
  if (internalStream_) {
    CUDA_CHECK(
        cuda_api_,
        cuda_api_->streamSynchronize(internalStream_),
        "Failed to synchronize internal stream");
    CUDA_CHECK(
        cuda_api_,
        cuda_api_->streamDestroy(internalStream_),
        "Failed to destroy internal stream");
    internalStream_ = nullptr;
  }

  // The internal store and options_.store are already cleaned up in init()
  // after initStaticRegime() completes. Only reset store_ if it's still
  // around (e.g., dynamic regime or external store that wasn't cleaned up).
  if (store_ != nullptr) {
    store_.reset();
  }

  if (options_.store != nullptr) {
    options_.store.reset();
  }

  // Destroy dependency event
  if (dependencyEvent_) {
    CUDA_CHECK(
        cuda_api_,
        cuda_api_->eventDestroy(dependencyEvent_),
        "Failed to destroy dependency event");
    dependencyEvent_ = nullptr;
  }

  // Free barrier buffer
  if (barrierBuffer_) {
    CUDA_CHECK(
        cuda_api_,
        cuda_api_->free(barrierBuffer_),
        "Failed to free barrier buffer");
    barrierBuffer_ = nullptr;
  }

  if (mccl_comm_) {
    detachMemoryHook();
    mccl_comm_.reset();
  }

  initState_ = InitializationState::FINALIZED;
  uuid_.clear();
  TC_LOG(INFO) << "TorchCommMCCL: finalized. " << rank_;
}

int TorchCommMCCL::getRank() const {
  TC_MCCL_CHECK_INITIALIZED();
  return rank_;
}

int TorchCommMCCL::getSize() const {
  TC_MCCL_CHECK_INITIALIZED();
  return commSize_;
}

bool TorchCommMCCL::isInitialized() const {
  return initState_ == InitializationState::INITIALIZED;
}

std::string_view TorchCommMCCL::getCommName() const {
  return commName_;
}

std::string_view TorchCommMCCL::getBackendName() const {
  return kBackendName;
}

const CommOptions& TorchCommMCCL::getOptions() const {
  return options_;
}

const at::Device& TorchCommMCCL::getDevice() const {
  return device_;
}

c10::intrusive_ptr<TorchWork> TorchCommMCCL::send(
    const at::Tensor& tensor,
    int dst,
    bool async_op,
    const SendOptions& options) {
  TC_MCCL_CHECK_INITIALIZED();
  McclCachingAllocatorHookImpl::drainPendingRegistrations();

  TracingGuard tracingGuard(commName_, commSize_, "send", dst, tensor, tensor);
  cudaStream_t streamToUse = getOperationStream(async_op);

  ::mccl::SingleSendOpts opts =
      getSingleSendOpts(tensor, cudaDeviceId_, streamToUse, dst, options);

  auto mcclWorkHandle = mccl_comm_->send(opts);

  auto torchWorkHandle = async_op
      ? createWork(streamToUse, std::move(mcclWorkHandle), tensor, {})
      : createWork(streamToUse, std::move(mcclWorkHandle));

  enqueueWork(torchWorkHandle, streamToUse);

  return torchWorkHandle;
}

c10::intrusive_ptr<TorchWork> TorchCommMCCL::recv(
    at::Tensor& tensor,
    int src,
    bool async_op,
    const RecvOptions& options) {
  TC_MCCL_CHECK_INITIALIZED();
  McclCachingAllocatorHookImpl::drainPendingRegistrations();

  TracingGuard tracingGuard(commName_, commSize_, "recv", src, tensor, tensor);
  cudaStream_t streamToUse = getOperationStream(async_op);

  ::mccl::SingleRecvOpts opts =
      getSingleRecvOpts(tensor, cudaDeviceId_, streamToUse, src, options);

  auto mcclWorkHandle = mccl_comm_->recv(opts);

  auto torchWorkHandle = async_op
      ? createWork(streamToUse, std::move(mcclWorkHandle), {}, tensor)
      : createWork(streamToUse, std::move(mcclWorkHandle));

  enqueueWork(torchWorkHandle, streamToUse);

  return torchWorkHandle;
}

c10::intrusive_ptr<TorchWork> TorchCommMCCL::batch_op_issue(
    const std::vector<BatchSendRecv::P2POp>& ops,
    bool async_op,
    const BatchP2POptions& options) {
  TC_MCCL_CHECK_INITIALIZED();
  McclCachingAllocatorHookImpl::drainPendingRegistrations();

  if (ops.empty()) {
    throw std::runtime_error("Cannot issue empty batch operation");
  }
  // Collect input and output tensors for work tracking
  std::vector<at::Tensor> input_tensors;
  std::vector<at::Tensor> output_tensors;
  cudaStream_t streamToUse = getOperationStream(async_op);
  std::unique_ptr<::mccl::IWorkHandle> workHandle;

  std::vector<::mccl::SendRecvOpts> optsList;
  for (const auto& op : ops) {
    if (op.type == BatchSendRecv::P2POp::OpType::SEND) {
      at::Tensor tensor = op.tensor;
      input_tensors.emplace_back(tensor);
      ::mccl::SendOpts opts = getSendOpts(tensor, cudaDeviceId_, op.peer);
      optsList.emplace_back(opts);
    } else if (op.type == BatchSendRecv::P2POp::OpType::RECV) {
      at::Tensor tensor = op.tensor;
      output_tensors.emplace_back(tensor);
      ::mccl::RecvOpts opts = getRecvOpts(tensor, cudaDeviceId_, op.peer);
      optsList.emplace_back(opts);
    }
  }

  ::mccl::BatchOpts opts = getBatchOpts(optsList, streamToUse, options);
  workHandle = mccl_comm_->batchSendRecv(opts);
  const std::vector<at::Tensor> empty_tensors;
  auto torchWorkHandle = createWork(
      streamToUse,
      std::move(workHandle),
      async_op ? input_tensors : empty_tensors,
      async_op ? output_tensors : empty_tensors);
  enqueueWork(torchWorkHandle, streamToUse);
  return torchWorkHandle;
}

// Collective Operations
c10::intrusive_ptr<TorchWork> TorchCommMCCL::broadcast(
    at::Tensor& tensor,
    int root,
    bool async_op,
    const BroadcastOptions& options) {
  TC_MCCL_CHECK_INITIALIZED();
  McclCachingAllocatorHookImpl::drainPendingRegistrations();

  if (tensor.is_cpu()) {
    TracingGuard tracingGuard(
        commName_, commSize_, "broadcast_cpu", root, tensor, tensor);

    const std::optional<std::chrono::milliseconds> operationTimeout =
        getOperationTimeout(options.timeout);

    ::mccl::BroadcastOpts opts = ::mccl::BroadcastOpts{
        .data = torch::comms::mccl::getTensorDataPtr(tensor),
        .numElements = static_cast<size_t>(tensor.numel()),
        .dataType = torchToMcclType(tensor.scalar_type()),
        .stream = nullptr,
        .root = root,
        .deviceType = ::mccl::DeviceType::Cpu,
        .timeout = operationTimeout,
        .kvPairs = options.hints};

    auto mcclWorkHandle = mccl_comm_->broadcast(opts);
    auto torchWorkHandle =
        createWork(internalStream_, std::move(mcclWorkHandle), tensor, tensor);
    torchWorkHandle->setCpuWork(true);
    enqueueWork(torchWorkHandle, internalStream_);
    if (!async_op) {
      torchWorkHandle->waitCPU();
    }
    return torchWorkHandle;
  }

  TracingGuard tracingGuard(
      commName_, commSize_, "broadcast", root, tensor, tensor);
  cudaStream_t streamToUse = getOperationStream(async_op);

  ::mccl::BroadcastOpts opts =
      getBroadcastOpts(tensor, cudaDeviceId_, streamToUse, root, options);

  auto mcclWorkHandle = mccl_comm_->broadcast(opts);

  // A handle that already carries a result failed before dispatch -- MCCL has
  // no device broadcast and reports that as a code. The device wait path
  // cannot surface it: such a handle has no CUDA event, so waitStream()
  // returns without raising and the caller would go on to read a buffer that
  // was never broadcast. Raise here, before the handle is enqueued.
  if (const auto result = mcclWorkHandle->getResult();
      result.has_value() && result->code != commSuccess) {
    TORCH_CHECK(
        false,
        "TorchCommMCCL: ",
        commName_,
        " broadcast failed: ",
        result->message);
  }

  auto torchWorkHandle = async_op
      ? createWork(streamToUse, std::move(mcclWorkHandle), tensor, tensor)
      : createWork(streamToUse, std::move(mcclWorkHandle));
  enqueueWork(torchWorkHandle, streamToUse);
  return torchWorkHandle;
}

c10::intrusive_ptr<TorchWork> TorchCommMCCL::all_reduce(
    at::Tensor& tensor,
    const ReduceOp& op,
    bool async_op,
    const AllReduceOptions& options) {
  TC_MCCL_CHECK_INITIALIZED();
  McclCachingAllocatorHookImpl::drainPendingRegistrations();

  if (tensor.is_cpu()) {
    TracingGuard tracingGuard(
        commName_, commSize_, "all_reduce_cpu", rank_, tensor, tensor);

    const std::optional<std::chrono::milliseconds> operationTimeout =
        getOperationTimeout(options.timeout);

    ::mccl::AllReduceOpts opts = ::mccl::AllReduceOpts{
        .dataSend = torch::comms::mccl::getTensorDataPtr(tensor),
        .dataRecv = torch::comms::mccl::getTensorDataPtr(tensor),
        .numElements = static_cast<size_t>(tensor.numel()),
        .dataType = torchToMcclType(tensor.scalar_type()),
        .reduceOpType = torchToCommRedOp(op),
        .stream = nullptr,
        .deviceType = ::mccl::DeviceType::Cpu,
        .timeout = operationTimeout,
        .kvPairs = options.hints};

    auto mcclWorkHandle = mccl_comm_->allReduce(opts);
    auto torchWorkHandle =
        createWork(internalStream_, std::move(mcclWorkHandle), tensor, tensor);
    torchWorkHandle->setCpuWork(true);
    enqueueWork(torchWorkHandle, internalStream_);
    if (!async_op) {
      torchWorkHandle->waitCPU();
    }
    return torchWorkHandle;
  }

  TracingGuard tracingGuard(
      commName_, commSize_, "all_reduce", rank_, tensor, tensor);
  cudaStream_t streamToUse = getOperationStream(async_op);

  ::mccl::AllReduceOpts opts =
      getAllReduceOpts(tensor, cudaDeviceId_, streamToUse, op, options);

  auto mcclWorkHandle = mccl_comm_->allReduce(opts);
  auto torchWorkHandle = async_op
      ? createWork(streamToUse, std::move(mcclWorkHandle), tensor, tensor)
      : createWork(streamToUse, std::move(mcclWorkHandle));
  enqueueWork(torchWorkHandle, streamToUse);

  return torchWorkHandle;
}

c10::intrusive_ptr<TorchWork> TorchCommMCCL::reduce(
    const at::Tensor& /* tensor */,
    int /* root */,
    const ReduceOp& /* op */,
    bool /* async_op */,
    const ReduceOptions& /* options */) {
  throw std::runtime_error("not implemented");
}

c10::intrusive_ptr<TorchWork> TorchCommMCCL::all_gather(
    const std::vector<at::Tensor>& tensor_list,
    const at::Tensor& tensor,
    bool async_op,
    const AllGatherOptions& options) {
  TC_MCCL_CHECK_INITIALIZED();
  McclCachingAllocatorHookImpl::drainPendingRegistrations();

  // Validate output tensor list size
  if (tensor_list.size() != static_cast<size_t>(commSize_)) {
    throw std::runtime_error(
        "tensor_list size must equal comm_size for all_gather");
  }

  // Validate all output tensors have the same size as input
  for (const auto& t : tensor_list) {
    if (t.numel() != tensor.numel()) {
      throw std::runtime_error(
          "All tensors in tensor_list must have same size as input tensor");
    }
  }

  TracingGuard tracingGuard(
      commName_, commSize_, "all_gather", rank_, {tensor}, tensor_list);

  cudaStream_t streamToUse = getOperationStream(async_op);
  const std::optional<std::chrono::milliseconds> operationTimeout =
      getOperationTimeout(options.timeout);

  // This approach avoids synchronization issues with temp buffer + memcpy.
  std::vector<::mccl::SendRecvOpts> optsList;
  for (int i = 0; i < commSize_; ++i) {
    ::mccl::SendOpts sendOpt = getSendOpts(tensor, cudaDeviceId_, i);
    optsList.emplace_back(sendOpt);
    ::mccl::RecvOpts recvOpt = getRecvOpts(tensor_list[i], cudaDeviceId_, i);
    optsList.emplace_back(recvOpt);
  }

  ::mccl::BatchOpts batchOpts = ::mccl::BatchOpts{
      .optsList = optsList,
      .stream = streamToUse,
      .timeout = operationTimeout,
      .kvPairs = options.hints,
  };

  auto workHandle = mccl_comm_->batchSendRecv(batchOpts);
  auto torchWorkHandle = async_op
      ? createWork(streamToUse, std::move(workHandle), {tensor}, tensor_list)
      : createWork(streamToUse, std::move(workHandle));
  enqueueWork(torchWorkHandle, streamToUse);

  return torchWorkHandle;
}

c10::intrusive_ptr<TorchWork> TorchCommMCCL::all_gather_v(
    const std::vector<at::Tensor>& /* tensor_list */,
    const at::Tensor& /* tensor */,
    bool /* async_op */,
    const AllGatherOptions& /* options */) {
  throw std::runtime_error("not implemented");
}

c10::intrusive_ptr<TorchWork> TorchCommMCCL::all_gather_single(
    at::Tensor& output,
    const at::Tensor& input,
    bool async_op,
    const AllGatherSingleOptions& /* options */) {
  TC_MCCL_CHECK_INITIALIZED();

  TORCH_CHECK(
      input.is_cpu() == output.is_cpu(),
      "all_gather_single: input and output must be on the same device type, "
      "but got input on ",
      input.device(),
      " and output on ",
      output.device());

  if (!input.is_cpu()) {
    throw std::runtime_error("CUDA all_gather_single is not supported by MCCL");
  }

  McclCachingAllocatorHookImpl::drainPendingRegistrations();
  TracingGuard tracingGuard(
      commName_, commSize_, "all_gather_single_cpu", rank_, input, output);

  void* outputPtr = torch::comms::mccl::getTensorDataPtr(output);
  void* inputPtr = torch::comms::mccl::getTensorDataPtr(input);

  ::mccl::AllGatherOpts opts = ::mccl::AllGatherOpts{
      .dataSend = inputPtr,
      .dataRecv = outputPtr,
      .numElements = static_cast<size_t>(input.numel()),
      .dataType = torchToMcclType(input.scalar_type()),
      .stream = nullptr,
      .deviceType = ::mccl::DeviceType::Cpu,
  };

  auto mcclWorkHandle = mccl_comm_->allGather(opts);
  auto torchWorkHandle =
      createWork(internalStream_, std::move(mcclWorkHandle), input, output);
  torchWorkHandle->setCpuWork(true);
  enqueueWork(torchWorkHandle, internalStream_);
  if (!async_op) {
    torchWorkHandle->waitCPU();
  }

  return torchWorkHandle;
}

// Persistent AllGather operations

TorchCommBackend::AllGatherPHandle TorchCommMCCL::all_gather_p_init(
    at::Tensor& output,
    const AllGatherPInitOptions& options) {
  TC_MCCL_CHECK_INITIALIZED();
  McclCachingAllocatorHookImpl::drainPendingRegistrations();

  // Persistent AllGather is pinned to the internal stream at init: ctran
  // freezes the execution stream to the one supplied here, so
  // all_gather_p_exec() routes input readiness onto internalStream_.
  ::mccl::AllGatherPInitOpts opts =
      getAllGatherPInitOpts(output, cudaDeviceId_, internalStream_, options);

  // Auto-register the receive buffer if not already registered (ctran requires
  // the recv buffer to be registered before allGatherPInit). register_address()
  // is a no-op if this address is already registered. Not deregistered by
  // all_gather_p_free(); callers release it via tensor_deregister().
  register_address(
      AddressWithLen{output.data_ptr(), static_cast<size_t>(output.nbytes())});

  AllGatherPHandle handle = nullptr;
  auto mcclWorkHandle = mccl_comm_->allGatherPInit(handle, opts);
  // The generic all_gather_p_init returns a bare handle with no deferred-error
  // channel, so surface an init failure synchronously by waiting and throwing.
  mcclWorkHandle->waitCpu();
  auto result = mcclWorkHandle->getResult();
  if (!result.has_value() || result->code != commSuccess) {
    throw std::runtime_error(
        "TorchCommMCCL: " + commName_ + " allGatherPInit failed: " +
        (result.has_value() ? result->message : "no result"));
  }
  return handle;
}

c10::intrusive_ptr<TorchWork> TorchCommMCCL::all_gather_p_exec(
    AllGatherPHandle handle,
    const at::Tensor& input,
    bool async_op,
    const AllGatherPExecOptions& options) {
  TC_MCCL_CHECK_INITIALIZED();
  McclCachingAllocatorHookImpl::drainPendingRegistrations();

  TORCH_CHECK(handle != nullptr, "all_gather_p_exec: handle must not be null");

  TracingGuard tracingGuard(
      commName_, commSize_, "all_gather_p_exec", rank_, input, input);

  ::mccl::AllGatherPExecOpts opts =
      getAllGatherPExecOpts(input, cudaDeviceId_, options);

  // ctran runs persistent exec on the stream captured at init (internalStream_)
  // and inserts no input-readiness edge (Ctran fork/join contract), so we must
  // order the caller's current stream -> internalStream_ before exec on EVERY
  // call, in both sync and async modes. getOperationStream() is intentionally
  // not used: its sync branch returns the current stream and would neither
  // route to nor order input on internalStream_.
  cudaStream_t currentStream = cuda_api_->getCurrentCUDAStream(device_.index());
  CUDA_CHECK(
      cuda_api_,
      cuda_api_->eventRecord(dependencyEvent_, currentStream),
      "Failed to record dependency event for all_gather_p_exec");
  CUDA_CHECK(
      cuda_api_,
      cuda_api_->streamWaitEvent(internalStream_, dependencyEvent_, 0),
      "Failed to make internal stream wait for dependency event");

  auto mcclWorkHandle = mccl_comm_->allGatherPExec(handle, opts);
  // Create/enqueue on internalStream_ (the true work stream) so workq bucketing
  // and graph brackets match where ctran actually runs the collective.
  auto torchWorkHandle =
      createWork(internalStream_, std::move(mcclWorkHandle), {input}, {});
  enqueueWork(torchWorkHandle, internalStream_);

  if (!async_op) {
    // Nonblocking CUDA stream join: order the caller's current stream after the
    // persistent collective (internalStream_ -> current) without blocking the
    // CPU. A deferred MCCL error is surfaced via waitCPU()/checkStatus()/
    // getResult(), not by wait() (which only inserts the stream join here).
    torchWorkHandle->wait();
  }

  return torchWorkHandle;
}

void TorchCommMCCL::all_gather_p_free(AllGatherPHandle handle) {
  if (handle == nullptr || !mccl_comm_) {
    return;
  }
  // Free only the persistent collective handle; the recv buffer registered in
  // all_gather_p_init() is intentionally NOT deregistered here (callers release
  // it via tensor_deregister()). Best-effort teardown: wait for completion but
  // do not throw on the result.
  auto mcclWorkHandle = mccl_comm_->allGatherPFree(handle);
  mcclWorkHandle->waitCpu();
}

c10::intrusive_ptr<TorchWork> TorchCommMCCL::reduce_scatter(
    at::Tensor& /* output */,
    const std::vector<at::Tensor>& /* input_list */,
    const ReduceOp& /* op */,
    bool /* async_op */,
    const ReduceScatterOptions& /* options */) {
  throw std::runtime_error("not implemented");
}

c10::intrusive_ptr<TorchWork> TorchCommMCCL::reduce_scatter_v(
    at::Tensor& /* output */,
    const std::vector<at::Tensor>& /* input_list */,
    const ReduceOp& /* op */,
    bool /* async_op */,
    const ReduceScatterOptions& /* options */) {
  throw std::runtime_error("not implemented");
}

c10::intrusive_ptr<TorchWork> TorchCommMCCL::reduce_scatter_single(
    at::Tensor& output,
    const at::Tensor& input,
    const ReduceOp& op,
    bool async_op,
    const ReduceScatterSingleOptions& options) {
  TC_MCCL_CHECK_INITIALIZED();
  McclCachingAllocatorHookImpl::drainPendingRegistrations();

  TracingGuard tracingGuard(
      commName_, commSize_, "reduce_scatter_single", rank_, input, output);

  cudaStream_t streamToUse = getOperationStream(async_op);

  ::mccl::ReduceScatterOpts opts = getReduceScatterOpts(
      output, input, cudaDeviceId_, streamToUse, op, options);

  auto mcclWorkHandle = mccl_comm_->reduceScatter(opts);
  auto torchWorkHandle = async_op
      ? createWork(streamToUse, std::move(mcclWorkHandle), input, output)
      : createWork(streamToUse, std::move(mcclWorkHandle));
  enqueueWork(torchWorkHandle, streamToUse);

  return torchWorkHandle;
}

c10::intrusive_ptr<TorchWork> TorchCommMCCL::all_to_all_single(
    at::Tensor& /* output */,
    const at::Tensor& /* input */,
    bool /* async_op */,
    const AllToAllSingleOptions& /* options */) {
  TC_MCCL_CHECK_INITIALIZED();
  throw std::runtime_error("all_to_all_single is not supported by MCCL");
}

c10::intrusive_ptr<TorchWork> TorchCommMCCL::all_to_all_v_single(
    at::Tensor& /* output */,
    const at::Tensor& /* input */,
    const std::vector<uint64_t>& /* output_split_sizes */,
    const std::vector<uint64_t>& /* input_split_sizes */,
    bool /* async_op */,
    const AllToAllvSingleOptions& /* options */) {
  TC_MCCL_CHECK_INITIALIZED();
  throw std::runtime_error("all_to_all_v_single is not supported by MCCL");
}

c10::intrusive_ptr<TorchWork> TorchCommMCCL::all_to_all(
    const std::vector<at::Tensor>& output_tensor_list,
    const std::vector<at::Tensor>& input_tensor_list,
    bool async_op,
    const AllToAllOptions& options) {
  TC_MCCL_CHECK_INITIALIZED();
  McclCachingAllocatorHookImpl::drainPendingRegistrations();

  TracingGuard tracingGuard(
      commName_,
      commSize_,
      "all_to_all",
      rank_,
      input_tensor_list,
      output_tensor_list);
  const std::optional<std::chrono::milliseconds> operationTimeout =
      getOperationTimeout(options.timeout);

  if (output_tensor_list.size() != static_cast<size_t>(commSize_) ||
      input_tensor_list.size() != static_cast<size_t>(commSize_)) {
    throw std::runtime_error(
        "Tensor list sizes must equal comm_size for all_to_all");
  }

  auto streamToUse = getOperationStream(async_op);

  std::vector<::mccl::SendRecvOpts> optsList;
  for (int i = 0; i < commSize_; i++) {
    ::mccl::SendOpts sendOpt =
        getSendOpts(input_tensor_list[i], cudaDeviceId_, i);
    optsList.emplace_back(sendOpt);
    ::mccl::RecvOpts recvOpt =
        getRecvOpts(output_tensor_list[i], cudaDeviceId_, i);
    optsList.emplace_back(recvOpt);
  }

  ::mccl::BatchOpts opts = ::mccl::BatchOpts{
      .optsList = optsList,
      .stream = streamToUse,
      .timeout = operationTimeout,
      .kvPairs = options.hints,
  };

  auto workHandle = mccl_comm_->batchSendRecv(opts);
  const std::vector<at::Tensor> empty_tensors;
  auto torchWorkHandle = createWork(
      streamToUse,
      std::move(workHandle),
      async_op ? input_tensor_list : empty_tensors,
      async_op ? output_tensor_list : empty_tensors);
  enqueueWork(torchWorkHandle, streamToUse);

  return torchWorkHandle;
}

c10::intrusive_ptr<TorchWork> TorchCommMCCL::barrier(
    bool async_op,
    const BarrierOptions& options) {
  TC_MCCL_CHECK_INITIALIZED();

  if (configs_.useCpuBarrier_) {
    TracingGuard tracingGuard(commName_, commSize_, "barrier_cpu", rank_);

    ::mccl::BarrierOpts barrierOpts{
        .deviceType = ::mccl::DeviceType::Cpu,
        .timeout = getOperationTimeout(options.timeout)};

    auto mcclWorkHandle = mccl_comm_->barrier(barrierOpts);
    auto torchWorkHandle =
        createWork(internalStream_, std::move(mcclWorkHandle));
    torchWorkHandle->setCpuWork(true);
    enqueueWork(torchWorkHandle, internalStream_);
    if (!async_op) {
      torchWorkHandle->waitCPU();
    }
    return torchWorkHandle;
  }

  TracingGuard tracingGuard(commName_, commSize_, "barrier", rank_);
  cudaStream_t streamToUse = getOperationStream(async_op);
  const std::optional<std::chrono::milliseconds> operationTimeout =
      getOperationTimeout(options.timeout);

  ::mccl::AllReduceOpts opts = ::mccl::AllReduceOpts{
      .dataSend = barrierBuffer_,
      .dataRecv = barrierBuffer_,
      .numElements = kMCCLBarrierBufferSize,
      .dataType = commDataType_t::commFloat32,
      .reduceOpType = commSum,
      .stream = streamToUse,
      .timeout = operationTimeout,
      .kvPairs = options.hints};

  auto mcclWorkHandle = mccl_comm_->allReduce(opts);

  auto torchWorkHandle = createWork(streamToUse, std::move(mcclWorkHandle));

  enqueueWork(torchWorkHandle, streamToUse);

  return torchWorkHandle;
}

// Scatter and Gather Operations
c10::intrusive_ptr<TorchWork> TorchCommMCCL::scatter(
    at::Tensor& /* output_tensor */,
    const std::vector<at::Tensor>& /* input_tensor_list */,
    int /* root */,
    bool /* async_op */,
    const ScatterOptions& /* options */) {
  throw std::runtime_error("not implemented");
}

c10::intrusive_ptr<TorchWork> TorchCommMCCL::gather(
    const std::vector<at::Tensor>& /* output_tensor_list */,
    const at::Tensor& /* input_tensor */,
    int /* root */,
    bool /* async_op */,
    const GatherOptions& /* options */) {
  throw std::runtime_error(
      "[TorchCommMCCL]: gather not implemented, use gather_single instead");
}

c10::intrusive_ptr<TorchWork> TorchCommMCCL::gather_single(
    at::Tensor& output,
    const at::Tensor& input,
    int root,
    bool async_op,
    const GatherSingleOptions& /* options */) {
  TC_MCCL_CHECK_INITIALIZED();

  if (input.is_cpu()) {
    // Validate output tensor is also CPU on root
    if (rank_ == root) {
      TORCH_CHECK(
          output.is_cpu(),
          "gather_single: output tensor must be on CPU when input is CPU, but got " +
              output.device().str());
    }

    TracingGuard tracingGuard(
        commName_, commSize_, "gather_single_cpu", rank_, input, input);

    size_t numElements = static_cast<size_t>(input.numel());
    void* inputPtr = torch::comms::mccl::getTensorDataPtr(input);

    void* outputPtr = nullptr;
    if (rank_ == root) {
      outputPtr = torch::comms::mccl::getTensorDataPtr(output);
    }

    ::mccl::GatherOpts opts = ::mccl::GatherOpts{
        .dataSend = inputPtr,
        .dataRecv = outputPtr,
        .numElements = numElements,
        .dataType = torchToMcclType(input.dtype().toScalarType()),
        .stream = nullptr,
        .root = root,
        .deviceType = ::mccl::DeviceType::Cpu,
    };

    auto mcclWorkHandle = mccl_comm_->gather(opts);
    auto torchWorkHandle =
        createWork(internalStream_, std::move(mcclWorkHandle), input, output);
    torchWorkHandle->setCpuWork(true);
    enqueueWork(torchWorkHandle, internalStream_);
    if (!async_op) {
      torchWorkHandle->waitCPU();
    }
    return torchWorkHandle;
  }

  throw std::runtime_error("device gather_single not implemented");
}

// Window & One-sided Operations
std::shared_ptr<TorchCommWindow> TorchCommMCCL::new_window(
    const std::optional<at::Tensor>& /* tensor */) {
  throw std::runtime_error("not implemented");
}

// Communicator Management
std::shared_ptr<TorchCommBackend> TorchCommMCCL::split(
    const std::vector<int>& ranks,
    const std::string& name,
    const CommOptions& options) {
  TC_MCCL_CHECK_INITIALIZED();

  // Validate that all ranks are valid
  for (int rank : ranks) {
    if (rank < 0 || rank >= commSize_) {
      throw std::runtime_error(
          fmt::format(
              "Invalid rank {} in ranks. Valid ranks are 0 to {}",
              rank,
              commSize_ - 1));
    }
  }

  // Check for duplicate ranks
  std::set<int> unique_ranks(ranks.begin(), ranks.end());
  if (unique_ranks.size() != ranks.size()) {
    throw std::runtime_error("Duplicate ranks found in ranks list");
  }
  int new_rank;

  if (ranks.empty()) {
    new_rank = -1; // Will not participate in new communicator
  } else {
    auto it = std::find(ranks.begin(), ranks.end(), rank_);
    if (it == ranks.end()) {
      throw std::runtime_error(
          fmt::format(
              "Current rank {} is not included in the provided ranks list",
              rank_));
    }
    new_rank = static_cast<int>(std::distance(ranks.begin(), it));
  }

  const std::optional<std::chrono::milliseconds> operationTimeout =
      getOperationTimeout(options.timeout);

  std::unique_ptr<::mccl::IComm> newComm = mccl_comm_->commSplit(
      ::mccl::SplitOpts{
          .uuid = "0",
          .ranks = ranks,
          .timeout = operationTimeout,
          .kvPairs = options.hints});

  if (new_rank == -1) {
    return nullptr;
  }

  // NOLINTNEXTLINE(facebook-hte-SharedPtrFromNew)
  auto new_torchcomm = std::make_shared<TorchCommMCCL>(
      std::move(newComm), new_rank, static_cast<int>(ranks.size()), cuda_api_);
  new_torchcomm->init(device_, name, options);

  return new_torchcomm;
}

void TorchCommMCCL::tensor_register(const at::Tensor& tensor) {
  TORCH_CHECK(
      tensor.is_contiguous(), "tensor_register requires contiguous tensor");
  register_address(
      AddressWithLen{tensor.data_ptr(), static_cast<size_t>(tensor.nbytes())});
}

void TorchCommMCCL::tensor_deregister(const at::Tensor& tensor) {
  TORCH_CHECK(
      tensor.is_contiguous(), "tensor_deregister requires contiguous tensor");
  deregister_address(Address{tensor.data_ptr()});
}

void TorchCommMCCL::register_address(
    const TorchCommMCCL::AddressWithLen& addr) {
  TC_MCCL_CHECK_INITIALIZED();

  if (!mccl_comm_) {
    throw std::runtime_error(
        "TorchCommMCCL: register_address called while McclComm is null!");
  }

  if (memoryRegistrationHandles_.contains(addr.addr)) {
    TC_LOG(WARNING) << "TorchCommMCCL: Memory already registered with MCCL!";
    return;
  }
  void* handle = nullptr;
  commResult_t result = mccl_comm_->commRegister(addr.addr, addr.len, &handle);
  if (result != commSuccess) {
    throw std::runtime_error(
        "TorchCommMCCL: " + commName_ +
        " Failed to register memory with MCCL!");
  }
  memoryRegistrationHandles_.emplace(addr.addr, RegistrationHandle(handle));
}

void TorchCommMCCL::deregister_address(const TorchCommMCCL::Address& addr) {
  TC_MCCL_CHECK_INITIALIZED();

  if (!mccl_comm_) {
    throw std::runtime_error(
        "TorchCommMCCL: deregister_address called while McclComm is null");
  }

  auto it = memoryRegistrationHandles_.find(addr.addr);
  if (it == memoryRegistrationHandles_.end()) {
    // it's possible that the memory was registered for a different comm,
    // however failed registration for this comm.
    TC_LOG(WARNING)
        << "TorchCommMCCL: " << commName_
        << " Could not find memory register handle, the memory may have been registered for a different comm";
    return;
  }

  void* handle = it->second.regHandle;
  commResult_t result = mccl_comm_->commDeregister(handle);
  if (result != commSuccess) {
    throw std::runtime_error(
        "TorchCommMCCL:  " + commName_ +
        " Failed to deregister memory with MCCL!");
  }

  memoryRegistrationHandles_.erase(it);
}

void TorchCommMCCL::global_register_address(
    const TorchCommMCCL::AddressWithLen& addr) {
  commResult_t result = ::mccl::globalRegister(addr.addr, addr.len);
  if (result != commSuccess) {
    TC_LOG(WARNING) << "TorchCommMCCL: Failed to globally register memory"
                    << " (addr=" << addr.addr << ", len=" << addr.len << ")";
  }
}

void TorchCommMCCL::global_deregister_address(
    const TorchCommMCCL::AddressWithLen& addr) {
  commResult_t result = ::mccl::globalDeregister(addr.addr, addr.len);
  if (result != commSuccess) {
    TC_LOG(WARNING) << "TorchCommMCCL: Failed to globally deregister memory"
                    << " (addr=" << addr.addr << ", len=" << addr.len << ")";
  }
}

void TorchCommMCCL::initStaticRegime(
    const std::string& initUrl,
    const std::chrono::milliseconds& initTimeout) {
  // TODO: Replace the uuid with a different variable as commHash to simplify
  // the logics. For example, use min(bootStrapUrl)
  auto uuid = "0";

  std::unordered_map<std::string, std::string> kvPairs;
  if (!commName_.empty()) {
    kvPairs["commDesc"] = commName_;
  }

  TC_LOG(INFO) << "TorchCommMCCL::initStaticRegime using initMode: "
               << configs_.initMode_;

  ::mccl::InitOpts opts;
  opts.uuid = uuid;
  opts.timeout = initTimeout;
  opts.kvPairs = std::move(kvPairs);

  if (configs_.initMode_ == "ring") {
    opts.urls = exchangeRingUrls(initUrl);
  } else {
    // Default: all-to-all exchange (existing behavior)
    auto allUrls = exchangeUrls(initUrl);
    CHECK_THROW(allUrls.size() == commSize_, std::runtime_error);
    opts.urls = std::move(allUrls);
  }

  auto initWorkHandle = mccl_comm_->init(opts);
  initWorkHandle->waitCpu();
  auto initResult = initWorkHandle->getResult();
  CHECK_THROW(
      initResult->code == commResult_t::commSuccess, std::runtime_error);
  uuid_ = uuid;
}

std::string TorchCommMCCL::getInitURL() const {
  return mccl_comm_->getInitURL();
}

std::unordered_map<std::string, ::mccl::CollectiveStat>
TorchCommMCCL::getAndClearCollectiveStats() {
  if (!mccl_comm_) {
    return {};
  }
  return mccl_comm_->getAndClearCollectiveStats();
}

std::optional<uint64_t> TorchCommMCCL::getLifecycleCommId() const {
  if (!mccl_comm_) {
    return std::nullopt;
  }
  return mccl_comm_->getLifecycleCommId();
}

std::optional<uint64_t> TorchCommMCCL::getLatestLifecycleCollectiveId() const {
  if (!mccl_comm_) {
    return std::nullopt;
  }
  return mccl_comm_->getLatestLifecycleCollectiveId();
}

std::vector<::mccl::LifecycleEvent> TorchCommMCCL::drainLifecycleEvents() {
  if (!mccl_comm_) {
    return {};
  }
  return mccl_comm_->drainLifecycleEvents();
}

InitHandle TorchCommMCCL::getInitHandle() const {
  return getInitURL();
}

void TorchCommMCCL::abort() {
  abort(AbortInfo{});
}

void TorchCommMCCL::abort(const AbortInfo& info) {
  validateTerminalAbortReason(info.reason);
  if (mccl_comm_) {
    mccl_comm_->abort(info);
  }
}

bool TorchCommMCCL::isAbortSupported() const {
  return mccl_comm_ && mccl_comm_->isAbortSupported();
}

bool TorchCommMCCL::isAborted() const {
  return mccl_comm_ && mccl_comm_->isAborted();
}

std::optional<AbortInfo> TorchCommMCCL::getAbortInfo() const {
  if (!mccl_comm_) {
    return std::nullopt;
  }
  return mccl_comm_->getAbortInfo();
}

void TorchCommMCCL::setTimeout(std::chrono::milliseconds duration) {
  if (mccl_comm_) {
    mccl_comm_->setTimeout(duration);
  }
}

std::optional<std::chrono::milliseconds> TorchCommMCCL::getTimeout() const {
  if (!mccl_comm_) {
    return std::nullopt;
  }
  return mccl_comm_->getTimeout();
}

void TorchCommMCCL::setHints(
    std::unordered_map<std::string, std::string> hints) {
  if (mccl_comm_) {
    mccl_comm_->setHints(std::move(hints));
  }
}

void TorchCommMCCL::checkInitializedImpl(const char* file, int line) const {
  if (initState_ != InitializationState::INITIALIZED) {
    throw std::runtime_error(
        fmt::format(
            "TorchCommMCCL not initialized. "
            "In static regime initialization happens at create time. "
            "In dynamic regime, call reconfigure() first. "
            "[{}:{}]",
            file,
            line));
  }
}

void TorchCommMCCL::cleanInternalStore() {
  store_.reset();
  // Delete the internal store object and do a barrier to ensure that all
  // processes have deleted their store object too.  This way, when we
  // create the next torchcomm, we can use the same port to create a new store
  // object.
  // TODO: replace this with this->barrier() once we expose the blocking wait
  // API to torchcomm.

  // This must be a fault tolerant barrier (collective) to support the exception
  // handling path. Some application exceptions may be handled with an attempt
  // to properly clean up, in which cases, the peers are not guaranteed to join.
  ::mccl::AllReduceOpts opts = ::mccl::AllReduceOpts{
      .dataSend = barrierBuffer_,
      .dataRecv = barrierBuffer_,
      .numElements = kMCCLBarrierBufferSize,
      .dataType = commDataType_t::commFloat32,
      .reduceOpType = commSum,
      .stream = getOperationStream(false),
      .timeout = options_.timeout};
  TC_LOG(INFO, this) << "Start barrier AllReduce at rank" << rank_;
  auto mcclWorkHandle = mccl_comm_->allReduce(opts);
  mcclWorkHandle->waitCpu();
  TC_LOG(INFO, this) << "Finish barrier AllReduce at rank" << rank_;
}

// The timeout thread is based on TorchCommNCCLx timeout thread. It will poll
// status of the works in workq and set corresponding comm status based on
// mccl work result
void TorchCommMCCL::startTimeoutWatchdog() {
  auto weakComm = weak_from_this();
  if (weakComm.expired()) {
    TC_LOG(ERROR, this)
        << "TorchCommMCCL is not owned by a shared_ptr; timeout watchdog not started";
    return;
  }
  timeout_thread_ = std::thread(
      &TorchCommMCCL::timeoutWatchdog,
      watchdog_control_,
      std::move(weakComm),
      std::chrono::milliseconds(configs_.garbage_collect_interval_ms_));
}

void TorchCommMCCL::signalWatchdogShutdown() {
  // Set under the mutex so a concurrent wait cannot miss the notification and
  // sleep out the full interval.
  {
    std::lock_guard<std::mutex> lock(watchdog_control_->mutex);
    watchdog_control_->shutdown.store(true);
  }
  watchdog_control_->cv.notify_all();
}

void TorchCommMCCL::prepareWatchdogThread() {
  // checkAll() queries CUDA events from this thread while the main thread may
  // be capturing a CUDA graph. Pin the device (avoid creating a primary context
  // on device 0) and switch to thread-local capture mode so these queries do
  // not interact with — or block on — the main thread's global-mode capture.
  // Without this, checkAll() can block on a query while holding the tracker's
  // mutex_ while the capturing thread blocks on that mutex_ in
  // initOnGraphStart(), deadlocking (observed during reconfigure + recapture).
  CUDA_CHECK_IGNORE(
      cuda_api_,
      cuda_api_->setDevice(device_.index()),
      "Failed to set CUDA device in timeout thread");
  // `mode` receives this thread's previous (default) capture mode, which we
  // intentionally discard and never restore: capture mode is thread-local, and
  // this watchdog thread runs only for the comm's lifetime and never captures a
  // graph, so there is no caller to restore it for.
  cudaStreamCaptureMode mode = cudaStreamCaptureModeThreadLocal;
  CUDA_CHECK_IGNORE(
      cuda_api_,
      cuda_api_->threadExchangeStreamCaptureMode(&mode),
      "Failed to swap capture mode for timeout thread");
}

void TorchCommMCCL::watchdogIteration() {
  // Check work objects for completion or timeout.
  checkWorkQueue();

  // Reconfigurable comms must survive a timeout/error so the application can
  // abort the comm, reconfigure to a new quorum, and recover in place rather
  // than crash the whole process.
  if (commState_ == CommState::NORMAL ||
      !options_.abort_process_on_timeout_or_error ||
      options_.enable_reconfigure) {
    return;
  }

  // One-shot, and below the guard above so a healthy iteration cannot consume
  // it. commState_ has no path back to NORMAL, so this is what stops the abort
  // hooks re-running every garbage-collect interval.
  if (failureHandled_.exchange(true)) {
    return;
  }

  if (commState_ == CommState::TIMEOUT) {
    TC_LOG(ERROR, this) << kWatchdogCommAbortToken
                        << " aborting the communicator due to timeout on rank "
                        << rank_
                        << " - timeout watchdog detected operation timeout";
  } else if (commState_ == CommState::ERROR) {
    TC_LOG(ERROR, this) << kWatchdogCommAbortToken
                        << " aborting the communicator due to error on rank "
                        << rank_
                        << " - timeout watchdog detected operation error";
  }

  runAbortHooks();

  // Communicator abort, not std::abort().
  //
  // TODO(T284527221): abort_process_on_timeout_or_error promises process
  // termination, which this does not do.
  this->abort();
}

void TorchCommMCCL::timeoutWatchdog(
    std::shared_ptr<WatchdogControl> control,
    std::weak_ptr<TorchCommMCCL> weakComm,
    std::chrono::milliseconds interval) noexcept {
  {
    auto comm = weakComm.lock();
    if (!comm) {
      return;
    }
    TC_LOG(INFO) << "Timeout thread starting for rank: " << comm->rank_;
    comm->prepareWatchdogThread();
  }

  while (!control->shutdown.load()) {
    {
      std::unique_lock<std::mutex> lock(control->mutex);
      control->cv.wait_for(
          lock, interval, [&control]() { return control->shutdown.load(); });

      if (control->shutdown.load()) {
        TC_LOG(INFO) << "Shutting down timeout thread";
        break;
      }
    }

    // Held for this iteration only; owning it across the wait is what let the
    // watchdog destroy the comm underneath itself.
    auto comm = weakComm.lock();
    if (!comm) {
      break;
    }

    comm->watchdogIteration();

    // May run ~TorchCommMCCL on this thread. No comm access past this point.
    comm.reset();
  }

  TC_LOG(INFO) << "Watchdog thread exiting";
}

void TorchCommMCCL::checkWorkQueue() {
  auto status = workq_.garbageCollect();
  switch (status) {
    case TorchWorkMCCL::WorkStatus::TIMEDOUT:
      commState_ = CommState::TIMEOUT;
      break;
    case TorchWorkMCCL::WorkStatus::ERROR:
      commState_ = CommState::ERROR;
      break;
    default:
      break;
  }

  // Poll CUDA-graph-captured collectives to fire per-replay clog S/E events
  // (the eager work queue above cannot see captured ops, which replay as a
  // unit). Observability-only: MCCL collectives self-timeout on-device and set
  // a comm-level abort flag, so the tracker records each collective's timeout
  // but does not enforce it and never reports TIMEOUT — it does not drive comm
  // abort. A genuine CUDA error from the tracker's own event queries is still
  // surfaced.
  switch (graph_event_tracker_.checkAll()) {
    case McclGraphEventTracker::CheckResult::ERROR:
      // Don't downgrade a TIMEOUT already latched by garbageCollect() above: a
      // real timeout is more actionable than a generic tracker error, and both
      // drive the same abort.
      if (commState_ == CommState::NORMAL) {
        commState_ = CommState::ERROR;
      }
      break;
    case McclGraphEventTracker::CheckResult::TIMEOUT:
    case McclGraphEventTracker::CheckResult::OK:
      break;
  }
}

namespace {
class MCCLRegistration {
 public:
  MCCLRegistration() {
    TorchCommFactory::get().register_backend(
        "mccl", []() { return std::make_shared<TorchCommMCCL>(); });

    // Register a VMM-backed CUDA allocator factory for the "mccl" backend so
    // torchcomms.get_mem_allocator("mccl") returns an allocator whose memory
    // Ctran can register (persistent collectives, windows/RMA). Mirrors the
    // NCCLX/NCCL/RCCLX backends, using the same commCudaMalloc/commCudaFree VMM
    // path the MCCL adapter routes ncclMemAlloc/ncclMemFree through.
    TorchCommFactory::get().register_allocator_factory("mccl", []() {
      static std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>
          mccl_allocator =
              torch::cuda::CUDAPluggableAllocator::createCustomAllocator(
                  // alloc_fn
                  [](size_t size, int device, cudaStream_t /* stream */) {
                    at::cuda::OptionalCUDAGuard gpuGuard(device);
                    ::mccl::initLib();
                    TORCH_CHECK(
                        ctran::utils::commCudaLibraryInit() == commSuccess,
                        "MCCL mem allocator: commCudaLibraryInit failed");
                    meta::comms::StreamCaptureModeGuard captureGuard{
                        cudaStreamCaptureModeRelaxed};
                    char* ptr = nullptr;
                    commResult_t result = ctran::utils::commCudaMalloc(
                        &ptr, size, nullptr, "torchcomms::mccl::memAlloc");
                    TORCH_CHECK(
                        result == commSuccess,
                        "MCCL mem allocator: commCudaMalloc failed");
                    return static_cast<void*>(ptr);
                  },
                  // free_fn
                  [](void* ptr,
                     size_t /* size */,
                     int device,
                     cudaStream_t /* stream */) {
                    at::cuda::OptionalCUDAGuard gpuGuard(device);
                    meta::comms::StreamCaptureModeGuard captureGuard{
                        cudaStreamCaptureModeRelaxed};
                    commResult_t result = ctran::utils::commCudaFree(
                        static_cast<char*>(ptr), nullptr);
                    TORCH_CHECK(
                        result == commSuccess,
                        "MCCL mem allocator: commCudaFree failed");
                  });
      return mccl_allocator;
    });
  }
};

static const MCCLRegistration registration{};
} // namespace

} // namespace torch::comms
