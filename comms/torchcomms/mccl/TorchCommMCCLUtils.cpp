// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/torchcomms/mccl/TorchCommMCCLUtils.hpp"
#include <fmt/core.h>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp> // @manual=//caffe2:torch-cpp-cpu
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include "comms/torchcomms/mccl/TorchCommMCCLCCA.hpp"
#include "comms/torchcomms/mccl/TorchWorkMCCL.hpp"
#include "comms/torchcomms/utils/StoreManager.hpp"

namespace torch::comms {
#define TORCHCOMM_MCCL_CHECK(_status, _error_msg) \
  TORCH_CHECK(_status, fmt::format("TorchCommMCCL: {}", _error_msg))

namespace {
inline void validateCUDATensor(
    const at::Tensor& inputTensor,
    int cudaDeviceId) {
  TORCHCOMM_MCCL_CHECK(
      inputTensor.is_contiguous(),
      "TorchCommMCCL only supports contiguous tensor");
  // TODO: only support cuda tensor on the same device
  TORCHCOMM_MCCL_CHECK(
      inputTensor.device().type() == at::kCUDA &&
          inputTensor.device().index() == cudaDeviceId,
      fmt::format(
          "TorchCommMCCL only supports CUDA tensors: tensor device {} - CUDA device {}",
          inputTensor.device().index(),
          cudaDeviceId));
}

} // namespace

commRedOp_t TorchCommMCCL::torchToCommRedOp(const ReduceOp& op) {
  switch (op) {
    case ReduceOp::RedOpType::SUM:
      return commRedOp_t::commSum;
    case ReduceOp::RedOpType::PRODUCT:
      return commRedOp_t::commProd;
    case ReduceOp::RedOpType::MIN:
      return commRedOp_t::commMin;
    case ReduceOp::RedOpType::MAX:
      return commRedOp_t::commMax;
    case ReduceOp::RedOpType::AVG:
      return commRedOp_t::commAvg;
    default:
      throw std::invalid_argument(
          "TorchCommMCCL: unsupported reduce op by mccl");
  }
}

void* mccl::getTensorDataPtr(const at::Tensor& inputTensor) {
  const auto dataType = inputTensor.dtype().toScalarType();
  switch (dataType) {
    case at::kChar:
      return (void*)inputTensor.data_ptr<int8_t>();
    case at::kByte:
      return (void*)inputTensor.data_ptr<uint8_t>();
    case at::kInt:
      return (void*)inputTensor.data_ptr<int32_t>();
    case at::kUInt32:
      return (void*)inputTensor.data_ptr<uint32_t>();
    case at::kLong:
      return (void*)inputTensor.data_ptr<int64_t>();
    case at::kUInt64:
      return (void*)inputTensor.data_ptr<uint64_t>();
    case at::kHalf:
      return (void*)inputTensor.data_ptr<c10::Half>();
    case at::kFloat:
      return (void*)inputTensor.data_ptr<float>();
    case at::kDouble:
      return (void*)inputTensor.data_ptr<double>();
    case at::kBFloat16:
      return (void*)inputTensor.data_ptr<c10::BFloat16>();
    case at::kFloat8_e4m3fn:
      return (void*)inputTensor.data_ptr<c10::Float8_e4m3fn>();
    case at::kFloat8_e5m2:
      return (void*)inputTensor.data_ptr<c10::Float8_e5m2>();
    default:
      throw std::invalid_argument(
          fmt::format("Unsupported dataType: {}", int(dataType)));
  }
}

void TorchCommMCCL::setTCPStore(c10::intrusive_ptr<c10d::Store> store) {
  if (store == nullptr) {
    // Create a temporary TCPStore through StoreManager
    bool is_tcp_store_enabled =
        std::getenv("MASTER_ADDR") && std::getenv("MASTER_PORT");
    if (!is_tcp_store_enabled) {
      throw std::runtime_error("No way to exchange unique ID");
    }
    store_ = createPrefixStore(commName_, options_.timeout);

    createdInternalStore_ = true;
  } else {
    bool isTcpStore = [&store]() {
      if (auto prefixStore =
              c10::dynamic_intrusive_pointer_cast<c10d::PrefixStore>(store)) {
        return c10::dynamic_intrusive_pointer_cast<c10d::TCPStore>(
                   prefixStore->getUnderlyingNonPrefixStore()) != nullptr;
      }
      return c10::dynamic_intrusive_pointer_cast<c10d::TCPStore>(store) !=
          nullptr;
    }();
    if (!isTcpStore) {
      throw std::invalid_argument("TcpStore is required for MCCL init");
    }
    // Clone the store to create an independent connection to the same
    // backing store server, ensuring thread-safe access from this
    // communicator without sharing socket state with the caller.
    store_ = c10::make_intrusive<c10d::PrefixStore>(
        fmt::format("torchcomm(backend={},name={})", kBackendName, commName_),
        store->clone());
  }
}

std::vector<::mccl::InitURL> TorchCommMCCL::exchangeUrls(
    const ::mccl::InitURL& url) {
  std::vector<::mccl::InitURL> urls(commSize_);

  auto key = fmt::format("mccl_setUrlKey_{}_{}", uuid_, rank_);
  store_->set(key, url);

  std::vector<std::string> keys;
  keys.reserve(commSize_);
  for (int i = 0; i < commSize_; i++) {
    keys.push_back(fmt::format("mccl_setUrlKey_{}_{}", uuid_, i));
  }

  store_->wait(keys, options_.timeout);
  auto values = store_->multiGet(keys);
  TORCHCOMM_MCCL_CHECK(
      values.size() == static_cast<size_t>(commSize_),
      fmt::format(
          "Expected {} results from multiGet, got {}",
          commSize_,
          values.size()));

  for (int i = 0; i < commSize_; i++) {
    urls[i] = std::string(values.at(i).begin(), values.at(i).end());
  }
  return urls;
}

::mccl::RingInitInfo TorchCommMCCL::exchangeRingUrls(
    const ::mccl::InitURL& url) {
  // Publish own URL to store (1 SET)
  auto key = fmt::format("mccl_setUrlKey_{}_{}", uuid_, rank_);
  store_->set(key, url);

  // Compute ring neighbors
  size_t myRank = static_cast<size_t>(rank_);
  size_t nRanks = static_cast<size_t>(commSize_);
  size_t left = (myRank + nRanks - 1) % nRanks;
  size_t right = (myRank + 1) % nRanks;

  // Fetch only 2 neighbor URLs (1 batched WAIT + 1 multiGET)
  auto leftKey = fmt::format("mccl_setUrlKey_{}_{}", uuid_, left);
  auto rightKey = fmt::format("mccl_setUrlKey_{}_{}", uuid_, right);
  std::vector<std::string> neighborKeys = {
      std::move(leftKey), std::move(rightKey)};
  store_->wait(neighborKeys, options_.timeout);
  auto results = store_->multiGet(neighborKeys);
  TORCHCOMM_MCCL_CHECK(
      results.size() == 2,
      fmt::format("Expected 2 results from multiGet, got {}", results.size()));
  auto& leftVec = results[0];
  auto& rightVec = results[1];

  return ::mccl::RingInitInfo{
      .myRank = myRank,
      .nRanks = nRanks,
      .leftNeighborUrl = std::string(leftVec.begin(), leftVec.end()),
      .rightNeighborUrl = std::string(rightVec.begin(), rightVec.end()),
  };
}

bool TorchCommMCCL::isGraphCapturing() const {
  cudaStream_t currentStream = cuda_api_->getCurrentCUDAStream(device_.index());
  cudaStreamCaptureStatus captureStatus;
  CUDA_CHECK(
      cuda_api_,
      cuda_api_->streamIsCapturing(currentStream, &captureStatus),
      "Failed to query stream capture status");
  return captureStatus == cudaStreamCaptureStatusActive;
}

cudaStream_t TorchCommMCCL::getOperationStream(bool async_op) {
  cudaStream_t stream;
  if (async_op) {
    // Get current PyTorch CUDA stream for this device
    cudaStream_t currentStream =
        cuda_api_->getCurrentCUDAStream(device_.index());

    // Record event on current stream and wait for it on internal stream
    CUDA_CHECK(
        cuda_api_,
        cuda_api_->eventRecord(dependencyEvent_, currentStream),
        "Failed to record dependency event");

    CUDA_CHECK(
        cuda_api_,
        cuda_api_->streamWaitEvent(internalStream_, dependencyEvent_, 0),
        "Failed to make internal stream wait for dependency event");

    stream = internalStream_;
  } else {
    // Use the current PyTorch CUDA stream for synchronous operations
    stream = cuda_api_->getCurrentCUDAStream(device_.index());
  }

  // Record the graph-clog start event on the op stream BEFORE the collective is
  // issued, so the start/end events bracket the collective and the watchdog
  // reports precise per-replay S/E timing. No-op unless graph monitoring is
  // enabled and `stream` is actively capturing. getOperationStream() is the
  // single pre-collective choke point for GPU collectives (CPU paths use
  // internalStream_ directly and are not tracked).
  maybeRecordGraphCollectiveStart(stream);
  return stream;
}

std::optional<std::chrono::milliseconds> TorchCommMCCL::getOperationTimeout(
    std::chrono::milliseconds timeout) {
  if (timeout == kNoTimeout) {
    return std::nullopt;
  }
  return timeout;
}

c10::intrusive_ptr<TorchWorkMCCL> TorchCommMCCL::createWork(
    cudaStream_t stream,
    std::unique_ptr<::mccl::IWorkHandle> mcclWorkHandle,
    const std::vector<at::Tensor>& inputTensors,
    const std::vector<at::Tensor>& outputTensors) {
  // Only create the work object without enqueuing it
  return c10::make_intrusive<TorchWorkMCCL>(
      shared_from_this(),
      stream,
      inputTensors,
      outputTensors,
      std::move(mcclWorkHandle),
      std::chrono::milliseconds(configs_.watchdog_timeout_ms_));
}

c10::intrusive_ptr<TorchWorkMCCL> TorchCommMCCL::createWork(
    cudaStream_t stream,
    std::unique_ptr<::mccl::IWorkHandle> mcclWorkHandle,
    const at::Tensor& inputTensor,
    const at::Tensor& outputTensor) {
  return c10::make_intrusive<TorchWorkMCCL>(
      shared_from_this(),
      stream,
      inputTensor,
      outputTensor,
      std::move(mcclWorkHandle),
      std::chrono::milliseconds(configs_.watchdog_timeout_ms_));
}

void TorchCommMCCL::enqueueWork(
    c10::intrusive_ptr<TorchWorkMCCL> work,
    cudaStream_t stream) {
  // During graph capture, skip work queue — per-operation tracking is not
  // applicable for captured graphs (they are replayed as a unit). Instead,
  // register the captured collective with the graph event tracker so clog
  // graph-replay events fire on each replay.
  if (!work->isCpuWork() && isGraphCapturing()) {
    // Observability-only: finish registering the captured collective. The start
    // event was recorded before the collective in getOperationStream() →
    // maybeRecordGraphCollectiveStart(); here (after the collective) we record
    // the end event and hand the bracketing pair to the tracker so it emits
    // precise per-replay clog S/E events. The recorded timeout is NOT enforced
    // (MCCL self-times-out on-device).
    maybeRecordGraphCollectiveEnd(stream);
    return;
  }
  workq_.enqueueWork(std::move(work), stream);
}

void TorchCommMCCL::maybeRecordGraphCollectiveStart(cudaStream_t stream) {
  if (!isMcclGraphTimeoutMonitoringEnabled()) {
    return;
  }
  // Sets up the per-graph state (replay counter) on first call; returns false
  // if `stream` is not actively being captured. Done before the collective so
  // the counter increment lands upstream of work.wait()'s join point.
  if (!graph_event_tracker_.initOnGraphStart(stream)) {
    return;
  }

  // Release an orphaned bracketing pair left on this stream by a prior
  // collective that recorded a Start but never reached
  // maybeRecordGraphCollectiveEnd() (e.g. it threw in between). Otherwise the
  // assignment below would overwrite and leak it.
  if (auto pit = pending_graph_events_.find(stream);
      pit != pending_graph_events_.end()) {
    graph_event_tracker_.releaseEvent(pit->second.first);
    graph_event_tracker_.releaseEvent(pit->second.second);
    pending_graph_events_.erase(pit);
  }

  // Acquire the bracketing start/end events from the tracker's reused pool.
  // Both use cudaEventRecordExternal so they stay host-queryable during replay.
  // The start event is recorded HERE (before the collective is issued on
  // `stream`); the end event is recorded after the collective in
  // maybeRecordGraphCollectiveEnd(). Recording start-before / end-after makes
  // the events bracket the collective, so the watchdog observes the collective
  // in flight (start done, end pending) and reports precise per-replay S/E.
  cudaEvent_t start_event = nullptr;
  cudaEvent_t end_event = nullptr;
  // Return the events to the pool if any step below throws before they are
  // handed to pending_graph_events_ — until then they are untracked, so the
  // destructor's orphan cleanup would not reclaim them.
  SCOPE_FAIL {
    if (start_event != nullptr) {
      graph_event_tracker_.releaseEvent(start_event);
    }
    if (end_event != nullptr) {
      graph_event_tracker_.releaseEvent(end_event);
    }
  };
  CUDA_CHECK(
      cuda_api_,
      graph_event_tracker_.acquireEvent(start_event),
      "Failed to acquire graph clog start event");
  CUDA_CHECK(
      cuda_api_,
      graph_event_tracker_.acquireEvent(end_event),
      "Failed to acquire graph clog end event");
  // acquireEvent always sets a non-null handle on success; make the invariant
  // explicit (the CUDA_CHECKs above abort on failure).
  TORCH_CHECK(
      start_event != nullptr && end_event != nullptr,
      "graph clog events unexpectedly null after acquire");

  // Record start via the graph-monitor side stream (fork from `stream`, record,
  // rejoin). Recording directly on the op stream would leave an unjoined tail
  // past work.wait()'s join point and fail capture with
  // cudaErrorStreamCaptureUnjoined.
  cudaError_t record_err = cudaSuccess;
  CUDA_CHECK(
      cuda_api_,
      forkGraphMonitorSideStream(
          stream,
          [this, start_event, &record_err](cudaStream_t s) {
            record_err = cuda_api_->eventRecordWithFlags(
                start_event, s, cudaEventRecordExternal);
          }),
      "Failed to fork side stream for graph clog start event");
  CUDA_CHECK(cuda_api_, record_err, "Failed to record graph clog start event");

  // Hand off to maybeRecordGraphCollectiveEnd() (called from enqueueWork after
  // the collective is issued). Op calls are serialized per comm, so at most one
  // pending entry exists per stream at a time.
  pending_graph_events_[stream] = {start_event, end_event};
}

void TorchCommMCCL::maybeRecordGraphCollectiveEnd(cudaStream_t stream) {
  auto it = pending_graph_events_.find(stream);
  if (it == pending_graph_events_.end()) {
    // No start was recorded (monitoring off, not capturing, or a path that did
    // not route through getOperationStream()).
    return;
  }
  cudaEvent_t start_event = it->second.first;
  cudaEvent_t end_event = it->second.second;
  pending_graph_events_.erase(it);

  // Once erased from the map the events are untracked until addEntry() hands
  // them to the tracker; return them to the pool if a step below throws first,
  // since the destructor's orphan cleanup would not reclaim them.
  SCOPE_FAIL {
    graph_event_tracker_.releaseEvent(start_event);
    graph_event_tracker_.releaseEvent(end_event);
  };

  // Record end AFTER the collective (via the side stream), bracketing it.
  cudaError_t record_err = cudaSuccess;
  CUDA_CHECK(
      cuda_api_,
      forkGraphMonitorSideStream(
          stream,
          [this, end_event, &record_err](cudaStream_t s) {
            record_err = cuda_api_->eventRecordWithFlags(
                end_event, s, cudaEventRecordExternal);
          }),
      "Failed to fork side stream for graph clog end event");
  CUDA_CHECK(cuda_api_, record_err, "Failed to record graph clog end event");

  // Record the collective's effective MCCL timeout for observability. MCCL uses
  // the per-op opts.timeout when set, otherwise the comm-level default queried
  // here; captured collectives (e.g. PAFT FTAR) use the comm-level default. A
  // per-op override is not visible at this funnel, so the comm-level value is
  // recorded (-1 ms if unset). The tracker does not enforce it.
  const std::chrono::milliseconds tracked_timeout =
      getTimeout().value_or(std::chrono::milliseconds(-1));
  graph_event_tracker_.addEntry(
      stream, start_event, end_event, tracked_timeout);
}

cudaError_t TorchCommMCCL::forkGraphMonitorSideStream(
    cudaStream_t stream,
    std::function<void(cudaStream_t)> fn) {
  if (graph_monitor_side_stream_) {
    return graph_monitor_side_stream_->fork_from(stream, std::move(fn));
  }
  // No side stream (monitoring disabled) — run directly on the stream.
  fn(stream);
  return cudaSuccess;
}

::mccl::SendOpts TorchCommMCCL::getSendOpts(
    const at::Tensor& inputTensor,
    int cudaDeviceId,
    int p2pPeerRank) {
  validateCUDATensor(inputTensor, cudaDeviceId);
  auto dataType = torchToMcclType(inputTensor.dtype().toScalarType());
  auto dataPtr = torch::comms::mccl::getTensorDataPtr(inputTensor);
  size_t numel = inputTensor.numel();
  return ::mccl::SendOpts({
      .data = dataPtr,
      .numElements = numel,
      .dataType = dataType,
      .dstRank = p2pPeerRank,
  });
}

::mccl::RecvOpts TorchCommMCCL::getRecvOpts(
    const at::Tensor& outputTensor,
    int cudaDeviceId,
    int p2pPeerRank) {
  validateCUDATensor(outputTensor, cudaDeviceId);
  auto dataType = torchToMcclType(outputTensor.dtype().toScalarType());
  auto dataPtr = torch::comms::mccl::getTensorDataPtr(outputTensor);
  size_t numel = outputTensor.numel();
  return ::mccl::RecvOpts({
      .data = dataPtr,
      .numElements = numel,
      .dataType = dataType,
      .srcRank = p2pPeerRank,
  });
}

::mccl::SingleSendOpts TorchCommMCCL::getSingleSendOpts(
    const at::Tensor& inputTensor,
    int cudaDeviceId,
    cudaStream_t stream,
    int p2pPeerRank,
    const SendOptions& options) {
  const std::optional<std::chrono::milliseconds> operationTimeout =
      getOperationTimeout(options.timeout);

  return ::mccl::SingleSendOpts({
      .op = getSendOpts(inputTensor, cudaDeviceId, p2pPeerRank),
      .stream = stream,
      .timeout = operationTimeout,
      .kvPairs = options.hints,
  });
}

::mccl::SingleRecvOpts TorchCommMCCL::getSingleRecvOpts(
    const at::Tensor& inputTensor,
    int cudaDeviceId,
    cudaStream_t stream,
    int p2pPeerRank,
    const RecvOptions& options) {
  const std::optional<std::chrono::milliseconds> operationTimeout =
      getOperationTimeout(options.timeout);

  return ::mccl::SingleRecvOpts({
      .op = getRecvOpts(inputTensor, cudaDeviceId, p2pPeerRank),
      .stream = stream,
      .timeout = operationTimeout,
      .kvPairs = options.hints,
  });
}

::mccl::BatchOpts TorchCommMCCL::getBatchOpts(
    const std::vector<::mccl::SendRecvOpts>& optsList,
    cudaStream_t stream,
    const BatchP2POptions& options) {
  const std::optional<std::chrono::milliseconds> operationTimeout =
      getOperationTimeout(options.timeout);

  return ::mccl::BatchOpts({
      .optsList = optsList,
      .stream = stream,
      .timeout = operationTimeout,
      .kvPairs = options.hints,
  });
}

::mccl::AllReduceOpts TorchCommMCCL::getAllReduceOpts(
    const at::Tensor& inputTensor,
    int cudaDeviceId,
    cudaStream_t stream,
    const ReduceOp& op,
    const AllReduceOptions& options) {
  validateCUDATensor(inputTensor, cudaDeviceId);
  auto dataType = torchToMcclType(inputTensor.dtype().toScalarType());
  auto dataPtr = torch::comms::mccl::getTensorDataPtr(inputTensor);
  size_t numel = inputTensor.numel();
  const std::optional<std::chrono::milliseconds> operationTimeout =
      getOperationTimeout(options.timeout);
  return ::mccl::AllReduceOpts(
      {.dataSend = dataPtr,
       .dataRecv = dataPtr, // should always be in-place for TorchComms
       .numElements = numel,
       .dataType = dataType,
       .reduceOpType = torchToCommRedOp(op),
       .stream = stream,
       .timeout = operationTimeout,
       .kvPairs = options.hints});
}

::mccl::BroadcastOpts TorchCommMCCL::getBroadcastOpts(
    const at::Tensor& inputTensor,
    int cudaDeviceId,
    cudaStream_t stream,
    int rootRank,
    const BroadcastOptions& options) {
  validateCUDATensor(inputTensor, cudaDeviceId);
  auto dataType = torchToMcclType(inputTensor.dtype().toScalarType());
  auto dataPtr = torch::comms::mccl::getTensorDataPtr(inputTensor);
  size_t numel = inputTensor.numel();
  const std::optional<std::chrono::milliseconds> operationTimeout =
      getOperationTimeout(options.timeout);
  return ::mccl::BroadcastOpts(
      {.data = dataPtr,
       .numElements = numel,
       .dataType = dataType,
       .stream = stream,
       .root = rootRank,
       .timeout = operationTimeout,
       .kvPairs = options.hints});
}

::mccl::AllGatherPInitOpts TorchCommMCCL::getAllGatherPInitOpts(
    const at::Tensor& outputTensor,
    int cudaDeviceId,
    cudaStream_t stream,
    const AllGatherPInitOptions& options) {
  validateCUDATensor(outputTensor, cudaDeviceId);
  TORCH_CHECK(
      outputTensor.numel() > 0,
      "all_gather_p_init: receive tensor must be non-empty");

  // timeout/kvPairs are forwarded for cross-collective API symmetry; McclComm
  // does not read them at init today (see AllGatherPInitOpts in McclTypes.h).
  return ::mccl::AllGatherPInitOpts{
      .recvbuff = torch::comms::mccl::getTensorDataPtr(outputTensor),
      .maxRecvCount = static_cast<size_t>(outputTensor.numel()),
      .dataType = torchToMcclType(outputTensor.scalar_type()),
      .stream = stream,
      .timeout = getOperationTimeout(options.timeout),
      .kvPairs = options.hints};
}

::mccl::AllGatherPExecOpts TorchCommMCCL::getAllGatherPExecOpts(
    const at::Tensor& inputTensor,
    int cudaDeviceId,
    const AllGatherPExecOptions& /* options */) {
  validateCUDATensor(inputTensor, cudaDeviceId);
  TORCH_CHECK(
      inputTensor.numel() > 0,
      "all_gather_p_exec: send tensor must be non-empty");

  // AllGatherPExecOpts has no timeout/hints fields, so AllGatherPExecOptions'
  // timeout/hints are currently unsupported and ignored at exec time.
  return ::mccl::AllGatherPExecOpts{
      .sendbuff = torch::comms::mccl::getTensorDataPtr(inputTensor),
      .count = static_cast<size_t>(inputTensor.numel()),
      .dataType = torchToMcclType(inputTensor.scalar_type())};
}

::mccl::ReduceScatterOpts TorchCommMCCL::getReduceScatterOpts(
    const at::Tensor& outputTensor,
    const at::Tensor& inputTensor,
    int cudaDeviceId,
    cudaStream_t stream,
    const ReduceOp& op,
    const ReduceScatterSingleOptions& options) {
  validateCUDATensor(inputTensor, cudaDeviceId);
  validateCUDATensor(outputTensor, cudaDeviceId);

  auto dataType_in = torchToMcclType(inputTensor.dtype().toScalarType());
  auto dataType_out = torchToMcclType(outputTensor.dtype().toScalarType());
  if (dataType_in != dataType_out) {
    throw std::runtime_error(
        fmt::format(
            "Type mismatch between input and output tensors: {} vs {}",
            dataType_in,
            dataType_out));
  }

  if (inputTensor.numel() != outputTensor.numel() * commSize_) {
    throw std::runtime_error(
        "Input tensor size must be output_size * comm_size for reduce_scatter / reduce_scatter_single");
  }

  const std::optional<std::chrono::milliseconds> operationTimeout =
      getOperationTimeout(options.timeout);
  return ::mccl::ReduceScatterOpts(
      {.dataSend = torch::comms::mccl::getTensorDataPtr(inputTensor),
       .dataRecv = torch::comms::mccl::getTensorDataPtr(outputTensor),
       .numElements = static_cast<size_t>(outputTensor.numel()),
       .dataType = dataType_in,
       .reduceOpType = torchToCommRedOp(op),
       .stream = stream,
       .timeout = operationTimeout,
       .kvPairs = options.hints});
}

void TorchCommMCCL::attachMemoryHook() {
  if (!McclCachingAllocatorHook::isEnabled()) {
    return;
  }
  McclCachingAllocatorHook::getInstance().registerComm(this);
}

void TorchCommMCCL::detachMemoryHook() {
  if (!McclCachingAllocatorHook::isEnabled()) {
    return;
  }
  McclCachingAllocatorHook::getInstance().deregisterComm(this);
}

namespace mccl {

TorchCommMCCL::Configs parseOptions(const CommOptions& options) {
  TorchCommMCCL::Configs configs;
  // Use the first-class enable_reconfigure field from CommOptions.
  // Also support legacy hints["initDynamicRegime"] for backward compatibility.
  if (options.enable_reconfigure) {
    configs.initDynamicRegime_ = true;
  } else {
    configs.initDynamicRegime_ =
        options.getHint<bool>("initDynamicRegime", false);
  }
  configs.garbage_collect_interval_ms_ = options.getHint<size_t>(
      kHintGarbageCollectIntervalMs, kDefaultGarbageCollectIntervalMs);
  configs.watchdog_timeout_ms_ = options.getHint<size_t>(
      kHintWatchdogTimeoutMs, kDefaultWatchdogTimeoutMs);
  auto initModeKey = std::string(kHintInitMode);
  if (options.hints.contains(initModeKey)) {
    auto mode = options.hints.at(initModeKey);
    if (mode != "full_mesh" && mode != "ring") {
      throw std::invalid_argument(
          fmt::format(
              "Invalid initMode hint '{}': must be 'full_mesh' or 'ring'",
              mode));
    }
    configs.initMode_ = mode;
  }
  configs.useCpuBarrier_ = options.getHint<bool>(kHintUseCpuBarrier, false);
  return configs;
}

} // namespace mccl

} // namespace torch::comms
