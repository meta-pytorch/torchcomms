// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ATen/ATen.h>
#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy
#include <torch/csrc/distributed/c10d/ParamCommsUtils.hpp> // @manual=//caffe2:torch-cpp
#include <torch/csrc/distributed/c10d/Store.hpp> // @manual=//caffe2:torch-cpp
#include "comms/mccl/mccl.h"

#include "comms/torchcomms/TorchComm.hpp"
#include "comms/torchcomms/TorchCommBackend.hpp"
#include "comms/torchcomms/TorchCommTypes.hpp"
#include "comms/torchcomms/TorchWork.hpp"
#include "comms/torchcomms/device/cuda/CudaApi.hpp"
#include "comms/torchcomms/mccl/McclGraphEventTracker.hpp"
#include "comms/torchcomms/mccl/TorchWorkMCCL.hpp"
#include "comms/torchcomms/utils/Logging.hpp"
#include "comms/utils/GraphCaptureSideStream.h"

namespace torch::comms {

// Forward declarations for test friend classes
namespace test {
class TorchCommMCCLTest;
class TorchCommMCCLTest_CleanInternalStoreDestroysStoreAndBarriers_Test;
class TorchCommMCCLTest_CleanInternalStoreBarrierFailure_Test;
class TorchCommMCCLTest_DefaultAllReduceTimeoutLeavesMcclTimeoutUnset_Test;
class TorchCommMCCLTest_ExplicitAllReduceTimeoutIsPreserved_Test;
class TorchCommMCCLTest_QueuedWorkDoesNotKeepCommAlive_Test;
class GetReduceScatterOptsTest_MismatchedDtypesThrows_Test;
class GetReduceScatterOptsTest_WrongSizeRatioThrows_Test;
class GetReduceScatterOptsTest_CpuTensorThrows_Test;
class GetReduceScatterOptsTest_ValidInputsProducesCorrectOpts_Test;
class TorchCommMCCLTest_GetAllGatherPInitOptsForwardsFields_Test;
class TorchCommMCCLTest_GetAllGatherPInitOptsDefaultTimeoutUnset_Test;
class TorchCommMCCLTest_GetAllGatherPInitOptsRejectsInvalidTensor_Test;
class TorchCommMCCLTest_GetAllGatherPExecOptsForwardsFields_Test;
class TorchCommMCCLTest_GetAllGatherPExecOptsRejectsInvalidTensor_Test;
class TorchCommMCCLTest_AllGatherPInitAutoRegistersAndReturnsHandle_Test;
class TorchCommMCCLTest_AllGatherPInitThrowsOnFailure_Test;
class TorchCommMCCLTest_AllGatherPExecInsertsEventBridgeAndEnqueues_Test;
class TorchCommMCCLTest_AllGatherPFreeCallsFreeAndDoesNotDeregister_Test;
} // namespace test

constexpr uint16_t kTCPStorePort = 29501;
constexpr size_t kMCCLBarrierBufferSize = 1;
constexpr size_t kDefaultGarbageCollectIntervalMs = 100;
constexpr size_t kDefaultWatchdogTimeoutMs = 1800000; // 30 minutes

/*
 * TorchCommMCCL - MCCL backend implementation for TorchComm.
 *
 * Provides collective communication operations backed by the MCCL library,
 * with built-in fault tolerance support. The communicator uses a CUDA device,
 * while selected control-plane collectives also accept CPU tensors.
 *
 * Lifecycle:
 *   Static regime:  construct -> init() -> collectives -> finalize()
 *   Dynamic regime: construct -> reconfigure() -> collectives -> finalize()
 *   After failure (dynamic regime):
 *     reconfigure() -> collectives -> finalize()
 *
 * Thread Safety:
 *   API calls (collectives, init, finalize, reconfigure) are NOT
 *   thread-safe and must be serialized by the caller. Operation
 *   timeouts are monitored internally.
 *
 * Not all collective operations are implemented. The following throw
 * std::runtime_error("not implemented"):
 *   reduce, all_gather_v, reduce_scatter, reduce_scatter_v,
 *   reduce_scatter_single, scatter,
 *   new_window, gather (GPU tensors only - CPU gather is supported)
 *
 * List-form all_gather is supported through batched Send/Recv, and
 * all_gather_single is supported for CPU tensors through Bootstrap. CUDA
 * all_gather_single is unsupported.
 */
class TorchCommMCCL : public TorchCommBackend,
                      public std::enable_shared_from_this<TorchCommMCCL> {
 public:
  static constexpr std::string_view kBackendName = "mccl";
  const std::chrono::milliseconds kMcclCommTimeout =
      std::chrono::milliseconds(120000);
  TorchCommMCCL(std::shared_ptr<CudaApi> cuda_api = nullptr);

  // The mcclComm must be initialized before calling this constructor
  // TODO: Expose the mcclComm->getRank() and mcclComm->getSize() to remove
  // these args.
  TorchCommMCCL(
      std::unique_ptr<::mccl::IComm> mcclComm,
      int rank,
      int size,
      std::shared_ptr<CudaApi> cuda_api = nullptr);

  ~TorchCommMCCL() override;

  // Delete copy and move operations
  TorchCommMCCL(const TorchCommMCCL&) = delete;
  TorchCommMCCL(TorchCommMCCL&&) = delete;
  TorchCommMCCL& operator=(const TorchCommMCCL&) = delete;
  TorchCommMCCL& operator=(TorchCommMCCL&&) = delete;

  std::string_view getBackendName() const override;
  std::string_view getCommName() const override;
  const CommOptions& getOptions() const override;
  const at::Device& getDevice() const override;
  // Accessor for the CUDA API abstraction. Used by McclGraphEventTracker.
  CudaApi* getCudaApi() const {
    return cuda_api_.get();
  }
  int getRank() const override;
  int getSize() const override;
  bool isInitialized() const override;

  // Configuration parsed from CommOptions at init() time.
  // TODO: We need to consolidate the utility function
  // use in TorchCommMCCL. T250016523
  struct Configs {
    // Interval (ms) between internal cleanup passes for completed operations.
    // Hint key: "garbage_collect_interval_ms".
    size_t garbage_collect_interval_ms_{kDefaultGarbageCollectIntervalMs};

    // Timeout (ms) for detecting hung operations.
    // Hint key: "watchdog_timeout_ms".
    size_t watchdog_timeout_ms_{kDefaultWatchdogTimeoutMs};

    // Enable dynamic reconfiguration.
    // Required for fault tolerance support (but can be used without it).
    // Hint key: "initDynamicRegime" (legacy) or
    // CommOptions::enable_reconfigure.
    bool initDynamicRegime_{false};

    // Init topology: "full_mesh" (all ranks exchange URLs) or
    // "ring" (only neighbor URLs exchanged).
    // Hint key: "initMode".
    std::string initMode_{"full_mesh"};

    // Use CPU-based barrier instead of GPU barrier.
    // Hint key: "use_cpu_barrier".
    bool useCpuBarrier_{false};
  };

  void init(
      at::Device device,
      const std::string& name,
      const CommOptions& options = {}) override;

  c10::intrusive_ptr<TorchWork> reconfigure(
      const ReconfigureOptions& opts) override;

  void finalize() override;

  // Point-to-Point Operations
  c10::intrusive_ptr<TorchWork> send(
      const at::Tensor& tensor,
      int dst,
      bool async_op,
      const SendOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> recv(
      at::Tensor& tensor,
      int src,
      bool async_op,
      const RecvOptions& options = {}) override;

  // Batch P2P Operations
  c10::intrusive_ptr<TorchWork> batch_op_issue(
      const std::vector<BatchSendRecv::P2POp>& ops,
      bool async_op,
      const BatchP2POptions& options = {}) override;

  // Collective Operations
  c10::intrusive_ptr<TorchWork> broadcast(
      at::Tensor& tensor,
      int root,
      bool async_op,
      const BroadcastOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> all_reduce(
      at::Tensor& tensor,
      const ReduceOp& op,
      bool async_op,
      const AllReduceOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> reduce(
      const at::Tensor& tensor,
      int root,
      const ReduceOp& op,
      bool async_op,
      const ReduceOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> all_gather(
      const std::vector<at::Tensor>& tensor_list,
      const at::Tensor& tensor,
      bool async_op,
      const AllGatherOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> all_gather_v(
      const std::vector<at::Tensor>& tensor_list,
      const at::Tensor& tensor,
      bool async_op,
      const AllGatherOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> all_gather_single(
      at::Tensor& output,
      const at::Tensor& input,
      bool async_op,
      const AllGatherSingleOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> reduce_scatter(
      at::Tensor& output,
      const std::vector<at::Tensor>& input_list,
      const ReduceOp& op,
      bool async_op,
      const ReduceScatterOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> reduce_scatter_v(
      at::Tensor& output,
      const std::vector<at::Tensor>& input_list,
      const ReduceOp& op,
      bool async_op,
      const ReduceScatterOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> reduce_scatter_single(
      at::Tensor& output,
      const at::Tensor& input,
      const ReduceOp& op,
      bool async_op,
      const ReduceScatterSingleOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> all_to_all_single(
      at::Tensor& output,
      const at::Tensor& input,
      bool async_op,
      const AllToAllSingleOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> all_to_all_v_single(
      at::Tensor& output,
      const at::Tensor& input,
      const std::vector<uint64_t>& output_split_sizes,
      const std::vector<uint64_t>& input_split_sizes,
      bool async_op,
      const AllToAllvSingleOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> all_to_all(
      const std::vector<at::Tensor>& output_tensor_list,
      const std::vector<at::Tensor>& input_tensor_list,
      bool async_op,
      const AllToAllOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> barrier(
      bool async_op,
      const BarrierOptions& options = {}) override;

  // Scatter and Gather Operations
  c10::intrusive_ptr<TorchWork> scatter(
      at::Tensor& output_tensor,
      const std::vector<at::Tensor>& input_tensor_list,
      int root,
      bool async_op,
      const ScatterOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> gather(
      const std::vector<at::Tensor>& output_tensor_list,
      const at::Tensor& input_tensor,
      int root,
      bool async_op,
      const GatherOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> gather_single(
      at::Tensor& output,
      const at::Tensor& input,
      int root,
      bool async_op,
      const GatherSingleOptions& options = {}) override;

  // Persistent AllGather Operations
  AllGatherPHandle all_gather_p_init(
      at::Tensor& output,
      const AllGatherPInitOptions& options = {}) override;
  c10::intrusive_ptr<TorchWork> all_gather_p_exec(
      AllGatherPHandle handle,
      const at::Tensor& input,
      bool async_op,
      const AllGatherPExecOptions& options = {}) override;
  void all_gather_p_free(AllGatherPHandle handle) override;

  // Window & One-sided Operations
  std::shared_ptr<TorchCommWindow> new_window(
      const std::optional<at::Tensor>& tensor = std::nullopt) override;

  // Communicator Management
  std::shared_ptr<TorchCommBackend> split(
      const std::vector<int>& ranks,
      const std::string& name,
      const CommOptions& options = {}) override;

  // Get the init URL from MCCL communicator (for Python bindings)
  std::string getInitURL() const;

  // MCCL-only: drain per-(collective_op.msg_size) collective timing stats.
  std::unordered_map<std::string, ::mccl::CollectiveStat>
  getAndClearCollectiveStats();

  std::optional<uint64_t> getLifecycleCommId() const;
  std::optional<uint64_t> getLatestLifecycleCollectiveId() const;
  std::vector<::mccl::LifecycleEvent> drainLifecycleEvents();

  const std::string& getUuid() const {
    return uuid_;
  }

  // Fault Tolerance API override
  bool supportsReconfigure() const override {
    return true;
  }
  InitHandle getInitHandle() const override;

  void abort() override;
  void abort(const AbortInfo& info) override;
  bool isAbortSupported() const override;
  bool isAborted() const override;
  std::optional<AbortInfo> getAbortInfo() const override;

  // Per-op timeout fallback. Mutable across CUDA-graph replays without
  // recapture. Forwards to the underlying MCCL communicator.
  void setTimeout(std::chrono::milliseconds duration) override;

  std::optional<std::chrono::milliseconds> getTimeout() const;

  // Communicator-level key-value hints. Mutable across CUDA-graph replays
  // without recapture. Forwards to the underlying MCCL communicator.
  void setHints(std::unordered_map<std::string, std::string> hints) override;

  // Public TorchComm API — tensor-level registration
  void tensor_register(const at::Tensor& tensor) override;
  void tensor_deregister(const at::Tensor& tensor) override;

  // Friend access for TorchWorkMCCL
  friend class TorchWorkMCCL;
  friend class McclCachingAllocatorHookImpl;
  // Tracker calls the protected base TorchCommBackend::fireGraphReplayHook().
  friend class McclGraphEventTracker;

 public:
  // Identifies a memory buffer for deregistration.
  struct Address {
    void* addr;
  };

  // Describes a memory buffer for registration.
  struct AddressWithLen {
    void* addr;
    size_t len;
  };

  // Register a GPU memory buffer for optimized communication.
  // Throws std::runtime_error if the communicator is not initialized or
  //        if registration fails.
  void register_address(const AddressWithLen& addr);

  // Deregister a previously registered GPU memory buffer.
  // Throws std::runtime_error if the communicator is not initialized or
  //        if deregistration fails.
  void deregister_address(const Address& addr);

  // Global pointer-based registration that doesn't require a comm instance.
  // Used by McclCachingAllocatorHook for pre-comm memory registration.
  static void global_register_address(const AddressWithLen& addr);
  static void global_deregister_address(const AddressWithLen& addr);

 private:
  c10::intrusive_ptr<TorchWorkMCCL> createWork(
      cudaStream_t stream,
      std::unique_ptr<::mccl::IWorkHandle> mcclWorkHandle,
      const std::vector<at::Tensor>& inputTensors = {},
      const std::vector<at::Tensor>& outputTensors = {});

  c10::intrusive_ptr<TorchWorkMCCL> createWork(
      cudaStream_t stream,
      std::unique_ptr<::mccl::IWorkHandle> mcclWorkHandle,
      const at::Tensor& inputTensor,
      const at::Tensor& outputTensor);

  // Communicator health state
  enum class CommState {
    NORMAL,
    ERROR,
    TIMEOUT,
  };

  // Member variables
  at::Device device_;
  int commSize_{0};
  int rank_{-1};
  std::string commName_;
  CommOptions options_;
  enum class InitializationState {
    UNINITIALIZED,
    INITIALIZED,
    FINALIZED,
  } initState_ = InitializationState::UNINITIALIZED;
  c10::intrusive_ptr<c10d::Store> store_;
  bool createdInternalStore_ = false;
  cudaStream_t internalStream_{};
  cudaEvent_t dependencyEvent_{}; // Pre-allocated event for stream dependencies
  std::string uuid_;
  int cudaDeviceId_{};
  void* barrierBuffer_{}; // Pre-allocated CUDA buffer for barrier operations

  // Struct to hold the registration handle for a buffer
  struct RegistrationHandle {
    void* regHandle;

    explicit RegistrationHandle(void* regHandle) : regHandle{regHandle} {}

    RegistrationHandle(RegistrationHandle&& other) noexcept
        : regHandle{other.regHandle} {
      other.regHandle = nullptr;
    }

    RegistrationHandle(const RegistrationHandle&) = delete;
    RegistrationHandle& operator=(const RegistrationHandle&) = delete;
    RegistrationHandle& operator=(RegistrationHandle&&) = delete;

    ~RegistrationHandle() = default;
  };

  // List of [comm, regHandlesMap] pairs.  Each regHandlesMap is a map from the
  // buffer address to the registeration handle
  std::map<void*, RegistrationHandle> memoryRegistrationHandles_;

  // Work tracking per stream
  TorchWorkMCCLQueue workq_;

  // MCCL API abstraction
  std::shared_ptr<CudaApi> cuda_api_;

  std::unique_ptr<::mccl::IComm> mccl_comm_;

  // Separately owned so the watchdog can keep waiting on it after
  // ~TorchCommMCCL() has run.
  struct WatchdogControl {
    std::atomic<bool> shutdown{false};
    std::mutex mutex;
    std::condition_variable cv;
  };

  std::shared_ptr<WatchdogControl> watchdog_control_{
      std::make_shared<WatchdogControl>()};
  std::thread timeout_thread_;
  std::atomic<CommState> commState_{CommState::NORMAL};
  // Makes the watchdog's failure branch one-shot; commState_ has no path back
  // to NORMAL. Kept separate because encoding "handled" in commState_ would be
  // overwritten by checkWorkQueue()'s re-report of the retained terminal work.
  std::atomic<bool> failureHandled_{false};

  void startTimeoutWatchdog();
  void signalWatchdogShutdown();
  static void timeoutWatchdog(
      std::shared_ptr<WatchdogControl> control,
      std::weak_ptr<TorchCommMCCL> weakComm,
      std::chrono::milliseconds interval) noexcept;
  void prepareWatchdogThread();
  void watchdogIteration();
  void checkWorkQueue();
  void cleanInternalStore();

  // The fixture itself, so shared test helpers can reach private state. The
  // per-test friends below are still required: friendship is not inherited, so
  // this does not cover the derived TEST_F bodies.
  friend class test::TorchCommMCCLTest;
  friend class test::
      TorchCommMCCLTest_CleanInternalStoreDestroysStoreAndBarriers_Test;
  friend class test::TorchCommMCCLTest_CleanInternalStoreBarrierFailure_Test;
  friend class test::
      TorchCommMCCLTest_DefaultAllReduceTimeoutLeavesMcclTimeoutUnset_Test;
  friend class test::TorchCommMCCLTest_ExplicitAllReduceTimeoutIsPreserved_Test;
  friend class test::TorchCommMCCLTest_QueuedWorkDoesNotKeepCommAlive_Test;
  friend class test::GetReduceScatterOptsTest_MismatchedDtypesThrows_Test;
  friend class test::GetReduceScatterOptsTest_WrongSizeRatioThrows_Test;
  friend class test::GetReduceScatterOptsTest_CpuTensorThrows_Test;
  friend class test::
      GetReduceScatterOptsTest_ValidInputsProducesCorrectOpts_Test;
  friend class test::TorchCommMCCLTest_GetAllGatherPInitOptsForwardsFields_Test;
  friend class test::
      TorchCommMCCLTest_GetAllGatherPInitOptsDefaultTimeoutUnset_Test;
  friend class test::
      TorchCommMCCLTest_GetAllGatherPInitOptsRejectsInvalidTensor_Test;
  friend class test::TorchCommMCCLTest_GetAllGatherPExecOptsForwardsFields_Test;
  friend class test::
      TorchCommMCCLTest_GetAllGatherPExecOptsRejectsInvalidTensor_Test;
  friend class test::
      TorchCommMCCLTest_AllGatherPInitAutoRegistersAndReturnsHandle_Test;
  friend class test::TorchCommMCCLTest_AllGatherPInitThrowsOnFailure_Test;
  friend class test::
      TorchCommMCCLTest_AllGatherPExecInsertsEventBridgeAndEnqueues_Test;
  friend class test::
      TorchCommMCCLTest_AllGatherPFreeCallsFreeAndDoesNotDeregister_Test;

  Configs configs_;

  // Side stream used to host the graph-monitor's external EVENT_RECORD nodes
  // and the replay-counter increment during CUDA graph capture. Recording them
  // directly on the op stream (after the collective) would leave an unjoined
  // tail past work.wait()'s join point and fail capture with
  // cudaErrorStreamCaptureUnjoined; fork_from() forks + rejoins so the work
  // stays joined and off the main stream's critical path. Created at init()
  // (outside capture); only when monitoring is enabled.
  // Fully qualified (::meta) to disambiguate from at::meta, which becomes
  // visible when a TU also includes ATen's NativeMetaFunctions.h.
  std::unique_ptr<::meta::comms::GraphSideStream> graph_monitor_side_stream_;

  // Tracks CUDA-graph-captured collectives so clog graph-replay events fire on
  // each replay (and for in-graph timeout detection). Holds only a back-pointer
  // to this comm, so declaration order does not matter.
  McclGraphEventTracker graph_event_tracker_{this};

  // helper functions
  std::vector<::mccl::InitURL> exchangeUrls(const ::mccl::InitURL& urls);
  ::mccl::RingInitInfo exchangeRingUrls(const ::mccl::InitURL& url);

  commDataType_t torchToMcclType(const auto& dataType) {
    switch (dataType) {
      case at::kChar:
        return commDataType_t::commInt8;
      case at::kByte:
        return commDataType_t::commUint8;
      case at::kInt:
        return commDataType_t::commInt32;
      case at::kUInt32:
        return commDataType_t::commUint32;
      case at::kLong:
        return commDataType_t::commInt64;
      case at::kUInt64:
        return commDataType_t::commUint64;
      case at::kHalf:
        return commDataType_t::commHalf;
      case at::kFloat:
        return commDataType_t::commFloat32;
      case at::kDouble:
        return commDataType_t::commDouble;
      case at::kBFloat16:
        return commDataType_t::commBfloat16;
      case at::kFloat8_e4m3fn:
        return commDataType_t::commFloat8e4m3;
      case at::kFloat8_e5m2:
        return commDataType_t::commFloat8e5m2;
      default:
        throw std::invalid_argument(
            fmt::format("Unsupported dataType: {}", int(dataType)));
    }
  }

  commRedOp_t torchToCommRedOp(const ReduceOp& op);

  void setTCPStore(c10::intrusive_ptr<c10d::Store> store = nullptr);

  cudaStream_t getOperationStream(bool async_op);

  // Returns true if the current stream is being captured into a CUDA graph.
  bool isGraphCapturing() const;

  void enqueueWork(c10::intrusive_ptr<TorchWorkMCCL> work, cudaStream_t stream);

  // Graph-clog tracking is split so the start/end events bracket the collective
  // (precise per-replay S/E timing):
  //  - maybeRecordGraphCollectiveStart() runs in getOperationStream(), BEFORE
  //    the collective is issued: sets up per-graph state and records the start
  //    event, stashing the bracketing pair in pending_graph_events_.
  //  - maybeRecordGraphCollectiveEnd() runs in enqueueWork(), AFTER the
  //    collective: records the end event and hands the pair to the tracker.
  // Both are no-ops unless graph monitoring is enabled and `stream` is actively
  // capturing. The recorded timeout (comm-level getTimeout()) is observability
  // only — it is not enforced.
  void maybeRecordGraphCollectiveStart(cudaStream_t stream);
  void maybeRecordGraphCollectiveEnd(cudaStream_t stream);

  // Bracketing clog events recorded in maybeRecordGraphCollectiveStart() and
  // consumed in maybeRecordGraphCollectiveEnd(). Keyed by op stream; touched
  // only on the (serialized) collective-issuing thread, so no lock is needed.
  std::unordered_map<cudaStream_t, std::pair<cudaEvent_t, cudaEvent_t>>
      pending_graph_events_;

  // Run `fn` on the graph-monitor side stream wrapped in fork/rejoin
  // scaffolding (no-op fork when `stream` is not capturing). Used to record the
  // graph-monitor's captured work (replay-counter increment, external clog
  // events) so it rejoins the captured graph instead of dangling past
  // work.wait()'s join point. Falls back to running `fn(stream)` directly if no
  // side stream exists.
  cudaError_t forkGraphMonitorSideStream(
      cudaStream_t stream,
      std::function<void(cudaStream_t)> fn);

  std::optional<std::chrono::milliseconds> getOperationTimeout(
      std::chrono::milliseconds timeout);

  ::mccl::SingleSendOpts getSingleSendOpts(
      const at::Tensor& inputTensor,
      int cudaDeviceId,
      cudaStream_t stream,
      int p2pPeerRank,
      const SendOptions& options);

  ::mccl::SingleRecvOpts getSingleRecvOpts(
      const at::Tensor& inputTensor,
      int cudaDeviceId,
      cudaStream_t stream,
      int p2pPeerRank,
      const RecvOptions& options);

  ::mccl::SendOpts
  getSendOpts(const at::Tensor& inputTensor, int cudaDeviceId, int p2pPeerRank);

  ::mccl::RecvOpts
  getRecvOpts(const at::Tensor& inputTensor, int cudaDeviceId, int p2pPeerRank);

  ::mccl::BatchOpts getBatchOpts(
      const std::vector<::mccl::SendRecvOpts>& optsList,
      cudaStream_t stream,
      const BatchP2POptions& options);

  ::mccl::AllReduceOpts getAllReduceOpts(
      const at::Tensor& inputTensor,
      int cudaDeviceId,
      cudaStream_t stream,
      const ReduceOp& op,
      const AllReduceOptions& options);

  ::mccl::BroadcastOpts getBroadcastOpts(
      const at::Tensor& inputTensor,
      int cudaDeviceId,
      cudaStream_t stream,
      int rootRank,
      const BroadcastOptions& options);

  ::mccl::AllGatherPInitOpts getAllGatherPInitOpts(
      const at::Tensor& outputTensor,
      int cudaDeviceId,
      cudaStream_t stream,
      const AllGatherPInitOptions& options);

  ::mccl::AllGatherPExecOpts getAllGatherPExecOpts(
      const at::Tensor& inputTensor,
      int cudaDeviceId,
      const AllGatherPExecOptions& options);

  ::mccl::ReduceScatterOpts getReduceScatterOpts(
      const at::Tensor& outputTensor,
      const at::Tensor& inputTensor,
      int cudaDeviceId,
      cudaStream_t stream,
      const ReduceOp& op,
      const ReduceScatterSingleOptions& options);

  void attachMemoryHook();
  void detachMemoryHook();
  void checkInitializedImpl(const char* file, int line) const;

  void initStaticRegime(
      const std::string& initUrl,
      const std::chrono::milliseconds& initTimeout);
};

#define TC_MCCL_CHECK_INITIALIZED() checkInitializedImpl(__FILE__, __LINE__)

} // namespace torch::comms
