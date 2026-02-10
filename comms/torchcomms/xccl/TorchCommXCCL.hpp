#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include <ATen/ATen.h>
#include <torch/csrc/distributed/c10d/Store.hpp> // @manual=//caffe2:torch-cpp

#include "comms/torchcomms/TorchComm.hpp"
#include "comms/torchcomms/TorchCommBackend.hpp"
#include "comms/torchcomms/TorchCommBatch.hpp"
#include "comms/torchcomms/TorchCommTracing.hpp"
#include "comms/torchcomms/device/xpu/XpuApi.hpp"
#include "comms/torchcomms/xccl/TorchWorkXCCL.hpp"
#include "comms/torchcomms/xccl/XcclApi.hpp"

namespace torch::comms {

constexpr size_t kMaxEventPoolSize = 1000;

// Custom exception class for better error handling
class XCCLException : public std::exception {
 public:
  XCCLException(
      XcclApi& api,
      const std::string& message,
      onecclResult_t result);

  const char* what() const noexcept override;
  onecclResult_t getResult() const;

 private:
  std::string message_;
  onecclResult_t result_;
};

class TorchCommXCCL : public TorchCommBackend,
                      public std::enable_shared_from_this<TorchCommXCCL> {
 public:
  static constexpr std::string_view kBackendName = "xccl";

  TorchCommXCCL();
  ~TorchCommXCCL() override;

  // Delete copy and move operations
  TorchCommXCCL(const TorchCommXCCL&) = delete;
  TorchCommXCCL(TorchCommXCCL&&) = delete;
  TorchCommXCCL& operator=(const TorchCommXCCL&) = delete;
  TorchCommXCCL& operator=(TorchCommXCCL&&) = delete;

  void init(
      at::Device device,
      const std::string& name,
      const CommOptions& options = {}) override;
  void finalize() override;
  int getRank() const override;
  int getSize() const override;
  std::string_view getBackendName() const override;
  std::string_view getCommName() const override;

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

  // Communicator Management
  std::shared_ptr<TorchCommBackend> split(
      const std::vector<int>& ranks,
      const std::string& name,
      const CommOptions& options = {}) override;

  // Friend access for TorchCommXCCL
  friend class TorchWorkXCCL;

  // Getter for XPU API (for friend classes)
  XpuApi* getXpuApi() const {
    return xpu_api_.get();
  }

  // Getter for XCCL API (for friend classes)
  XcclApi* getXcclApi() const {
    return xccl_api_.get();
  }

  // Method to override the XCCL API implementation for testing
  void setXcclApi(std::shared_ptr<XcclApi> api) {
    xccl_api_ = std::move(api);
  }

  // Method to override the XPU API implementation for testing
  void setXpuApi(std::shared_ptr<XpuApi> api) {
    xpu_api_ = std::move(api);
  }

  const CommOptions& getOptions() const override {
    return options_;
  }

  const at::Device& getDevice() const override {
    return device_;
  }

 protected:
  // Event management for friend classes
  xpuEvent_t getEvent();
  void returnEvent(xpuEvent_t&& event);
  void abortXcclComm();

  enum class CommState {
    NORMAL,
    ERROR,
    TIMEOUT,
  };

  std::atomic<CommState> comm_state_{
      CommState::NORMAL}; // State of the communicator

  onecclDataType_t getXcclDataType(const at::Tensor& tensor);
  c10::intrusive_ptr<TorchWorkXCCL> createWork(
      xpuStream_t stream,
      std::chrono::milliseconds timeout,
      const std::vector<at::Tensor>& inputTensors);

 private:
  // Helper that automatically cleans up premul sums.
  struct RedOpRAII {
    /* implicit */ RedOpRAII(onecclRedOp_t op);

    // Constructor for Premulsum Reduction
    explicit RedOpRAII(
        const ReduceOp& op,
        const onecclComm_t comm,
        const onecclDataType_t dataType,
        std::shared_ptr<XcclApi> xccl_api);

    RedOpRAII() = delete;
    RedOpRAII(const RedOpRAII&) = delete;
    RedOpRAII& operator=(const RedOpRAII&) = delete;
    RedOpRAII(RedOpRAII&& tmp) = delete;
    RedOpRAII& operator=(RedOpRAII&&) = delete;
    ~RedOpRAII();

    operator onecclRedOp_t() const {
      return xcclRedOp_;
    }

    onecclRedOp_t xcclRedOp_{onecclMaxRedOp};
    onecclComm_t comm_{nullptr};
    std::shared_ptr<XcclApi> xccl_api_;
  };

  // Constructor for split communicators
  explicit TorchCommXCCL(const onecclComm_t xccl_comm);

  // Private utility methods
  size_t wordSize(onecclDataType_t type) const;
  RedOpRAII getXcclReduceOp(
      const ReduceOp& op,
      const onecclComm_t comm,
      const onecclDataType_t dataType);
  void timeoutWatchdog() noexcept;
  void checkInitialized() const;
  void checkAndAbortIfTimedOutOrError();
  void checkWorkQueue(bool isMainThread);
  void enqueueWork(c10::intrusive_ptr<TorchWorkXCCL> work, xpuStream_t stream);
  xpuStream_t getOperationStream(bool async_op);
  void ensureTensorContiguous(const at::Tensor& tensor);

  // Member variables
  onecclComm_t xccl_comm_{};
  at::Device device_;
  int comm_size_{};
  int rank_{};
  CommOptions options_;
  size_t max_event_pool_size_{};
  std::optional<xpuStream_t> internal_stream_; // Initialized in init()
  std::optional<xpuEvent_t>
      dependency_event_; // Pre-allocated event for stream dependencies
  void* barrier_buffer_{}; // Pre-allocated XPU buffer for barrier operations
  enum class InitializationState {
    UNINITIALIZED,
    INITIALIZED,
    FINALIZED,
  } init_state_;

  // XCCL API abstraction
  std::shared_ptr<XcclApi> xccl_api_;

  // XPU API abstraction
  std::shared_ptr<XpuApi> xpu_api_;

  // Event pool management
  std::queue<xpuEvent_t> event_pool_;
  std::mutex event_pool_mutex_;

  // Work tracking per stream
  TorchWorkXCCLQueue workq_;

  // Timeout monitoring
  std::thread timeout_thread_;
  std::atomic<bool> shutdown_;
  std::condition_variable timeout_cv_;
  std::mutex timeout_mutex_;

  std::shared_ptr<TorchCommTracing> tracing_;
  bool high_priority_stream_{false};
  std::string name_;
};

} // namespace torch::comms
