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
#include "comms/torchcomms/device/npu/NpuApi.hpp"
#include "comms/torchcomms/hccl/HcclApi.hpp"
#include "comms/torchcomms/hccl/TorchWorkHCCL.hpp"

namespace torch::comms {

constexpr size_t kMaxEventPoolSize = 1000;

// Custom exception class for better error handling
class HCCLException : public std::exception {
 public:
  HCCLException(HcclApi& api, const std::string& message, HcclResult result);

  const char* what() const noexcept override;
  HcclResult getResult() const;

 private:
  std::string message_;
  HcclResult result_;
};

class TorchCommHCCL : public TorchCommBackend,
                      public std::enable_shared_from_this<TorchCommHCCL> {
 public:
  static constexpr std::string_view kBackendName = "hccl";

  TorchCommHCCL();
  ~TorchCommHCCL() override;

  // Delete copy and move operations
  TorchCommHCCL(const TorchCommHCCL&) = delete;
  TorchCommHCCL(TorchCommHCCL&&) = delete;
  TorchCommHCCL& operator=(const TorchCommHCCL&) = delete;
  TorchCommHCCL& operator=(TorchCommHCCL&&) = delete;

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

  std::shared_ptr<c10::Allocator> getMemAllocator();

  // Friend access for TorchCommHCCL
  friend class TorchWorkHCCL;

  // Getter for NPU API (for friend classes)
  NpuApi* getNpuApi() const {
    return npu_api_.get();
  }

  // Getter for HCCL API (for friend classes)
  HcclApi* getHcclApi() const {
    return hccl_api_.get();
  }

  // Method to override the HCCL API implementation for testing
  void setHcclApi(std::shared_ptr<HcclApi> api) {
    hccl_api_ = std::move(api);
  }

  // Method to override the NPU API implementation for testing
  void setNpuApi(std::shared_ptr<NpuApi> api) {
    npu_api_ = std::move(api);
  }

  const CommOptions& getOptions() const override {
    return options_;
  }

  const at::Device& getDevice() const override {
    return device_;
  }

 protected:
  // Event management for friend classes
  npuEvent_t getEvent();
  void returnEvent(npuEvent_t&& event);
  void abortHcclComm();

  enum class CommState {
    NORMAL,
    ERROR,
    TIMEOUT,
  };

  std::atomic<CommState> comm_state_{
      CommState::NORMAL}; // State of the communicator

  HcclDataType getHcclDataType(const at::Tensor& tensor);
  c10::intrusive_ptr<TorchWorkHCCL> createWork(
      npuStream_t stream,
      std::chrono::milliseconds timeout,
      const std::vector<at::Tensor>& inputTensors);

 private:
  // Helper that automatically cleans up premul sums.
  struct RedOpRAII {
    /* implicit */ RedOpRAII(HcclReduceOp op);

    // Constructor for Premulsum Reduction
    explicit RedOpRAII(
        const ReduceOp& op,
        const HcclComm comm,
        const HcclDataType dataType,
        std::shared_ptr<HcclApi> hccl_api);

    RedOpRAII() = delete;
    RedOpRAII(const RedOpRAII&) = delete;
    RedOpRAII& operator=(const RedOpRAII&) = delete;
    RedOpRAII(RedOpRAII&& tmp) = delete;
    RedOpRAII& operator=(RedOpRAII&&) = delete;
    ~RedOpRAII();

    operator HcclReduceOp() const {
      return hcclRedOp_;
    }

    HcclReduceOp hcclRedOp_{HCCL_REDUCE_SUM};
    HcclComm comm_{nullptr};
    std::shared_ptr<HcclApi> hccl_api_;
  };

  // Constructor for split communicators
  explicit TorchCommHCCL(const HcclComm hccl_comm);

  // Private utility methods
  size_t wordSize(HcclDataType type) const;
  RedOpRAII getHcclReduceOp(
      const ReduceOp& op,
      const HcclComm comm,
      const HcclDataType dataType);
  void timeoutWatchdog() noexcept;
  void checkInitialized() const;
  void checkAndAbortIfTimedOutOrError();
  void checkWorkQueue(bool isMainThread);
  void enqueueWork(c10::intrusive_ptr<TorchWorkHCCL> work, npuStream_t stream);
  npuStream_t getOperationStream(bool async_op);
  void ensureTensorContiguous(const at::Tensor& tensor);

  // Member variables
  HcclComm hccl_comm_{};
  at::Device device_;
  int comm_size_{};
  int rank_{};
  CommOptions options_;
  size_t max_event_pool_size_{};
  std::optional<npuStream_t> internal_stream_; // Initialized in init()
  std::optional<npuEvent_t>
      dependency_event_; // Pre-allocated event for stream dependencies
  void* barrier_buffer_{}; // Pre-allocated NPU buffer for barrier operations
  enum class InitializationState {
    UNINITIALIZED,
    INITIALIZED,
    FINALIZED,
  } init_state_;

  // HCCL API abstraction
  std::shared_ptr<HcclApi> hccl_api_;

  // NPU API abstraction
  std::shared_ptr<NpuApi> npu_api_;

  // Event pool management
  std::queue<npuEvent_t> event_pool_;
  std::mutex event_pool_mutex_;

  // Work tracking per stream
  TorchWorkHCCLQueue workq_;

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
