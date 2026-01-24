// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <vector>

#include <ATen/ATen.h>
#include <hip/hip_runtime.h> // @manual=third-party//cuda:cuda-lazy
#include <torch/csrc/distributed/c10d/Store.hpp> // @manual=//caffe2:torch-cpp

#include "comms/torchcomms/TorchComm.hpp" // @manual=//comms/torchcomms:torchcomms-headers-cpp
#include "comms/torchcomms/TorchCommBackend.hpp" // @manual=//comms/torchcomms:torchcomms-headers-cpp
#include "comms/torchcomms/TorchCommBatch.hpp" // @manual=//comms/torchcomms:torchcomms-headers-cpp
#include "comms/torchcomms/rccl/HipApi.hpp" // @manual
#include "comms/torchcomms/rccl/RcclApi.hpp" // @manual
#include "comms/torchcomms/rccl/TorchWorkRCCL.hpp" // @manual

namespace torch {
namespace comms {

constexpr size_t kMaxEventPoolSize = 1000;

// Custom exception class for better error handling
class RCCLException : public std::exception {
 public:
  RCCLException(
      RcclApi& api,
      const std::string& message,
      ncclResult_t result,
      ncclComm_t comm);

  const char* what() const noexcept override;
  ncclResult_t getResult() const;

 private:
  std::string message_;
  ncclResult_t result_;
};

class TorchCommRCCL : public TorchCommBackend,
                      public std::enable_shared_from_this<TorchCommRCCL> {
 public:
  static constexpr std::string_view kBackendName = "rccl";
  TorchCommRCCL();
  ~TorchCommRCCL() override;

  // Delete copy and move operations
  TorchCommRCCL(const TorchCommRCCL&) = delete;
  TorchCommRCCL(TorchCommRCCL&&) = delete;
  TorchCommRCCL& operator=(const TorchCommRCCL&) = delete;
  TorchCommRCCL& operator=(TorchCommRCCL&&) = delete;

  void init(
      at::Device device,
      const std::string& name,
      const CommOptions& options = {}) override;
  void finalize() override;
  int getRank() const override;
  int getSize() const override;

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
  std::string_view getBackendName() const override;
  std::string_view getCommName() const override;
  // Communicator Management
  std::shared_ptr<TorchCommBackend> split(
      const std::vector<int>& ranks,
      const std::string& name,
      const CommOptions& options = {}) override;

  // Friend access for TorchCommRCCL
  friend class TorchWorkRCCL;
  friend class CachingAllocatorHookImpl;

  // Getter for CUDA API (for friend classes)
  HipApi* getHipApi() const {
    return hip_api_.get();
  }

  // Method to override the RCCL API implementation for testing
  void setRcclApi(std::shared_ptr<RcclApi> api) {
    rccl_api_ = std::move(api);
  }

  // Method to override the CUDA API implementation for testing
  void setHipApi(std::shared_ptr<HipApi> api) {
    hip_api_ = std::move(api);
  }

  const CommOptions& getOptions() const override {
    return options_;
  }

  const at::Device& getDevice() const override {
    return device_;
  }

 protected:
  // Event management for friend classes
  hipEvent_t getEvent();
  void returnEvent(hipEvent_t event);
  void abortRcclComm();

  enum class CommState {
    NORMAL,
    ERROR,
    TIMEOUT,
  };

  struct Address {
    void* addr;
  };

  struct AddressWithLen {
    void* addr;
    size_t len;
  };

  std::atomic<CommState> comm_state_{
      CommState::NORMAL}; // State of the communicator

  void register_address(const AddressWithLen& addr);
  void deregister_address(const Address& addr);

 private:
  // Adding this class to align with NCCL/NCCLX
  struct RedOpRAII {
    /* implicit */ RedOpRAII(ncclRedOp_t op);

    // Constructor for Premulsum Reduction
    explicit RedOpRAII(
        const ReduceOp& op,
        const ncclComm_t comm,
        const ncclDataType_t dataType,
        std::shared_ptr<RcclApi> rccl_api);

    RedOpRAII() = delete;
    RedOpRAII(const RedOpRAII&) = delete;
    RedOpRAII& operator=(const RedOpRAII&) = delete;
    RedOpRAII(RedOpRAII&& tmp) = delete;
    RedOpRAII& operator=(RedOpRAII&&) = delete;
    ~RedOpRAII();

    operator ncclRedOp_t() const {
      return ncclRedOp_;
    }

    ncclRedOp_t ncclRedOp_;
    ncclComm_t comm_;
    std::shared_ptr<RcclApi> rccl_api_;
  };

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

  // Constructor for split communicators
  TorchCommRCCL(
      const ncclComm_t nccl_comm,
      at::Device device,
      const CommOptions& options);

  // Private utility methods
  ncclDataType_t getNcclDataType(const at::Tensor& tensor);
  RedOpRAII getNcclReduceOp(
      const ReduceOp& op,
      const ncclComm_t comm,
      const ncclDataType_t dataType);
  void timeoutWatchdog() noexcept;
  void checkInitialized() const;
  void checkAndAbortIfTimedOutOrError();
  void garbageCollectWorkQueues();
  c10::intrusive_ptr<TorchWorkRCCL> createWork(
      hipStream_t stream,
      std::chrono::milliseconds timeout,
      const std::vector<at::Tensor>& inputTensors);
  void enqueueWork(c10::intrusive_ptr<TorchWorkRCCL> work, hipStream_t stream);
  hipStream_t getOperationStream(bool async_op);
  void ensureTensorContiguous(const at::Tensor& tensor);

  void attachMemoryHook();
  void detachMemoryHook();

  // Member variables
  ncclComm_t nccl_comm_{};
  at::Device device_;
  int comm_size_{};
  int rank_{};
  CommOptions options_;
  size_t max_event_pool_size_{};
  hipStream_t internal_stream_{};
  hipEvent_t dependency_event_{}; // Pre-allocated event for stream dependencies
  void* barrier_buffer_{}; // Pre-allocated CUDA buffer for barrier operations
  enum class InitializationState {
    UNINITIALIZED,
    INITIALIZED,
    FINALIZED,
  } init_state_;

  // List of [comm, regHandlesMap] pairs.  Each regHandlesMap is a map from the
  // buffer address to the registeration handle
  std::map<void*, RegistrationHandle> memoryRegistrationHandles_;

  // RCCL API abstraction
  std::shared_ptr<RcclApi> rccl_api_;

  // CUDA API abstraction
  std::shared_ptr<HipApi> hip_api_;

  // Event pool management
  std::queue<hipEvent_t> event_pool_;
  std::mutex event_pool_mutex_;

  // Work tracking per stream
  std::unordered_map<hipStream_t, std::queue<c10::intrusive_ptr<TorchWorkRCCL>>>
      stream_work_queues_;
  std::queue<c10::intrusive_ptr<TorchWorkRCCL>> completed_works_;
  std::mutex work_queues_mutex_;

  // Timeout monitoring
  std::thread timeout_thread_;
  std::atomic<bool> shutdown_;
  std::condition_variable timeout_cv_;
  std::mutex timeout_mutex_;

  bool high_priority_stream_{false};
  std::string name_;
};

} // namespace comms
} // namespace torch
