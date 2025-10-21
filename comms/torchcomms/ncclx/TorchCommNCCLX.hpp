// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include <ATen/ATen.h>
#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy
#include <torch/csrc/distributed/c10d/Store.hpp> // @manual=//caffe2:torch-cpp

#include "comms/torchcomms/TorchComm.hpp"
#include "comms/torchcomms/TorchCommBackend.hpp"
#include "comms/torchcomms/TorchCommBatch.hpp"
#include "comms/torchcomms/device/CudaApi.hpp"
#include "comms/torchcomms/ncclx/NcclxApi.hpp"
#include "comms/torchcomms/ncclx/TorchCommWindowNCCLX.hpp"
#include "comms/torchcomms/ncclx/TorchWorkNCCLX.hpp"

namespace torch {
namespace comms {

constexpr size_t kMaxEventPoolSize = 1000;

// Custom exception class for better error handling
class NCCLException : public std::exception {
 public:
  NCCLException(NcclxApi& api, const std::string& message, ncclResult_t result);

  const char* what() const noexcept override;
  ncclResult_t getResult() const;

 private:
  std::string message_;
  ncclResult_t result_;
};

class TorchCommNCCLX : public TorchCommBackend,
                       public std::enable_shared_from_this<TorchCommNCCLX> {
 public:
  static constexpr std::string_view kBackendName = "ncclx";

  TorchCommNCCLX();
  ~TorchCommNCCLX() override;

  // Delete copy and move operations
  TorchCommNCCLX(const TorchCommNCCLX&) = delete;
  TorchCommNCCLX(TorchCommNCCLX&&) = delete;
  TorchCommNCCLX& operator=(const TorchCommNCCLX&) = delete;
  TorchCommNCCLX& operator=(TorchCommNCCLX&&) = delete;

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
  std::shared_ptr<TorchWork> send(
      const at::Tensor& tensor,
      int dst,
      bool async_op,
      const SendOptions& options = {}) override;
  std::shared_ptr<TorchWork> recv(
      at::Tensor& tensor,
      int src,
      bool async_op,
      const RecvOptions& options = {}) override;

  // Batch P2P Operations
  std::shared_ptr<TorchWork> batch_op_issue(
      const std::vector<BatchSendRecv::P2POp>& ops,
      bool async_op,
      const BatchP2POptions& options = {}) override;

  // Collective Operations
  std::shared_ptr<TorchWork> broadcast(
      at::Tensor& tensor,
      int root,
      bool async_op,
      const BroadcastOptions& options = {}) override;
  std::shared_ptr<TorchWork> all_reduce(
      at::Tensor& tensor,
      ReduceOp op,
      bool async_op,
      const AllReduceOptions& options = {}) override;
  std::shared_ptr<TorchWork> reduce(
      const at::Tensor& tensor,
      int root,
      ReduceOp op,
      bool async_op,
      const ReduceOptions& options = {}) override;
  std::shared_ptr<TorchWork> all_gather(
      const std::vector<at::Tensor>& tensor_list,
      const at::Tensor& tensor,
      bool async_op,
      const AllGatherOptions& options = {}) override;
  std::shared_ptr<TorchWork> all_gather_single(
      at::Tensor& output,
      const at::Tensor& input,
      bool async_op,
      const AllGatherSingleOptions& options = {}) override;
  std::shared_ptr<TorchWork> reduce_scatter(
      at::Tensor& output,
      const std::vector<at::Tensor>& input_list,
      ReduceOp op,
      bool async_op,
      const ReduceScatterOptions& options = {}) override;
  std::shared_ptr<TorchWork> reduce_scatter_single(
      at::Tensor& output,
      const at::Tensor& input,
      ReduceOp op,
      bool async_op,
      const ReduceScatterSingleOptions& options = {}) override;
  std::shared_ptr<TorchWork> all_to_all_single(
      at::Tensor& output,
      const at::Tensor& input,
      bool async_op,
      const AllToAllSingleOptions& options = {}) override;
  std::shared_ptr<TorchWork> all_to_all_v_single(
      at::Tensor& output,
      const at::Tensor& input,
      const std::vector<uint64_t>& output_split_sizes,
      const std::vector<uint64_t>& input_split_sizes,
      bool async_op,
      const AllToAllvSingleOptions& options = {}) override;
  std::shared_ptr<TorchWork> all_to_all(
      const std::vector<at::Tensor>& output_tensor_list,
      const std::vector<at::Tensor>& input_tensor_list,
      bool async_op,
      const AllToAllOptions& options = {}) override;
  std::shared_ptr<TorchWork> barrier(
      bool async_op,
      const BarrierOptions& options = {}) override;

  // Scatter and Gather Operations
  std::shared_ptr<TorchWork> scatter(
      at::Tensor& output_tensor,
      const std::vector<at::Tensor>& input_tensor_list,
      int root,
      bool async_op,
      const ScatterOptions& options = {}) override;
  std::shared_ptr<TorchWork> gather(
      const std::vector<at::Tensor>& output_tensor_list,
      const at::Tensor& input_tensor,
      int root,
      bool async_op,
      const GatherOptions& options = {}) override;

  // Window & One-sidede Operations
  std::shared_ptr<TorchCommWindow> window_allocate(
      const size_t window_size,
      bool cpu_buf = false,
      const size_t signal_size = 256) override;

  // Communicator Management
  std::shared_ptr<TorchCommBackend> split(
      const std::vector<int>& ranks,
      const std::string& name,
      const CommOptions& options = {}) override;

  // Friend access for TorchCommNCCLX
  friend class TorchWorkNCCLX;
  friend class CachingAllocatorHookImpl;
  friend class TorchCommWindowNCCLX;

  // Getter for CUDA API (for friend classes)
  CudaApi* getCudaApi() const {
    return cuda_api_.get();
  }

  // Getter for NCCL API (for friend classes)
  NcclxApi* getNcclApi() const {
    return nccl_api_.get();
  }

  // Method to override the NCCL API implementation for testing
  void setNcclApi(std::shared_ptr<NcclxApi> api) {
    nccl_api_ = std::move(api);
  }

  // Method to override the CUDA API implementation for testing
  void setCudaApi(std::shared_ptr<CudaApi> api) {
    cuda_api_ = std::move(api);
  }

  const CommOptions& getOptions() const override {
    return options_;
  }

  const at::Device& getDevice() const override {
    return device_;
  }

 protected:
  // Event management for friend classes
  cudaEvent_t getEvent();
  void returnEvent(cudaEvent_t event);
  void abortNcclComm();

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

  cudaEvent_t
      dependency_event_{}; // Pre-allocated event for stream dependencies

  void register_address(const AddressWithLen& addr);
  void deregister_address(const Address& addr);
  ncclDataType_t getNcclDataType(const at::Tensor& tensor);
  std::shared_ptr<TorchWorkNCCLX> createWork(
      cudaStream_t stream,
      std::chrono::milliseconds timeout,
      const std::vector<at::Tensor>& inputTensors);
  NcclxWindowCmpOp getNcclSignalCmpOp(SignalCmpOp op);

 private:
  // Helper that automatically cleans up premul sums.
  struct RedOpRAII {
    /* implicit */ RedOpRAII(ncclRedOp_t op);

    // Constructor for Premulsum Reduction
    explicit RedOpRAII(
        const ReduceOp& op,
        const ncclComm_t comm,
        const ncclDataType_t dataType,
        std::shared_ptr<NcclxApi> nccl_api);

    RedOpRAII() = delete;
    RedOpRAII(const RedOpRAII&) = delete;
    RedOpRAII& operator=(const RedOpRAII&) = delete;
    RedOpRAII(RedOpRAII&& tmp) = delete;
    RedOpRAII& operator=(RedOpRAII&&) = delete;
    ~RedOpRAII();

    operator ncclRedOp_t() const {
      return ncclRedOp_;
    }

    ncclRedOp_t ncclRedOp_{ncclMaxRedOp};
    ncclComm_t comm_{nullptr};
    std::shared_ptr<NcclxApi> nccl_api_;
  };

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

  // Constructor for split communicators
  explicit TorchCommNCCLX(const ncclComm_t nccl_comm);

  // Private utility methods
  RedOpRAII getNcclReduceOp(
      const ReduceOp& op,
      const ncclComm_t comm,
      const ncclDataType_t dataType);
  void timeoutWatchdog() noexcept;
  void checkInitialized() const;
  void checkAndAbortIfTimedOutOrError();
  void checkWorkQueue(bool isMainThread);
  void enqueueWork(std::shared_ptr<TorchWorkNCCLX> work, cudaStream_t stream);
  bool getGraphCaptureMode();
  cudaStream_t getOperationStream(bool async_op);
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
  cudaStream_t internal_stream_{};
  void* barrier_buffer_{}; // Pre-allocated CUDA buffer for barrier operations
  enum class InitializationState {
    UNINITIALIZED,
    INITIALIZED,
    FINALIZED,
  } init_state_;

  // List of [comm, regHandlesMap] pairs.  Each regHandlesMap is a map from the
  // buffer address to the registeration handle
  std::map<void*, RegistrationHandle> memoryRegistrationHandles_;

  // NCCL API abstraction
  std::shared_ptr<NcclxApi> nccl_api_;

  // CUDA API abstraction
  std::shared_ptr<CudaApi> cuda_api_;

  // Event pool management
  std::queue<cudaEvent_t> event_pool_;
  std::mutex event_pool_mutex_;

  // Work tracking per stream
  TorchWorkNCCLXQueue workq_;

  // Timeout monitoring
  std::thread timeout_thread_;
  std::atomic<bool> shutdown_;
  std::condition_variable timeout_cv_;
  std::mutex timeout_mutex_;

  bool high_priority_stream_{false};
  std::string name_;

  // Graph capture mode work references
  // Keep references to work objects during graph capture to prevent premature
  // destruction, organized per graph using capture ID
  std::unordered_map<
      unsigned long long,
      std::vector<std::shared_ptr<TorchWorkNCCLX>>>
      graph_capture_work_refs_;
  std::mutex graph_capture_work_mutex_;

  // Structure to hold cleanup data for CUDA user objects
  struct GraphCleanupData {
    TorchCommNCCLX* comm;
    unsigned long long graph_id;

    GraphCleanupData(TorchCommNCCLX* comm_, unsigned long long id)
        : comm(comm_), graph_id(id) {}
  };

  // Static callback function for CUDA user object cleanup
  static void CUDART_CB graphCleanupCallback(void* userData);

  friend class TorchWorkNCCLXQueueCommTest;
};

} // namespace comms
} // namespace torch
