// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <c10/util/intrusive_ptr.h>
#include <comms/torchcomms/RemovableHandle.hpp>
#include <comms/torchcomms/TorchCommBackend.hpp>
#include <comms/torchcomms/TorchCommBatch.hpp>
#include <comms/torchcomms/TorchCommHooks.hpp>
#include <comms/torchcomms/TorchCommOptions.hpp>
#include <comms/torchcomms/TorchCommTypes.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp> // @manual=//caffe2:torch-cpp-cpu
#include <memory>
#include <string>

namespace torch::comms {

// Reset the global op_id generator. Used when creating isolated
// FlightRecorder instances to ensure each test gets a fresh op_id space.
void resetGlobalOpIdGenerator();

// Forward declarations
class TorchCommNCCLX;
class TorchWin;

/**
 * TorchComm - Main communication abstraction for TorchComms.
 *
 * Thread Safety:
 * TorchComm is NOT thread-safe. Users must not call TorchComm operations
 * from multiple threads simultaneously. All operations (collectives,
 * point-to-point, memory registration, finalize, etc.) must be serialized
 * by the caller.
 */
class TorchComm : public std::enable_shared_from_this<TorchComm> {
 public:
  ~TorchComm() = default;

  void finalize();
  int getRank() const;
  int getSize() const;
  std::vector<int> getRanks() const;
  std::string_view getCommName() const;

  // Point-to-Point Operations
  c10::intrusive_ptr<TorchWork> send(
      const at::Tensor& tensor,
      int dst,
      bool async_op,
      const SendOptions& options = {});
  c10::intrusive_ptr<TorchWork> recv(
      at::Tensor& tensor,
      int src,
      bool async_op,
      const RecvOptions& options = {});

  // Collective Operations
  c10::intrusive_ptr<TorchWork> broadcast(
      at::Tensor& tensor,
      int root,
      bool async_op,
      const BroadcastOptions& options = {});
  c10::intrusive_ptr<TorchWork> all_reduce(
      at::Tensor& tensor,
      const ReduceOp& op,
      bool async_op,
      const AllReduceOptions& options = {});
  c10::intrusive_ptr<TorchWork> reduce(
      const at::Tensor& tensor,
      int root,
      const ReduceOp& op,
      bool async_op,
      const ReduceOptions& options = {});
  c10::intrusive_ptr<TorchWork> all_gather(
      const std::vector<at::Tensor>& tensor_list,
      const at::Tensor& tensor,
      bool async_op,
      const AllGatherOptions& options = {});
  c10::intrusive_ptr<TorchWork> all_gather_v(
      const std::vector<at::Tensor>& tensor_list,
      const at::Tensor& tensor,
      bool async_op,
      const AllGatherOptions& options = {});
  c10::intrusive_ptr<TorchWork> all_gather_single(
      at::Tensor& output,
      const at::Tensor& input,
      bool async_op,
      const AllGatherSingleOptions& options = {});
  c10::intrusive_ptr<TorchWork> reduce_scatter(
      at::Tensor& output,
      const std::vector<at::Tensor>& input_list,
      const ReduceOp& op,
      bool async_op,
      const ReduceScatterOptions& options = {});
  c10::intrusive_ptr<TorchWork> reduce_scatter_v(
      at::Tensor& output,
      const std::vector<at::Tensor>& input_list,
      const ReduceOp& op,
      bool async_op,
      const ReduceScatterOptions& options = {});
  c10::intrusive_ptr<TorchWork> reduce_scatter_single(
      at::Tensor& output,
      const at::Tensor& input,
      const ReduceOp& op,
      bool async_op,
      const ReduceScatterSingleOptions& options = {});
  c10::intrusive_ptr<TorchWork> all_to_all_single(
      at::Tensor& output,
      const at::Tensor& input,
      bool async_op,
      const AllToAllSingleOptions& options = {});
  c10::intrusive_ptr<TorchWork> all_to_all_v_single(
      at::Tensor& output,
      const at::Tensor& input,
      const std::vector<uint64_t>& output_split_sizes,
      const std::vector<uint64_t>& input_split_sizes,
      bool async_op,
      const AllToAllvSingleOptions& options = {});
  c10::intrusive_ptr<TorchWork> all_to_all(
      const std::vector<at::Tensor>& output_tensor_list,
      const std::vector<at::Tensor>& input_tensor_list,
      bool async_op,
      const AllToAllOptions& options = {});
  c10::intrusive_ptr<TorchWork> barrier(
      bool async_op,
      const BarrierOptions& options = {});

  // Scatter and Gather Operations
  c10::intrusive_ptr<TorchWork> scatter(
      at::Tensor& output_tensor,
      const std::vector<at::Tensor>& input_tensor_list,
      int root,
      bool async_op,
      const ScatterOptions& options = {});
  c10::intrusive_ptr<TorchWork> gather(
      const std::vector<at::Tensor>& output_tensor_list,
      const at::Tensor& input_tensor,
      int root,
      bool async_op,
      const GatherOptions& options = {});

  // Communicator Management
  std::shared_ptr<TorchComm> split(
      const std::vector<int>& ranks,
      const std::string& name,
      const CommOptions& options = {});

  // Batch Operations
  BatchSendRecv batch_op_create();

  const CommOptions& getOptions() const;

  const at::Device& getDevice() const;

  const std::string& getBackend() const {
    return backend_;
  }

  std::string_view getBackendVersion() const;

  // Device Transport API — returns device pointer as int64 for Triton kernels.
  // Throws if not supported by the backend.
  int64_t get_device_transport();

  std::shared_ptr<TorchCommBackend> getBackendImpl() const {
    return impl_;
  }

  std::shared_ptr<TorchCommWindow> new_window(
      const std::optional<at::Tensor>& tensor = std::nullopt);

  // Persistent AllGather operations
  using AllGatherPHandle = TorchCommBackend::AllGatherPHandle;

  AllGatherPHandle all_gather_p_init(
      at::Tensor& output,
      const AllGatherPInitOptions& options = {});

  c10::intrusive_ptr<TorchWork> all_gather_p_exec(
      AllGatherPHandle handle,
      const at::Tensor& input,
      bool async_op,
      const AllGatherPExecOptions& options = {});

  void all_gather_p_free(AllGatherPHandle handle);

  // Fault Tolerance API

  /**
   * Get the initialization handle for this communicator.
   * In dynamic regime, this handle encodes information required by the backend
   * to complete the initialization process via reconfigure().
   *
   * @return An InitHandle containing the initialization URL/handle.
   * @throws std::runtime_error if not implemented by the backend.
   */
  InitHandle getInitHandle() const;

  /**
   * Reconfigure the communicator with a new set of peers.
   * In dynamic regime, this method initializes the communicator with the
   * provided set of peers. After a successful reconfigure call, the
   * communicator is fully initialized and collective operations are permitted.
   *
   * @param opts ReconfigureOptions containing uuid, handles, timeout, and
   * hints.
   * @return A TorchWork handle that can be used to wait for completion.
   * @throws std::runtime_error if not implemented by the backend.
   */
  c10::intrusive_ptr<TorchWork> reconfigure(const ReconfigureOptions& opts);

  // Hook types (defined in TorchCommHooks.hpp; aliased for backward compat)
  using PreHookArgs = ::torch::comms::PreHookArgs;
  using PostHookArgs = ::torch::comms::PostHookArgs;
  using PreHook = ::torch::comms::PreHook;
  using PostHook = ::torch::comms::PostHook;
  using AbortHook = ::torch::comms::AbortHook;

  // Hook registration (not thread-safe; must not be called while a collective
  // is in progress)
  std::unique_ptr<RemovableHandle> registerPreHook(PreHook preHook);
  std::unique_ptr<RemovableHandle> registerPostHook(PostHook postHook);
  std::unique_ptr<RemovableHandle> registerAbortHook(AbortHook hook);

  // Disable copy and move semantics
  TorchComm(const TorchComm&) = delete;
  TorchComm& operator=(const TorchComm&) = delete;
  TorchComm(TorchComm&&) = delete;
  TorchComm& operator=(TorchComm&&) = delete;

  friend class BatchSendRecv;
  friend std::shared_ptr<TorchComm> new_comm(
      const std::string& backend_name,
      at::Device device,
      const std::string& name,
      const CommOptions& options);

 private:
  // constructor for root communicators
  explicit TorchComm(
      const std::string& backend,
      std::shared_ptr<TorchCommBackend> impl);

  // constructor for split communicators
  TorchComm(
      const std::string& backend,
      std::shared_ptr<TorchCommBackend> impl,
      std::vector<int> ranks);

  void preHook(PreHookArgs&& args);
  void postHook(PostHookArgs&& args);

  // Rank validation helper
  void validateRank(int rank, const char* param_name) const;

  // Backend name
  std::string backend_;
  std::string backend_version_;
  // Implementation object
  std::shared_ptr<TorchCommBackend> impl_;

  int64_t nextHookId_ = 0;
  std::unordered_map<int64_t, PreHook> preHooks_;
  std::unordered_map<int64_t, PostHook> postHooks_;
  // Global ranks of the members of this communicator.
  // For root communicators: [0, 1, 2, ..., size-1]
  // For split communicators: global ranks from the parent communicator
  std::vector<int> ranks_;
};

// Constructor that creates the appropriate backend implementation
std::shared_ptr<TorchComm> new_comm(
    const std::string& backend_name,
    at::Device device,
    const std::string& name,
    const CommOptions& options = {});

// Global memory allocator function
// Returns a static allocator for the specified backend
// Note: Allocator is created once per backend and reused across all instances
std::shared_ptr<c10::Allocator> get_mem_allocator(const std::string& backend);

} // namespace torch::comms
