// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <c10/util/intrusive_ptr.h>
#include <comms/torchcomms/CollectiveCoalescer.hpp>
#include <comms/torchcomms/TorchCommBatch.hpp>
#include <comms/torchcomms/TorchCommOptions.hpp>
#include <comms/torchcomms/TorchCommTypes.hpp>
#include <comms/torchcomms/TorchCommUtils.hpp>
#include <comms/torchcomms/TorchCommWindow.hpp>
#include <comms/torchcomms/TorchWork.hpp>
#include <memory>
#include <vector>

namespace torch {
namespace comms {

inline constexpr const char* TORCHCOMM_BACKEND_ABI_VERSION = "1.0";

class TorchCommBackend : public CollectiveCoalescer {
 public:
  virtual ~TorchCommBackend() = default;

  // Initialize the communication backend
  virtual void init(
      at::Device device,
      const std::string& name,
      const CommOptions& options = {}) = 0;
  virtual void finalize() = 0;
  virtual int getRank() const = 0;
  virtual int getSize() const = 0;

  // Name of the backend impl that's the same for all instances of a backend.
  virtual std::string_view getBackendName() const = 0;
  // Unique name for this instance of the communicator.
  virtual std::string_view getCommName() const = 0;

  // Point-to-Point Operations
  virtual c10::intrusive_ptr<TorchWork> send(
      const at::Tensor& tensor,
      int dst,
      bool async_op,
      const SendOptions& options = {}) = 0;
  virtual c10::intrusive_ptr<TorchWork> recv(
      at::Tensor& tensor,
      int src,
      bool async_op,
      const RecvOptions& options = {}) = 0;

  virtual c10::intrusive_ptr<TorchWork> batch_op_issue(
      const std::vector<BatchSendRecv::P2POp>& ops,
      bool async_op,
      const BatchP2POptions& options = {}) = 0;

  // Collective Operations
  virtual c10::intrusive_ptr<TorchWork> broadcast(
      at::Tensor& tensor,
      int root,
      bool async_op,
      const BroadcastOptions& options = {}) = 0;
  virtual c10::intrusive_ptr<TorchWork> all_reduce(
      at::Tensor& tensor,
      const ReduceOp& op,
      bool async_op,
      const AllReduceOptions& options = {}) = 0;
  virtual c10::intrusive_ptr<TorchWork> reduce(
      const at::Tensor& tensor,
      int root,
      const ReduceOp& op,
      bool async_op,
      const ReduceOptions& options = {}) = 0;
  virtual c10::intrusive_ptr<TorchWork> all_gather(
      const std::vector<at::Tensor>& tensor_list,
      const at::Tensor& tensor,
      bool async_op,
      const AllGatherOptions& options = {}) = 0;
  virtual c10::intrusive_ptr<TorchWork> all_gather_v(
      const std::vector<at::Tensor>& tensor_list,
      const at::Tensor& tensor,
      bool async_op,
      const AllGatherOptions& options = {}) = 0;
  virtual c10::intrusive_ptr<TorchWork> all_gather_single(
      at::Tensor& output,
      const at::Tensor& input,
      bool async_op,
      const AllGatherSingleOptions& options = {}) = 0;
  virtual c10::intrusive_ptr<TorchWork> reduce_scatter(
      at::Tensor& output,
      const std::vector<at::Tensor>& input_list,
      const ReduceOp& op,
      bool async_op,
      const ReduceScatterOptions& options = {}) = 0;
  virtual c10::intrusive_ptr<TorchWork> reduce_scatter_v(
      at::Tensor& output,
      const std::vector<at::Tensor>& input_list,
      const ReduceOp& op,
      bool async_op,
      const ReduceScatterOptions& options = {}) = 0;
  virtual c10::intrusive_ptr<TorchWork> reduce_scatter_single(
      at::Tensor& output,
      const at::Tensor& input,
      const ReduceOp& op,
      bool async_op,
      const ReduceScatterSingleOptions& options = {}) = 0;
  virtual c10::intrusive_ptr<TorchWork> all_to_all_single(
      at::Tensor& output,
      const at::Tensor& input,
      bool async_op,
      const AllToAllSingleOptions& options = {}) = 0;
  virtual c10::intrusive_ptr<TorchWork> all_to_all_v_single(
      at::Tensor& output,
      const at::Tensor& input,
      const std::vector<uint64_t>& output_split_sizes,
      const std::vector<uint64_t>& input_split_sizes,
      bool async_op,
      const AllToAllvSingleOptions& options = {}) = 0;
  virtual c10::intrusive_ptr<TorchWork> all_to_all(
      const std::vector<at::Tensor>& output_tensor_list,
      const std::vector<at::Tensor>& input_tensor_list,
      bool async_op,
      const AllToAllOptions& options = {}) = 0;
  virtual c10::intrusive_ptr<TorchWork> barrier(
      bool async_op,
      const BarrierOptions& options = {}) = 0;

  // Scatter and Gather Operations
  virtual c10::intrusive_ptr<TorchWork> scatter(
      at::Tensor& output_tensor,
      const std::vector<at::Tensor>& input_tensor_list,
      int root,
      bool async_op,
      const ScatterOptions& options = {}) = 0;
  virtual c10::intrusive_ptr<TorchWork> gather(
      const std::vector<at::Tensor>& output_tensor_list,
      const at::Tensor& input_tensor,
      int root,
      bool async_op,
      const GatherOptions& options = {}) = 0;

  // Coalescing Operations inherited from CollectiveCoalescer:
  // - startCoalescing()
  // - endCoalescing()
  // Backends implement: onCoalescingStart(), onCoalescingEnd(),
  // createCoalescedWork()

  // Communicator Management
  virtual std::shared_ptr<TorchCommBackend> split(
      const std::vector<int>& ranks,
      const std::string& name,
      const CommOptions& options = {}) = 0;

  virtual const CommOptions& getOptions() const = 0;

  virtual const at::Device& getDevice() const = 0;
  // Window & One-sidede Operations, not required for all backends, so we added
  // default implementation here
  virtual std::shared_ptr<TorchCommWindow> new_window() {
    throw std::logic_error(
        "[TorchCommBackend]: new_window not implemented for communicator:" +
        std::string(getCommName()));
    return nullptr;
  }
};

/**
 * Interface for a dynamic loader to be able to load a backend library
 * from a dynamic library.
 */
struct DynamicLoaderInterface {
  // Function pointers
  TorchCommBackend* (*new_comm)(void);
  void (*destroy_comm)(TorchCommBackend* comm);
  const char* (*get_supported_version)();
};

// Factory function signature (implemented in each .so)
using CreateDynamicLoaderFn = DynamicLoaderInterface (*)();

} // namespace comms
} // namespace torch
