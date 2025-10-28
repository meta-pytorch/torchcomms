// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <comms/torchcomms/TorchCommBackend.hpp>
#include <comms/torchcomms/TorchWork.hpp>
#include <vector>

namespace torch {
namespace comms {

class DummyTorchCommBackend : public TorchCommBackend {
 public:
  static constexpr std::string_view kBackendName = "dummy";

  DummyTorchCommBackend();
  ~DummyTorchCommBackend() override = default;

  // Initialize the communication backend
  void init(
      at::Device device,
      const std::string& name,
      const CommOptions& options = {}) override;
  void finalize() override;
  int getRank() const override;
  int getSize() const override;
  std::string_view getCommName() const override;
  std::string_view getBackendName() const override;

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
  std::shared_ptr<TorchWork> all_gather_v(
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

  const CommOptions& getOptions() const override;
  const at::Device& getDevice() const override;

 private:
  bool initialized_;
  at::Device device_;
  CommOptions options_;
  int rank_;
  int size_;
  std::string name_;
};

} // namespace comms
} // namespace torch
