// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <torch/csrc/distributed/c10d/Backend.hpp> // @manual=//caffe2:torch-cpp-cpu
#include <torch/csrc/distributed/c10d/Store.hpp> // @manual=//caffe2:torch-cpp-cpu
#include <torch/csrc/distributed/c10d/Work.hpp> // @manual=//caffe2:torch-cpp-cpu

#include "comms/torchcomms/TorchCommBackend.hpp"
#include "comms/torchcomms/TorchCommTypes.hpp"
#include "comms/torchcomms/TorchWork.hpp"

namespace torch::comms {

class WorkWrapper : public c10d::Work {
 public:
  explicit WorkWrapper(c10::intrusive_ptr<TorchWork> work);
  ~WorkWrapper() override = default;

  bool isCompleted() override;
  bool isSuccess() const override;
  std::exception_ptr exception() const override;
  void synchronize() override;
  bool wait(std::chrono::milliseconds timeout) override;
  std::vector<at::Tensor> result() override;

 private:
  friend class BackendWrapper;
  c10::intrusive_ptr<TorchWork> work_;
};

using c10d::kUnsetTimeout;

class BackendWrapper : public c10d::Backend {
 public:
  struct TORCH_API Options : c10d::Backend::Options {
    bool abort_process_on_timeout_or_error{true};
    std::chrono::milliseconds timeout{kDefaultTimeout};
    bool high_priority_stream{false};
    c10::intrusive_ptr<c10d::Store> store{nullptr};
    std::unordered_map<std::string, std::string> hints;

    explicit Options() : c10d::Backend::Options("torchcomms") {}
  };

  explicit BackendWrapper(std::shared_ptr<TorchComm> comm);
  ~BackendWrapper() override = default;

  c10::intrusive_ptr<c10d::Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const c10d::BroadcastOptions& opts = c10d::BroadcastOptions()) override;
  c10::intrusive_ptr<c10d::Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const c10d::AllreduceOptions& opts = c10d::AllreduceOptions()) override;
  c10::intrusive_ptr<c10d::Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const c10d::AllreduceCoalescedOptions& opts =
          c10d::AllreduceCoalescedOptions()) override;
  c10::intrusive_ptr<c10d::Work> reduce(
      std::vector<at::Tensor>& tensors,
      const c10d::ReduceOptions& opts = c10d::ReduceOptions()) override;
  c10::intrusive_ptr<c10d::Work> allgather(
      std::vector<std::vector<at::Tensor>>& output_tensors,
      std::vector<at::Tensor>& input_tensors,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;
  c10::intrusive_ptr<c10d::Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& output_lists,
      std::vector<at::Tensor>& input_list,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;
  c10::intrusive_ptr<c10d::Work> allgather_into_tensor_coalesced(
      std::vector<at::Tensor>& output_tensors,
      std::vector<at::Tensor>& input_tensors,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;
  c10::intrusive_ptr<c10d::Work> _allgather_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;
  c10::intrusive_ptr<c10d::Work> gather(
      std::vector<std::vector<at::Tensor>>& output_tensors,
      std::vector<at::Tensor>& input_tensors,
      const c10d::GatherOptions& opts = c10d::GatherOptions()) override;
  c10::intrusive_ptr<c10d::Work> scatter(
      std::vector<at::Tensor>& output_tensors,
      std::vector<std::vector<at::Tensor>>& input_tensors,
      const c10d::ScatterOptions& opts = c10d::ScatterOptions()) override;
  c10::intrusive_ptr<c10d::Work> reduce_scatter(
      std::vector<at::Tensor>& output_tensors,
      std::vector<std::vector<at::Tensor>>& input_tensors,
      const c10d::ReduceScatterOptions& opts =
          c10d::ReduceScatterOptions()) override;
  c10::intrusive_ptr<c10d::Work> reduce_scatter_tensor_coalesced(
      std::vector<at::Tensor>& output_tensors,
      std::vector<at::Tensor>& input_tensors,
      const c10d::ReduceScatterOptions& opts =
          c10d::ReduceScatterOptions()) override;
  c10::intrusive_ptr<c10d::Work> _reduce_scatter_base(
      at::Tensor& output_tensor,
      at::Tensor& input_tensor,
      const c10d::ReduceScatterOptions& opts =
          c10d::ReduceScatterOptions()) override;
  c10::intrusive_ptr<c10d::Work> alltoall_base(
      at::Tensor& output_tensor,
      at::Tensor& input_tensor,
      std::vector<int64_t>& output_split_sizes,
      std::vector<int64_t>& input_split_sizes,
      const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) override;
  c10::intrusive_ptr<c10d::Work> alltoall(
      std::vector<at::Tensor>& output_tensors,
      std::vector<at::Tensor>& input_tensors,
      const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) override;
  c10::intrusive_ptr<c10d::Work> barrier(
      const c10d::BarrierOptions& opts = c10d::BarrierOptions()) override;
  c10::intrusive_ptr<c10d::Work>
  send(std::vector<at::Tensor>& tensors, int dstRank, int tag) override;
  c10::intrusive_ptr<c10d::Work>
  recv(std::vector<at::Tensor>& tensors, int srcRank, int tag) override;

  // Get the underlying backend comm for backend-specific operations
  std::shared_ptr<TorchComm> getComm() const;

  c10::intrusive_ptr<Options> getOptions() {
    return options_;
  }

  const std::string getBackendName() const override;

  c10::intrusive_ptr<c10d::Backend::Options> getBackendOptions() override;

  // Verify that a work object has the expected timeout.
  // Used for testing timeout propagation.
  bool verifyWorkTimeoutForTest(
      const c10::intrusive_ptr<c10d::Work>& work,
      const std::chrono::milliseconds& timeout);

  // Set the default timeout for this backend.
  void setTimeout(std::chrono::milliseconds timeout) override;

  // Split communicator into a subgroup and return a new BackendWrapper
  c10::intrusive_ptr<Backend> split(
      const c10::intrusive_ptr<c10d::Store>& store,
      const std::vector<int>& ranks,
      const c10::intrusive_ptr<c10d::Backend::Options>& opts) override;

 private:
  std::shared_ptr<TorchComm> comm_;
  std::shared_ptr<TorchCommBackend> backend_;
  c10::intrusive_ptr<Options> options_;
};

} // namespace torch::comms
