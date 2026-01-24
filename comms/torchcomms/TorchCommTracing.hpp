// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/core/ivalue.h>
#include <string_view>
#include <vector>

#include <ATen/ATen.h>
#include <torch/csrc/distributed/c10d/ParamCommsUtils.hpp> // @manual=//caffe2:torch-cpp-cpu

namespace torch::comms {

// TODO: remove once other backends were migrated to TorchCommTracingGuard.
class TorchCommTracing {
 public:
  TorchCommTracing(std::string name, int comm_size, int rank)
      : name_(std::move(name)), comm_size_(comm_size), rank_(rank) {}

  void recordEvent(std::string_view collective_name);
  void recordEventWithInputOutput(
      std::string_view collective_name,
      int collective_rank,
      const std::vector<at::Tensor>& input_tensor_list,
      const std::vector<at::Tensor>& output_tensor_list);
  void recordEventWithInputOutput(
      std::string_view collective_name,
      int collective_rank,
      const std::vector<at::Tensor>& input_tensor_list,
      const std::vector<at::Tensor>& output_tensor_list,
      const std::vector<int64_t>& in_split_sizes,
      const std::vector<int64_t>& out_split_sizes);

 private:
  std::string name_;
  int comm_size_;
  int rank_;
};

class TorchCommTracingGuard {
 public:
  TorchCommTracingGuard(
      std::string_view comm_name,
      int comm_size,
      std::string_view collective_name,
      int collective_rank,
      const std::vector<at::Tensor>& input_tensor_list = {},
      const std::vector<at::Tensor>& output_tensor_list = {});

  TorchCommTracingGuard(
      std::string_view comm_name,
      int comm_size,
      std::string_view collective_name,
      int collective_rank,
      const at::Tensor& input_tensor,
      const at::Tensor& output_tensor);

  void initializeTracingCommon(
      std::string_view comm_name,
      int comm_size,
      std::string_view collective_name,
      int collective_rank,
      const std::vector<at::Tensor>& input_tensor_list,
      const std::vector<at::Tensor>& output_tensor_list);

  std::shared_ptr<torch::ParamCommsDebugInfo> getDebugInfo(
      std::string_view comm_name,
      int comm_size,
      std::string_view collective_name,
      int collective_rank,
      const std::vector<at::Tensor>& input_tensor_list,
      const std::vector<at::Tensor>& output_tensor_list,
      const std::vector<int64_t>& input_split_sizes,
      const std::vector<int64_t>& output_split_sizes);

 private:
  std::unique_ptr<c10::DebugInfoGuard> debug_info_guard_;
  std::optional<at::RecordFunction> record_function_guard_;

  inline static int sequence_number_ = 0;
};

} // namespace torch::comms
