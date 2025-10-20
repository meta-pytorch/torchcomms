// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/TorchCommTracing.hpp"

#include <ATen/core/ivalue.h>
#include <ATen/record_function.h>
#include <torch/csrc/distributed/c10d/ParamCommsUtils.hpp> // @manual=//caffe2:torch-cpp-cpu
#include <string>

namespace torch {
namespace comms {

void TorchCommTracing::recordEvent(const std::string& collective_name) {
  RECORD_PARAM_COMMS(
      std::make_tuple(0, false), // sequence number tuple
      std::make_tuple(
          name_,
          ""), // PG name/description tuple
      rank_,
      collective_name.c_str(),
      0, // inNelems
      0, // outNelems
      at::kByte, // dType
      std::vector<int64_t>(), // inSplitSizes
      std::vector<int64_t>(), // outSplitSizes
      -1, // TODO: fix global rank start
      -1, // TODO: fix global rank stride
      comm_size_); // worldSize
}

void TorchCommTracing::recordEventWithInputOutput(
    const std::string& collective_name,
    int rank,
    const std::vector<at::Tensor>& input_tensor_list,
    const std::vector<at::Tensor>& output_tensor_list) {
  std::vector<int64_t> in_split_sizes;
  for (const auto r : c10::irange(input_tensor_list.size())) {
    in_split_sizes.push_back(input_tensor_list[r].numel());
  }
  std::vector<int64_t> out_split_sizes;
  for (const auto r : c10::irange(output_tensor_list.size())) {
    out_split_sizes.push_back(output_tensor_list[r].numel());
  }

  recordEventWithInputOutput(
      collective_name,
      rank,
      input_tensor_list,
      output_tensor_list,
      in_split_sizes,
      out_split_sizes);
}

void TorchCommTracing::recordEventWithInputOutput(
    const std::string& collective_name,
    int collective_rank,
    const std::vector<at::Tensor>& input_tensor_list,
    const std::vector<at::Tensor>& output_tensor_list,
    const std::vector<int64_t>& input_split_sizes,
    const std::vector<int64_t>& output_split_sizes) {
  int64_t input_total_numel = 0;
  for (const auto r : c10::irange(input_tensor_list.size())) {
    input_total_numel += input_tensor_list[r].numel();
  }
  int64_t output_total_numel = 0;
  for (const auto r : c10::irange(output_tensor_list.size())) {
    output_total_numel += output_tensor_list[r].numel();
  }

  auto data_type = input_tensor_list.size() > 0
      ? input_tensor_list.front().scalar_type()
      : output_tensor_list.front().scalar_type();

  RECORD_PARAM_COMMS_DATA(
      std::make_tuple(0, false), // sequence number tuple
      std::make_tuple(
          name_,
          ""), // PG name/description tuple
      input_tensor_list, // inputTensors
      output_tensor_list, // outputTensors
      collective_rank,
      collective_name.c_str(), // collective name
      input_total_numel, // inNelems
      output_total_numel, // outNelems
      data_type, // dType
      input_split_sizes, // inSplitSizes
      output_split_sizes, // outSplitSizes
      -1, // TODO: fix global rank start
      -1, // TODO: fix global rank stride
      comm_size_); // worldSize
}

std::shared_ptr<torch::ParamCommsDebugInfo> TorchCommTracingGuard::getDebugInfo(
    const std::string& comm_name,
    int comm_size,
    const std::string& collective_name,
    int collective_rank,
    const std::vector<at::Tensor>& input_tensor_list,
    const std::vector<at::Tensor>& output_tensor_list,
    const std::vector<int64_t>& input_split_sizes,
    const std::vector<int64_t>& output_split_sizes) {
  int64_t input_total_numel = 0;
  for (const auto r : c10::irange(input_tensor_list.size())) {
    input_total_numel += input_tensor_list[r].numel();
  }
  int64_t output_total_numel = 0;
  for (const auto r : c10::irange(output_tensor_list.size())) {
    output_total_numel += output_tensor_list[r].numel();
  }

  // If both input and output tensor lists are empty, use a default data type.
  auto data_type = at::kByte;
  if (input_tensor_list.size() > 0) {
    data_type = input_tensor_list.front().scalar_type();
  } else if (output_tensor_list.size() > 0) {
    data_type = output_tensor_list.front().scalar_type();
  }

  return std::make_shared<torch::ParamCommsDebugInfo>(
      std::make_tuple(comm_name, ""),
      collective_rank,
      collective_name.c_str(),
      input_total_numel,
      output_total_numel,
      data_type,
      input_split_sizes,
      output_split_sizes,
      -1, // TODO: fix global rank start
      -1, // TODO: fix global rank stride
      comm_size);
}

TorchCommTracingGuard::TorchCommTracingGuard(
    const std::string& comm_name,
    int comm_size,
    const std::string& collective_name,
    int collective_rank,
    const std::vector<at::Tensor>& input_tensor_list,
    const std::vector<at::Tensor>& output_tensor_list) {
  std::vector<int64_t> in_split_sizes;
  for (const auto r : c10::irange(input_tensor_list.size())) {
    in_split_sizes.push_back(input_tensor_list[r].numel());
  }
  std::vector<int64_t> out_split_sizes;
  for (const auto r : c10::irange(output_tensor_list.size())) {
    out_split_sizes.push_back(output_tensor_list[r].numel());
  }

  debug_info_guard_ = std::make_unique<c10::DebugInfoGuard>(
      c10::DebugInfoKind::PARAM_COMMS_INFO,
      getDebugInfo(
          comm_name,
          comm_size,
          collective_name,
          collective_rank,
          input_tensor_list,
          output_tensor_list,
          in_split_sizes,
          out_split_sizes));

  std::initializer_list<const c10::IValue> paramList = {
      c10::IValue(input_tensor_list),
      std::make_tuple(++sequence_number_, false),
      std::make_tuple(comm_name, ""),
      collective_rank,
      collective_name,
      in_split_sizes,
      out_split_sizes,
      -1, // Global rank start isn't set in TorchComms.
      -1, // Global rank stride isn't set in TorchComms.
      comm_size};
  c10::ArrayRef<const c10::IValue> paramInputs(paramList);

  record_function_guard_ =
      std::make_unique<at::RecordFunction>(at::RecordScope::FUNCTION);
  if (record_function_guard_->isActive()) {
    if (record_function_guard_->needsInputs()) {
      record_function_guard_->before(at::kParamCommsCallName, paramInputs);
    } else {
      record_function_guard_->before(at::kParamCommsCallName);
    }
    if (record_function_guard_->needsOutputs()) {
      record_function_guard_->setOutputs(
          std::vector<c10::IValue>(1, c10::IValue(output_tensor_list)));
    }
  }
}

} // namespace comms
} // namespace torch
