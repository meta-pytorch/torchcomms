// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/ATen.h>
#include <c10/util/intrusive_ptr.h>
#include <comms/torchcomms/TorchWork.hpp>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace torch::comms {

// Forward declarations
class TorchComm;
class TorchCommWindow;

// Enum for collective operation names
enum class OpName {
  send,
  recv,
  broadcast,
  all_reduce,
  reduce,
  all_gather,
  all_gather_v,
  all_gather_single,
  reduce_scatter,
  reduce_scatter_v,
  reduce_scatter_single,
  all_to_all_single,
  all_to_all_v_single,
  all_to_all,
  barrier,
  scatter,
  gather,
  split,
  new_window,
};

// Convert OpName enum to string
constexpr std::string_view toString(OpName name) {
  switch (name) {
    case OpName::send:
      return "send";
    case OpName::recv:
      return "recv";
    case OpName::broadcast:
      return "broadcast";
    case OpName::all_reduce:
      return "all_reduce";
    case OpName::reduce:
      return "reduce";
    case OpName::all_gather:
      return "all_gather";
    case OpName::all_gather_v:
      return "all_gather_v";
    case OpName::all_gather_single:
      return "all_gather_single";
    case OpName::reduce_scatter:
      return "reduce_scatter";
    case OpName::reduce_scatter_v:
      return "reduce_scatter_v";
    case OpName::reduce_scatter_single:
      return "reduce_scatter_single";
    case OpName::all_to_all_single:
      return "all_to_all_single";
    case OpName::all_to_all_v_single:
      return "all_to_all_v_single";
    case OpName::all_to_all:
      return "all_to_all";
    case OpName::barrier:
      return "barrier";
    case OpName::scatter:
      return "scatter";
    case OpName::gather:
      return "gather";
    case OpName::split:
      return "split";
    case OpName::new_window:
      return "new_window";
  }
  return "unknown";
}

struct PreHookArgs {
  OpName name{};
  bool async_op{false};
  std::vector<at::Tensor>* input_tensors{nullptr};
  std::vector<at::Tensor>* output_tensors{nullptr};
  const at::Tensor* input_tensor{nullptr};
  const at::Tensor* output_tensor{nullptr};
  int root{-1};
  // For all_to_all_v_single
  const std::vector<uint64_t>* output_split_sizes{nullptr};
  const std::vector<uint64_t>* input_split_sizes{nullptr};
  // For split
  const std::vector<int>* ranks{nullptr};
  const std::string* split_name{nullptr};
  // Unique operation ID to correlate pre-hook and post-hook calls
  size_t op_id{0};
};

using PreHook = std::function<void(PreHookArgs)>;

struct PostHookArgs {
  OpName name;
  std::optional<c10::weak_intrusive_ptr<TorchWork>> work{};
  std::weak_ptr<TorchComm> new_comm{};
  std::weak_ptr<TorchCommWindow> new_window{};
  // Unique operation ID to correlate pre-hook and post-hook calls
  size_t op_id{0};
};

using PostHook = std::function<void(PostHookArgs)>;

// Abort hook - called before aborting when a collective times out or fails.
// This allows users to capture debug information before the abort.
using AbortHook = std::function<void()>;

} // namespace torch::comms
