// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <string>
#include <string_view>
#include <vector>

#include <ATen/ATen.h>
#include "comms/torchcomms/TorchCommHooks.hpp"
#include "comms/torchcomms/TorchCommTypes.hpp"

namespace torch::comms {

std::string_view dtypeToString(at::ScalarType dtype);
std::string_view reduceOpToString(const ReduceOp& op);
std::string formatPtr(const void* ptr);
std::string formatPtrs(const std::vector<at::Tensor>& tensors);
std::string formatCounts(const std::vector<at::Tensor>& tensors);
std::string formatCounts(const std::vector<uint64_t>& counts);
std::string buildSignature(
    std::string_view comm_name,
    const PreHookArgs& args,
    bool include_buffers);
std::string
buildNewCommSignature(std::string_view comm_name, int rank, int world_size);
std::string buildSplitLine(
    std::string_view parent_name,
    const SplitPreHookArgs& split);

} // namespace torch::comms
