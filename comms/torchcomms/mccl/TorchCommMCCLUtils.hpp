// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <string_view>
#include "comms/torchcomms/mccl/TorchCommMCCL.hpp"

namespace torch::comms {

class CommOptions;

namespace mccl {

// Hint key names for MCCL backend configuration
constexpr std::string_view kHintGarbageCollectIntervalMs =
    "garbage_collect_interval_ms";
constexpr std::string_view kHintWatchdogTimeoutMs = "watchdog_timeout_ms";
constexpr std::string_view kHintInitMode = "initMode";
constexpr std::string_view kHintUseCpuBarrier = "use_cpu_barrier";

TorchCommMCCL::Configs parseOptions(const CommOptions& options);

void* getTensorDataPtr(const at::Tensor& inputTensor);

} // namespace mccl
} // namespace torch::comms
