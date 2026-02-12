// Copyright (c) Meta Platforms, Inc. and affiliates.
// TorchComms Device API - NCCL Backend Type Definitions
//
// This header provides type aliases for NCCL device backend that can be safely
// included from both CUDA (.cu) and non-CUDA (.cpp) code compiled with clang.
//
// For device-side implementations (ncclGin usage), include
// TorchCommDeviceNCCLX.cuh instead - but ONLY from .cu files compiled with
// nvcc.

#pragma once

#include "comms/torchcomms/device/DeviceBackendTraits.hpp"
#include "comms/torchcomms/device/TorchCommDeviceWindow.hpp"

namespace torchcomms::device {

// =============================================================================
// Type Aliases (safe for non-CUDA code)
// =============================================================================

using DeviceWindowNCCL = TorchCommDeviceWindow<NCCLDeviceBackend>;
using RegisteredBufferNCCL = RegisteredBuffer;

} // namespace torchcomms::device
