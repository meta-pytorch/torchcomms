// Copyright (c) Meta Platforms, Inc. and affiliates.
// TorchComms Device API - NCCL GIN Backend Type Definitions
//
// This header provides type aliases for NCCL GIN backend that can be safely
// included from both CUDA (.cu) and non-CUDA (.cpp) code compiled with clang.
//
// For device-side implementations (ncclGin usage), include
// TorchCommDeviceNCCLX.cuh instead - but ONLY from .cu files compiled with
// nvcc.

#pragma once

#include "comms/torchcomms/device/DeviceBackendTraits.hpp"
#include "comms/torchcomms/device/TorchCommDeviceComm.hpp"

namespace torchcomms::device {

// =============================================================================
// Type Aliases (safe for non-CUDA code)
// =============================================================================

using DeviceWindowNCCL = TorchCommDeviceWindow<NCCLGinBackend>;
using RegisteredBufferNCCL = RegisteredBuffer;

} // namespace torchcomms::device
