// Copyright (c) Meta Platforms, Inc. and affiliates.
// TorchComms Device API - Pipes Backend Type Definitions
//
// Provides type aliases for the Pipes device backend that are safe to include
// from both CUDA (.cu) and non-CUDA (.cpp/.cc) code compiled with clang.
//
// For device-side implementations (IBGDA/NVLink usage), include
// TorchCommDevicePipes.cuh instead - but ONLY from .cu files compiled with
// nvcc.

#pragma once

#if defined(ENABLE_PIPES)

#include "comms/torchcomms/device/TorchCommDeviceWindow.hpp"
#include "comms/torchcomms/device/pipes/PipesDeviceBackend.hpp"

namespace torchcomms::device {

// =============================================================================
// Type Aliases (safe for non-CUDA code)
// =============================================================================

using DeviceWindowPipes = TorchCommDeviceWindow<PipesDeviceBackend>;
using RegisteredBufferPipes = RegisteredBuffer;

} // namespace torchcomms::device

#endif // ENABLE_PIPES
