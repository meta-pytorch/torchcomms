// Copyright (c) Meta Platforms, Inc. and affiliates.
// TorchComms Device API - Prims Backend Type Definitions
//
// Provides type aliases for the Prims device backend that are safe to include
// from both CUDA (.cu) and non-CUDA (.cpp/.cc) code compiled with clang.
//
// For device-side implementations (IBGDA/NVLink usage), include
// TorchCommDevicePrims.cuh instead - but ONLY from .cu files compiled with
// nvcc.

#pragma once

#if defined(ENABLE_PRIMS)

#include "comms/torchcomms/device/TorchCommDeviceWindow.hpp"
#include "comms/torchcomms/device/prims/PrimsDeviceBackend.hpp"

namespace torchcomms::device {

// =============================================================================
// Type Aliases (safe for non-CUDA code)
// =============================================================================

using DeviceWindowPrims = TorchCommDeviceWindow<PrimsDeviceBackend>;
using RegisteredBufferPrims = torch::comms::RegisteredBuffer;

using DeviceWindowPipes = DeviceWindowPrims;
using RegisteredBufferPipes = RegisteredBufferPrims;

} // namespace torchcomms::device

#endif // ENABLE_PRIMS
