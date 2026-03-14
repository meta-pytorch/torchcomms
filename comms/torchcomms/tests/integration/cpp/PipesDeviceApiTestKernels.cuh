// Copyright (c) Meta Platforms, Inc. and affiliates.
// CUDA kernel declarations for PipesDeviceApiTest
//
// This header provides function declarations that can be included from
// both .cpp (host code, compiled by clang) and .cu (CUDA code, compiled
// by nvcc) files.
//
// The type aliases (DeviceWindowPipes, RegisteredBufferPipes) are defined in
// TorchCommDevicePipesTypes.hpp which is safe to include from host code.
// The full device implementations (Pipes transport usage, etc.) are only in
// the .cu file which is compiled by nvcc.

// NOLINTNEXTLINE(clang-diagnostic-pragma-once-outside-header)
#pragma once

#include <cuda_runtime.h>
// Include the host-safe header that provides type aliases
// (DeviceWindowPipes = TorchCommDeviceWindow<PipesDeviceBackend>)
// This does NOT include the device implementation code that requires nvcc.
#include "comms/torchcomms/device/pipes/TorchCommDevicePipesTypes.hpp"

namespace torchcomms::device::test {

// Host-callable wrapper functions to launch CUDA kernels
// These are defined in PipesDeviceApiTestKernels.cu

// Launch standalone signal kernel - signals a peer without data transfer.
// Tests per-peer signal model via Pipes transport (NVLink/IBGDA).
void launchPipesSignalKernel(
    DeviceWindowPipes* win,
    int peer,
    int signal_id,
    SignalOp op,
    uint64_t value,
    cudaStream_t stream);

// Launch device wait signal kernel - waits for aggregated signal from all peers
// Note: DeviceWindowPipes* is a DEVICE pointer (allocated via cudaMalloc)
void launchPipesWaitSignalKernel(
    DeviceWindowPipes* win,
    int signal_id,
    uint64_t expected_value,
    cudaStream_t stream);

// Launch device reset signal kernel - resets all signal slots to 0
// Note: DeviceWindowPipes* is a DEVICE pointer (allocated via cudaMalloc)
void launchPipesResetSignalKernel(
    DeviceWindowPipes* win,
    int signal_id,
    cudaStream_t stream);

// Launch read signal kernel - reads aggregated signal value into output buffer.
// out must be a device pointer to a single uint64_t.
void launchPipesReadSignalKernel(
    DeviceWindowPipes* win,
    int signal_id,
    uint64_t* out,
    cudaStream_t stream);

// Launch wait signal from specific peer kernel - waits for signal from a
// single peer (not aggregated). Tests point-to-point synchronization.
void launchPipesWaitSignalFromKernel(
    DeviceWindowPipes* win,
    int peer,
    int signal_id,
    CmpOp cmp,
    uint64_t value,
    cudaStream_t stream);

// Launch device barrier kernel - synchronizes all ranks via barrier
void launchPipesBarrierKernel(
    DeviceWindowPipes* win,
    int barrier_id,
    cudaStream_t stream);

} // namespace torchcomms::device::test
