// Copyright (c) Meta Platforms, Inc. and affiliates.
// CUDA kernel declarations for DeviceApiTest

#pragma once

#include <cuda_runtime.h>
#include "comms/torchcomms/device/TorchCommDeviceComm.hpp"

namespace torch::comms::device::test {

// Host-callable wrapper functions to launch CUDA kernels
// These are defined in DeviceApiTestKernels.cu

// Launch device put kernel - performs put from src_buf to window on dst_rank
// Uses src_offset=0 and dst_offset=rank*bytes pattern
void launchDevicePutKernel(
    TorchCommDeviceWindow* win,
    RegisteredBuffer src_buf,
    size_t bytes,
    int dst_rank,
    int signal_id,
    cudaStream_t stream);

// Launch device put kernel with explicit offsets - performs put with custom
// src/dst offsets This is useful when using a single window buffer for both
// source and destination sections.
void launchDevicePutKernelWithOffsets(
    TorchCommDeviceWindow* win,
    RegisteredBuffer src_buf,
    size_t src_offset,
    size_t dst_offset,
    size_t bytes,
    int dst_rank,
    int signal_id,
    cudaStream_t stream);

// Launch device wait signal kernel - waits for signal from peer
void launchDeviceWaitSignalKernel(
    TorchCommDeviceWindow* win,
    int signal_id,
    uint64_t expected_value,
    cudaStream_t stream);

// Launch device reset signal kernel - resets signal to 0
void launchDeviceResetSignalKernel(
    TorchCommDeviceWindow* win,
    int signal_id,
    cudaStream_t stream);

} // namespace torch::comms::device::test
