// Copyright (c) Meta Platforms, Inc. and affiliates.
// CUDA kernel declarations for DeviceApiTest
//
// This header provides function declarations that can be included from
// both .cpp (host code, compiled by clang) and .cu (CUDA code, compiled
// by nvcc) files.
//
// The type aliases (DeviceWindowNCCL, RegisteredBufferNCCL) are defined in
// TorchCommDeviceNCCLXTypes.hpp which is safe to include from host code.
// The full device implementations (ncclGin usage, etc.) are only in the
// .cu file which is compiled by nvcc.

#pragma once

#include <cuda_runtime.h>
// Include the host-safe header that provides type aliases
// (DeviceWindowNCCL = TorchCommDeviceWindow<NCCLDeviceBackend>)
// This does NOT include the device implementation code that requires nvcc.
#include "comms/torchcomms/device/ncclx/TorchCommDeviceNCCLXTypes.hpp"

namespace torchcomms::device::test {

// Host-callable wrapper functions to launch CUDA kernels
// These are defined in DeviceApiTestKernels.cu

// Launch device put kernel - performs put from src_buf to window on dst_rank
// Uses src_offset=0 and dst_offset=rank*bytes pattern
// Note: DeviceWindowNCCL* is a DEVICE pointer (allocated via cudaMalloc)
void launchDevicePutKernel(
    DeviceWindowNCCL* win,
    RegisteredBufferNCCL src_buf,
    size_t bytes,
    int dst_rank,
    int signal_id,
    cudaStream_t stream);

// Launch device put kernel with explicit offsets - performs put with custom
// src/dst offsets This is useful when using a single window buffer for both
// source and destination sections.
// Note: DeviceWindowNCCL* is a DEVICE pointer (allocated via cudaMalloc)
void launchDevicePutKernelWithOffsets(
    DeviceWindowNCCL* win,
    RegisteredBufferNCCL src_buf,
    size_t src_offset,
    size_t dst_offset,
    size_t bytes,
    int dst_rank,
    int signal_id,
    cudaStream_t stream);

// Launch device wait signal kernel - waits for signal from peer
// Note: DeviceWindowNCCL* is a DEVICE pointer (allocated via cudaMalloc)
void launchDeviceWaitSignalKernel(
    DeviceWindowNCCL* win,
    int signal_id,
    uint64_t expected_value,
    cudaStream_t stream);

// Launch device reset signal kernel - resets signal to 0
// Note: DeviceWindowNCCL* is a DEVICE pointer (allocated via cudaMalloc)
void launchDeviceResetSignalKernel(
    DeviceWindowNCCL* win,
    int signal_id,
    cudaStream_t stream);

// Launch read signal kernel - reads aggregated signal value into output buffer.
// out must be a device pointer to a single uint64_t.
void launchDeviceReadSignalKernel(
    DeviceWindowNCCL* win,
    int signal_id,
    uint64_t* out,
    cudaStream_t stream);

// Launch GIN atomicAdd test kernel - performs remote atomic fetch-and-add
// on a uint64_t in the destination window, then signals the destination rank
// using resource buffer signals.
// Note: DeviceWindowNCCL* is a DEVICE pointer (allocated via cudaMalloc)
void launchDeviceGinAtomicAddKernel(
    DeviceWindowNCCL* win,
    size_t dst_offset,
    uint64_t add_value,
    int dst_rank,
    int signal_id,
    cudaStream_t stream);

// Launch standalone signal kernel - signals a peer without data transfer.
// Tests per-peer resource buffer signal model in isolation.
void launchDeviceSignalKernel(
    DeviceWindowNCCL* win,
    int peer,
    int signal_id,
    SignalOp op,
    uint64_t value,
    cudaStream_t stream);

// Launch wait signal from specific peer kernel - waits for signal from a
// single peer (not aggregated). Tests point-to-point synchronization.
void launchDeviceWaitSignalFromKernel(
    DeviceWindowNCCL* win,
    int peer,
    int signal_id,
    CmpOp cmp,
    uint64_t value,
    cudaStream_t stream);

} // namespace torchcomms::device::test
