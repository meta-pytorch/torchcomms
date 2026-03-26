// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// CUDA kernel declarations for PipesDeviceApiIteratedTest (Pipes backend).
// Host-safe header — can be included from .cpp files compiled by clang.

// NOLINTNEXTLINE(clang-diagnostic-pragma-once-outside-header)
#pragma once

#include <cuda_runtime.h>
#include "comms/torchcomms/device/pipes/TorchCommDevicePipesTypes.hpp"

namespace torchcomms::device::test {

// Iterated put kernel for Pipes backend.
// Uses monotonic signal values (Pipes does NOT support reset_signal).
void launchPipesIteratedPutKernel(
    DeviceWindowPipes* win,
    RegisteredBufferPipes src_buf,
    float* src_ptr,
    float* win_base,
    size_t src_offset,
    size_t dst_offset,
    size_t bytes,
    size_t count,
    int dst_rank,
    int src_rank,
    int signal_id,
    int iterations,
    CoopScope scope,
    int num_threads,
    int* results,
    cudaStream_t stream);

// Iterated signal kernel for Pipes backend.
// Ring pattern with monotonic signal values.
void launchPipesIteratedSignalKernel(
    DeviceWindowPipes* win,
    int dst_rank,
    int src_rank,
    int signal_id,
    int iterations,
    CoopScope scope,
    int num_threads,
    cudaStream_t stream);

// Iterated barrier kernel for Pipes backend.
void launchPipesIteratedBarrierKernel(
    DeviceWindowPipes* win,
    int iterations,
    CoopScope scope,
    int num_threads,
    cudaStream_t stream);

// Combined ops kernel for Pipes backend.
// barrier -> fill -> put -> wait_signal -> verify -> barrier per iteration.
void launchPipesIteratedCombinedKernel(
    DeviceWindowPipes* win,
    RegisteredBufferPipes src_buf,
    float* src_ptr,
    float* win_base,
    size_t src_offset,
    size_t dst_offset,
    size_t bytes,
    size_t count,
    int dst_rank,
    int src_rank,
    int signal_id,
    int barrier_id_base,
    int iterations,
    int* results,
    cudaStream_t stream);

} // namespace torchcomms::device::test
