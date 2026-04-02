// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// CUDA kernel declarations for DeviceApiIteratedTest (NCCLx backend).
// Host-safe header — can be included from .cpp files compiled by clang.

// NOLINTNEXTLINE(clang-diagnostic-pragma-once-outside-header)
#pragma once

#include <cuda_runtime.h>
#include "comms/torchcomms/device/ncclx/TorchCommDeviceNCCLXTypes.hpp"

namespace torchcomms::device::test {

// Iterated put kernel: performs put+signal in a loop, writing src_buf to
// dst_rank's window. Uses monotonic signal values (signal value = iteration+1).
// Receiver waits for signal to reach the monotonic target before verifying.
// The kernel fills src with a rank+iteration pattern each iteration.
//
// Parameters:
//   win         - device window pointer (device memory)
//   src_buf     - registered local source buffer
//   src_ptr     - raw float pointer to source buffer (for fill pattern)
//   win_base    - raw float pointer to window base (for verification)
//   src_offset  - byte offset within src_buf
//   dst_offset  - byte offset within destination window
//   bytes       - bytes to put per iteration
//   count       - number of float elements (bytes / sizeof(float))
//   dst_rank    - destination rank in ring
//   src_rank    - source rank (who sends to us)
//   signal_id   - signal slot to use
//   iterations  - number of put iterations
//   scope       - cooperative scope (THREAD, WARP, BLOCK)
//   results     - device int array[iterations], 1=pass 0=fail per iteration
void launchIteratedPutKernel(
    DeviceWindowNCCL* win,
    RegisteredBufferNCCL src_buf,
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

// Iterated signal kernel: sends signal to dst_rank in a ring, waits for
// signal from src_rank, repeated iterations times. Uses monotonic values.
void launchIteratedSignalKernel(
    DeviceWindowNCCL* win,
    int dst_rank,
    int src_rank,
    int signal_id,
    int iterations,
    CoopScope scope,
    int num_threads,
    cudaStream_t stream);

// Iterated barrier kernel: calls barrier iterations times, alternating
// between barrier_id 0 and 1.
void launchIteratedBarrierKernel(
    DeviceWindowNCCL* win,
    int iterations,
    CoopScope scope,
    int num_threads,
    cudaStream_t stream);

// Combined ops kernel: barrier -> fill -> put -> wait_signal -> verify ->
// barrier per iteration. Tests interleaved operations.
void launchIteratedCombinedKernel(
    DeviceWindowNCCL* win,
    RegisteredBufferNCCL src_buf,
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

// Iterated aggregated wait_signal kernel: ring signal -> aggregated
// wait_signal (not per-peer) -> read_signal -> reset_signal -> verify
// read_signal returns 0. Exercises the non-per-peer signal path.
void launchIteratedAggregatedSignalKernel(
    DeviceWindowNCCL* win,
    int dst_rank,
    int signal_id,
    int iterations,
    int* results,
    cudaStream_t stream);

// Iterated half-precision put kernel: same as iteratedPutKernel but with
// __half data type. Uses half-precision fill/verify patterns.
void launchIteratedPutHalfKernel(
    DeviceWindowNCCL* win,
    RegisteredBufferNCCL src_buf,
    void* src_ptr,
    void* win_base,
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

} // namespace torchcomms::device::test
