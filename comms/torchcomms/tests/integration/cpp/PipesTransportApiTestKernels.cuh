// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// CUDA kernel declarations for PipesTransportApiTest.
// Host-safe header — can be included from .cpp files compiled by clang.

// NOLINTNEXTLINE(clang-diagnostic-pragma-once-outside-header)
#pragma once

#include <cuda_runtime.h>
#include "comms/pipes/MultiPeerDeviceHandle.cuh"

namespace torchcomms::device::test {

// Stress send/recv: rank 0 sends, rank 1 receives, with signal-based sync
// between iterations. Verifies data integrity on receiver.
void launchTransportStressSendRecvKernel(
    comms::pipes::MultiPeerDeviceHandle handle,
    float* buf,
    size_t count,
    int peer,
    int iterations,
    int num_threads,
    int* results,
    cudaStream_t stream);

// Stress signal/wait: ring pattern, monotonic ADD signals with GE waits.
void launchTransportStressSignalKernel(
    comms::pipes::MultiPeerDeviceHandle handle,
    int peer,
    int iterations,
    int num_threads,
    cudaStream_t stream);

// Combined: signal-sync + send/recv + signal/wait + verify per iteration.
void launchTransportStressCombinedKernel(
    comms::pipes::MultiPeerDeviceHandle handle,
    float* buf,
    size_t count,
    int peer,
    int iterations,
    int num_threads,
    int* results,
    cudaStream_t stream);

// LL128 send/recv: warp-only, small messages.
void launchTransportStressLl128Kernel(
    comms::pipes::MultiPeerDeviceHandle handle,
    char* buf,
    size_t nbytes,
    int peer,
    int iterations,
    int* results,
    cudaStream_t stream);

} // namespace torchcomms::device::test
