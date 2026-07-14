// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// CUDA kernel declarations for PipesTransportApiTest.
// Host-safe header — can be included from .cpp files compiled by clang.

// NOLINTNEXTLINE(clang-diagnostic-pragma-once-outside-header)
#pragma once

#include <cuda_runtime.h>
#include "comms/prims/transport/MultiPeerDeviceHandle.cuh"

namespace torchcomms::device::test {

// Stress signal/wait: ring pattern, monotonic ADD signals with GE waits.
void launchTransportStressSignalKernel(
    comms::prims::MultiPeerDeviceHandle handle,
    int peer,
    int iterations,
    int num_threads,
    cudaStream_t stream);

// LL128 send/recv: warp-only, small messages.
void launchTransportStressLl128Kernel(
    comms::prims::MultiPeerDeviceHandle handle,
    char* buf,
    size_t nbytes,
    int peer,
    int iterations,
    int* results,
    cudaStream_t stream);

} // namespace torchcomms::device::test
