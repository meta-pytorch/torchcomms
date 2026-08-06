// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace comms::prims::test {

// Test kernel that spins forever on SignalState, should trigger timeout
cudaError_t launchSignalStateTimeoutKernel(int device, uint32_t timeout_ms);

// Test kernel that completes before timeout (should NOT trap)
void launchNoTimeoutKernel(int device, uint32_t timeout_ms);

// Test kernel that uses ThreadGroup-based timeout checking for SignalState
cudaError_t launchSignalStateThreadGroupTimeoutKernel(
    int device,
    uint32_t timeout_ms);

// Test that when a kernel traps, subsequent kernels on the same stream don't
// run Returns true if second kernel did NOT run (expected behavior)
cudaError_t launchMultipleKernelsOnStreamTest(
    int device,
    uint32_t timeout_ms,
    bool* secondKernelDidNotRun);

} // namespace comms::prims::test
