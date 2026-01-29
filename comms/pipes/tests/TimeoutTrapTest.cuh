// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>

namespace comms::pipes::test {

// Test kernel that spins forever, should trigger timeout
void launchChunkStateTimeoutKernel(int device, uint32_t timeout_ms);

// Test kernel that spins forever on SignalState, should trigger timeout
void launchSignalStateTimeoutKernel(int device, uint32_t timeout_ms);

// Test kernel that completes before timeout (should NOT trap)
void launchNoTimeoutKernel(int device, uint32_t timeout_ms);

// Test kernel that uses ThreadGroup-based timeout checking for ChunkState
void launchChunkStateThreadGroupTimeoutKernel(int device, uint32_t timeout_ms);

// Test kernel that uses ThreadGroup-based timeout checking for SignalState
void launchSignalStateThreadGroupTimeoutKernel(int device, uint32_t timeout_ms);

// Test kernel that calls start() twice (should trap on double-start)
void launchDoubleStartKernel(int device, uint32_t timeout_ms);

// Test that when a kernel traps, subsequent kernels on the same stream don't
// run Returns true if second kernel did NOT run (expected behavior)
bool launchMultipleKernelsOnStreamTest(int device, uint32_t timeout_ms);

} // namespace comms::pipes::test
