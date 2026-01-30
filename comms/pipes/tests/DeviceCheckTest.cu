// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <cstdint>

#include "comms/pipes/DeviceCheck.cuh"

namespace comms::pipes::test {

// =============================================================================
// Passing case kernels - PIPES_DEVICE_CHECK with true expressions
// =============================================================================

// Kernel that uses PIPES_DEVICE_CHECK with an expression that evaluates to
// true. Should complete without triggering a trap.
__global__ void testDeviceCheckPassingKernel() {
  // Simple true expression - should not trap
  PIPES_DEVICE_CHECK(1 == 1);

  // Thread-based check - always true since threadIdx is always >= 0
  PIPES_DEVICE_CHECK(threadIdx.x >= 0);
}

void testDeviceCheckPassing(int numBlocks, int blockSize) {
  testDeviceCheckPassingKernel<<<numBlocks, blockSize>>>();
}

// Kernel that uses PIPES_DEVICE_CHECK_MSG with an expression that evaluates to
// true. Should complete without triggering a trap.
__global__ void testDeviceCheckMsgPassingKernel() {
  // Simple true expression with custom message - should not trap
  PIPES_DEVICE_CHECK_MSG(1 == 1, "This should never print");

  // Block-based check - always true since blockIdx is always >= 0
  PIPES_DEVICE_CHECK_MSG(blockIdx.x >= 0, "Block index is never negative");
}

void testDeviceCheckMsgPassing(int numBlocks, int blockSize) {
  testDeviceCheckMsgPassingKernel<<<numBlocks, blockSize>>>();
}

// Kernel with multiple PIPES_DEVICE_CHECK calls, all passing.
// Tests that multiple checks in sequence work correctly.
__global__ void testMultipleDeviceChecksPassingKernel(
    uint32_t value,
    uint32_t threshold) {
  // Multiple checks that should all pass when value <= threshold
  PIPES_DEVICE_CHECK(value <= threshold);
  PIPES_DEVICE_CHECK(value >= 0); // Always true for unsigned
  PIPES_DEVICE_CHECK_MSG(threshold > 0, "Threshold must be positive");
}

void testMultipleDeviceChecksPassing(
    uint32_t value,
    uint32_t threshold,
    int numBlocks,
    int blockSize) {
  testMultipleDeviceChecksPassingKernel<<<numBlocks, blockSize>>>(
      value, threshold);
}

// =============================================================================
// Trap kernels - PIPES_DEVICE_CHECK with false expressions
// =============================================================================

// Kernel that uses PIPES_DEVICE_CHECK with a false expression.
// Should trigger __trap() and cause a CUDA error.
__global__ void testDeviceCheckFailingKernel() {
  // False expression - should trigger trap
  PIPES_DEVICE_CHECK(1 == 0);
}

void testDeviceCheckFailing(int numBlocks, int blockSize) {
  // NOLINT because we intentionally don't check launch errors - the kernel
  // is expected to trap
  testDeviceCheckFailingKernel<<<
      numBlocks,
      blockSize>>>(); // NOLINT(facebook-cuda-safe-kernel-call-check)
}

// Kernel that uses PIPES_DEVICE_CHECK_MSG with a false expression.
// Should trigger __trap() and cause a CUDA error.
__global__ void testDeviceCheckMsgFailingKernel() {
  // False expression with custom message - should trigger trap
  PIPES_DEVICE_CHECK_MSG(1 == 0, "This intentionally fails for testing");
}

void testDeviceCheckMsgFailing(int numBlocks, int blockSize) {
  // NOLINT because we intentionally don't check launch errors - the kernel
  // is expected to trap
  testDeviceCheckMsgFailingKernel<<<
      numBlocks,
      blockSize>>>(); // NOLINT(facebook-cuda-safe-kernel-call-check)
}

} // namespace comms::pipes::test
