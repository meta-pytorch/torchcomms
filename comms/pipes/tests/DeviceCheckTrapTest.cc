// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "comms/pipes/tests/DeviceCheckTest.cuh"
#include "comms/testinfra/TestXPlatUtils.h"

namespace comms::pipes {

// =============================================================================
// Trap Test - PIPES_DEVICE_CHECK
// =============================================================================
//
// This test is in a separate binary because:
// 1. __trap() puts the CUDA device into an unrecoverable error state
// 2. cudaDeviceReset() is required to recover, but it invalidates all contexts
// 3. This would break any subsequent tests in the same process

class DeviceCheckTrapTestFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    // Ensure we have a valid CUDA device
    int deviceCount = 0;
    cudaError_t deviceErr = cudaGetDeviceCount(&deviceCount);
    if (deviceErr != cudaSuccess || deviceCount == 0) {
      GTEST_SKIP() << "No CUDA devices available";
    }
    CUDACHECK_TEST(cudaSetDevice(0));
  }

  void TearDown() override {
    // Clear any CUDA errors from trap tests
    cudaGetLastError(); // NOLINT(facebook-cuda-safe-api-call-check)
  }
};

// Test: Verify that PIPES_DEVICE_CHECK with a false expression triggers a
// device-side trap. This validates that the macro correctly aborts kernel
// execution when the invariant is violated.
//
// Note: __trap() causes an illegal instruction that stops kernel execution.
// After the trap fires, cudaDeviceSynchronize() returns an error.
TEST_F(DeviceCheckTrapTestFixture, DeviceCheckWithFalseExpressionTraps) {
  const int numBlocks = 1;
  const int blockSize = 32;

  // Launch the kernel - this should trigger the device trap
  test::testDeviceCheckFailing(numBlocks, blockSize);

  // Synchronize and check for error
  cudaError_t syncError = cudaDeviceSynchronize();

  // The trap should have fired, causing a CUDA error
  EXPECT_TRUE(
      syncError == cudaErrorIllegalInstruction ||
      syncError == cudaErrorAssert || syncError == cudaErrorLaunchFailure)
      << "Expected CUDA error when PIPES_DEVICE_CHECK expression is false, "
         "but got: "
      << cudaGetErrorString(syncError);

  // Reset the device to clear the sticky error state
  cudaDeviceReset(); // NOLINT(facebook-cuda-safe-api-call-check)
  cudaSetDevice(0); // NOLINT(facebook-cuda-safe-api-call-check)
  cudaGetLastError(); // NOLINT(facebook-cuda-safe-api-call-check)
}

} // namespace comms::pipes

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
