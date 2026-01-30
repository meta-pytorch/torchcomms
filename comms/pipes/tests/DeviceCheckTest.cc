// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "comms/pipes/tests/DeviceCheckTest.cuh"
#include "comms/testinfra/TestXPlatUtils.h"

namespace comms::pipes {

class DeviceCheckTestFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    int deviceCount = 0; // Ensure valid CUDA device
    cudaError_t deviceErr = cudaGetDeviceCount(&deviceCount);
    if (deviceErr != cudaSuccess || deviceCount == 0) {
      GTEST_SKIP() << "No CUDA devices available";
    }
    CUDACHECK_TEST(cudaSetDevice(0));
  }

  void TearDown() override {
    // Clear any lingering CUDA errors
    cudaGetLastError(); // NOLINT(facebook-cuda-safe-api-call-check)
    CUDACHECK_TEST(cudaDeviceSynchronize());
  }
};

// ==================================================================
// Passing Case Tests (kernel should complete without error)
// ==================================================================

// Test: PIPES_DEVICE_CHECK with true expression completes successfully.
// Verifies that the macro doesn't trigger a trap when the condition is true.
TEST_F(DeviceCheckTestFixture, DeviceCheckPassingDoesNotTrap) {
  const int numBlocks = 1;
  const int blockSize = 32;

  // Launch the kernel - should complete without error
  test::testDeviceCheckPassing(numBlocks, blockSize);

  // Verify kernel completed successfully
  cudaError_t syncError = cudaDeviceSynchronize();
  EXPECT_EQ(syncError, cudaSuccess)
      << "PIPES_DEVICE_CHECK with true expression should not cause error, got: "
      << cudaGetErrorString(syncError);
}

// Test: PIPES_DEVICE_CHECK_MSG with true expression completes successfully.
// Verifies that macro with custom message doesn't trap when condition is true.
TEST_F(DeviceCheckTestFixture, DeviceCheckMsgPassingDoesNotTrap) {
  const int numBlocks = 1;
  const int blockSize = 32;

  // Launch the kernel - should complete without error
  test::testDeviceCheckMsgPassing(numBlocks, blockSize);

  // Verify kernel completed successfully
  cudaError_t syncError = cudaDeviceSynchronize();
  EXPECT_EQ(syncError, cudaSuccess)
      << "PIPES_DEVICE_CHECK_MSG with true expression should not cause error, "
         "got: "
      << cudaGetErrorString(syncError);
}

// Test: Multiple PIPES_DEVICE_CHECK calls in a kernel all pass.
// Verifies that multiple checks can coexist in the same kernel and all
// evaluate correctly.
TEST_F(DeviceCheckTestFixture, MultipleDeviceChecksPassingAllSucceed) {
  const int numBlocks = 1;
  const int blockSize = 32;
  const uint32_t value = 5;
  const uint32_t threshold = 10;

  // Launch the kernel with value < threshold - all checks should pass
  test::testMultipleDeviceChecksPassing(value, threshold, numBlocks, blockSize);

  // Verify kernel completed successfully
  cudaError_t syncError = cudaDeviceSynchronize();
  EXPECT_EQ(syncError, cudaSuccess)
      << "Multiple passing PIPES_DEVICE_CHECK calls should not cause error, "
         "got: "
      << cudaGetErrorString(syncError);
}

// Test: PIPES_DEVICE_CHECK works correctly with larger block configurations.
// Verifies that the macro works across different thread configurations.
TEST_F(DeviceCheckTestFixture, DeviceCheckPassingMultipleBlocks) {
  const int numBlocks = 4;
  const int blockSize = 128;

  // Launch with multiple blocks - should complete without error
  test::testDeviceCheckPassing(numBlocks, blockSize);

  // Verify kernel completed successfully
  cudaError_t syncError = cudaDeviceSynchronize();
  EXPECT_EQ(syncError, cudaSuccess)
      << "PIPES_DEVICE_CHECK should work with multiple blocks, got: "
      << cudaGetErrorString(syncError);
}

// Test: PIPES_DEVICE_CHECK_MSG works correctly with larger block
// configurations. Verifies that the macro with message works across different
// thread configurations.
TEST_F(DeviceCheckTestFixture, DeviceCheckMsgPassingMultipleBlocks) {
  const int numBlocks = 4;
  const int blockSize = 128;

  // Launch with multiple blocks - should complete without error
  test::testDeviceCheckMsgPassing(numBlocks, blockSize);

  // Verify kernel completed successfully
  cudaError_t syncError = cudaDeviceSynchronize();
  EXPECT_EQ(syncError, cudaSuccess)
      << "PIPES_DEVICE_CHECK_MSG should work with multiple blocks, got: "
      << cudaGetErrorString(syncError);
}

} // namespace comms::pipes
