// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "comms/pipes/tests/ThreadGroupTrapTest.cuh"

// CUDA error checking macro for tests
#define CUDACHECK_TEST(cmd)                                      \
  do {                                                           \
    cudaError_t err = (cmd);                                     \
    ASSERT_EQ(err, cudaSuccess)                                  \
        << "CUDA error: " << __FILE__ << ":" << __LINE__ << " '" \
        << cudaGetErrorString(err) << "'";                       \
  } while (0)

namespace comms::pipes {

// =============================================================================
// Trap Test - to_warp_group() with invalid group_size
// =============================================================================
//
// This test is in a separate binary because:
// 1. __trap() puts the CUDA device into an unrecoverable error state
// 2. cudaDeviceReset() is required to recover, but it invalidates all contexts
// 3. This would break any subsequent tests in the same process

class ThreadGroupTrapTestFixture : public ::testing::Test {
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

// Test: Verify that to_warp_group() traps when called on a group with
// group_size < 32 (e.g., make_thread_solo() which has group_size == 1).
// This validates the invariant that group_size must be >= 32 and a multiple
// of 32.
TEST_F(ThreadGroupTrapTestFixture, ToWarpGroupTrapsOnInvalidGroupSize) {
  const int numBlocks = 1;
  const int blockSize = 32;

  // Launch the kernel - this should trigger the device trap
  test::testToWarpGroupTrap(numBlocks, blockSize);

  // Synchronize and check for error
  cudaError_t syncError = cudaDeviceSynchronize();

  // The trap should have fired, causing a CUDA error
  EXPECT_TRUE(
      syncError == cudaErrorIllegalInstruction ||
      syncError == cudaErrorAssert || syncError == cudaErrorLaunchFailure)
      << "Expected CUDA error when to_warp_group() is called with "
         "group_size < 32, but got: "
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
