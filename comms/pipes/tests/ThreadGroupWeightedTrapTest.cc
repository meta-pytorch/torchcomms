// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

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
// Trap Test - Weighted Partition
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

// Test: Verify that partition(weights) with more partitions than groups
// triggers a device-side trap. This validates the invariant that num_partitions
// <= total_groups.
//
// Note: __trap() causes an illegal instruction that stops kernel execution.
// After the trap fires, cudaDeviceSynchronize() returns an error.
TEST_F(
    ThreadGroupTrapTestFixture,
    WeightedPartitionMorePartitionsThanGroupsTraps) {
  // Setup: 1 block × 32 threads = 1 warp (total_groups = 1)
  // But we'll request 4 partitions, which is invalid (4 > 1)
  const int numBlocks = 1;
  const int blockSize = 32;
  const std::vector<uint32_t> weights = {1, 1, 1, 1}; // 4 partitions
  const uint32_t numPartitions = static_cast<uint32_t>(weights.size());

  // Use raw cudaMalloc instead of DeviceBuffer because we'll reset the device
  uint32_t* weights_d = nullptr;
  CUDACHECK_TEST(cudaMalloc(&weights_d, numPartitions * sizeof(uint32_t)));

  CUDACHECK_TEST(cudaMemcpy(
      weights_d,
      weights.data(),
      numPartitions * sizeof(uint32_t),
      cudaMemcpyHostToDevice));

  // Launch the kernel - this should trigger the device trap
  test::testWeightedPartitionMorePartitionsThanGroups(
      weights_d, numPartitions, numBlocks, blockSize);

  // Synchronize and check for error
  cudaError_t syncError = cudaDeviceSynchronize();

  // The trap should have fired, causing a CUDA error
  EXPECT_TRUE(
      syncError == cudaErrorIllegalInstruction ||
      syncError == cudaErrorAssert || syncError == cudaErrorLaunchFailure)
      << "Expected CUDA error when num_partitions > total_groups, but got: "
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
