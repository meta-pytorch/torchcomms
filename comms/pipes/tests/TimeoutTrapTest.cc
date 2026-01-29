// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/tests/TimeoutTrapTest.cuh"
#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace comms::pipes::test {

class TimeoutTrapTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
      GTEST_SKIP() << "No CUDA devices available";
    }
    err = cudaSetDevice(0);
    ASSERT_EQ(err, cudaSuccess) << "Failed to set CUDA device";
  }

  void TearDown() override {
    // Reset device to clear any trap state.
    // These calls are intentionally unchecked - device may be in corrupted
    // state.
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaDeviceReset();
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaSetDevice(0);
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaGetLastError(); // Clear any pending errors
  }

  bool isExpectedTrapError(cudaError_t err) {
    // Different CUDA versions/drivers report traps differently
    return err == cudaErrorIllegalInstruction || err == cudaErrorAssert ||
        err == cudaErrorLaunchFailure;
  }
};

// Test that ChunkState::wait_ready_to_recv times out and traps
TEST_F(TimeoutTrapTest, ChunkStateWaitReadyToRecvTimeout) {
  // Use a short timeout (10ms) - should trigger quickly
  launchChunkStateTimeoutKernel(0, 10);

  cudaError_t err = cudaGetLastError();
  EXPECT_TRUE(isExpectedTrapError(err))
      << "Expected trap error, got: " << cudaGetErrorString(err);
}

// Test that SignalState::wait_until times out and traps
TEST_F(TimeoutTrapTest, SignalStateWaitUntilTimeout) {
  // Use a short timeout (10ms) - should trigger quickly
  launchSignalStateTimeoutKernel(0, 10);

  cudaError_t err = cudaGetLastError();
  EXPECT_TRUE(isExpectedTrapError(err))
      << "Expected trap error, got: " << cudaGetErrorString(err);
}

// Test that a kernel calling start() and check() completes successfully
// when the timeout has not expired (positive test case)
TEST_F(TimeoutTrapTest, NoTimeoutWhenKernelCompletes) {
  // Use a long timeout (1000ms) - kernel calls start() then check() once
  // and completes immediately without trapping
  launchNoTimeoutKernel(0, 1000);

  cudaError_t err = cudaGetLastError();
  EXPECT_EQ(err, cudaSuccess)
      << "Expected success, got: " << cudaGetErrorString(err);
}

// Test that ChunkState with ThreadGroup-based timeout checking works
TEST_F(TimeoutTrapTest, ChunkStateThreadGroupTimeout) {
  // Use a short timeout (10ms) - should trigger quickly
  // This tests the leader-only check(ThreadGroup&) path
  launchChunkStateThreadGroupTimeoutKernel(0, 10);

  cudaError_t err = cudaGetLastError();
  EXPECT_TRUE(isExpectedTrapError(err))
      << "Expected trap error, got: " << cudaGetErrorString(err);
}

// Test that SignalState with ThreadGroup-based timeout checking works
TEST_F(TimeoutTrapTest, SignalStateThreadGroupTimeout) {
  // Use a short timeout (10ms) - should trigger quickly
  // This tests the leader-only check(ThreadGroup&) path
  launchSignalStateThreadGroupTimeoutKernel(0, 10);

  cudaError_t err = cudaGetLastError();
  EXPECT_TRUE(isExpectedTrapError(err))
      << "Expected trap error, got: " << cudaGetErrorString(err);
}

// Test that calling start() twice traps (programming error detection)
TEST_F(TimeoutTrapTest, DoubleStartTraps) {
  // Calling start() twice is a programming error and should trap
  launchDoubleStartKernel(0, 1000);

  cudaError_t err = cudaGetLastError();
  EXPECT_TRUE(isExpectedTrapError(err))
      << "Expected trap error from double-start, got: "
      << cudaGetErrorString(err);
}

// Test that when a kernel traps, subsequent kernels on the same stream don't
// run
TEST_F(TimeoutTrapTest, TrapPreventsSubsequentKernelsOnStream) {
  // Launch two kernels on the same stream - first will trap, second should not
  // run
  bool secondKernelDidNotRun = launchMultipleKernelsOnStreamTest(0, 10);

  cudaError_t err = cudaGetLastError();
  EXPECT_TRUE(isExpectedTrapError(err))
      << "Expected trap error from first kernel, got: "
      << cudaGetErrorString(err);

  // Verify the second kernel did not execute
  EXPECT_TRUE(secondKernelDidNotRun)
      << "Second kernel should NOT have run after first kernel trapped";
}

} // namespace comms::pipes::test
