// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/ll128/tests/Ll128TimeoutTrapTest.cuh"
#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace comms::pipes::test {

class Ll128TimeoutTrapTest : public ::testing::Test {
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
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaDeviceReset();
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaSetDevice(0);
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaGetLastError();
  }

  bool isExpectedTrapError(cudaError_t err) {
    return err == cudaErrorIllegalInstruction || err == cudaErrorAssert ||
        err == cudaErrorLaunchFailure;
  }
};

TEST_F(Ll128TimeoutTrapTest, SendNoRecv) {
  // Launch send with short timeout (10ms) and no receiver.
  // The sender polls for ACKs that never arrive, triggering __trap().
  launch_ll128_send_no_recv_timeout(0, 10);

  cudaError_t err = cudaGetLastError();
  EXPECT_TRUE(isExpectedTrapError(err))
      << "Expected trap error, got: " << cudaGetErrorString(err);
}

TEST_F(Ll128TimeoutTrapTest, SendRecv_Chunked_UndersizedBuffer_Trap) {
  // Launch send/recv with buffer_num_packets=2, below kLl128PacketsPerWarp=4.
  // PIPES_DEVICE_CHECK should fire, triggering __trap().
  launch_ll128_send_recv_undersized_buffer(0);

  cudaError_t err = cudaGetLastError();
  EXPECT_TRUE(isExpectedTrapError(err))
      << "Expected trap error for undersized buffer, got: "
      << cudaGetErrorString(err);
}

} // namespace comms::pipes::test
