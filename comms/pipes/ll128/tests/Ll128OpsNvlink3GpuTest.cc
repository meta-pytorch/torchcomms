// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstddef>
#include <vector>

#include "comms/pipes/ll128/Ll128Packet.cuh"
#include "comms/pipes/ll128/tests/Ll128OpsNvlinkTest.cuh"
#include "comms/testinfra/TestXPlatUtils.h"

namespace comms::pipes {

// =============================================================================
// Three-GPU Test Fixture for 3-distinct-GPU LL128 forward tests (Gap 2)
// =============================================================================

class Ll128OpsNvlink3GpuTestFixture : public ::testing::Test {
 protected:
  static constexpr int kGpu0 = 0;
  static constexpr int kGpu1 = 1;
  static constexpr int kGpu2 = 2;

  void SetUp() override {
    int deviceCount = 0;
    CUDACHECK_TEST(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 3) {
      GTEST_SKIP() << "Test requires at least 3 GPUs";
    }

    // Check P2P access between all pairs
    int canAccess01 = 0, canAccess10 = 0;
    int canAccess12 = 0, canAccess21 = 0;
    int canAccess02 = 0, canAccess20 = 0;
    CUDACHECK_TEST(cudaDeviceCanAccessPeer(&canAccess01, kGpu0, kGpu1));
    CUDACHECK_TEST(cudaDeviceCanAccessPeer(&canAccess10, kGpu1, kGpu0));
    CUDACHECK_TEST(cudaDeviceCanAccessPeer(&canAccess12, kGpu1, kGpu2));
    CUDACHECK_TEST(cudaDeviceCanAccessPeer(&canAccess21, kGpu2, kGpu1));
    CUDACHECK_TEST(cudaDeviceCanAccessPeer(&canAccess02, kGpu0, kGpu2));
    CUDACHECK_TEST(cudaDeviceCanAccessPeer(&canAccess20, kGpu2, kGpu0));
    if (!canAccess01 || !canAccess10 || !canAccess12 || !canAccess21 ||
        !canAccess02 || !canAccess20) {
      GTEST_SKIP() << "Test requires P2P access between GPUs 0, 1, and 2";
    }

    // Enable P2P access between all pairs
    auto enable_peer = [](int from, int to) {
      CUDACHECK_TEST(cudaSetDevice(from));
      auto err = cudaDeviceEnablePeerAccess(to, 0);
      if (err == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
      } else if (err != cudaSuccess) {
        CUDACHECK_TEST(err);
      }
    };
    enable_peer(kGpu0, kGpu1);
    enable_peer(kGpu0, kGpu2);
    enable_peer(kGpu1, kGpu0);
    enable_peer(kGpu1, kGpu2);
    enable_peer(kGpu2, kGpu0);
    enable_peer(kGpu2, kGpu1);
  }

  void TearDown() override {
    cudaSetDevice(kGpu0);
    cudaDeviceSynchronize();
    cudaSetDevice(kGpu1);
    cudaDeviceSynchronize();
    cudaSetDevice(kGpu2);
    cudaDeviceSynchronize();
  }

  std::vector<char> make_pattern(size_t nbytes, int seed = 0) {
    std::vector<char> pattern(nbytes);
    for (size_t i = 0; i < nbytes; ++i) {
      pattern[i] = static_cast<char>((i + seed) & 0xFF);
    }
    return pattern;
  }

  /// Run a 3-GPU forward test: GPU0 sends, GPU1 forwards, GPU2 receives.
  void run_forward_3gpu_test(
      size_t nbytes,
      int num_blocks = 1,
      int block_size = 256,
      size_t buffer_num_packets = 0,
      int num_steps = 1) {
    auto pattern = make_pattern(nbytes);

    // src on GPU0 (sender)
    CUDACHECK_TEST(cudaSetDevice(kGpu0));
    char* src_d;
    CUDACHECK_TEST(cudaMalloc(&src_d, nbytes));
    CUDACHECK_TEST(
        cudaMemcpy(src_d, pattern.data(), nbytes, cudaMemcpyHostToDevice));

    // fwd_dst + buf_a on GPU1 (forwarder)
    CUDACHECK_TEST(cudaSetDevice(kGpu1));
    char* fwd_dst_d;
    CUDACHECK_TEST(cudaMalloc(&fwd_dst_d, nbytes));
    CUDACHECK_TEST(cudaMemset(fwd_dst_d, 0, nbytes));
    size_t buf_size = (buffer_num_packets > 0)
        ? buffer_num_packets * kLl128PacketSize
        : ll128_buffer_size(nbytes);
    Ll128Packet* ll128_buf_a;
    CUDACHECK_TEST(cudaMalloc(&ll128_buf_a, buf_size));

    // recv_dst + buf_b on GPU2 (receiver)
    CUDACHECK_TEST(cudaSetDevice(kGpu2));
    char* recv_dst_d;
    CUDACHECK_TEST(cudaMalloc(&recv_dst_d, nbytes));
    CUDACHECK_TEST(cudaMemset(recv_dst_d, 0, nbytes));
    Ll128Packet* ll128_buf_b;
    CUDACHECK_TEST(cudaMalloc(&ll128_buf_b, buf_size));

    test::test_ll128_nvlink_forward_3gpu(
        kGpu0,
        kGpu1,
        kGpu2,
        src_d,
        fwd_dst_d,
        recv_dst_d,
        nbytes,
        ll128_buf_a,
        ll128_buf_b,
        num_blocks,
        block_size,
        buffer_num_packets,
        num_steps);

    // Verify forwarder's local copy
    std::vector<char> fwd_result(nbytes);
    CUDACHECK_TEST(cudaSetDevice(kGpu1));
    CUDACHECK_TEST(cudaMemcpy(
        fwd_result.data(), fwd_dst_d, nbytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < nbytes; ++i) {
      ASSERT_EQ(fwd_result[i], pattern[i])
          << "3GPU Forward: fwd_dst mismatch at byte " << i;
    }

    // Verify receiver's output
    std::vector<char> recv_result(nbytes);
    CUDACHECK_TEST(cudaSetDevice(kGpu2));
    CUDACHECK_TEST(cudaMemcpy(
        recv_result.data(), recv_dst_d, nbytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < nbytes; ++i) {
      ASSERT_EQ(recv_result[i], pattern[i])
          << "3GPU Forward: recv_dst mismatch at byte " << i;
    }

    // Cleanup
    CUDACHECK_TEST(cudaSetDevice(kGpu0));
    CUDACHECK_TEST(cudaFree(src_d));
    CUDACHECK_TEST(cudaSetDevice(kGpu1));
    CUDACHECK_TEST(cudaFree(fwd_dst_d));
    CUDACHECK_TEST(cudaFree(ll128_buf_a));
    CUDACHECK_TEST(cudaSetDevice(kGpu2));
    CUDACHECK_TEST(cudaFree(recv_dst_d));
    CUDACHECK_TEST(cudaFree(ll128_buf_b));
  }
};

// =============================================================================
// 3-GPU Forward tests — sender ≠ forwarder ≠ receiver
// =============================================================================

TEST_F(Ll128OpsNvlink3GpuTestFixture, Forward_4KB_3GPU) {
  run_forward_3gpu_test(4096);
}

TEST_F(Ll128OpsNvlink3GpuTestFixture, Forward_4KB_3GPU_MultiStep_10) {
  run_forward_3gpu_test(
      4096,
      /*num_blocks=*/1,
      /*block_size=*/256,
      /*buffer_num_packets=*/0,
      /*num_steps=*/10);
}

TEST_F(Ll128OpsNvlink3GpuTestFixture, Forward_4KB_3GPU_Chunked_8pkt) {
  run_forward_3gpu_test(
      4096,
      /*num_blocks=*/1,
      /*block_size=*/256,
      /*buffer_num_packets=*/8);
}

// =============================================================================
// 3-GPU extended tests (Gap 9)
// =============================================================================

TEST_F(Ll128OpsNvlink3GpuTestFixture, Forward_64KB_3GPU_MultiBlock) {
  run_forward_3gpu_test(65536, /*num_blocks=*/4, /*block_size=*/256);
}

TEST_F(Ll128OpsNvlink3GpuTestFixture, Forward_3GPU_Chunked_MultiStep) {
  run_forward_3gpu_test(
      4096,
      /*num_blocks=*/1,
      /*block_size=*/256,
      /*buffer_num_packets=*/8,
      /*num_steps=*/10);
}

} // namespace comms::pipes
