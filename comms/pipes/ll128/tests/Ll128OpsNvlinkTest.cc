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
// Two-GPU Test Fixture for cross-GPU LL128 ops
// =============================================================================

class Ll128OpsNvlinkTestFixture : public ::testing::Test {
 protected:
  static constexpr int kGpu0 = 0;
  static constexpr int kGpu1 = 1;

  void SetUp() override {
    int deviceCount = 0;
    CUDACHECK_TEST(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2) {
      GTEST_SKIP() << "Test requires at least 2 GPUs";
    }

    int canAccessPeer01 = 0;
    int canAccessPeer10 = 0;
    CUDACHECK_TEST(cudaDeviceCanAccessPeer(&canAccessPeer01, kGpu0, kGpu1));
    CUDACHECK_TEST(cudaDeviceCanAccessPeer(&canAccessPeer10, kGpu1, kGpu0));
    if (!canAccessPeer01 || !canAccessPeer10) {
      GTEST_SKIP() << "Test requires P2P access between GPU 0 and GPU 1";
    }

    // Enable bidirectional P2P access
    CUDACHECK_TEST(cudaSetDevice(kGpu0));
    auto err0 = cudaDeviceEnablePeerAccess(kGpu1, 0);
    if (err0 == cudaErrorPeerAccessAlreadyEnabled) {
      cudaGetLastError();
    } else if (err0 != cudaSuccess) {
      CUDACHECK_TEST(err0);
    }

    CUDACHECK_TEST(cudaSetDevice(kGpu1));
    auto err1 = cudaDeviceEnablePeerAccess(kGpu0, 0);
    if (err1 == cudaErrorPeerAccessAlreadyEnabled) {
      cudaGetLastError();
    } else if (err1 != cudaSuccess) {
      CUDACHECK_TEST(err1);
    }
  }

  void TearDown() override {
    cudaSetDevice(kGpu0);
    cudaDeviceSynchronize();
    cudaSetDevice(kGpu1);
    cudaDeviceSynchronize();
  }

  std::vector<char> make_pattern(size_t nbytes, int seed = 0) {
    std::vector<char> pattern(nbytes);
    for (size_t i = 0; i < nbytes; ++i) {
      pattern[i] = static_cast<char>((i + seed) & 0xFF);
    }
    return pattern;
  }

  /// Run a cross-GPU send/recv test: GPU0 sends, GPU1 receives via NVLink.
  void run_send_recv_test(
      size_t nbytes,
      int num_blocks = 1,
      int block_size = 256,
      size_t buffer_num_packets = 0,
      int num_steps = 1,
      int seed = 0) {
    auto pattern = make_pattern(nbytes, seed);

    // Allocate src on GPU0
    CUDACHECK_TEST(cudaSetDevice(kGpu0));
    char* src_d;
    CUDACHECK_TEST(cudaMalloc(&src_d, nbytes));
    CUDACHECK_TEST(
        cudaMemcpy(src_d, pattern.data(), nbytes, cudaMemcpyHostToDevice));

    // Allocate dst + LL128 buffer on GPU1
    CUDACHECK_TEST(cudaSetDevice(kGpu1));
    char* dst_d;
    CUDACHECK_TEST(cudaMalloc(&dst_d, nbytes));
    CUDACHECK_TEST(cudaMemset(dst_d, 0, nbytes));

    size_t buf_size = (buffer_num_packets > 0)
        ? buffer_num_packets * kLl128PacketSize
        : ll128_buffer_size(nbytes);
    Ll128Packet* ll128_buf;
    CUDACHECK_TEST(cudaMalloc(&ll128_buf, buf_size));

    // Run test
    test::test_ll128_nvlink_send_recv(
        kGpu0,
        kGpu1,
        src_d,
        dst_d,
        nbytes,
        ll128_buf,
        buffer_num_packets,
        num_steps,
        num_blocks,
        block_size);

    // Verify
    std::vector<char> result(nbytes);
    CUDACHECK_TEST(
        cudaMemcpy(result.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < nbytes; ++i) {
      ASSERT_EQ(result[i], pattern[i]) << "Mismatch at byte " << i;
    }

    // Cleanup
    CUDACHECK_TEST(cudaSetDevice(kGpu0));
    CUDACHECK_TEST(cudaFree(src_d));
    CUDACHECK_TEST(cudaSetDevice(kGpu1));
    CUDACHECK_TEST(cudaFree(dst_d));
    CUDACHECK_TEST(cudaFree(ll128_buf));
  }
};

// =============================================================================
// Send/Recv — various sizes over NVLink
// =============================================================================

TEST_F(Ll128OpsNvlinkTestFixture, SendRecv_16B) {
  run_send_recv_test(16);
}

TEST_F(Ll128OpsNvlinkTestFixture, SendRecv_112B) {
  // Exact 7-lane payload, no thread 7 payload data
  run_send_recv_test(112);
}

TEST_F(Ll128OpsNvlinkTestFixture, SendRecv_128B) {
  // Crosses packet boundary (2 packets, 2nd has 8B)
  run_send_recv_test(128);
}

TEST_F(Ll128OpsNvlinkTestFixture, SendRecv_480B) {
  // Exact 1-warp payload (4 full packets)
  run_send_recv_test(480);
}

TEST_F(Ll128OpsNvlinkTestFixture, SendRecv_4KB) {
  run_send_recv_test(4096);
}

TEST_F(Ll128OpsNvlinkTestFixture, SendRecv_64KB) {
  run_send_recv_test(65536, /*num_blocks=*/4, /*block_size=*/256);
}

// =============================================================================
// Multi-step — buffer reuse and ABA prevention over NVLink
// =============================================================================

TEST_F(Ll128OpsNvlinkTestFixture, SendRecv_MultiStep_10) {
  run_send_recv_test(
      4096,
      /*num_blocks=*/1,
      /*block_size=*/256,
      /*buffer_num_packets=*/0,
      /*num_steps=*/10);
}

TEST_F(Ll128OpsNvlinkTestFixture, SendRecv_MultiStep_ABA_100) {
  run_send_recv_test(
      4096,
      /*num_blocks=*/1,
      /*block_size=*/256,
      /*buffer_num_packets=*/0,
      /*num_steps=*/100);
}

// =============================================================================
// Chunked — windowed buffer over NVLink
// =============================================================================

TEST_F(Ll128OpsNvlinkTestFixture, SendRecv_Chunked_4KB_8pkt) {
  run_send_recv_test(
      4096, /*num_blocks=*/1, /*block_size=*/256, /*buffer_num_packets=*/8);
}

TEST_F(Ll128OpsNvlinkTestFixture, SendRecv_Chunked_MultiStep_10) {
  run_send_recv_test(
      4096,
      /*num_blocks=*/1,
      /*block_size=*/256,
      /*buffer_num_packets=*/8,
      /*num_steps=*/10);
}

// =============================================================================
// Stress — host-side loop with different patterns over NVLink
// =============================================================================

TEST_F(Ll128OpsNvlinkTestFixture, SendRecv_Stress_50) {
  constexpr int kStressIterations = 50;
  const size_t nbytes = 4096;

  // Allocate once, reuse across iterations
  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  char* src_d;
  CUDACHECK_TEST(cudaMalloc(&src_d, nbytes));

  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  char* dst_d;
  CUDACHECK_TEST(cudaMalloc(&dst_d, nbytes));
  size_t buf_size = ll128_buffer_size(nbytes);
  Ll128Packet* ll128_buf;
  CUDACHECK_TEST(cudaMalloc(&ll128_buf, buf_size));

  for (int iter = 0; iter < kStressIterations; ++iter) {
    auto pattern = make_pattern(nbytes, /*seed=*/iter);

    CUDACHECK_TEST(cudaSetDevice(kGpu0));
    CUDACHECK_TEST(
        cudaMemcpy(src_d, pattern.data(), nbytes, cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaSetDevice(kGpu1));
    CUDACHECK_TEST(cudaMemset(dst_d, 0, nbytes));

    test::test_ll128_nvlink_send_recv(
        kGpu0,
        kGpu1,
        src_d,
        dst_d,
        nbytes,
        ll128_buf,
        /*buffer_num_packets=*/0,
        /*num_steps=*/1,
        /*num_blocks=*/1,
        /*block_size=*/256);

    std::vector<char> result(nbytes);
    CUDACHECK_TEST(
        cudaMemcpy(result.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < nbytes; ++i) {
      ASSERT_EQ(result[i], pattern[i])
          << "Stress iter " << iter << ": mismatch at byte " << i;
    }
  }

  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  CUDACHECK_TEST(cudaFree(src_d));
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  CUDACHECK_TEST(cudaFree(dst_d));
  CUDACHECK_TEST(cudaFree(ll128_buf));
}

// =============================================================================
// Large messages — ops-level coverage beyond 64KB (G4)
// =============================================================================

TEST_F(Ll128OpsNvlinkTestFixture, SendRecv_256KB) {
  run_send_recv_test(256 * 1024, /*num_blocks=*/4, /*block_size=*/512);
}

// =============================================================================
// 512-thread blocks — matches production auto-tune config (G3)
// =============================================================================

TEST_F(Ll128OpsNvlinkTestFixture, SendRecv_512t_4KB) {
  run_send_recv_test(4096, /*num_blocks=*/1, /*block_size=*/512);
}

TEST_F(Ll128OpsNvlinkTestFixture, SendRecv_512t_64KB) {
  run_send_recv_test(65536, /*num_blocks=*/4, /*block_size=*/512);
}

// =============================================================================
// Chunked multi-block over NVLink (G9)
// =============================================================================

TEST_F(Ll128OpsNvlinkTestFixture, SendRecv_Chunked_64KB_8pkt_MultiBlock) {
  run_send_recv_test(
      65536, /*num_blocks=*/4, /*block_size=*/256, /*buffer_num_packets=*/8);
}

// =============================================================================
// Windowed mode over NVLink (G2)
// =============================================================================

TEST_F(Ll128OpsNvlinkTestFixture, SendRecv_Windowed_4KB_8pkt) {
  run_send_recv_test(
      4096, /*num_blocks=*/1, /*block_size=*/64, /*buffer_num_packets=*/8);
}

TEST_F(Ll128OpsNvlinkTestFixture, SendRecv_Windowed_64KB_64pkt) {
  run_send_recv_test(
      65536, /*num_blocks=*/2, /*block_size=*/128, /*buffer_num_packets=*/64);
}

// =============================================================================
// Zero-byte over NVLink (G8)
// =============================================================================

TEST_F(Ll128OpsNvlinkTestFixture, SendRecv_ZeroBytes) {
  run_send_recv_test(0);
}

// =============================================================================
// Forward — 3-role pipeline over NVLink on both hops
// =============================================================================

TEST_F(Ll128OpsNvlinkTestFixture, Forward_4KB) {
  const size_t nbytes = 4096;
  auto pattern = make_pattern(nbytes);

  // src on GPU0
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
  size_t buf_size = ll128_buffer_size(nbytes);
  Ll128Packet* ll128_buf_a;
  CUDACHECK_TEST(cudaMalloc(&ll128_buf_a, buf_size));

  // recv_dst + buf_b on GPU0 (receiver = sender GPU)
  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  char* recv_dst_d;
  CUDACHECK_TEST(cudaMalloc(&recv_dst_d, nbytes));
  CUDACHECK_TEST(cudaMemset(recv_dst_d, 0, nbytes));
  Ll128Packet* ll128_buf_b;
  CUDACHECK_TEST(cudaMalloc(&ll128_buf_b, buf_size));

  test::test_ll128_nvlink_forward(
      kGpu0,
      kGpu1,
      kGpu0,
      src_d,
      fwd_dst_d,
      recv_dst_d,
      nbytes,
      ll128_buf_a,
      ll128_buf_b,
      /*num_blocks=*/1,
      /*block_size=*/256);

  // Verify forwarder's local copy
  std::vector<char> fwd_result(nbytes);
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  CUDACHECK_TEST(
      cudaMemcpy(fwd_result.data(), fwd_dst_d, nbytes, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < nbytes; ++i) {
    ASSERT_EQ(fwd_result[i], pattern[i])
        << "Forward: fwd_dst mismatch at byte " << i;
  }

  // Verify receiver's output
  std::vector<char> recv_result(nbytes);
  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  CUDACHECK_TEST(cudaMemcpy(
      recv_result.data(), recv_dst_d, nbytes, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < nbytes; ++i) {
    ASSERT_EQ(recv_result[i], pattern[i])
        << "Forward: recv_dst mismatch at byte " << i;
  }

  // Cleanup
  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  CUDACHECK_TEST(cudaFree(src_d));
  CUDACHECK_TEST(cudaFree(recv_dst_d));
  CUDACHECK_TEST(cudaFree(ll128_buf_b));
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  CUDACHECK_TEST(cudaFree(fwd_dst_d));
  CUDACHECK_TEST(cudaFree(ll128_buf_a));
}

// =============================================================================
// Bidirectional — GPU0↔GPU1 simultaneous, 4 concurrent kernels
// =============================================================================

TEST_F(Ll128OpsNvlinkTestFixture, Bidirectional_4KB) {
  const size_t nbytes = 4096;
  auto pattern0 = make_pattern(nbytes, /*seed=*/0);
  auto pattern1 = make_pattern(nbytes, /*seed=*/42);

  // GPU0: src0, dst0, ll128_buf_on_gpu0
  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  char* src0_d;
  CUDACHECK_TEST(cudaMalloc(&src0_d, nbytes));
  CUDACHECK_TEST(
      cudaMemcpy(src0_d, pattern0.data(), nbytes, cudaMemcpyHostToDevice));
  char* dst0_d;
  CUDACHECK_TEST(cudaMalloc(&dst0_d, nbytes));
  CUDACHECK_TEST(cudaMemset(dst0_d, 0, nbytes));
  size_t buf_size = ll128_buffer_size(nbytes);
  Ll128Packet* ll128_buf_on_gpu0;
  CUDACHECK_TEST(cudaMalloc(&ll128_buf_on_gpu0, buf_size));

  // GPU1: src1, dst1, ll128_buf_on_gpu1
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  char* src1_d;
  CUDACHECK_TEST(cudaMalloc(&src1_d, nbytes));
  CUDACHECK_TEST(
      cudaMemcpy(src1_d, pattern1.data(), nbytes, cudaMemcpyHostToDevice));
  char* dst1_d;
  CUDACHECK_TEST(cudaMalloc(&dst1_d, nbytes));
  CUDACHECK_TEST(cudaMemset(dst1_d, 0, nbytes));
  Ll128Packet* ll128_buf_on_gpu1;
  CUDACHECK_TEST(cudaMalloc(&ll128_buf_on_gpu1, buf_size));

  test::test_ll128_nvlink_bidirectional(
      src0_d,
      dst0_d,
      src1_d,
      dst1_d,
      nbytes,
      ll128_buf_on_gpu0,
      ll128_buf_on_gpu1,
      /*num_blocks=*/1,
      /*block_size=*/256);

  // Verify GPU0→GPU1: dst1 should match pattern0
  std::vector<char> result1(nbytes);
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  CUDACHECK_TEST(
      cudaMemcpy(result1.data(), dst1_d, nbytes, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < nbytes; ++i) {
    ASSERT_EQ(result1[i], pattern0[i])
        << "Bidirectional GPU0→GPU1: mismatch at byte " << i;
  }

  // Verify GPU1→GPU0: dst0 should match pattern1
  std::vector<char> result0(nbytes);
  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  CUDACHECK_TEST(
      cudaMemcpy(result0.data(), dst0_d, nbytes, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < nbytes; ++i) {
    ASSERT_EQ(result0[i], pattern1[i])
        << "Bidirectional GPU1→GPU0: mismatch at byte " << i;
  }

  // Cleanup
  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  CUDACHECK_TEST(cudaFree(src0_d));
  CUDACHECK_TEST(cudaFree(dst0_d));
  CUDACHECK_TEST(cudaFree(ll128_buf_on_gpu0));
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  CUDACHECK_TEST(cudaFree(src1_d));
  CUDACHECK_TEST(cudaFree(dst1_d));
  CUDACHECK_TEST(cudaFree(ll128_buf_on_gpu1));
}

// =============================================================================
// Forward variants — multi-step, multi-block, chunked over NVLink (G1, G12)
// =============================================================================

TEST_F(Ll128OpsNvlinkTestFixture, Forward_4KB_MultiStep_10) {
  const size_t nbytes = 4096;
  auto pattern = make_pattern(nbytes);

  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  char* src_d;
  CUDACHECK_TEST(cudaMalloc(&src_d, nbytes));
  CUDACHECK_TEST(
      cudaMemcpy(src_d, pattern.data(), nbytes, cudaMemcpyHostToDevice));

  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  char* fwd_dst_d;
  CUDACHECK_TEST(cudaMalloc(&fwd_dst_d, nbytes));
  CUDACHECK_TEST(cudaMemset(fwd_dst_d, 0, nbytes));
  size_t buf_size = ll128_buffer_size(nbytes);
  Ll128Packet* ll128_buf_a;
  CUDACHECK_TEST(cudaMalloc(&ll128_buf_a, buf_size));

  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  char* recv_dst_d;
  CUDACHECK_TEST(cudaMalloc(&recv_dst_d, nbytes));
  CUDACHECK_TEST(cudaMemset(recv_dst_d, 0, nbytes));
  Ll128Packet* ll128_buf_b;
  CUDACHECK_TEST(cudaMalloc(&ll128_buf_b, buf_size));

  test::test_ll128_nvlink_forward(
      kGpu0,
      kGpu1,
      kGpu0,
      src_d,
      fwd_dst_d,
      recv_dst_d,
      nbytes,
      ll128_buf_a,
      ll128_buf_b,
      /*num_blocks=*/1,
      /*block_size=*/256,
      /*buffer_num_packets=*/0,
      /*num_steps=*/10);

  std::vector<char> fwd_result(nbytes);
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  CUDACHECK_TEST(
      cudaMemcpy(fwd_result.data(), fwd_dst_d, nbytes, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < nbytes; ++i) {
    ASSERT_EQ(fwd_result[i], pattern[i])
        << "Forward multi-step: fwd_dst mismatch at byte " << i;
  }

  std::vector<char> recv_result(nbytes);
  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  CUDACHECK_TEST(cudaMemcpy(
      recv_result.data(), recv_dst_d, nbytes, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < nbytes; ++i) {
    ASSERT_EQ(recv_result[i], pattern[i])
        << "Forward multi-step: recv_dst mismatch at byte " << i;
  }

  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  CUDACHECK_TEST(cudaFree(src_d));
  CUDACHECK_TEST(cudaFree(recv_dst_d));
  CUDACHECK_TEST(cudaFree(ll128_buf_b));
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  CUDACHECK_TEST(cudaFree(fwd_dst_d));
  CUDACHECK_TEST(cudaFree(ll128_buf_a));
}

TEST_F(Ll128OpsNvlinkTestFixture, Forward_64KB_MultiBlock) {
  const size_t nbytes = 65536;
  auto pattern = make_pattern(nbytes);

  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  char* src_d;
  CUDACHECK_TEST(cudaMalloc(&src_d, nbytes));
  CUDACHECK_TEST(
      cudaMemcpy(src_d, pattern.data(), nbytes, cudaMemcpyHostToDevice));

  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  char* fwd_dst_d;
  CUDACHECK_TEST(cudaMalloc(&fwd_dst_d, nbytes));
  CUDACHECK_TEST(cudaMemset(fwd_dst_d, 0, nbytes));
  size_t buf_size = ll128_buffer_size(nbytes);
  Ll128Packet* ll128_buf_a;
  CUDACHECK_TEST(cudaMalloc(&ll128_buf_a, buf_size));

  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  char* recv_dst_d;
  CUDACHECK_TEST(cudaMalloc(&recv_dst_d, nbytes));
  CUDACHECK_TEST(cudaMemset(recv_dst_d, 0, nbytes));
  Ll128Packet* ll128_buf_b;
  CUDACHECK_TEST(cudaMalloc(&ll128_buf_b, buf_size));

  test::test_ll128_nvlink_forward(
      kGpu0,
      kGpu1,
      kGpu0,
      src_d,
      fwd_dst_d,
      recv_dst_d,
      nbytes,
      ll128_buf_a,
      ll128_buf_b,
      /*num_blocks=*/4,
      /*block_size=*/256);

  std::vector<char> fwd_result(nbytes);
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  CUDACHECK_TEST(
      cudaMemcpy(fwd_result.data(), fwd_dst_d, nbytes, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < nbytes; ++i) {
    ASSERT_EQ(fwd_result[i], pattern[i])
        << "Forward multi-block: fwd_dst mismatch at byte " << i;
  }

  std::vector<char> recv_result(nbytes);
  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  CUDACHECK_TEST(cudaMemcpy(
      recv_result.data(), recv_dst_d, nbytes, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < nbytes; ++i) {
    ASSERT_EQ(recv_result[i], pattern[i])
        << "Forward multi-block: recv_dst mismatch at byte " << i;
  }

  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  CUDACHECK_TEST(cudaFree(src_d));
  CUDACHECK_TEST(cudaFree(recv_dst_d));
  CUDACHECK_TEST(cudaFree(ll128_buf_b));
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  CUDACHECK_TEST(cudaFree(fwd_dst_d));
  CUDACHECK_TEST(cudaFree(ll128_buf_a));
}

TEST_F(Ll128OpsNvlinkTestFixture, Forward_4KB_Chunked_8pkt) {
  const size_t nbytes = 4096;
  auto pattern = make_pattern(nbytes);

  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  char* src_d;
  CUDACHECK_TEST(cudaMalloc(&src_d, nbytes));
  CUDACHECK_TEST(
      cudaMemcpy(src_d, pattern.data(), nbytes, cudaMemcpyHostToDevice));

  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  char* fwd_dst_d;
  CUDACHECK_TEST(cudaMalloc(&fwd_dst_d, nbytes));
  CUDACHECK_TEST(cudaMemset(fwd_dst_d, 0, nbytes));
  size_t buf_size = 8 * kLl128PacketSize;
  Ll128Packet* ll128_buf_a;
  CUDACHECK_TEST(cudaMalloc(&ll128_buf_a, buf_size));

  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  char* recv_dst_d;
  CUDACHECK_TEST(cudaMalloc(&recv_dst_d, nbytes));
  CUDACHECK_TEST(cudaMemset(recv_dst_d, 0, nbytes));
  Ll128Packet* ll128_buf_b;
  CUDACHECK_TEST(cudaMalloc(&ll128_buf_b, buf_size));

  test::test_ll128_nvlink_forward(
      kGpu0,
      kGpu1,
      kGpu0,
      src_d,
      fwd_dst_d,
      recv_dst_d,
      nbytes,
      ll128_buf_a,
      ll128_buf_b,
      /*num_blocks=*/1,
      /*block_size=*/256,
      /*buffer_num_packets=*/8);

  std::vector<char> fwd_result(nbytes);
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  CUDACHECK_TEST(
      cudaMemcpy(fwd_result.data(), fwd_dst_d, nbytes, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < nbytes; ++i) {
    ASSERT_EQ(fwd_result[i], pattern[i])
        << "Forward chunked: fwd_dst mismatch at byte " << i;
  }

  std::vector<char> recv_result(nbytes);
  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  CUDACHECK_TEST(cudaMemcpy(
      recv_result.data(), recv_dst_d, nbytes, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < nbytes; ++i) {
    ASSERT_EQ(recv_result[i], pattern[i])
        << "Forward chunked: recv_dst mismatch at byte " << i;
  }

  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  CUDACHECK_TEST(cudaFree(src_d));
  CUDACHECK_TEST(cudaFree(recv_dst_d));
  CUDACHECK_TEST(cudaFree(ll128_buf_b));
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  CUDACHECK_TEST(cudaFree(fwd_dst_d));
  CUDACHECK_TEST(cudaFree(ll128_buf_a));
}

// =============================================================================
// Bidirectional variants — chunked and multi-step over NVLink (G6)
// =============================================================================

TEST_F(Ll128OpsNvlinkTestFixture, Bidirectional_Chunked_4KB_8pkt) {
  const size_t nbytes = 4096;
  auto pattern0 = make_pattern(nbytes, /*seed=*/0);
  auto pattern1 = make_pattern(nbytes, /*seed=*/42);

  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  char* src0_d;
  CUDACHECK_TEST(cudaMalloc(&src0_d, nbytes));
  CUDACHECK_TEST(
      cudaMemcpy(src0_d, pattern0.data(), nbytes, cudaMemcpyHostToDevice));
  char* dst0_d;
  CUDACHECK_TEST(cudaMalloc(&dst0_d, nbytes));
  CUDACHECK_TEST(cudaMemset(dst0_d, 0, nbytes));
  size_t buf_size = 8 * kLl128PacketSize;
  Ll128Packet* ll128_buf_on_gpu0;
  CUDACHECK_TEST(cudaMalloc(&ll128_buf_on_gpu0, buf_size));

  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  char* src1_d;
  CUDACHECK_TEST(cudaMalloc(&src1_d, nbytes));
  CUDACHECK_TEST(
      cudaMemcpy(src1_d, pattern1.data(), nbytes, cudaMemcpyHostToDevice));
  char* dst1_d;
  CUDACHECK_TEST(cudaMalloc(&dst1_d, nbytes));
  CUDACHECK_TEST(cudaMemset(dst1_d, 0, nbytes));
  Ll128Packet* ll128_buf_on_gpu1;
  CUDACHECK_TEST(cudaMalloc(&ll128_buf_on_gpu1, buf_size));

  test::test_ll128_nvlink_bidirectional(
      src0_d,
      dst0_d,
      src1_d,
      dst1_d,
      nbytes,
      ll128_buf_on_gpu0,
      ll128_buf_on_gpu1,
      /*num_blocks=*/1,
      /*block_size=*/256,
      /*buffer_num_packets=*/8);

  std::vector<char> result1(nbytes);
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  CUDACHECK_TEST(
      cudaMemcpy(result1.data(), dst1_d, nbytes, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < nbytes; ++i) {
    ASSERT_EQ(result1[i], pattern0[i])
        << "Bidirectional chunked GPU0→GPU1: mismatch at byte " << i;
  }

  std::vector<char> result0(nbytes);
  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  CUDACHECK_TEST(
      cudaMemcpy(result0.data(), dst0_d, nbytes, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < nbytes; ++i) {
    ASSERT_EQ(result0[i], pattern1[i])
        << "Bidirectional chunked GPU1→GPU0: mismatch at byte " << i;
  }

  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  CUDACHECK_TEST(cudaFree(src0_d));
  CUDACHECK_TEST(cudaFree(dst0_d));
  CUDACHECK_TEST(cudaFree(ll128_buf_on_gpu0));
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  CUDACHECK_TEST(cudaFree(src1_d));
  CUDACHECK_TEST(cudaFree(dst1_d));
  CUDACHECK_TEST(cudaFree(ll128_buf_on_gpu1));
}

TEST_F(Ll128OpsNvlinkTestFixture, Bidirectional_MultiStep_10) {
  const size_t nbytes = 4096;
  auto pattern0 = make_pattern(nbytes, /*seed=*/0);
  auto pattern1 = make_pattern(nbytes, /*seed=*/42);

  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  char* src0_d;
  CUDACHECK_TEST(cudaMalloc(&src0_d, nbytes));
  CUDACHECK_TEST(
      cudaMemcpy(src0_d, pattern0.data(), nbytes, cudaMemcpyHostToDevice));
  char* dst0_d;
  CUDACHECK_TEST(cudaMalloc(&dst0_d, nbytes));
  CUDACHECK_TEST(cudaMemset(dst0_d, 0, nbytes));
  size_t buf_size = ll128_buffer_size(nbytes);
  Ll128Packet* ll128_buf_on_gpu0;
  CUDACHECK_TEST(cudaMalloc(&ll128_buf_on_gpu0, buf_size));

  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  char* src1_d;
  CUDACHECK_TEST(cudaMalloc(&src1_d, nbytes));
  CUDACHECK_TEST(
      cudaMemcpy(src1_d, pattern1.data(), nbytes, cudaMemcpyHostToDevice));
  char* dst1_d;
  CUDACHECK_TEST(cudaMalloc(&dst1_d, nbytes));
  CUDACHECK_TEST(cudaMemset(dst1_d, 0, nbytes));
  Ll128Packet* ll128_buf_on_gpu1;
  CUDACHECK_TEST(cudaMalloc(&ll128_buf_on_gpu1, buf_size));

  test::test_ll128_nvlink_bidirectional(
      src0_d,
      dst0_d,
      src1_d,
      dst1_d,
      nbytes,
      ll128_buf_on_gpu0,
      ll128_buf_on_gpu1,
      /*num_blocks=*/1,
      /*block_size=*/256,
      /*buffer_num_packets=*/0,
      /*num_steps=*/10);

  std::vector<char> result1(nbytes);
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  CUDACHECK_TEST(
      cudaMemcpy(result1.data(), dst1_d, nbytes, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < nbytes; ++i) {
    ASSERT_EQ(result1[i], pattern0[i])
        << "Bidirectional multi-step GPU0→GPU1: mismatch at byte " << i;
  }

  std::vector<char> result0(nbytes);
  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  CUDACHECK_TEST(
      cudaMemcpy(result0.data(), dst0_d, nbytes, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < nbytes; ++i) {
    ASSERT_EQ(result0[i], pattern1[i])
        << "Bidirectional multi-step GPU1→GPU0: mismatch at byte " << i;
  }

  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  CUDACHECK_TEST(cudaFree(src0_d));
  CUDACHECK_TEST(cudaFree(dst0_d));
  CUDACHECK_TEST(cudaFree(ll128_buf_on_gpu0));
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  CUDACHECK_TEST(cudaFree(src1_d));
  CUDACHECK_TEST(cudaFree(dst1_d));
  CUDACHECK_TEST(cudaFree(ll128_buf_on_gpu1));
}

} // namespace comms::pipes
