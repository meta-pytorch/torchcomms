// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstddef>
#include <string>
#include <vector>

#include "comms/pipes/ll128/Ll128Packet.cuh"
#include "comms/pipes/ll128/tests/Ll128OpsNvlinkTest.cuh"
#include "comms/testinfra/TestXPlatUtils.h"

namespace comms::pipes {

// =============================================================================
// Test parameters
// =============================================================================

struct TestParam {
  std::string name;
  size_t nbytes;
  int num_blocks = 1;
  int block_size = 256;
  size_t buffer_num_packets = 0;
  int num_steps = 1;
};

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

  /// Run a 3-role forward test: GPU0→GPU1(forward)→GPU0.
  void run_forward_test(
      size_t nbytes,
      int num_blocks = 1,
      int block_size = 256,
      size_t buffer_num_packets = 0,
      int num_steps = 1) {
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
    size_t buf_size = (buffer_num_packets > 0)
        ? buffer_num_packets * kLl128PacketSize
        : ll128_buffer_size(nbytes);
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

  /// Run a bidirectional test: GPU0↔GPU1 simultaneous send/recv.
  void run_bidirectional_test(
      size_t nbytes,
      int num_blocks = 1,
      int block_size = 256,
      size_t buffer_num_packets = 0,
      int num_steps = 1) {
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
    size_t buf_size = (buffer_num_packets > 0)
        ? buffer_num_packets * kLl128PacketSize
        : ll128_buffer_size(nbytes);
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
        num_blocks,
        block_size,
        buffer_num_packets,
        num_steps);

    // Verify GPU0->GPU1
    std::vector<char> result1(nbytes);
    CUDACHECK_TEST(cudaSetDevice(kGpu1));
    CUDACHECK_TEST(
        cudaMemcpy(result1.data(), dst1_d, nbytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < nbytes; ++i) {
      ASSERT_EQ(result1[i], pattern0[i])
          << "Bidirectional GPU0->GPU1: mismatch at byte " << i;
    }

    // Verify GPU1->GPU0
    std::vector<char> result0(nbytes);
    CUDACHECK_TEST(cudaSetDevice(kGpu0));
    CUDACHECK_TEST(
        cudaMemcpy(result0.data(), dst0_d, nbytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < nbytes; ++i) {
      ASSERT_EQ(result0[i], pattern1[i])
          << "Bidirectional GPU1->GPU0: mismatch at byte " << i;
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
};

// =============================================================================
// SendRecv — parameterized suite
// =============================================================================

class Ll128OpsNvlinkSendRecvTest
    : public Ll128OpsNvlinkTestFixture,
      public ::testing::WithParamInterface<TestParam> {};

TEST_P(Ll128OpsNvlinkSendRecvTest, SendRecv) {
  const auto& p = GetParam();
  run_send_recv_test(
      p.nbytes, p.num_blocks, p.block_size, p.buffer_num_packets, p.num_steps);
}

INSTANTIATE_TEST_SUITE_P(
    Ll128NvlinkSendRecv,
    Ll128OpsNvlinkSendRecvTest,
    ::testing::Values(
        // Sub-packet and boundary sizes
        TestParam{"16B", 16},
        TestParam{"112B", 112}, // 7-lane payload, no flag-lane data
        TestParam{"128B", 128}, // Crosses packet boundary (2nd pkt has 8B)
        TestParam{"480B", 480}, // Exact 1-warp payload (4 full packets)
        // Standard sizes
        TestParam{"4KB", 4096},
        TestParam{"64KB", 65536, 4, 256},
        TestParam{"256KB", 256 * 1024, 4, 512},
        TestParam{"1MB", 1024 * 1024, 8, 512},
        TestParam{"2MB", 2 * 1024 * 1024, 16, 512},
        // 512-thread blocks (production auto-tune config)
        TestParam{"512t_4KB", 4096, 1, 512},
        TestParam{"512t_64KB", 65536, 4, 512},
        // Edge case
        TestParam{"ZeroBytes", 0},
        // Multi-step (buffer reuse / ABA prevention)
        TestParam{"MultiStep_10", 4096, 1, 256, 0, 10},
        TestParam{"MultiStep_ABA_100", 4096, 1, 256, 0, 100},
        // Chunked (windowed buffer)
        TestParam{"Chunked_4KB_8pkt", 4096, 1, 256, 8},
        TestParam{"Chunked_MultiStep_10", 4096, 1, 256, 8, 10},
        TestParam{"Chunked_64KB_8pkt_MultiBlock", 65536, 4, 256, 8},
        TestParam{"Chunked_512t_64KB_8pkt", 65536, 4, 512, 8},
        TestParam{"Chunked_256KB_8Pkt", 256 * 1024, 4, 256, 8},
        // Windowed mode
        TestParam{"Windowed_4KB_8pkt", 4096, 1, 64, 8},
        TestParam{"Windowed_64KB_64pkt", 65536, 2, 128, 64}),
    [](const auto& info) { return info.param.name; });

// =============================================================================
// SendRecv — stress tests (standalone, unique host-side iteration structure)
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

TEST_F(Ll128OpsNvlinkTestFixture, SendRecv_Chunked_Stress_50) {
  constexpr int kStressIterations = 50;
  const size_t nbytes = 4096;
  const size_t buffer_num_packets = 8;

  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  char* src_d;
  CUDACHECK_TEST(cudaMalloc(&src_d, nbytes));

  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  char* dst_d;
  CUDACHECK_TEST(cudaMalloc(&dst_d, nbytes));
  size_t buf_size = buffer_num_packets * kLl128PacketSize;
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
        buffer_num_packets,
        /*num_steps=*/1,
        /*num_blocks=*/1,
        /*block_size=*/256);

    std::vector<char> result(nbytes);
    CUDACHECK_TEST(
        cudaMemcpy(result.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < nbytes; ++i) {
      ASSERT_EQ(result[i], pattern[i])
          << "Chunked stress iter " << iter << ": mismatch at byte " << i;
    }
  }

  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  CUDACHECK_TEST(cudaFree(src_d));
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  CUDACHECK_TEST(cudaFree(dst_d));
  CUDACHECK_TEST(cudaFree(ll128_buf));
}

// =============================================================================
// SendRecv — varying-data multi-step (unique API:
// test_ll128_nvlink_varying_send_recv)
// =============================================================================

TEST_F(Ll128OpsNvlinkTestFixture, SendRecv_VaryingData_MultiStep) {
  const size_t nbytes = 4096;
  const int num_steps = 10;
  const size_t total_bytes = num_steps * nbytes;

  // Build per-step patterns
  std::vector<char> src_host(total_bytes);
  for (int step = 0; step < num_steps; ++step) {
    for (size_t i = 0; i < nbytes; ++i) {
      src_host[step * nbytes + i] = static_cast<char>((i + step * 37) & 0xFF);
    }
  }

  // Allocate src on GPU0
  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  char* src_d;
  CUDACHECK_TEST(cudaMalloc(&src_d, total_bytes));
  CUDACHECK_TEST(
      cudaMemcpy(src_d, src_host.data(), total_bytes, cudaMemcpyHostToDevice));

  // Allocate dst + LL128 buffer on GPU1
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  char* dst_d;
  CUDACHECK_TEST(cudaMalloc(&dst_d, total_bytes));
  CUDACHECK_TEST(cudaMemset(dst_d, 0, total_bytes));
  size_t buf_size = ll128_buffer_size(nbytes);
  Ll128Packet* ll128_buf;
  CUDACHECK_TEST(cudaMalloc(&ll128_buf, buf_size));

  test::test_ll128_nvlink_varying_send_recv(
      kGpu0,
      kGpu1,
      src_d,
      dst_d,
      nbytes,
      ll128_buf,
      /*buffer_num_packets=*/0,
      num_steps,
      /*num_blocks=*/1,
      /*block_size=*/256);

  std::vector<char> result(total_bytes);
  CUDACHECK_TEST(
      cudaMemcpy(result.data(), dst_d, total_bytes, cudaMemcpyDeviceToHost));
  for (int step = 0; step < num_steps; ++step) {
    for (size_t i = 0; i < nbytes; ++i) {
      ASSERT_EQ(result[step * nbytes + i], src_host[step * nbytes + i])
          << "VaryingData NVLink step " << step << ": mismatch at byte " << i;
    }
  }

  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  CUDACHECK_TEST(cudaFree(src_d));
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  CUDACHECK_TEST(cudaFree(dst_d));
  CUDACHECK_TEST(cudaFree(ll128_buf));
}

TEST_F(Ll128OpsNvlinkTestFixture, SendRecv_VaryingData_MultiStep_Chunked) {
  const size_t nbytes = 4096;
  const int num_steps = 10;
  const size_t buffer_num_packets = 8;
  const size_t total_bytes = num_steps * nbytes;

  std::vector<char> src_host(total_bytes);
  for (int step = 0; step < num_steps; ++step) {
    for (size_t i = 0; i < nbytes; ++i) {
      src_host[step * nbytes + i] = static_cast<char>((i + step * 37) & 0xFF);
    }
  }

  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  char* src_d;
  CUDACHECK_TEST(cudaMalloc(&src_d, total_bytes));
  CUDACHECK_TEST(
      cudaMemcpy(src_d, src_host.data(), total_bytes, cudaMemcpyHostToDevice));

  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  char* dst_d;
  CUDACHECK_TEST(cudaMalloc(&dst_d, total_bytes));
  CUDACHECK_TEST(cudaMemset(dst_d, 0, total_bytes));
  size_t buf_size = buffer_num_packets * kLl128PacketSize;
  Ll128Packet* ll128_buf;
  CUDACHECK_TEST(cudaMalloc(&ll128_buf, buf_size));

  test::test_ll128_nvlink_varying_send_recv(
      kGpu0,
      kGpu1,
      src_d,
      dst_d,
      nbytes,
      ll128_buf,
      buffer_num_packets,
      num_steps,
      /*num_blocks=*/1,
      /*block_size=*/256);

  std::vector<char> result(total_bytes);
  CUDACHECK_TEST(
      cudaMemcpy(result.data(), dst_d, total_bytes, cudaMemcpyDeviceToHost));
  for (int step = 0; step < num_steps; ++step) {
    for (size_t i = 0; i < nbytes; ++i) {
      ASSERT_EQ(result[step * nbytes + i], src_host[step * nbytes + i])
          << "VaryingData chunked NVLink step " << step << ": mismatch at byte "
          << i;
    }
  }

  CUDACHECK_TEST(cudaSetDevice(kGpu0));
  CUDACHECK_TEST(cudaFree(src_d));
  CUDACHECK_TEST(cudaSetDevice(kGpu1));
  CUDACHECK_TEST(cudaFree(dst_d));
  CUDACHECK_TEST(cudaFree(ll128_buf));
}

// =============================================================================
// Forward — parameterized suite
// =============================================================================

class Ll128OpsNvlinkForwardTest
    : public Ll128OpsNvlinkTestFixture,
      public ::testing::WithParamInterface<TestParam> {};

TEST_P(Ll128OpsNvlinkForwardTest, Forward) {
  const auto& p = GetParam();
  run_forward_test(
      p.nbytes, p.num_blocks, p.block_size, p.buffer_num_packets, p.num_steps);
}

INSTANTIATE_TEST_SUITE_P(
    Ll128NvlinkForward,
    Ll128OpsNvlinkForwardTest,
    ::testing::Values(
        TestParam{"4KB", 4096, 1, 256},
        TestParam{"4KB_MultiStep_10", 4096, 1, 256, 0, 10},
        TestParam{"64KB_MultiBlock", 65536, 4, 256},
        TestParam{"4KB_Chunked_8pkt", 4096, 1, 256, 8},
        TestParam{"4KB_Chunked_8pkt_MultiStep_10", 4096, 1, 256, 8, 10},
        TestParam{"Chunked_MultiStep_MultiBlock", 65536, 4, 256, 8, 10}),
    [](const auto& info) { return info.param.name; });

// =============================================================================
// Bidirectional — parameterized suite
// =============================================================================

class Ll128OpsNvlinkBidirectionalTest
    : public Ll128OpsNvlinkTestFixture,
      public ::testing::WithParamInterface<TestParam> {};

TEST_P(Ll128OpsNvlinkBidirectionalTest, Bidirectional) {
  const auto& p = GetParam();
  run_bidirectional_test(
      p.nbytes, p.num_blocks, p.block_size, p.buffer_num_packets, p.num_steps);
}

INSTANTIATE_TEST_SUITE_P(
    Ll128NvlinkBidirectional,
    Ll128OpsNvlinkBidirectionalTest,
    ::testing::Values(
        TestParam{"4KB", 4096, 1, 256},
        TestParam{"Chunked_4KB_8pkt", 4096, 1, 256, 8},
        TestParam{"Chunked_MultiStep_10", 4096, 1, 256, 8, 10},
        TestParam{"MultiStep_10", 4096, 1, 256, 0, 10}),
    [](const auto& info) { return info.param.name; });

} // namespace comms::pipes
