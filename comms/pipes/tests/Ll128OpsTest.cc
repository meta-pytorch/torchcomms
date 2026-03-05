// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <vector>

#include "comms/pipes/ll128/Ll128Packet.cuh"
#include "comms/pipes/tests/Ll128OpsTest.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::DeviceBuffer;

namespace comms::pipes {

class Ll128OpsTestFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    CUDACHECK_TEST(cudaSetDevice(0));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaDeviceSynchronize());
  }

  /// Create a sequential byte pattern.
  std::vector<char> make_pattern(size_t nbytes, int seed = 0) {
    std::vector<char> pattern(nbytes);
    for (size_t i = 0; i < nbytes; ++i) {
      pattern[i] = static_cast<char>((i + seed) & 0xFF);
    }
    return pattern;
  }

  /// Run send/recv test for a given size and verify output.
  void
  run_send_recv_test(size_t nbytes, int num_blocks = 1, int block_size = 256) {
    auto pattern = make_pattern(nbytes);

    DeviceBuffer srcBuffer(nbytes);
    DeviceBuffer dstBuffer(nbytes);
    size_t ll128BufSize = ll128_buffer_size(nbytes);
    DeviceBuffer ll128Buffer(ll128BufSize);

    auto* src_d = static_cast<char*>(srcBuffer.get());
    auto* dst_d = static_cast<char*>(dstBuffer.get());
    auto* ll128_buf = static_cast<Ll128Packet*>(ll128Buffer.get());

    CUDACHECK_TEST(
        cudaMemcpy(src_d, pattern.data(), nbytes, cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemset(dst_d, 0, nbytes));

    test::test_ll128_send_recv(
        src_d, dst_d, nbytes, ll128_buf, num_blocks, block_size);

    std::vector<char> result(nbytes);
    CUDACHECK_TEST(
        cudaMemcpy(result.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < nbytes; ++i) {
      ASSERT_EQ(result[i], pattern[i])
          << "Mismatch at byte " << i << " for nbytes=" << nbytes;
    }
  }

  /// Pack user data into LL128 packet format on the host, setting flags to
  /// flag_value. This simulates what a predecessor send would produce.
  std::vector<char> pack_ll128_host(
      const std::vector<char>& payload,
      int64_t flag_value) {
    size_t nbytes = payload.size();
    size_t num_packets = ll128_num_packets(nbytes);
    size_t buf_size = num_packets * kLl128PacketSize;
    std::vector<char> buf(buf_size, 0);

    for (size_t p = 0; p < num_packets; ++p) {
      size_t valid = ll128_packet_payload_size(p, nbytes);
      char* pkt = buf.data() + p * kLl128PacketSize;

      // Copy payload bytes
      size_t src_offset = p * kLl128PayloadSize;
      memcpy(pkt, payload.data() + src_offset, valid);

      // Set flag at offset 120
      auto* flag_ptr = reinterpret_cast<int64_t*>(pkt + kLl128FlagOffset);
      *flag_ptr = flag_value;
    }
    return buf;
  }
};

// =============================================================================
// Send/Recv — various sizes (all 16B-aligned)
// =============================================================================

TEST_F(Ll128OpsTestFixture, SendRecv_16Bytes) {
  run_send_recv_test(16);
}

TEST_F(Ll128OpsTestFixture, SendRecv_112Bytes) {
  run_send_recv_test(112);
}

TEST_F(Ll128OpsTestFixture, SendRecv_128Bytes) {
  run_send_recv_test(128);
}

TEST_F(Ll128OpsTestFixture, SendRecv_480Bytes) {
  run_send_recv_test(480);
}

TEST_F(Ll128OpsTestFixture, SendRecv_1008Bytes) {
  run_send_recv_test(1008);
}

TEST_F(Ll128OpsTestFixture, SendRecv_4KB) {
  run_send_recv_test(4096);
}

TEST_F(Ll128OpsTestFixture, SendRecv_64KB) {
  run_send_recv_test(65536);
}

// =============================================================================
// Send/Recv — 0 bytes
// =============================================================================

TEST_F(Ll128OpsTestFixture, SendRecv_ZeroBytes) {
  DeviceBuffer ll128Buffer(128);
  auto* ll128_buf = static_cast<Ll128Packet*>(ll128Buffer.get());
  CUDACHECK_TEST(cudaMemset(ll128_buf, kLl128MemsetInitByte, 128));
  // Should not crash
  test::test_ll128_send_recv(nullptr, nullptr, 0, ll128_buf, 1, 32);
}

// =============================================================================
// Send/Recv — multi-block
// =============================================================================

TEST_F(Ll128OpsTestFixture, SendRecv_MultiBlock_64KB) {
  run_send_recv_test(65536, /*num_blocks=*/8, /*block_size=*/256);
}

// =============================================================================
// Forward — populate local LL128 from host, call forward, verify dst + remote
// =============================================================================

TEST_F(Ll128OpsTestFixture, Forward_4KB) {
  const size_t nbytes = 4096;
  auto pattern = make_pattern(nbytes);

  size_t ll128BufSize = ll128_buffer_size(nbytes);
  DeviceBuffer dstBuffer(nbytes);
  DeviceBuffer localLl128Buffer(ll128BufSize);
  DeviceBuffer remoteLl128Buffer(ll128BufSize);

  auto* dst_d = static_cast<char*>(dstBuffer.get());
  auto* local_ll128 = static_cast<Ll128Packet*>(localLl128Buffer.get());
  auto* remote_ll128 = static_cast<Ll128Packet*>(remoteLl128Buffer.get());

  // Pack payload into LL128 format with flag_value=1
  auto packed = pack_ll128_host(pattern, /*flag_value=*/1);
  CUDACHECK_TEST(cudaMemcpy(
      local_ll128, packed.data(), ll128BufSize, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(dst_d, 0, nbytes));

  test::test_ll128_forward(dst_d, nbytes, local_ll128, remote_ll128, 1, 256);

  // Verify dst has the correct payload
  std::vector<char> result(nbytes);
  CUDACHECK_TEST(
      cudaMemcpy(result.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < nbytes; ++i) {
    ASSERT_EQ(result[i], pattern[i]) << "Forward: dst mismatch at byte " << i;
  }

  // Verify remote_ll128 has data with flag_value=1 flags
  std::vector<char> remote_host(ll128BufSize);
  CUDACHECK_TEST(cudaMemcpy(
      remote_host.data(), remote_ll128, ll128BufSize, cudaMemcpyDeviceToHost));
  size_t num_packets = ll128_num_packets(nbytes);
  for (size_t p = 0; p < num_packets; ++p) {
    int64_t flag = *reinterpret_cast<int64_t*>(
        remote_host.data() + p * kLl128PacketSize + kLl128FlagOffset);
    EXPECT_EQ(flag, 1) << "Remote packet " << p
                       << " flag should be flag_value=1";
  }
}

TEST_F(Ll128OpsTestFixture, Forward_112Bytes) {
  const size_t nbytes = 112;
  auto pattern = make_pattern(nbytes);

  size_t ll128BufSize = ll128_buffer_size(nbytes);
  DeviceBuffer dstBuffer(nbytes);
  DeviceBuffer localLl128Buffer(ll128BufSize);
  DeviceBuffer remoteLl128Buffer(ll128BufSize);

  auto* dst_d = static_cast<char*>(dstBuffer.get());
  auto* local_ll128 = static_cast<Ll128Packet*>(localLl128Buffer.get());
  auto* remote_ll128 = static_cast<Ll128Packet*>(remoteLl128Buffer.get());

  auto packed = pack_ll128_host(pattern, 1);
  CUDACHECK_TEST(cudaMemcpy(
      local_ll128, packed.data(), ll128BufSize, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(dst_d, 0, nbytes));

  test::test_ll128_forward(dst_d, nbytes, local_ll128, remote_ll128, 1, 256);

  std::vector<char> result(nbytes);
  CUDACHECK_TEST(
      cudaMemcpy(result.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < nbytes; ++i) {
    ASSERT_EQ(result[i], pattern[i]) << "Forward: dst mismatch at byte " << i;
  }
}

// =============================================================================
// Multi-step in-kernel tests
// =============================================================================

TEST_F(Ll128OpsTestFixture, SendRecv_MultiStep_InKernel) {
  const size_t nbytes = 4096;
  auto pattern = make_pattern(nbytes);

  DeviceBuffer srcBuffer(nbytes);
  DeviceBuffer dstBuffer(nbytes);
  size_t ll128BufSize = ll128_buffer_size(nbytes);
  DeviceBuffer ll128Buffer(ll128BufSize);

  auto* src_d = static_cast<char*>(srcBuffer.get());
  auto* dst_d = static_cast<char*>(dstBuffer.get());
  auto* ll128_buf = static_cast<Ll128Packet*>(ll128Buffer.get());

  CUDACHECK_TEST(
      cudaMemcpy(src_d, pattern.data(), nbytes, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(dst_d, 0, nbytes));

  // Send same data 10 times with flag_value 1..10, verify final recv is correct
  test::test_ll128_multi_step_send_recv(
      src_d,
      dst_d,
      nbytes,
      ll128_buf,
      /*start_flag_value=*/1,
      /*num_steps=*/10,
      /*num_blocks=*/1,
      /*block_size=*/256);

  std::vector<char> result(nbytes);
  CUDACHECK_TEST(
      cudaMemcpy(result.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < nbytes; ++i) {
    ASSERT_EQ(result[i], pattern[i]) << "MultiStep: mismatch at byte " << i;
  }
}

// =============================================================================
// ABA wraparound test — many steps on same buffer
// =============================================================================

TEST_F(Ll128OpsTestFixture, SendRecv_MultiStep_ABA) {
  const size_t nbytes = 4096;
  auto pattern = make_pattern(nbytes);

  DeviceBuffer srcBuffer(nbytes);
  DeviceBuffer dstBuffer(nbytes);
  size_t ll128BufSize = ll128_buffer_size(nbytes);
  DeviceBuffer ll128Buffer(ll128BufSize);

  auto* src_d = static_cast<char*>(srcBuffer.get());
  auto* dst_d = static_cast<char*>(dstBuffer.get());
  auto* ll128_buf = static_cast<Ll128Packet*>(ll128Buffer.get());

  CUDACHECK_TEST(
      cudaMemcpy(src_d, pattern.data(), nbytes, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(dst_d, 0, nbytes));

  // 100 steps on same buffer tests that flag_value incrementing prevents ABA
  test::test_ll128_multi_step_send_recv(
      src_d,
      dst_d,
      nbytes,
      ll128_buf,
      /*start_flag_value=*/1,
      /*num_steps=*/100,
      /*num_blocks=*/1,
      /*block_size=*/256);

  std::vector<char> result(nbytes);
  CUDACHECK_TEST(
      cudaMemcpy(result.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < nbytes; ++i) {
    ASSERT_EQ(result[i], pattern[i]) << "ABA: mismatch at byte " << i;
  }
}

// =============================================================================
// Multi-block multi-step test
// =============================================================================

TEST_F(Ll128OpsTestFixture, SendRecv_MultiStep_MultiBlock) {
  const size_t nbytes = 65536;
  auto pattern = make_pattern(nbytes);

  DeviceBuffer srcBuffer(nbytes);
  DeviceBuffer dstBuffer(nbytes);
  size_t ll128BufSize = ll128_buffer_size(nbytes);
  DeviceBuffer ll128Buffer(ll128BufSize);

  auto* src_d = static_cast<char*>(srcBuffer.get());
  auto* dst_d = static_cast<char*>(dstBuffer.get());
  auto* ll128_buf = static_cast<Ll128Packet*>(ll128Buffer.get());

  CUDACHECK_TEST(
      cudaMemcpy(src_d, pattern.data(), nbytes, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(dst_d, 0, nbytes));

  // Multi-block with multi-step exercises packet assignment across blocks
  test::test_ll128_multi_step_send_recv(
      src_d,
      dst_d,
      nbytes,
      ll128_buf,
      /*start_flag_value=*/1,
      /*num_steps=*/10,
      /*num_blocks=*/4,
      /*block_size=*/256);

  std::vector<char> result(nbytes);
  CUDACHECK_TEST(
      cudaMemcpy(result.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < nbytes; ++i) {
    ASSERT_EQ(result[i], pattern[i])
        << "MultiStepMultiBlock: mismatch at byte " << i;
  }
}

// =============================================================================
// Stress test — host-side loop, across kernels
// =============================================================================

TEST_F(Ll128OpsTestFixture, SendRecv_Stress) {
  constexpr int kStressIterations = 50;
  const size_t nbytes = 4096;

  DeviceBuffer srcBuffer(nbytes);
  DeviceBuffer dstBuffer(nbytes);
  size_t ll128BufSize = ll128_buffer_size(nbytes);
  DeviceBuffer ll128Buffer(ll128BufSize);

  auto* src_d = static_cast<char*>(srcBuffer.get());
  auto* dst_d = static_cast<char*>(dstBuffer.get());
  auto* ll128_buf = static_cast<Ll128Packet*>(ll128Buffer.get());

  for (int iter = 0; iter < kStressIterations; ++iter) {
    auto pattern = make_pattern(nbytes, /*seed=*/iter);

    CUDACHECK_TEST(
        cudaMemcpy(src_d, pattern.data(), nbytes, cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemset(dst_d, 0, nbytes));

    // Single send/recv with flag_value=1 each time (buffer re-initialized
    // inside test_ll128_send_recv via cudaMemset 0xFF)
    test::test_ll128_send_recv(src_d, dst_d, nbytes, ll128_buf, 1, 256);

    std::vector<char> result(nbytes);
    CUDACHECK_TEST(
        cudaMemcpy(result.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < nbytes; ++i) {
      ASSERT_EQ(result[i], pattern[i])
          << "Stress iter " << iter << ": mismatch at byte " << i;
    }
  }
}

} // namespace comms::pipes
