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

  /// Run single-shot forward test: pre-populate local LL128 buffer from host,
  /// call forward, verify dst payload and remote LL128 buffer flags.
  void
  run_forward_test(size_t nbytes, int num_blocks = 1, int block_size = 256) {
    auto pattern = make_pattern(nbytes);
    size_t ll128BufSize = ll128_buffer_size(nbytes);

    DeviceBuffer dstBuffer(nbytes);
    DeviceBuffer localLl128Buffer(ll128BufSize);
    DeviceBuffer remoteLl128Buffer(ll128BufSize);

    auto* dst_d = static_cast<char*>(dstBuffer.get());
    auto* local_ll128 = static_cast<Ll128Packet*>(localLl128Buffer.get());
    auto* remote_ll128 = static_cast<Ll128Packet*>(remoteLl128Buffer.get());

    auto packed = pack_ll128_host(pattern, /*flag_value=*/1);
    CUDACHECK_TEST(cudaMemcpy(
        local_ll128, packed.data(), ll128BufSize, cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemset(dst_d, 0, nbytes));

    test::test_ll128_forward(
        dst_d, nbytes, local_ll128, remote_ll128, num_blocks, block_size);

    // Verify dst payload matches source
    std::vector<char> result(nbytes);
    CUDACHECK_TEST(
        cudaMemcpy(result.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < nbytes; ++i) {
      ASSERT_EQ(result[i], pattern[i]) << "Forward: dst mismatch at byte " << i;
    }

    // Verify remote LL128 buffer flags are flag_value=1
    std::vector<char> remote_host(ll128BufSize);
    CUDACHECK_TEST(cudaMemcpy(
        remote_host.data(),
        remote_ll128,
        ll128BufSize,
        cudaMemcpyDeviceToHost));
    size_t num_packets = ll128_num_packets(nbytes);
    for (size_t p = 0; p < num_packets; ++p) {
      int64_t flag = *reinterpret_cast<int64_t*>(
          remote_host.data() + p * kLl128PacketSize + kLl128FlagOffset);
      EXPECT_EQ(flag, 1) << "Remote packet " << p
                         << " flag should be flag_value=1";
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

  /// Run a chunked send/recv test with buffer_num_packets < total packets.
  void run_send_recv_chunked_test(
      size_t nbytes,
      size_t buffer_num_packets,
      int num_blocks = 1,
      int block_size = 256) {
    auto pattern = make_pattern(nbytes);

    DeviceBuffer srcBuffer(nbytes);
    DeviceBuffer dstBuffer(nbytes);
    size_t ll128BufSize = buffer_num_packets * kLl128PacketSize;
    DeviceBuffer ll128Buffer(ll128BufSize);

    auto* src_d = static_cast<char*>(srcBuffer.get());
    auto* dst_d = static_cast<char*>(dstBuffer.get());
    auto* ll128_buf = static_cast<Ll128Packet*>(ll128Buffer.get());

    CUDACHECK_TEST(
        cudaMemcpy(src_d, pattern.data(), nbytes, cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemset(dst_d, 0, nbytes));

    test::test_ll128_send_recv_chunked(
        src_d,
        dst_d,
        nbytes,
        ll128_buf,
        buffer_num_packets,
        num_blocks,
        block_size);

    std::vector<char> result(nbytes);
    CUDACHECK_TEST(
        cudaMemcpy(result.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < nbytes; ++i) {
      ASSERT_EQ(result[i], pattern[i])
          << "Chunked: mismatch at byte " << i << " for nbytes=" << nbytes
          << " buffer_num_packets=" << buffer_num_packets;
    }
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

TEST_F(Ll128OpsTestFixture, Forward_112Bytes) {
  run_forward_test(112);
}

TEST_F(Ll128OpsTestFixture, Forward_4KB) {
  run_forward_test(4096);
}

TEST_F(Ll128OpsTestFixture, Forward_64KB) {
  run_forward_test(65536);
}

TEST_F(Ll128OpsTestFixture, Forward_MultiBlock_64KB) {
  run_forward_test(65536, /*num_blocks=*/8, /*block_size=*/256);
}

// =============================================================================
// Forward — multi-step send→forward→recv pipeline
// =============================================================================

TEST_F(Ll128OpsTestFixture, Forward_MultiStep) {
  const size_t nbytes = 4096;
  const int num_steps = 10;
  auto pattern = make_pattern(nbytes);

  size_t ll128BufSize = ll128_buffer_size(nbytes);
  DeviceBuffer srcBuffer(nbytes);
  DeviceBuffer fwdDstBuffer(nbytes);
  DeviceBuffer recvDstBuffer(nbytes);
  DeviceBuffer ll128BufA(ll128BufSize);
  DeviceBuffer ll128BufB(ll128BufSize);

  auto* src_d = static_cast<char*>(srcBuffer.get());
  auto* fwd_dst_d = static_cast<char*>(fwdDstBuffer.get());
  auto* recv_dst_d = static_cast<char*>(recvDstBuffer.get());
  auto* ll128_buf_a = static_cast<Ll128Packet*>(ll128BufA.get());
  auto* ll128_buf_b = static_cast<Ll128Packet*>(ll128BufB.get());

  CUDACHECK_TEST(
      cudaMemcpy(src_d, pattern.data(), nbytes, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(fwd_dst_d, 0, nbytes));
  CUDACHECK_TEST(cudaMemset(recv_dst_d, 0, nbytes));

  test::test_ll128_multi_step_forward(
      src_d,
      fwd_dst_d,
      recv_dst_d,
      nbytes,
      ll128_buf_a,
      ll128_buf_b,
      num_steps,
      /*num_blocks=*/1,
      /*block_size=*/256);

  // Verify forwarder's local copy matches source
  std::vector<char> fwd_result(nbytes);
  CUDACHECK_TEST(
      cudaMemcpy(fwd_result.data(), fwd_dst_d, nbytes, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < nbytes; ++i) {
    ASSERT_EQ(fwd_result[i], pattern[i])
        << "Forward_MultiStep: fwd_dst mismatch at byte " << i;
  }

  // Verify receiver's output matches source
  std::vector<char> recv_result(nbytes);
  CUDACHECK_TEST(cudaMemcpy(
      recv_result.data(), recv_dst_d, nbytes, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < nbytes; ++i) {
    ASSERT_EQ(recv_result[i], pattern[i])
        << "Forward_MultiStep: recv_dst mismatch at byte " << i;
  }
}

TEST_F(Ll128OpsTestFixture, Forward_MultiStep_MultiBlock) {
  const size_t nbytes = 65536;
  const int num_steps = 10;
  auto pattern = make_pattern(nbytes);

  size_t ll128BufSize = ll128_buffer_size(nbytes);
  DeviceBuffer srcBuffer(nbytes);
  DeviceBuffer fwdDstBuffer(nbytes);
  DeviceBuffer recvDstBuffer(nbytes);
  DeviceBuffer ll128BufA(ll128BufSize);
  DeviceBuffer ll128BufB(ll128BufSize);

  auto* src_d = static_cast<char*>(srcBuffer.get());
  auto* fwd_dst_d = static_cast<char*>(fwdDstBuffer.get());
  auto* recv_dst_d = static_cast<char*>(recvDstBuffer.get());
  auto* ll128_buf_a = static_cast<Ll128Packet*>(ll128BufA.get());
  auto* ll128_buf_b = static_cast<Ll128Packet*>(ll128BufB.get());

  CUDACHECK_TEST(
      cudaMemcpy(src_d, pattern.data(), nbytes, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(fwd_dst_d, 0, nbytes));
  CUDACHECK_TEST(cudaMemset(recv_dst_d, 0, nbytes));

  test::test_ll128_multi_step_forward(
      src_d,
      fwd_dst_d,
      recv_dst_d,
      nbytes,
      ll128_buf_a,
      ll128_buf_b,
      num_steps,
      /*num_blocks=*/3,
      /*block_size=*/256);

  // Verify forwarder's local copy matches source
  std::vector<char> fwd_result(nbytes);
  CUDACHECK_TEST(
      cudaMemcpy(fwd_result.data(), fwd_dst_d, nbytes, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < nbytes; ++i) {
    ASSERT_EQ(fwd_result[i], pattern[i])
        << "Forward_MultiStep_MultiBlock: fwd_dst mismatch at byte " << i;
  }

  // Verify receiver's output matches source
  std::vector<char> recv_result(nbytes);
  CUDACHECK_TEST(cudaMemcpy(
      recv_result.data(), recv_dst_d, nbytes, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < nbytes; ++i) {
    ASSERT_EQ(recv_result[i], pattern[i])
        << "Forward_MultiStep_MultiBlock: recv_dst mismatch at byte " << i;
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

  // Send same data 10 times, verify final recv is correct
  test::test_ll128_multi_step_send_recv(
      src_d,
      dst_d,
      nbytes,
      ll128_buf,
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

  // 100 steps on same buffer tests that the two-state protocol prevents ABA
  test::test_ll128_multi_step_send_recv(
      src_d,
      dst_d,
      nbytes,
      ll128_buf,
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

// =============================================================================
// Chunked send/recv tests — buffer smaller than message
// =============================================================================

TEST_F(Ll128OpsTestFixture, SendRecv_Chunked_4KB_4Pkt) {
  run_send_recv_chunked_test(4096, 4);
}

TEST_F(Ll128OpsTestFixture, SendRecv_Chunked_4KB_8Pkt) {
  run_send_recv_chunked_test(4096, 8);
}

TEST_F(Ll128OpsTestFixture, SendRecv_Chunked_64KB_8Pkt) {
  run_send_recv_chunked_test(65536, 8);
}

TEST_F(Ll128OpsTestFixture, SendRecv_Chunked_64KB_32Pkt_MultiBlock) {
  run_send_recv_chunked_test(65536, 32, /*num_blocks=*/4, /*block_size=*/256);
}

TEST_F(Ll128OpsTestFixture, SendRecv_Chunked_ExactFit) {
  // Buffer fits message exactly → no-op chunking (buf_packets == total_packets)
  const size_t nbytes = 4096;
  const size_t total_packets = ll128_num_packets(nbytes);
  run_send_recv_chunked_test(nbytes, total_packets);
}

TEST_F(Ll128OpsTestFixture, SendRecv_Chunked_MultiStep) {
  // Multi-step with chunked buffer — tests ABA prevention via per-packet
  // flag_value
  const size_t nbytes = 4096;
  const size_t buffer_num_packets = 8;
  const int num_steps = 10;
  auto pattern = make_pattern(nbytes);

  DeviceBuffer srcBuffer(nbytes);
  DeviceBuffer dstBuffer(nbytes);
  size_t ll128BufSize = buffer_num_packets * kLl128PacketSize;
  DeviceBuffer ll128Buffer(ll128BufSize);

  auto* src_d = static_cast<char*>(srcBuffer.get());
  auto* dst_d = static_cast<char*>(dstBuffer.get());
  auto* ll128_buf = static_cast<Ll128Packet*>(ll128Buffer.get());

  CUDACHECK_TEST(
      cudaMemcpy(src_d, pattern.data(), nbytes, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(dst_d, 0, nbytes));

  test::test_ll128_multi_step_send_recv_chunked(
      src_d,
      dst_d,
      nbytes,
      ll128_buf,
      buffer_num_packets,
      num_steps,
      /*num_blocks=*/1,
      /*block_size=*/256);

  std::vector<char> result(nbytes);
  CUDACHECK_TEST(
      cudaMemcpy(result.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < nbytes; ++i) {
    ASSERT_EQ(result[i], pattern[i])
        << "Chunked MultiStep: mismatch at byte " << i;
  }
}

TEST_F(Ll128OpsTestFixture, Forward_Chunked_MultiStep) {
  // Send→forward→recv with chunked buffer and multiple steps
  const size_t nbytes = 4096;
  const size_t buffer_num_packets = 8;
  const int num_steps = 10;
  auto pattern = make_pattern(nbytes);

  size_t ll128BufSize = buffer_num_packets * kLl128PacketSize;
  DeviceBuffer srcBuffer(nbytes);
  DeviceBuffer fwdDstBuffer(nbytes);
  DeviceBuffer recvDstBuffer(nbytes);
  DeviceBuffer ll128BufA(ll128BufSize);
  DeviceBuffer ll128BufB(ll128BufSize);

  auto* src_d = static_cast<char*>(srcBuffer.get());
  auto* fwd_dst_d = static_cast<char*>(fwdDstBuffer.get());
  auto* recv_dst_d = static_cast<char*>(recvDstBuffer.get());
  auto* ll128_buf_a = static_cast<Ll128Packet*>(ll128BufA.get());
  auto* ll128_buf_b = static_cast<Ll128Packet*>(ll128BufB.get());

  CUDACHECK_TEST(
      cudaMemcpy(src_d, pattern.data(), nbytes, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(fwd_dst_d, 0, nbytes));
  CUDACHECK_TEST(cudaMemset(recv_dst_d, 0, nbytes));

  test::test_ll128_multi_step_forward_chunked(
      src_d,
      fwd_dst_d,
      recv_dst_d,
      nbytes,
      ll128_buf_a,
      ll128_buf_b,
      buffer_num_packets,
      num_steps,
      /*num_blocks=*/1,
      /*block_size=*/256);

  std::vector<char> fwd_result(nbytes);
  CUDACHECK_TEST(
      cudaMemcpy(fwd_result.data(), fwd_dst_d, nbytes, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < nbytes; ++i) {
    ASSERT_EQ(fwd_result[i], pattern[i])
        << "Forward_Chunked_MultiStep: fwd_dst mismatch at byte " << i;
  }

  std::vector<char> recv_result(nbytes);
  CUDACHECK_TEST(cudaMemcpy(
      recv_result.data(), recv_dst_d, nbytes, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < nbytes; ++i) {
    ASSERT_EQ(recv_result[i], pattern[i])
        << "Forward_Chunked_MultiStep: recv_dst mismatch at byte " << i;
  }
}

TEST_F(Ll128OpsTestFixture, SendRecv_Chunked_OneByteOver) {
  // Edge case: nbytes = capacity + 16 → minimum 2-round chunking
  const size_t buffer_num_packets = 8;
  const size_t capacity =
      ll128_buffer_payload_capacity(buffer_num_packets * kLl128PacketSize);
  const size_t nbytes = capacity + 16;
  run_send_recv_chunked_test(nbytes, buffer_num_packets);
}

TEST_F(Ll128OpsTestFixture, SendRecv_Chunked_UnevenSize) {
  // Non-power-of-2 message size exercises uneven last-round handling
  const size_t buffer_num_packets = 8;
  const size_t nbytes = 4960; // 16-byte aligned, not a power of 2
  run_send_recv_chunked_test(nbytes, buffer_num_packets);
}

// From D95387114

// =============================================================================
// Windowed (capped buffer) tests — buffer smaller than message
// =============================================================================
//
// CONSTRAINT: max_ll128_packets must be >= total_sender_warps *
// kLl128PacketsPerWarp. Each warp processes 4 packets per round, and all
// warps within a round access buffer slots concurrently. If multiple warps
// map to the same slot, they race. The test wrapper launches
// 2 * num_blocks total blocks (sender + receiver), so total_sender_warps =
// num_blocks * (block_size / 32).

TEST_F(Ll128OpsTestFixture, Windowed_SmallBuffer_4KBMessage) {
  // 1 block × 64 threads = 2 warps → 2 sender warps → need >= 8 slots
  // 8 packets = 1KB of buffer for a 4KB message (~35 packets)
  const size_t nbytes = 4096;
  const size_t max_packets = 8; // power of 2, >= 2 warps * 4
  auto pattern = make_pattern(nbytes);

  DeviceBuffer srcBuffer(nbytes);
  DeviceBuffer dstBuffer(nbytes);
  DeviceBuffer ll128Buffer(max_packets * kLl128PacketSize);

  auto* src_d = static_cast<char*>(srcBuffer.get());
  auto* dst_d = static_cast<char*>(dstBuffer.get());
  auto* ll128_buf = static_cast<Ll128Packet*>(ll128Buffer.get());

  CUDACHECK_TEST(
      cudaMemcpy(src_d, pattern.data(), nbytes, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(dst_d, 0, nbytes));

  test::test_ll128_windowed_send_recv(
      src_d,
      dst_d,
      nbytes,
      ll128_buf,
      max_packets,
      /*num_blocks=*/1,
      /*block_size=*/64);

  std::vector<char> result(nbytes);
  CUDACHECK_TEST(
      cudaMemcpy(result.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < nbytes; ++i) {
    ASSERT_EQ(result[i], pattern[i])
        << "Windowed small: mismatch at byte " << i;
  }
}

TEST_F(Ll128OpsTestFixture, Windowed_MediumBuffer_64KBMessage) {
  // 2 blocks × 128 threads = 8 warps → 8 sender warps → need >= 32 slots
  // 64 packets = 8KB of buffer for a 64KB message (~547 packets)
  const size_t nbytes = 64 * 1024;
  const size_t max_packets = 64; // power of 2, >= 8 warps * 4
  auto pattern = make_pattern(nbytes);

  DeviceBuffer srcBuffer(nbytes);
  DeviceBuffer dstBuffer(nbytes);
  DeviceBuffer ll128Buffer(max_packets * kLl128PacketSize);

  auto* src_d = static_cast<char*>(srcBuffer.get());
  auto* dst_d = static_cast<char*>(dstBuffer.get());
  auto* ll128_buf = static_cast<Ll128Packet*>(ll128Buffer.get());

  CUDACHECK_TEST(
      cudaMemcpy(src_d, pattern.data(), nbytes, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(dst_d, 0, nbytes));

  test::test_ll128_windowed_send_recv(
      src_d,
      dst_d,
      nbytes,
      ll128_buf,
      max_packets,
      /*num_blocks=*/2,
      /*block_size=*/128);

  std::vector<char> result(nbytes);
  CUDACHECK_TEST(
      cudaMemcpy(result.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < nbytes; ++i) {
    ASSERT_EQ(result[i], pattern[i])
        << "Windowed medium: mismatch at byte " << i;
  }
}

TEST_F(Ll128OpsTestFixture, Windowed_ExactFit) {
  // Buffer exactly fits the message — no wrapping should occur
  const size_t nbytes = 480; // 4 packets worth of payload
  const size_t max_packets = 4;
  auto pattern = make_pattern(nbytes);

  DeviceBuffer srcBuffer(nbytes);
  DeviceBuffer dstBuffer(nbytes);
  DeviceBuffer ll128Buffer(max_packets * kLl128PacketSize);

  auto* src_d = static_cast<char*>(srcBuffer.get());
  auto* dst_d = static_cast<char*>(dstBuffer.get());
  auto* ll128_buf = static_cast<Ll128Packet*>(ll128Buffer.get());

  CUDACHECK_TEST(
      cudaMemcpy(src_d, pattern.data(), nbytes, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(dst_d, 0, nbytes));

  test::test_ll128_windowed_send_recv(
      src_d,
      dst_d,
      nbytes,
      ll128_buf,
      max_packets,
      /*num_blocks=*/1,
      /*block_size=*/32);

  std::vector<char> result(nbytes);
  CUDACHECK_TEST(
      cudaMemcpy(result.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < nbytes; ++i) {
    ASSERT_EQ(result[i], pattern[i])
        << "Windowed exact fit: mismatch at byte " << i;
  }
}

TEST_F(Ll128OpsTestFixture, Windowed_MinimumFourPackets) {
  // Minimum viable window: 4 packets (one warp's worth per round).
  // 1 block × 32 threads = 1 warp → 1 sender warp → need >= 4 slots
  // 8 packets of data (960 bytes), 4-packet buffer wraps once
  const size_t nbytes = 960; // 8 packets
  const size_t max_packets = 4; // power of 2, = 1 warp * 4
  auto pattern = make_pattern(nbytes);

  DeviceBuffer srcBuffer(nbytes);
  DeviceBuffer dstBuffer(nbytes);
  DeviceBuffer ll128Buffer(max_packets * kLl128PacketSize);

  auto* src_d = static_cast<char*>(srcBuffer.get());
  auto* dst_d = static_cast<char*>(dstBuffer.get());
  auto* ll128_buf = static_cast<Ll128Packet*>(ll128Buffer.get());

  CUDACHECK_TEST(
      cudaMemcpy(src_d, pattern.data(), nbytes, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(dst_d, 0, nbytes));

  test::test_ll128_windowed_send_recv(
      src_d,
      dst_d,
      nbytes,
      ll128_buf,
      max_packets,
      /*num_blocks=*/1,
      /*block_size=*/32);

  std::vector<char> result(nbytes);
  CUDACHECK_TEST(
      cudaMemcpy(result.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < nbytes; ++i) {
    ASSERT_EQ(result[i], pattern[i])
        << "Windowed 4-packet: mismatch at byte " << i;
  }
}

} // namespace comms::pipes
