// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <vector>

#include "comms/pipes/TimeoutUtils.h"
#include "comms/pipes/ll128/Ll128Packet.cuh"
#include "comms/pipes/ll128/tests/Ll128OpsTest.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::DeviceBuffer;

namespace comms::pipes {

// =============================================================================
// Param struct and name generator
// =============================================================================

struct Ll128OpsTestParam {
  std::string name;
  size_t nbytes;
  int num_blocks = 1;
  int block_size = 256;
  size_t buffer_num_packets = 0;
  int num_steps = 1;
};

std::string ll128_test_name(
    const ::testing::TestParamInfo<Ll128OpsTestParam>& info) {
  return info.param.name;
}

// =============================================================================
// Test fixture
// =============================================================================

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

  /// Run multi-step send/recv test with full-sized buffer.
  void run_multi_step_send_recv_test(
      size_t nbytes,
      int num_steps,
      int num_blocks = 1,
      int block_size = 256) {
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

    test::test_ll128_multi_step_send_recv(
        src_d, dst_d, nbytes, ll128_buf, num_steps, num_blocks, block_size);

    std::vector<char> result(nbytes);
    CUDACHECK_TEST(
        cudaMemcpy(result.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < nbytes; ++i) {
      ASSERT_EQ(result[i], pattern[i]) << "MultiStep: mismatch at byte " << i;
    }
  }

  /// Run multi-step chunked send/recv test.
  void run_multi_step_chunked_test(
      size_t nbytes,
      size_t buffer_num_packets,
      int num_steps,
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

    test::test_ll128_multi_step_send_recv_chunked(
        src_d,
        dst_d,
        nbytes,
        ll128_buf,
        buffer_num_packets,
        num_steps,
        num_blocks,
        block_size);

    std::vector<char> result(nbytes);
    CUDACHECK_TEST(
        cudaMemcpy(result.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < nbytes; ++i) {
      ASSERT_EQ(result[i], pattern[i])
          << "Chunked MultiStep: mismatch at byte " << i;
    }
  }

  /// Run multi-step forward test (send→forward→recv pipeline).
  void run_multi_step_forward_test(
      size_t nbytes,
      int num_steps,
      int num_blocks = 1,
      int block_size = 256) {
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
        num_blocks,
        block_size);

    std::vector<char> fwd_result(nbytes);
    CUDACHECK_TEST(cudaMemcpy(
        fwd_result.data(), fwd_dst_d, nbytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < nbytes; ++i) {
      ASSERT_EQ(fwd_result[i], pattern[i])
          << "Forward MultiStep: fwd_dst mismatch at byte " << i;
    }

    std::vector<char> recv_result(nbytes);
    CUDACHECK_TEST(cudaMemcpy(
        recv_result.data(), recv_dst_d, nbytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < nbytes; ++i) {
      ASSERT_EQ(recv_result[i], pattern[i])
          << "Forward MultiStep: recv_dst mismatch at byte " << i;
    }
  }

  /// Run windowed send/recv test with capped buffer.
  void run_windowed_test(
      size_t nbytes,
      size_t max_packets,
      int num_blocks = 1,
      int block_size = 256) {
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
        src_d, dst_d, nbytes, ll128_buf, max_packets, num_blocks, block_size);

    std::vector<char> result(nbytes);
    CUDACHECK_TEST(
        cudaMemcpy(result.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < nbytes; ++i) {
      ASSERT_EQ(result[i], pattern[i]) << "Windowed: mismatch at byte " << i;
    }
  }

  /// Run multi-step forward chunked test.
  void run_multi_step_forward_chunked_test(
      size_t nbytes,
      size_t buffer_num_packets,
      int num_steps,
      int num_blocks = 1,
      int block_size = 256) {
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
        num_blocks,
        block_size);

    std::vector<char> fwd_result(nbytes);
    CUDACHECK_TEST(cudaMemcpy(
        fwd_result.data(), fwd_dst_d, nbytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < nbytes; ++i) {
      ASSERT_EQ(fwd_result[i], pattern[i])
          << "Forward Chunked MultiStep: fwd_dst mismatch at byte " << i;
    }

    std::vector<char> recv_result(nbytes);
    CUDACHECK_TEST(cudaMemcpy(
        recv_result.data(), recv_dst_d, nbytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < nbytes; ++i) {
      ASSERT_EQ(recv_result[i], pattern[i])
          << "Forward Chunked MultiStep: recv_dst mismatch at byte " << i;
    }
  }

  /// Run varying-data multi-step send/recv test.
  /// When buffer_num_packets == 0, uses full-sized buffer.
  void run_varying_data_multi_step_test(
      size_t nbytes,
      size_t buffer_num_packets,
      int num_steps,
      int num_blocks = 1,
      int block_size = 256) {
    const size_t total_bytes = num_steps * nbytes;

    size_t ll128BufSize = (buffer_num_packets == 0)
        ? ll128_buffer_size(nbytes)
        : buffer_num_packets * kLl128PacketSize;

    DeviceBuffer srcBuffer(total_bytes);
    DeviceBuffer dstBuffer(total_bytes);
    DeviceBuffer ll128Buffer(ll128BufSize);

    auto* src_d = static_cast<char*>(srcBuffer.get());
    auto* dst_d = static_cast<char*>(dstBuffer.get());
    auto* ll128_buf = static_cast<Ll128Packet*>(ll128Buffer.get());

    std::vector<char> src_host(total_bytes);
    for (int step = 0; step < num_steps; ++step) {
      auto pattern = make_pattern(nbytes, /*seed=*/step * 37);
      memcpy(src_host.data() + step * nbytes, pattern.data(), nbytes);
    }
    CUDACHECK_TEST(cudaMemcpy(
        src_d, src_host.data(), total_bytes, cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemset(dst_d, 0, total_bytes));

    test::test_ll128_varying_data_multi_step_send_recv(
        src_d,
        dst_d,
        nbytes,
        ll128_buf,
        buffer_num_packets,
        num_steps,
        num_blocks,
        block_size);

    std::vector<char> result(total_bytes);
    CUDACHECK_TEST(
        cudaMemcpy(result.data(), dst_d, total_bytes, cudaMemcpyDeviceToHost));
    for (int step = 0; step < num_steps; ++step) {
      auto pattern = make_pattern(nbytes, /*seed=*/step * 37);
      for (size_t i = 0; i < nbytes; ++i) {
        ASSERT_EQ(result[step * nbytes + i], pattern[i])
            << "VaryingData step " << step << ": mismatch at byte " << i;
      }
    }
  }

  /// Run varying-data multi-step forward test.
  /// When buffer_num_packets == 0, uses full-sized buffer.
  void run_varying_data_multi_step_forward_test(
      size_t nbytes,
      size_t buffer_num_packets,
      int num_steps,
      int num_blocks = 1,
      int block_size = 256) {
    const size_t total_bytes = num_steps * nbytes;

    size_t ll128BufSize = (buffer_num_packets == 0)
        ? ll128_buffer_size(nbytes)
        : buffer_num_packets * kLl128PacketSize;

    DeviceBuffer srcBuffer(total_bytes);
    DeviceBuffer fwdDstBuffer(total_bytes);
    DeviceBuffer recvDstBuffer(total_bytes);
    DeviceBuffer ll128BufA(ll128BufSize);
    DeviceBuffer ll128BufB(ll128BufSize);

    auto* src_d = static_cast<char*>(srcBuffer.get());
    auto* fwd_dst_d = static_cast<char*>(fwdDstBuffer.get());
    auto* recv_dst_d = static_cast<char*>(recvDstBuffer.get());
    auto* ll128_buf_a = static_cast<Ll128Packet*>(ll128BufA.get());
    auto* ll128_buf_b = static_cast<Ll128Packet*>(ll128BufB.get());

    std::vector<char> src_host(total_bytes);
    for (int step = 0; step < num_steps; ++step) {
      auto pattern = make_pattern(nbytes, /*seed=*/step * 37);
      memcpy(src_host.data() + step * nbytes, pattern.data(), nbytes);
    }
    CUDACHECK_TEST(cudaMemcpy(
        src_d, src_host.data(), total_bytes, cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemset(fwd_dst_d, 0, total_bytes));
    CUDACHECK_TEST(cudaMemset(recv_dst_d, 0, total_bytes));

    test::test_ll128_varying_data_multi_step_forward(
        src_d,
        fwd_dst_d,
        recv_dst_d,
        nbytes,
        ll128_buf_a,
        ll128_buf_b,
        buffer_num_packets,
        num_steps,
        num_blocks,
        block_size);

    std::vector<char> fwd_result(total_bytes);
    CUDACHECK_TEST(cudaMemcpy(
        fwd_result.data(), fwd_dst_d, total_bytes, cudaMemcpyDeviceToHost));
    std::vector<char> recv_result(total_bytes);
    CUDACHECK_TEST(cudaMemcpy(
        recv_result.data(), recv_dst_d, total_bytes, cudaMemcpyDeviceToHost));

    for (int step = 0; step < num_steps; ++step) {
      auto pattern = make_pattern(nbytes, /*seed=*/step * 37);
      for (size_t i = 0; i < nbytes; ++i) {
        ASSERT_EQ(fwd_result[step * nbytes + i], pattern[i])
            << "Forward VaryingData step " << step
            << ": fwd_dst mismatch at byte " << i;
        ASSERT_EQ(recv_result[step * nbytes + i], pattern[i])
            << "Forward VaryingData step " << step
            << ": recv_dst mismatch at byte " << i;
      }
    }
  }
};

// =============================================================================
// Group A: SendRecv — various sizes
// =============================================================================

class Ll128OpsSendRecvTest
    : public Ll128OpsTestFixture,
      public ::testing::WithParamInterface<Ll128OpsTestParam> {};

TEST_P(Ll128OpsSendRecvTest, SendRecv) {
  const auto& p = GetParam();
  run_send_recv_test(p.nbytes, p.num_blocks, p.block_size);
}

INSTANTIATE_TEST_SUITE_P(
    Ll128Ops,
    Ll128OpsSendRecvTest,
    ::testing::Values(
        Ll128OpsTestParam{.name = "16Bytes", .nbytes = 16},
        Ll128OpsTestParam{.name = "112Bytes", .nbytes = 112},
        Ll128OpsTestParam{.name = "128Bytes", .nbytes = 128},
        Ll128OpsTestParam{.name = "480Bytes", .nbytes = 480},
        Ll128OpsTestParam{.name = "960Bytes", .nbytes = 960},
        Ll128OpsTestParam{.name = "1008Bytes", .nbytes = 1008},
        Ll128OpsTestParam{.name = "1440Bytes", .nbytes = 1440},
        Ll128OpsTestParam{.name = "4KB", .nbytes = 4096},
        Ll128OpsTestParam{.name = "64KB", .nbytes = 65536},
        Ll128OpsTestParam{
            .name = "MultiBlock_64KB",
            .nbytes = 65536,
            .num_blocks = 8}),
    ll128_test_name);

// =============================================================================
// Group B: Forward — populate local LL128, call forward, verify dst + remote
// =============================================================================

class Ll128OpsForwardTest
    : public Ll128OpsTestFixture,
      public ::testing::WithParamInterface<Ll128OpsTestParam> {};

TEST_P(Ll128OpsForwardTest, Forward) {
  const auto& p = GetParam();
  run_forward_test(p.nbytes, p.num_blocks, p.block_size);
}

INSTANTIATE_TEST_SUITE_P(
    Ll128Ops,
    Ll128OpsForwardTest,
    ::testing::Values(
        Ll128OpsTestParam{.name = "112Bytes", .nbytes = 112},
        Ll128OpsTestParam{.name = "4KB", .nbytes = 4096},
        Ll128OpsTestParam{.name = "64KB", .nbytes = 65536},
        Ll128OpsTestParam{
            .name = "MultiBlock_64KB",
            .nbytes = 65536,
            .num_blocks = 8}),
    ll128_test_name);

// =============================================================================
// Group C: Chunked SendRecv — buffer smaller than message
// =============================================================================

class Ll128OpsChunkedSendRecvTest
    : public Ll128OpsTestFixture,
      public ::testing::WithParamInterface<Ll128OpsTestParam> {};

TEST_P(Ll128OpsChunkedSendRecvTest, ChunkedSendRecv) {
  const auto& p = GetParam();
  run_send_recv_chunked_test(
      p.nbytes, p.buffer_num_packets, p.num_blocks, p.block_size);
}

INSTANTIATE_TEST_SUITE_P(
    Ll128Ops,
    Ll128OpsChunkedSendRecvTest,
    ::testing::Values(
        Ll128OpsTestParam{
            .name = "4KB_4Pkt",
            .nbytes = 4096,
            .buffer_num_packets = 4},
        Ll128OpsTestParam{
            .name = "4KB_8Pkt",
            .nbytes = 4096,
            .buffer_num_packets = 8},
        Ll128OpsTestParam{
            .name = "64KB_8Pkt",
            .nbytes = 65536,
            .buffer_num_packets = 8},
        Ll128OpsTestParam{
            .name = "64KB_32Pkt_MultiBlock",
            .nbytes = 65536,
            .num_blocks = 4,
            .buffer_num_packets = 32},
        // ll128_num_packets(4096) = ceil(4096/120) = 35
        Ll128OpsTestParam{
            .name = "ExactFit",
            .nbytes = 4096,
            .buffer_num_packets = 35},
        Ll128OpsTestParam{
            .name = "5Pkt",
            .nbytes = 4096,
            .buffer_num_packets = 5},
        Ll128OpsTestParam{
            .name = "6Pkt",
            .nbytes = 4096,
            .buffer_num_packets = 6},
        Ll128OpsTestParam{
            .name = "7Pkt",
            .nbytes = 4096,
            .buffer_num_packets = 7},
        Ll128OpsTestParam{
            .name = "12Pkt",
            .nbytes = 4096,
            .buffer_num_packets = 12},
        Ll128OpsTestParam{
            .name = "WarpClampingStress",
            .nbytes = 65536,
            .num_blocks = 8,
            .buffer_num_packets = 4},
        Ll128OpsTestParam{
            .name = "OversizedBuffer",
            .nbytes = 480,
            .buffer_num_packets = 8},
        Ll128OpsTestParam{
            .name = "SmallMessage_960B_4Pkt",
            .nbytes = 960,
            .buffer_num_packets = 4},
        // ll128_buffer_payload_capacity(8 * kLl128PacketSize) + 16
        // = (8 * 120) + 16 = 976
        Ll128OpsTestParam{
            .name = "OneByteOver",
            .nbytes = 976,
            .buffer_num_packets = 8},
        Ll128OpsTestParam{
            .name = "UnevenSize",
            .nbytes = 4960,
            .buffer_num_packets = 8}),
    ll128_test_name);

// =============================================================================
// Group D: MultiStep SendRecv
// =============================================================================

class Ll128OpsMultiStepSendRecvTest
    : public Ll128OpsTestFixture,
      public ::testing::WithParamInterface<Ll128OpsTestParam> {};

TEST_P(Ll128OpsMultiStepSendRecvTest, MultiStepSendRecv) {
  const auto& p = GetParam();
  run_multi_step_send_recv_test(
      p.nbytes, p.num_steps, p.num_blocks, p.block_size);
}

INSTANTIATE_TEST_SUITE_P(
    Ll128Ops,
    Ll128OpsMultiStepSendRecvTest,
    ::testing::Values(
        Ll128OpsTestParam{.name = "InKernel", .nbytes = 4096, .num_steps = 10},
        Ll128OpsTestParam{.name = "ABA", .nbytes = 4096, .num_steps = 100},
        Ll128OpsTestParam{
            .name = "MultiBlock",
            .nbytes = 65536,
            .num_blocks = 4,
            .num_steps = 10}),
    ll128_test_name);

// =============================================================================
// Group E: MultiStep Chunked SendRecv
// =============================================================================

class Ll128OpsMultiStepChunkedSendRecvTest
    : public Ll128OpsTestFixture,
      public ::testing::WithParamInterface<Ll128OpsTestParam> {};

TEST_P(Ll128OpsMultiStepChunkedSendRecvTest, MultiStepChunkedSendRecv) {
  const auto& p = GetParam();
  run_multi_step_chunked_test(
      p.nbytes, p.buffer_num_packets, p.num_steps, p.num_blocks, p.block_size);
}

INSTANTIATE_TEST_SUITE_P(
    Ll128Ops,
    Ll128OpsMultiStepChunkedSendRecvTest,
    ::testing::Values(
        Ll128OpsTestParam{
            .name = "4KB_8pkt_10steps",
            .nbytes = 4096,
            .buffer_num_packets = 8,
            .num_steps = 10},
        Ll128OpsTestParam{
            .name = "64KB_32pkt_MultiBlock",
            .nbytes = 65536,
            .num_blocks = 4,
            .buffer_num_packets = 32,
            .num_steps = 10},
        Ll128OpsTestParam{
            .name = "ABA_4pkt_100steps",
            .nbytes = 4096,
            .block_size = 32,
            .buffer_num_packets = 4,
            .num_steps = 100},
        Ll128OpsTestParam{
            .name = "ABA_64KB_8pkt_MultiBlock",
            .nbytes = 65536,
            .num_blocks = 4,
            .buffer_num_packets = 8,
            .num_steps = 100},
        Ll128OpsTestParam{
            .name = "Windowed_4KB_8pkt_10steps",
            .nbytes = 4096,
            .block_size = 64,
            .buffer_num_packets = 8,
            .num_steps = 10}),
    ll128_test_name);

// =============================================================================
// Group F: MultiStep Forward
// =============================================================================

class Ll128OpsMultiStepForwardTest
    : public Ll128OpsTestFixture,
      public ::testing::WithParamInterface<Ll128OpsTestParam> {};

TEST_P(Ll128OpsMultiStepForwardTest, MultiStepForward) {
  const auto& p = GetParam();
  run_multi_step_forward_test(
      p.nbytes, p.num_steps, p.num_blocks, p.block_size);
}

INSTANTIATE_TEST_SUITE_P(
    Ll128Ops,
    Ll128OpsMultiStepForwardTest,
    ::testing::Values(
        Ll128OpsTestParam{
            .name = "4KB_10steps",
            .nbytes = 4096,
            .num_steps = 10},
        Ll128OpsTestParam{
            .name = "64KB_MultiBlock",
            .nbytes = 65536,
            .num_blocks = 3,
            .num_steps = 10}),
    ll128_test_name);

// =============================================================================
// Group G: Windowed SendRecv — capped buffer tests
// =============================================================================

class Ll128OpsWindowedSendRecvTest
    : public Ll128OpsTestFixture,
      public ::testing::WithParamInterface<Ll128OpsTestParam> {};

TEST_P(Ll128OpsWindowedSendRecvTest, WindowedSendRecv) {
  const auto& p = GetParam();
  run_windowed_test(p.nbytes, p.buffer_num_packets, p.num_blocks, p.block_size);
}

INSTANTIATE_TEST_SUITE_P(
    Ll128Ops,
    Ll128OpsWindowedSendRecvTest,
    ::testing::Values(
        Ll128OpsTestParam{
            .name = "SmallBuffer_4KB",
            .nbytes = 4096,
            .block_size = 64,
            .buffer_num_packets = 8},
        Ll128OpsTestParam{
            .name = "MediumBuffer_64KB",
            .nbytes = 65536,
            .num_blocks = 2,
            .block_size = 128,
            .buffer_num_packets = 64},
        Ll128OpsTestParam{
            .name = "ExactFit",
            .nbytes = 480,
            .block_size = 32,
            .buffer_num_packets = 4},
        Ll128OpsTestParam{
            .name = "MinimumFourPackets",
            .nbytes = 960,
            .block_size = 32,
            .buffer_num_packets = 4}),
    ll128_test_name);

// =============================================================================
// Group H: MultiStep Forward Chunked
// =============================================================================

class Ll128OpsMultiStepForwardChunkedTest
    : public Ll128OpsTestFixture,
      public ::testing::WithParamInterface<Ll128OpsTestParam> {};

TEST_P(Ll128OpsMultiStepForwardChunkedTest, MultiStepForwardChunked) {
  const auto& p = GetParam();
  run_multi_step_forward_chunked_test(
      p.nbytes, p.buffer_num_packets, p.num_steps, p.num_blocks, p.block_size);
}

INSTANTIATE_TEST_SUITE_P(
    Ll128Ops,
    Ll128OpsMultiStepForwardChunkedTest,
    ::testing::Values(
        Ll128OpsTestParam{
            .name = "4KB_8pkt_1step",
            .nbytes = 4096,
            .block_size = 64,
            .buffer_num_packets = 8,
            .num_steps = 1},
        Ll128OpsTestParam{
            .name = "4KB_8pkt_10steps",
            .nbytes = 4096,
            .buffer_num_packets = 8,
            .num_steps = 10},
        Ll128OpsTestParam{
            .name = "64KB_32pkt_MultiBlock",
            .nbytes = 65536,
            .num_blocks = 3,
            .buffer_num_packets = 32,
            .num_steps = 10}),
    ll128_test_name);

// =============================================================================
// Group I: VaryingData MultiStep SendRecv
// =============================================================================

class Ll128OpsVaryingDataSendRecvTest
    : public Ll128OpsTestFixture,
      public ::testing::WithParamInterface<Ll128OpsTestParam> {};

TEST_P(Ll128OpsVaryingDataSendRecvTest, VaryingDataSendRecv) {
  const auto& p = GetParam();
  run_varying_data_multi_step_test(
      p.nbytes, p.buffer_num_packets, p.num_steps, p.num_blocks, p.block_size);
}

INSTANTIATE_TEST_SUITE_P(
    Ll128Ops,
    Ll128OpsVaryingDataSendRecvTest,
    ::testing::Values(
        Ll128OpsTestParam{
            .name = "NonChunked",
            .nbytes = 4096,
            .num_steps = 10},
        Ll128OpsTestParam{
            .name = "Chunked",
            .nbytes = 4096,
            .buffer_num_packets = 8,
            .num_steps = 10}),
    ll128_test_name);

// =============================================================================
// Group J: VaryingData MultiStep Forward
// =============================================================================

class Ll128OpsVaryingDataForwardTest
    : public Ll128OpsTestFixture,
      public ::testing::WithParamInterface<Ll128OpsTestParam> {};

TEST_P(Ll128OpsVaryingDataForwardTest, VaryingDataForward) {
  const auto& p = GetParam();
  run_varying_data_multi_step_forward_test(
      p.nbytes, p.buffer_num_packets, p.num_steps, p.num_blocks, p.block_size);
}

INSTANTIATE_TEST_SUITE_P(
    Ll128Ops,
    Ll128OpsVaryingDataForwardTest,
    ::testing::Values(
        Ll128OpsTestParam{
            .name = "NonChunked",
            .nbytes = 4096,
            .num_steps = 10},
        Ll128OpsTestParam{
            .name = "Chunked",
            .nbytes = 4096,
            .buffer_num_packets = 8,
            .num_steps = 10}),
    ll128_test_name);

// =============================================================================
// Standalone tests — unique control flow, not parameterizable
// =============================================================================

TEST_F(Ll128OpsTestFixture, SendRecv_ZeroBytes) {
  DeviceBuffer ll128Buffer(128);
  auto* ll128_buf = static_cast<Ll128Packet*>(ll128Buffer.get());
  CUDACHECK_TEST(cudaMemset(ll128_buf, kLl128MemsetInitByte, 128));
  // Should not crash
  test::test_ll128_send_recv(nullptr, nullptr, 0, ll128_buf, 1, 32);
}

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

TEST_F(Ll128OpsTestFixture, FlagState_AfterMultiStep_Chunked) {
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

  // Verify data correctness first
  std::vector<char> result(nbytes);
  CUDACHECK_TEST(
      cudaMemcpy(result.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < nbytes; ++i) {
    ASSERT_EQ(result[i], pattern[i])
        << "FlagState: data mismatch at byte " << i;
  }

  // Read back LL128 buffer and verify all flags are READY_TO_WRITE (-1)
  // After completion, the receiver has ACKed all packets.
  std::vector<char> buf_host(ll128BufSize);
  CUDACHECK_TEST(cudaMemcpy(
      buf_host.data(), ll128_buf, ll128BufSize, cudaMemcpyDeviceToHost));

  for (size_t p = 0; p < buffer_num_packets; ++p) {
    auto* flag_ptr = reinterpret_cast<int64_t*>(
        buf_host.data() + p * kLl128PacketSize + kLl128FlagOffset);
    EXPECT_EQ(*flag_ptr, kLl128ReadyToWrite)
        << "Buffer packet " << p
        << " flag should be READY_TO_WRITE after completion";
  }
}

TEST_F(Ll128OpsTestFixture, FlagState_AfterMultiStep_NonChunked) {
  const size_t nbytes = 4096;
  const int num_steps = 10;
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

  test::test_ll128_multi_step_send_recv(
      src_d,
      dst_d,
      nbytes,
      ll128_buf,
      num_steps,
      /*num_blocks=*/1,
      /*block_size=*/256);

  // Verify data correctness
  std::vector<char> result(nbytes);
  CUDACHECK_TEST(
      cudaMemcpy(result.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < nbytes; ++i) {
    ASSERT_EQ(result[i], pattern[i])
        << "FlagState non-chunked: data mismatch at byte " << i;
  }

  // Verify all flags are READY_TO_WRITE (-1) after completion
  size_t num_packets = ll128_num_packets(nbytes);
  std::vector<char> buf_host(ll128BufSize);
  CUDACHECK_TEST(cudaMemcpy(
      buf_host.data(), ll128_buf, ll128BufSize, cudaMemcpyDeviceToHost));
  for (size_t p = 0; p < num_packets; ++p) {
    auto* flag_ptr = reinterpret_cast<int64_t*>(
        buf_host.data() + p * kLl128PacketSize + kLl128FlagOffset);
    EXPECT_EQ(*flag_ptr, kLl128ReadyToWrite)
        << "Non-chunked packet " << p
        << " flag should be READY_TO_WRITE after multi-step completion";
  }
}

TEST_F(Ll128OpsTestFixture, Timeout_Constructors) {
  // Default timeout is disabled
  Timeout default_timeout;
  EXPECT_FALSE(default_timeout.isEnabled());

  // Timeout with cycles is enabled
  Timeout enabled_timeout(1000);
  EXPECT_TRUE(enabled_timeout.isEnabled());

  // makeTimeout(0) creates disabled timeout
  auto disabled = makeTimeout(0);
  EXPECT_FALSE(disabled.isEnabled());

  // makeTimeout(ms) creates enabled timeout with correct cycles
  auto enabled = makeTimeout(1000);
  EXPECT_TRUE(enabled.isEnabled());
  EXPECT_GT(enabled.timeout_cycles, 0u);
}

} // namespace comms::pipes
