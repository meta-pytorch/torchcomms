// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <vector>

#include "comms/pipes/TimeoutUtils.h"
#include "comms/pipes/ll/LlPacket.cuh"
#include "comms/pipes/ll/tests/LlOpsTest.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::DeviceBuffer;

namespace comms::pipes {

// =============================================================================
// Param struct and name generator
// =============================================================================

struct LlOpsTestParam {
  std::string name;
  size_t nbytes;
  int num_blocks = 1;
  int block_size = 256;
  size_t buffer_num_lines = 0;
  int num_steps = 1;
};

std::string ll_test_name(const ::testing::TestParamInfo<LlOpsTestParam>& info) {
  return info.param.name;
}

// =============================================================================
// Test fixture
// =============================================================================

class LlOpsTestFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    CUDACHECK_TEST(cudaSetDevice(0));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaDeviceSynchronize());
  }

  std::vector<char> make_pattern(size_t nbytes, int seed = 0) {
    std::vector<char> pattern(nbytes);
    for (size_t i = 0; i < nbytes; ++i) {
      pattern[i] = static_cast<char>((i + seed) & 0xFF);
    }
    return pattern;
  }

  void
  run_send_recv_test(size_t nbytes, int num_blocks = 1, int block_size = 256) {
    auto pattern = make_pattern(nbytes);

    DeviceBuffer srcBuffer(nbytes);
    DeviceBuffer dstBuffer(nbytes);
    size_t llBufSize = ll_buffer_size(nbytes);
    DeviceBuffer llBuffer(llBufSize);

    auto* src_d = static_cast<char*>(srcBuffer.get());
    auto* dst_d = static_cast<char*>(dstBuffer.get());
    auto* ll_buf = static_cast<LlLine*>(llBuffer.get());

    CUDACHECK_TEST(
        cudaMemcpy(src_d, pattern.data(), nbytes, cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemset(dst_d, 0, nbytes));

    test::test_ll_send_recv(
        src_d, dst_d, nbytes, ll_buf, num_blocks, block_size);

    std::vector<char> result(nbytes);
    CUDACHECK_TEST(
        cudaMemcpy(result.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < nbytes; ++i) {
      ASSERT_EQ(result[i], pattern[i])
          << "Mismatch at byte " << i << " for nbytes=" << nbytes;
    }
  }

  void
  run_forward_test(size_t nbytes, int num_blocks = 1, int block_size = 256) {
    auto pattern = make_pattern(nbytes);
    size_t llBufSize = ll_buffer_size(nbytes);

    DeviceBuffer dstBuffer(nbytes);
    DeviceBuffer localLlBuffer(llBufSize);
    DeviceBuffer remoteLlBuffer(llBufSize);

    auto* dst_d = static_cast<char*>(dstBuffer.get());
    auto* local_ll = static_cast<LlLine*>(localLlBuffer.get());
    auto* remote_ll = static_cast<LlLine*>(remoteLlBuffer.get());

    auto packed = pack_ll_host(pattern, /*flag_value=*/1);
    CUDACHECK_TEST(
        cudaMemcpy(local_ll, packed.data(), llBufSize, cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemset(dst_d, 0, nbytes));

    test::test_ll_forward(
        dst_d, nbytes, local_ll, remote_ll, num_blocks, block_size);

    std::vector<char> result(nbytes);
    CUDACHECK_TEST(
        cudaMemcpy(result.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < nbytes; ++i) {
      ASSERT_EQ(result[i], pattern[i]) << "Forward: dst mismatch at byte " << i;
    }

    // Verify remote LL buffer flags are flag_value=1
    std::vector<char> remote_host(llBufSize);
    CUDACHECK_TEST(cudaMemcpy(
        remote_host.data(), remote_ll, llBufSize, cudaMemcpyDeviceToHost));
    size_t num_lines = ll_num_lines(nbytes);
    for (size_t l = 0; l < num_lines; ++l) {
      auto* line =
          reinterpret_cast<LlLine*>(remote_host.data() + l * kLlLineSize);
      EXPECT_EQ(line->flag1, 1u) << "Remote line " << l << " flag1 should be 1";
      EXPECT_EQ(line->flag2, 1u) << "Remote line " << l << " flag2 should be 1";
    }
  }

  /// Pack user data into LL line format on the host.
  std::vector<char> pack_ll_host(
      const std::vector<char>& payload,
      uint32_t flag_value) {
    size_t nbytes = payload.size();
    size_t num_lines = ll_num_lines(nbytes);
    size_t buf_size = num_lines * kLlLineSize;
    std::vector<char> buf(buf_size, 0);

    for (size_t l = 0; l < num_lines; ++l) {
      size_t payload_offset = l * kLlPayloadPerLine;
      auto* line = reinterpret_cast<LlLine*>(buf.data() + l * kLlLineSize);

      const auto* src_u32 =
          reinterpret_cast<const uint32_t*>(payload.data() + payload_offset);
      line->data1 = src_u32[0];
      line->flag1 = flag_value;
      line->data2 = src_u32[1];
      line->flag2 = flag_value;
    }
    return buf;
  }

  void run_send_recv_chunked_test(
      size_t nbytes,
      size_t buffer_num_lines,
      int num_blocks = 1,
      int block_size = 256) {
    auto pattern = make_pattern(nbytes);

    DeviceBuffer srcBuffer(nbytes);
    DeviceBuffer dstBuffer(nbytes);
    size_t llBufSize = buffer_num_lines * kLlLineSize;
    DeviceBuffer llBuffer(llBufSize);

    auto* src_d = static_cast<char*>(srcBuffer.get());
    auto* dst_d = static_cast<char*>(dstBuffer.get());
    auto* ll_buf = static_cast<LlLine*>(llBuffer.get());

    CUDACHECK_TEST(
        cudaMemcpy(src_d, pattern.data(), nbytes, cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemset(dst_d, 0, nbytes));

    test::test_ll_send_recv_chunked(
        src_d, dst_d, nbytes, ll_buf, buffer_num_lines, num_blocks, block_size);

    std::vector<char> result(nbytes);
    CUDACHECK_TEST(
        cudaMemcpy(result.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < nbytes; ++i) {
      ASSERT_EQ(result[i], pattern[i])
          << "Chunked: mismatch at byte " << i << " for nbytes=" << nbytes
          << " buffer_num_lines=" << buffer_num_lines;
    }
  }

  void run_multi_step_send_recv_test(
      size_t nbytes,
      int num_steps,
      int num_blocks = 1,
      int block_size = 256) {
    auto pattern = make_pattern(nbytes);

    DeviceBuffer srcBuffer(nbytes);
    DeviceBuffer dstBuffer(nbytes);
    size_t llBufSize = ll_buffer_size(nbytes);
    DeviceBuffer llBuffer(llBufSize);

    auto* src_d = static_cast<char*>(srcBuffer.get());
    auto* dst_d = static_cast<char*>(dstBuffer.get());
    auto* ll_buf = static_cast<LlLine*>(llBuffer.get());

    CUDACHECK_TEST(
        cudaMemcpy(src_d, pattern.data(), nbytes, cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemset(dst_d, 0, nbytes));

    test::test_ll_multi_step_send_recv(
        src_d, dst_d, nbytes, ll_buf, num_steps, num_blocks, block_size);

    std::vector<char> result(nbytes);
    CUDACHECK_TEST(
        cudaMemcpy(result.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < nbytes; ++i) {
      ASSERT_EQ(result[i], pattern[i]) << "MultiStep: mismatch at byte " << i;
    }
  }

  void run_multi_step_chunked_test(
      size_t nbytes,
      size_t buffer_num_lines,
      int num_steps,
      int num_blocks = 1,
      int block_size = 256) {
    auto pattern = make_pattern(nbytes);

    DeviceBuffer srcBuffer(nbytes);
    DeviceBuffer dstBuffer(nbytes);
    size_t llBufSize = buffer_num_lines * kLlLineSize;
    DeviceBuffer llBuffer(llBufSize);

    auto* src_d = static_cast<char*>(srcBuffer.get());
    auto* dst_d = static_cast<char*>(dstBuffer.get());
    auto* ll_buf = static_cast<LlLine*>(llBuffer.get());

    CUDACHECK_TEST(
        cudaMemcpy(src_d, pattern.data(), nbytes, cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemset(dst_d, 0, nbytes));

    test::test_ll_multi_step_send_recv_chunked(
        src_d,
        dst_d,
        nbytes,
        ll_buf,
        buffer_num_lines,
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

  void run_multi_step_forward_test(
      size_t nbytes,
      int num_steps,
      int num_blocks = 1,
      int block_size = 256) {
    auto pattern = make_pattern(nbytes);

    size_t llBufSize = ll_buffer_size(nbytes);
    DeviceBuffer srcBuffer(nbytes);
    DeviceBuffer fwdDstBuffer(nbytes);
    DeviceBuffer recvDstBuffer(nbytes);
    DeviceBuffer llBufA(llBufSize);
    DeviceBuffer llBufB(llBufSize);

    auto* src_d = static_cast<char*>(srcBuffer.get());
    auto* fwd_dst_d = static_cast<char*>(fwdDstBuffer.get());
    auto* recv_dst_d = static_cast<char*>(recvDstBuffer.get());
    auto* ll_buf_a = static_cast<LlLine*>(llBufA.get());
    auto* ll_buf_b = static_cast<LlLine*>(llBufB.get());

    CUDACHECK_TEST(
        cudaMemcpy(src_d, pattern.data(), nbytes, cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemset(fwd_dst_d, 0, nbytes));
    CUDACHECK_TEST(cudaMemset(recv_dst_d, 0, nbytes));

    test::test_ll_multi_step_forward(
        src_d,
        fwd_dst_d,
        recv_dst_d,
        nbytes,
        ll_buf_a,
        ll_buf_b,
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

  void run_varying_data_multi_step_test(
      size_t nbytes,
      size_t buffer_num_lines,
      int num_steps,
      int num_blocks = 1,
      int block_size = 256) {
    const size_t total_bytes = num_steps * nbytes;

    size_t llBufSize = (buffer_num_lines == 0) ? ll_buffer_size(nbytes)
                                               : buffer_num_lines * kLlLineSize;

    DeviceBuffer srcBuffer(total_bytes);
    DeviceBuffer dstBuffer(total_bytes);
    DeviceBuffer llBuffer(llBufSize);

    auto* src_d = static_cast<char*>(srcBuffer.get());
    auto* dst_d = static_cast<char*>(dstBuffer.get());
    auto* ll_buf = static_cast<LlLine*>(llBuffer.get());

    std::vector<char> src_host(total_bytes);
    for (int step = 0; step < num_steps; ++step) {
      auto pattern = make_pattern(nbytes, /*seed=*/step * 37);
      memcpy(src_host.data() + step * nbytes, pattern.data(), nbytes);
    }
    CUDACHECK_TEST(cudaMemcpy(
        src_d, src_host.data(), total_bytes, cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemset(dst_d, 0, total_bytes));

    test::test_ll_varying_data_multi_step_send_recv(
        src_d,
        dst_d,
        nbytes,
        ll_buf,
        buffer_num_lines,
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

  void run_varying_data_multi_step_forward_test(
      size_t nbytes,
      size_t buffer_num_lines,
      int num_steps,
      int num_blocks = 1,
      int block_size = 256) {
    const size_t total_bytes = num_steps * nbytes;

    size_t llBufSize = (buffer_num_lines == 0) ? ll_buffer_size(nbytes)
                                               : buffer_num_lines * kLlLineSize;

    DeviceBuffer srcBuffer(total_bytes);
    DeviceBuffer fwdDstBuffer(total_bytes);
    DeviceBuffer recvDstBuffer(total_bytes);
    DeviceBuffer llBufA(llBufSize);
    DeviceBuffer llBufB(llBufSize);

    auto* src_d = static_cast<char*>(srcBuffer.get());
    auto* fwd_dst_d = static_cast<char*>(fwdDstBuffer.get());
    auto* recv_dst_d = static_cast<char*>(recvDstBuffer.get());
    auto* ll_buf_a = static_cast<LlLine*>(llBufA.get());
    auto* ll_buf_b = static_cast<LlLine*>(llBufB.get());

    std::vector<char> src_host(total_bytes);
    for (int step = 0; step < num_steps; ++step) {
      auto pattern = make_pattern(nbytes, /*seed=*/step * 37);
      memcpy(src_host.data() + step * nbytes, pattern.data(), nbytes);
    }
    CUDACHECK_TEST(cudaMemcpy(
        src_d, src_host.data(), total_bytes, cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemset(fwd_dst_d, 0, total_bytes));
    CUDACHECK_TEST(cudaMemset(recv_dst_d, 0, total_bytes));

    test::test_ll_varying_data_multi_step_forward(
        src_d,
        fwd_dst_d,
        recv_dst_d,
        nbytes,
        ll_buf_a,
        ll_buf_b,
        buffer_num_lines,
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
// Group A: SendRecv — various sizes (LL uses 8-byte alignment)
// =============================================================================

class LlOpsSendRecvTest : public LlOpsTestFixture,
                          public ::testing::WithParamInterface<LlOpsTestParam> {
};

TEST_P(LlOpsSendRecvTest, SendRecv) {
  const auto& p = GetParam();
  run_send_recv_test(p.nbytes, p.num_blocks, p.block_size);
}

INSTANTIATE_TEST_SUITE_P(
    LlOps,
    LlOpsSendRecvTest,
    ::testing::Values(
        LlOpsTestParam{.name = "8Bytes", .nbytes = 8},
        LlOpsTestParam{.name = "16Bytes", .nbytes = 16},
        LlOpsTestParam{.name = "64Bytes", .nbytes = 64},
        LlOpsTestParam{.name = "256Bytes", .nbytes = 256},
        LlOpsTestParam{.name = "1KB", .nbytes = 1024},
        LlOpsTestParam{.name = "4KB", .nbytes = 4096},
        LlOpsTestParam{.name = "64KB", .nbytes = 65536},
        LlOpsTestParam{
            .name = "MultiBlock_64KB",
            .nbytes = 65536,
            .num_blocks = 8}),
    ll_test_name);

// =============================================================================
// Group C: Forward
// =============================================================================

class LlOpsForwardTest : public LlOpsTestFixture,
                         public ::testing::WithParamInterface<LlOpsTestParam> {
};

TEST_P(LlOpsForwardTest, Forward) {
  const auto& p = GetParam();
  run_forward_test(p.nbytes, p.num_blocks, p.block_size);
}

INSTANTIATE_TEST_SUITE_P(
    LlOps,
    LlOpsForwardTest,
    ::testing::Values(
        LlOpsTestParam{.name = "64Bytes", .nbytes = 64},
        LlOpsTestParam{.name = "4KB", .nbytes = 4096},
        LlOpsTestParam{.name = "64KB", .nbytes = 65536},
        LlOpsTestParam{
            .name = "MultiBlock_64KB",
            .nbytes = 65536,
            .num_blocks = 8}),
    ll_test_name);

// =============================================================================
// Group C: Chunked SendRecv
// =============================================================================

class LlOpsChunkedSendRecvTest
    : public LlOpsTestFixture,
      public ::testing::WithParamInterface<LlOpsTestParam> {};

TEST_P(LlOpsChunkedSendRecvTest, ChunkedSendRecv) {
  const auto& p = GetParam();
  run_send_recv_chunked_test(
      p.nbytes, p.buffer_num_lines, p.num_blocks, p.block_size);
}

INSTANTIATE_TEST_SUITE_P(
    LlOps,
    LlOpsChunkedSendRecvTest,
    ::testing::Values(
        // ll_num_lines(4096) = 4096/8 = 512 lines
        LlOpsTestParam{
            .name = "4KB_32Lines",
            .nbytes = 4096,
            .buffer_num_lines = 32},
        LlOpsTestParam{
            .name = "4KB_64Lines",
            .nbytes = 4096,
            .buffer_num_lines = 64},
        LlOpsTestParam{
            .name = "64KB_64Lines",
            .nbytes = 65536,
            .buffer_num_lines = 64},
        LlOpsTestParam{
            .name = "64KB_256Lines_MultiBlock",
            .nbytes = 65536,
            .num_blocks = 4,
            .buffer_num_lines = 256},
        // Exact fit: 4096/8 = 512 lines
        LlOpsTestParam{
            .name = "ExactFit",
            .nbytes = 4096,
            .buffer_num_lines = 512},
        LlOpsTestParam{
            .name = "WarpClampingStress",
            .nbytes = 65536,
            .num_blocks = 8,
            .buffer_num_lines = 32}),
    ll_test_name);

// =============================================================================
// Group D: MultiStep SendRecv
// =============================================================================

class LlOpsMultiStepSendRecvTest
    : public LlOpsTestFixture,
      public ::testing::WithParamInterface<LlOpsTestParam> {};

TEST_P(LlOpsMultiStepSendRecvTest, MultiStepSendRecv) {
  const auto& p = GetParam();
  run_multi_step_send_recv_test(
      p.nbytes, p.num_steps, p.num_blocks, p.block_size);
}

INSTANTIATE_TEST_SUITE_P(
    LlOps,
    LlOpsMultiStepSendRecvTest,
    ::testing::Values(
        LlOpsTestParam{.name = "InKernel", .nbytes = 4096, .num_steps = 10},
        LlOpsTestParam{.name = "ABA", .nbytes = 4096, .num_steps = 100},
        LlOpsTestParam{
            .name = "MultiBlock",
            .nbytes = 65536,
            .num_blocks = 4,
            .num_steps = 10}),
    ll_test_name);

// =============================================================================
// Group E: MultiStep Chunked SendRecv
// =============================================================================

class LlOpsMultiStepChunkedSendRecvTest
    : public LlOpsTestFixture,
      public ::testing::WithParamInterface<LlOpsTestParam> {};

TEST_P(LlOpsMultiStepChunkedSendRecvTest, MultiStepChunkedSendRecv) {
  const auto& p = GetParam();
  run_multi_step_chunked_test(
      p.nbytes, p.buffer_num_lines, p.num_steps, p.num_blocks, p.block_size);
}

INSTANTIATE_TEST_SUITE_P(
    LlOps,
    LlOpsMultiStepChunkedSendRecvTest,
    ::testing::Values(
        LlOpsTestParam{
            .name = "4KB_64lines_10steps",
            .nbytes = 4096,
            .buffer_num_lines = 64,
            .num_steps = 10},
        LlOpsTestParam{
            .name = "64KB_256lines_MultiBlock",
            .nbytes = 65536,
            .num_blocks = 4,
            .buffer_num_lines = 256,
            .num_steps = 10},
        LlOpsTestParam{
            .name = "ABA_32lines_100steps",
            .nbytes = 4096,
            .block_size = 32,
            .buffer_num_lines = 32,
            .num_steps = 100}),
    ll_test_name);

// =============================================================================
// Group F: MultiStep Forward
// =============================================================================

class LlOpsMultiStepForwardTest
    : public LlOpsTestFixture,
      public ::testing::WithParamInterface<LlOpsTestParam> {};

TEST_P(LlOpsMultiStepForwardTest, MultiStepForward) {
  const auto& p = GetParam();
  run_multi_step_forward_test(
      p.nbytes, p.num_steps, p.num_blocks, p.block_size);
}

INSTANTIATE_TEST_SUITE_P(
    LlOps,
    LlOpsMultiStepForwardTest,
    ::testing::Values(
        LlOpsTestParam{.name = "4KB_10steps", .nbytes = 4096, .num_steps = 10},
        LlOpsTestParam{
            .name = "64KB_MultiBlock",
            .nbytes = 65536,
            .num_blocks = 3,
            .num_steps = 10}),
    ll_test_name);

// =============================================================================
// Group G: VaryingData MultiStep SendRecv
// =============================================================================

class LlOpsVaryingDataSendRecvTest
    : public LlOpsTestFixture,
      public ::testing::WithParamInterface<LlOpsTestParam> {};

TEST_P(LlOpsVaryingDataSendRecvTest, VaryingDataSendRecv) {
  const auto& p = GetParam();
  run_varying_data_multi_step_test(
      p.nbytes, p.buffer_num_lines, p.num_steps, p.num_blocks, p.block_size);
}

INSTANTIATE_TEST_SUITE_P(
    LlOps,
    LlOpsVaryingDataSendRecvTest,
    ::testing::Values(
        LlOpsTestParam{.name = "NonChunked", .nbytes = 4096, .num_steps = 10},
        LlOpsTestParam{
            .name = "Chunked",
            .nbytes = 4096,
            .buffer_num_lines = 64,
            .num_steps = 10}),
    ll_test_name);

// =============================================================================
// Group H: VaryingData MultiStep Forward
// =============================================================================

class LlOpsVaryingDataForwardTest
    : public LlOpsTestFixture,
      public ::testing::WithParamInterface<LlOpsTestParam> {};

TEST_P(LlOpsVaryingDataForwardTest, VaryingDataForward) {
  const auto& p = GetParam();
  run_varying_data_multi_step_forward_test(
      p.nbytes, p.buffer_num_lines, p.num_steps, p.num_blocks, p.block_size);
}

INSTANTIATE_TEST_SUITE_P(
    LlOps,
    LlOpsVaryingDataForwardTest,
    ::testing::Values(
        LlOpsTestParam{.name = "NonChunked", .nbytes = 4096, .num_steps = 10},
        LlOpsTestParam{
            .name = "Chunked",
            .nbytes = 4096,
            .buffer_num_lines = 64,
            .num_steps = 10}),
    ll_test_name);

// =============================================================================
// Standalone tests
// =============================================================================

TEST_F(LlOpsTestFixture, SendRecv_ZeroBytes) {
  DeviceBuffer llBuffer(kLlLineSize);
  auto* ll_buf = static_cast<LlLine*>(llBuffer.get());
  CUDACHECK_TEST(cudaMemset(ll_buf, kLlMemsetInitByte, kLlLineSize));
  test::test_ll_send_recv(nullptr, nullptr, 0, ll_buf, 1, 32);
}

TEST_F(LlOpsTestFixture, SendRecv_Stress) {
  constexpr int kStressIterations = 50;
  const size_t nbytes = 4096;

  DeviceBuffer srcBuffer(nbytes);
  DeviceBuffer dstBuffer(nbytes);
  size_t llBufSize = ll_buffer_size(nbytes);
  DeviceBuffer llBuffer(llBufSize);

  auto* src_d = static_cast<char*>(srcBuffer.get());
  auto* dst_d = static_cast<char*>(dstBuffer.get());
  auto* ll_buf = static_cast<LlLine*>(llBuffer.get());

  for (int iter = 0; iter < kStressIterations; ++iter) {
    auto pattern = make_pattern(nbytes, /*seed=*/iter);

    CUDACHECK_TEST(
        cudaMemcpy(src_d, pattern.data(), nbytes, cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemset(dst_d, 0, nbytes));

    test::test_ll_send_recv(src_d, dst_d, nbytes, ll_buf, 1, 256);

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
