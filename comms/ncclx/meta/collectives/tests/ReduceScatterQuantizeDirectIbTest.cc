// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <comm.h>
#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include <nccl.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "comms/ncclx/meta/tests/NcclCommUtils.h"
#include "comms/ncclx/meta/tests/NcclxBaseTest.h"
#include "comms/ncclx/meta/tests/VerifyAlgoStatsUtil.h"
#include "comms/testinfra/TestUtils.h"

class ReduceScatterQuantizeDirectIbTest : public NcclxBaseTestFixture {
 protected:
  void SetUp() override {
    NcclxBaseTestFixture::SetUp({
        {"NCCL_CTRAN_ENABLE", "1"},
        {"NCCL_CTRAN_USE_PIPES", "1"},
        {"NCCL_COMM_STATE_DEBUG_TOPO", "nolocal"},
        {"NCCL_MNNVL_ENABLE", "0"},
        {"NCCL_P2P_DISABLE", "1"},
        {"NCCL_REDUCESCATTER_QUANTIZED_ALGO", "ctdirect_ib"},
    });
    algoStats_.enable();
    ncclx::Hints hints{{"useCtran", "1"}};
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.hints = &hints;
    comm_.emplace(
        globalRank, numRanks, localRank, bootstrap_.get(), false, &config);
    CUDACHECK_TEST(cudaStreamCreate(&stream_));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaStreamDestroy(stream_));
    comm_.reset();
    NcclxBaseTestFixture::TearDown();
  }

  void run(ncclRedOp_t op, std::size_t count) {
    const std::size_t sendCount = count * static_cast<std::size_t>(numRanks);
    std::vector<float> input(sendCount, 1.0f);
    std::vector<float> output(count);

    float* send = nullptr;
    float* recv = nullptr;
    std::uint64_t* seed = nullptr;
    if (count != 0) {
      CUDACHECK_TEST(cudaMalloc(&send, sendCount * sizeof(float)));
      CUDACHECK_TEST(cudaMalloc(&recv, count * sizeof(float)));
      CUDACHECK_TEST(cudaMemcpy(
          send,
          input.data(),
          sendCount * sizeof(float),
          cudaMemcpyHostToDevice));
    }
    CUDACHECK_TEST(cudaMalloc(&seed, sizeof(*seed)));
    constexpr std::uint64_t kSeed = 0x123456789abcdef0ULL;
    CUDACHECK_TEST(
        cudaMemcpy(seed, &kSeed, sizeof(kSeed), cudaMemcpyHostToDevice));

    ASSERT_EQ(
        ncclReduceScatterQuantize(
            send,
            recv,
            count,
            ncclFloat32,
            ncclBfloat16,
            op,
            seed,
            comm_->get(),
            stream_),
        ncclSuccess);
    CUDACHECK_TEST(cudaStreamSynchronize(stream_));

    if (count != 0) {
      CUDACHECK_TEST(cudaMemcpy(
          output.data(), recv, count * sizeof(float), cudaMemcpyDeviceToHost));
      const float expected =
          op == ncclSum ? static_cast<float>(numRanks) : 1.0f;
      for (float value : output) {
        EXPECT_EQ(value, expected);
      }
      CUDACHECK_TEST(cudaFree(send));
      CUDACHECK_TEST(cudaFree(recv));
    }
    CUDACHECK_TEST(cudaFree(seed));
  }

  std::optional<ncclx::test::NcclCommRAII> comm_;
  cudaStream_t stream_{nullptr};
  ncclx::test::VerifyAlgoStatsHelper algoStats_;
};

class ReduceScatterQuantizePatTest : public ReduceScatterQuantizeDirectIbTest {
 protected:
  void SetUp() override {
    NcclxBaseTestFixture::SetUp({{"NCCL_CTRAN_ENABLE", "0"}});
    algoStats_.enable();
    comm_.emplace(globalRank, numRanks, localRank, bootstrap_.get());
    CUDACHECK_TEST(cudaStreamCreate(&stream_));
  }
};

class ReduceScatterQuantizeDefaultPatTest
    : public ReduceScatterQuantizeDirectIbTest {
 protected:
  void SetUp() override {
    NcclxBaseTestFixture::SetUp({
        {"NCCL_CTRAN_ENABLE", "1"},
        {"NCCL_CTRAN_USE_PIPES", "1"},
        {"NCCL_COMM_STATE_DEBUG_TOPO", "nolocal"},
        {"NCCL_MNNVL_ENABLE", "0"},
        {"NCCL_P2P_DISABLE", "1"},
    });
    algoStats_.enable();
    ncclx::Hints hints{{"useCtran", "1"}};
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.hints = &hints;
    comm_.emplace(
        globalRank, numRanks, localRank, bootstrap_.get(), false, &config);
    CUDACHECK_TEST(cudaStreamCreate(&stream_));
  }
};

TEST_F(ReduceScatterQuantizeDirectIbTest, SumUsesDirectIbForOddTail) {
  run(ncclSum, 1025);
  algoStats_.verify(
      comm_->get(), "ReduceScatter", "CtranReduceScatterDirectIb");
}

TEST_F(ReduceScatterQuantizeDirectIbTest, ZeroCountSucceeds) {
  run(ncclSum, 0);
}

TEST_F(ReduceScatterQuantizePatTest, CtranDisabledUsesPat) {
  run(ncclSum, 1025);
  algoStats_.verify(comm_->get(), "ReduceScatter", "PAT");
  algoStats_.verifyNot(
      comm_->get(), "ReduceScatter", "CtranReduceScatterDirectIb");
}

TEST_F(ReduceScatterQuantizeDefaultPatTest, DefaultUsesPat) {
  run(ncclSum, 1025);
  algoStats_.verify(comm_->get(), "ReduceScatter", "PAT");
  algoStats_.verifyNot(
      comm_->get(), "ReduceScatter", "CtranReduceScatterDirectIb");
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
