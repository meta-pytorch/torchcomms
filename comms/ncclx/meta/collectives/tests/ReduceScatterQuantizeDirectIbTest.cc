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

class ReduceScatterQuantizeDirectIbTest : public NcclxBaseTestFixture {
 protected:
  void SetUp() override {
    NcclxBaseTestFixture::SetUp({
        {"NCCL_CTRAN_ENABLE", "1"},
        {"NCCL_CTRAN_USE_PIPES", "1"},
        {"NCCL_COMM_STATE_DEBUG_TOPO", "nolocal"},
        {"MCCL_CHANNEL_BUFFER_SIZE", "4194304"},
        {"NCCL_MNNVL_ENABLE", "0"},
        {"NCCL_P2P_DISABLE", "1"},
        {"NCCL_REDUCESCATTER_ALGO", "ctdirect_ib"},
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

  void runOnTwoStreams(bool quantized) {
    constexpr std::size_t kCount = 1 << 20;
    constexpr int kIterations = 4;
    const std::size_t sendCount = kCount * static_cast<std::size_t>(numRanks);

    cudaStream_t streams[2]{};
    float* send[2]{};
    float* recv[2][kIterations]{};
    std::uint64_t* seed[2]{};
    std::vector<float> output(kCount);
    float expected[2]{};

    for (int lane = 0; lane < 2; ++lane) {
      CUDACHECK_TEST(
          cudaStreamCreateWithFlags(&streams[lane], cudaStreamNonBlocking));
      CUDACHECK_TEST(cudaMalloc(&send[lane], sendCount * sizeof(float)));
      CUDACHECK_TEST(cudaMalloc(&seed[lane], sizeof(*seed[lane])));
      for (int iteration = 0; iteration < kIterations; ++iteration) {
        CUDACHECK_TEST(
            cudaMalloc(&recv[lane][iteration], kCount * sizeof(float)));
      }

      const float rankValue = static_cast<float>((lane + 1) * 16 + globalRank);
      std::vector<float> input(sendCount, rankValue);
      CUDACHECK_TEST(cudaMemcpy(
          send[lane],
          input.data(),
          sendCount * sizeof(float),
          cudaMemcpyHostToDevice));
      const std::uint64_t hostSeed =
          0x123456789abcdef0ULL + static_cast<std::uint64_t>(lane);
      CUDACHECK_TEST(cudaMemcpy(
          seed[lane], &hostSeed, sizeof(hostSeed), cudaMemcpyHostToDevice));
      expected[lane] = static_cast<float>(
          numRanks * (lane + 1) * 16 + numRanks * (numRanks - 1) / 2);
    }

    for (int iteration = 0; iteration < kIterations; ++iteration) {
      for (int lane = 0; lane < 2; ++lane) {
        const ncclResult_t result = quantized ? ncclReduceScatterQuantize(
                                                    send[lane],
                                                    recv[lane][iteration],
                                                    kCount,
                                                    ncclFloat32,
                                                    ncclBfloat16,
                                                    ncclSum,
                                                    seed[lane],
                                                    comm_->get(),
                                                    streams[lane])
                                              : ncclReduceScatter(
                                                    send[lane],
                                                    recv[lane][iteration],
                                                    kCount,
                                                    ncclFloat32,
                                                    ncclSum,
                                                    comm_->get(),
                                                    streams[lane]);
        ASSERT_EQ(result, ncclSuccess);
      }
    }

    for (int lane = 0; lane < 2; ++lane) {
      CUDACHECK_TEST(cudaStreamSynchronize(streams[lane]));
      for (int iteration = 0; iteration < kIterations; ++iteration) {
        CUDACHECK_TEST(cudaMemcpy(
            output.data(),
            recv[lane][iteration],
            kCount * sizeof(float),
            cudaMemcpyDeviceToHost));
        for (float value : output) {
          EXPECT_EQ(value, expected[lane]);
        }
        CUDACHECK_TEST(cudaFree(recv[lane][iteration]));
      }
      CUDACHECK_TEST(cudaFree(send[lane]));
      CUDACHECK_TEST(cudaFree(seed[lane]));
      CUDACHECK_TEST(cudaStreamDestroy(streams[lane]));
    }
  }

  void runCapturedOnTwoStreams() {
    constexpr std::size_t kCount = 1 << 20;
    const std::size_t sendCount = kCount * static_cast<std::size_t>(numRanks);

    cudaStream_t primary{};
    cudaStream_t streams[2]{};
    cudaEvent_t forkEvent{};
    cudaEvent_t joinEvents[2]{};
    cudaGraph_t graph{};
    cudaGraphExec_t graphExec{};
    float* send[2]{};
    float* recv[3]{};
    std::uint64_t* seed[2]{};
    float expected[2]{};

    CUDACHECK_TEST(cudaStreamCreateWithFlags(&primary, cudaStreamNonBlocking));
    CUDACHECK_TEST(
        cudaEventCreateWithFlags(&forkEvent, cudaEventDisableTiming));
    for (int lane = 0; lane < 2; ++lane) {
      CUDACHECK_TEST(
          cudaStreamCreateWithFlags(&streams[lane], cudaStreamNonBlocking));
      CUDACHECK_TEST(
          cudaEventCreateWithFlags(&joinEvents[lane], cudaEventDisableTiming));
      CUDACHECK_TEST(cudaMalloc(&send[lane], sendCount * sizeof(float)));
      CUDACHECK_TEST(cudaMalloc(&recv[lane], kCount * sizeof(float)));
      CUDACHECK_TEST(cudaMalloc(&seed[lane], sizeof(*seed[lane])));

      const float rankValue = static_cast<float>((lane + 1) * 32 + globalRank);
      std::vector<float> input(sendCount, rankValue);
      CUDACHECK_TEST(cudaMemcpy(
          send[lane],
          input.data(),
          sendCount * sizeof(float),
          cudaMemcpyHostToDevice));
      const std::uint64_t hostSeed =
          0x23456789abcdef01ULL + static_cast<std::uint64_t>(lane);
      CUDACHECK_TEST(cudaMemcpy(
          seed[lane], &hostSeed, sizeof(hostSeed), cudaMemcpyHostToDevice));
      expected[lane] = static_cast<float>(
          numRanks * (lane + 1) * 32 + numRanks * (numRanks - 1) / 2);

      ASSERT_EQ(
          ncclReduceScatterQuantize(
              send[lane],
              recv[lane],
              kCount,
              ncclFloat32,
              ncclBfloat16,
              ncclSum,
              seed[lane],
              comm_->get(),
              streams[lane]),
          ncclSuccess);
    }
    CUDACHECK_TEST(cudaStreamSynchronize(streams[0]));
    CUDACHECK_TEST(cudaStreamSynchronize(streams[1]));
    CUDACHECK_TEST(cudaMalloc(&recv[2], kCount * sizeof(float)));

    CUDACHECK_TEST(
        cudaStreamBeginCapture(primary, cudaStreamCaptureModeRelaxed));
    CUDACHECK_TEST(cudaEventRecord(forkEvent, primary));
    for (int lane = 0; lane < 2; ++lane) {
      CUDACHECK_TEST(cudaStreamWaitEvent(streams[lane], forkEvent, 0));
      ASSERT_EQ(
          ncclReduceScatterQuantize(
              send[lane],
              recv[lane],
              kCount,
              ncclFloat32,
              ncclBfloat16,
              ncclSum,
              seed[lane],
              comm_->get(),
              streams[lane]),
          ncclSuccess);
      CUDACHECK_TEST(cudaEventRecord(joinEvents[lane], streams[lane]));
      CUDACHECK_TEST(cudaStreamWaitEvent(primary, joinEvents[lane], 0));
    }
    CUDACHECK_TEST(cudaStreamEndCapture(primary, &graph));
    ASSERT_NE(graph, nullptr);
    CUDACHECK_TEST(
        cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));
    CUDACHECK_TEST(cudaGraphLaunch(graphExec, primary));
    CUDACHECK_TEST(cudaGraphLaunch(graphExec, primary));

    ASSERT_EQ(
        ncclReduceScatterQuantize(
            send[0],
            recv[2],
            kCount,
            ncclFloat32,
            ncclBfloat16,
            ncclSum,
            seed[0],
            comm_->get(),
            streams[0]),
        ncclSuccess);
    CUDACHECK_TEST(cudaStreamSynchronize(primary));
    CUDACHECK_TEST(cudaStreamSynchronize(streams[0]));

    std::vector<float> output(kCount);
    for (int outputIndex = 0; outputIndex < 3; ++outputIndex) {
      CUDACHECK_TEST(cudaMemcpy(
          output.data(),
          recv[outputIndex],
          kCount * sizeof(float),
          cudaMemcpyDeviceToHost));
      const float expectedValue = expected[outputIndex == 1 ? 1 : 0];
      for (float value : output) {
        EXPECT_EQ(value, expectedValue);
      }
    }

    CUDACHECK_TEST(cudaGraphExecDestroy(graphExec));
    CUDACHECK_TEST(cudaGraphDestroy(graph));
    CUDACHECK_TEST(cudaEventDestroy(forkEvent));
    CUDACHECK_TEST(cudaStreamDestroy(primary));
    for (int lane = 0; lane < 2; ++lane) {
      CUDACHECK_TEST(cudaEventDestroy(joinEvents[lane]));
      CUDACHECK_TEST(cudaStreamDestroy(streams[lane]));
      CUDACHECK_TEST(cudaFree(send[lane]));
      CUDACHECK_TEST(cudaFree(seed[lane]));
    }
    for (float* outputBuffer : recv) {
      CUDACHECK_TEST(cudaFree(outputBuffer));
    }
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

TEST_F(
    ReduceScatterQuantizeDirectIbTest,
    QuantizedCallsAreOrderedAcrossStreams) {
  runOnTwoStreams(true);
  algoStats_.verify(
      comm_->get(), "ReduceScatter", "CtranReduceScatterDirectIb");
}

TEST_F(
    ReduceScatterQuantizeDirectIbTest,
    UnquantizedCallsAreOrderedAcrossStreams) {
  runOnTwoStreams(false);
  algoStats_.verify(
      comm_->get(), "ReduceScatter", "CtranReduceScatterDirectIb");
}

TEST_F(
    ReduceScatterQuantizeDirectIbTest,
    CapturedCallsAreOrderedAcrossStreamsAndEagerWork) {
  runCapturedOnTwoStreams();
  algoStats_.verify(
      comm_->get(), "ReduceScatter", "CtranReduceScatterDirectIb");
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
