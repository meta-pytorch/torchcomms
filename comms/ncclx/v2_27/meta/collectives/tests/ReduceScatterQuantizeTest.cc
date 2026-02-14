// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <comm.h>
#include <fmt/core.h>
#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>
#include <cstddef>
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsCuUtils.h"
#include "comms/testinfra/TestsDistUtils.h"

class ReduceScatterQuantizeTest : public NcclxBaseTest {
 public:
  ReduceScatterQuantizeTest() = default;
  void SetUp() override {
    NcclxBaseTest::SetUp();
    comm = createNcclComm(globalRank, numRanks, localRank);
    CUDACHECK_TEST(cudaStreamCreate(&stream));
  }

  void TearDown() override {
    NCCLCHECK_TEST(ncclCommDestroy(comm));
    CUDACHECK_TEST(cudaStreamDestroy(stream));
    NcclxBaseTest::TearDown();
  }

 protected:
  ncclComm_t comm;
  cudaStream_t stream;
};

// Test parameters: <ncclRedOp_t, size_t count, uint64_t seed>
class ReduceScatterQuantizeTestParam
    : public ReduceScatterQuantizeTest,
      public ::testing::WithParamInterface<
          std::tuple<ncclRedOp_t, size_t, uint64_t>> {};

TEST_P(ReduceScatterQuantizeTestParam, CorrectReduction) {
  const auto& [redOp, count, seed] = GetParam();

  // Allocate buffers - input is FP32, output is also FP32
  float *sendBuf = nullptr, *recvBuf = nullptr;
  size_t sendSize = count * numRanks * sizeof(float);
  size_t recvSize = count * sizeof(float);

  CUDACHECK_TEST(cudaMalloc(&sendBuf, sendSize));
  CUDACHECK_TEST(cudaMalloc(&recvBuf, recvSize));

  // Initialize send buffer with deterministic values
  // Each rank r sends: sendBuf[chunk_for_rank_c][i] = r * numRanks + c + i *
  // 0.001 This allows us to compute expected values after reduce scatter
  std::vector<float> hostSendBuf(count * numRanks);
  for (int c = 0; c < numRanks; c++) {
    for (size_t i = 0; i < count; i++) {
      hostSendBuf[c * count + i] =
          static_cast<float>(globalRank * numRanks + c) +
          static_cast<float>(i) * 0.001f;
    }
  }
  CUDACHECK_TEST(cudaMemcpy(
      sendBuf, hostSendBuf.data(), sendSize, cudaMemcpyHostToDevice));

  // Initialize recv buffer with sentinel value
  CUDACHECK_TEST(cudaMemset(recvBuf, 0xFF, recvSize));

  // Initialize seed
  uint64_t* seedBuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&seedBuf, sizeof(uint64_t)));

  // Perform reduce scatter with quantization
  auto res = ncclReduceScatterQuantize(
      sendBuf,
      recvBuf,
      count,
      ncclFloat32, // inputType
      ncclBfloat16, // transportType
      redOp,
      seedBuf,
      comm,
      stream);
  ASSERT_EQ(res, ncclSuccess);

  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Copy result back to host for verification
  std::vector<float> hostRecvBuf(count);
  CUDACHECK_TEST(cudaMemcpy(
      hostRecvBuf.data(), recvBuf, recvSize, cudaMemcpyDeviceToHost));

  // Compute expected values:
  // For rank 'globalRank', we receive the reduction of chunk 'globalRank' from
  // all ranks. Each rank r contributes: r * numRanks + globalRank + i * 0.001
  // Sum = sum over r of (r * numRanks + globalRank + i * 0.001)
  //     = numRanks * (0 + 1 + ... + numRanks-1) + numRanks * globalRank + i *
  //     0.001 * numRanks = numRanks * numRanks * (numRanks-1) / 2 + numRanks *
  //     globalRank + i * 0.001 * numRanks
  int errs = 0;
  for (size_t i = 0; i < count; i++) {
    float expectedSum = static_cast<float>(numRanks) *
            static_cast<float>(numRanks) * static_cast<float>(numRanks - 1) /
            2.0f +
        static_cast<float>(numRanks * globalRank) +
        static_cast<float>(i) * 0.001f * static_cast<float>(numRanks);

    float expected = expectedSum;
    if (redOp == ncclAvg) {
      expected = expectedSum / static_cast<float>(numRanks);
    }

    // Use larger tolerance due to BF16 quantization
    constexpr float tolerance = 0.1f;
    if (std::abs(hostRecvBuf[i] - expected) > tolerance) {
      if (errs < 10) {
        printf(
            "Rank %d: Mismatch at index %zu: expected=%f, got=%f (diff=%f)\n",
            globalRank,
            i,
            expected,
            hostRecvBuf[i],
            std::abs(hostRecvBuf[i] - expected));
      }
      errs++;
    }
  }

  EXPECT_EQ(errs, 0) << "Rank " << globalRank << " found " << errs
                     << " mismatches out of " << count << " elements";

  CUDACHECK_TEST(cudaFree(sendBuf));
  CUDACHECK_TEST(cudaFree(recvBuf));
}

// Test invalid input type (should fail)
TEST_F(ReduceScatterQuantizeTest, InvalidInputType) {
  size_t count = 1024;
  void *sendBuf = nullptr, *recvBuf = nullptr;

  CUDACHECK_TEST(cudaMalloc(&sendBuf, count * numRanks * sizeof(float)));
  CUDACHECK_TEST(cudaMalloc(&recvBuf, count * sizeof(float)));

  // Using ncclFloat16 as input type should fail
  auto res = ncclReduceScatterQuantize(
      sendBuf,
      recvBuf,
      count,
      ncclFloat16, // invalid - must be ncclFloat32
      ncclBfloat16,
      ncclSum,
      0,
      comm,
      stream);
  EXPECT_EQ(res, ncclInvalidArgument);

  CUDACHECK_TEST(cudaFree(sendBuf));
  CUDACHECK_TEST(cudaFree(recvBuf));
}

// Test invalid transport type (should fail)
TEST_F(ReduceScatterQuantizeTest, InvalidTransportType) {
  size_t count = 1024;
  void *sendBuf = nullptr, *recvBuf = nullptr;

  CUDACHECK_TEST(cudaMalloc(&sendBuf, count * numRanks * sizeof(float)));
  CUDACHECK_TEST(cudaMalloc(&recvBuf, count * sizeof(float)));

  // Using ncclFloat16 as transport type should fail
  auto res = ncclReduceScatterQuantize(
      sendBuf,
      recvBuf,
      count,
      ncclFloat32,
      ncclFloat16, // invalid - must be ncclBfloat16
      ncclSum,
      0,
      comm,
      stream);
  EXPECT_EQ(res, ncclInvalidArgument);

  CUDACHECK_TEST(cudaFree(sendBuf));
  CUDACHECK_TEST(cudaFree(recvBuf));
}

// Test invalid reduction operation (should fail)
TEST_F(ReduceScatterQuantizeTest, InvalidRedOp) {
  size_t count = 1024;
  void *sendBuf = nullptr, *recvBuf = nullptr;

  CUDACHECK_TEST(cudaMalloc(&sendBuf, count * numRanks * sizeof(float)));
  CUDACHECK_TEST(cudaMalloc(&recvBuf, count * sizeof(float)));

  // Using ncclMax should fail - only ncclSum and ncclAvg are supported
  auto res = ncclReduceScatterQuantize(
      sendBuf,
      recvBuf,
      count,
      ncclFloat32,
      ncclBfloat16,
      ncclMax, // invalid - must be ncclSum or ncclAvg
      0,
      comm,
      stream);
  EXPECT_EQ(res, ncclInvalidArgument);

  CUDACHECK_TEST(cudaFree(sendBuf));
  CUDACHECK_TEST(cudaFree(recvBuf));
}

INSTANTIATE_TEST_SUITE_P(
    ReduceScatterQuantizeTestInstance,
    ReduceScatterQuantizeTestParam,
    ::testing::Values(
        // redOp, count, seed
        std::make_tuple(ncclSum, 1024, 0UL),
        std::make_tuple(ncclSum, 8192, 42UL),
        std::make_tuple(ncclSum, 65536, 12345UL),
        std::make_tuple(ncclAvg, 1024, 0UL),
        std::make_tuple(ncclAvg, 8192, 42UL),
        std::make_tuple(ncclAvg, 65536, 12345UL)),
    [](const testing::TestParamInfo<ReduceScatterQuantizeTestParam::ParamType>&
           info) {
      const char* opName;
      switch (std::get<0>(info.param)) {
        case ncclSum:
          opName = "Sum";
          break;
        case ncclAvg:
          opName = "Avg";
          break;
        default:
          opName = "Unknown";
          break;
      }
      return fmt::format(
          "{}_{}count_seed{}",
          opName,
          std::get<1>(info.param),
          std::get<2>(info.param));
    });

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
