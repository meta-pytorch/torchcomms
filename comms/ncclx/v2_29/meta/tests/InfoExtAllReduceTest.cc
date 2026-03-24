// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <folly/init/Init.h>

#include <nccl.h>
#include "VerifyAlgoStatsUtil.h"
#include "collectives.h"
#include "comms/testinfra/TestsDistUtils.h"

#include "comms/ncclx/meta/tests/NcclCommUtils.h"
#include "device.h"
#include "enqueue.h"
#include "meta/algoconf/InfoExt.h"

using ncclx::algoconf::ncclInfoExt;

struct InfoExtAllReduceParams {
  int algorithm;
  int protocol;
  int nMaxChannels;
  int nWarps;
  size_t count;
  std::string expectedAlgoSubstr; // e.g., "SIMPLE_RING_8"

  std::string name() const {
    return expectedAlgoSubstr;
  }
};

class InfoExtAllReduceTest
    : public NcclxBaseTest,
      public ::testing::WithParamInterface<InfoExtAllReduceParams> {
 protected:
  ncclx::test::VerifyAlgoStatsHelper algoStats_;

  void SetUp() override {
    NcclxBaseTest::SetUp();
    algoStats_.enable(); // Must be called before comm creation
  }
};

TEST_P(InfoExtAllReduceTest, Override) {
  const auto& param = GetParam();

  ncclx::test::NcclCommRAII commGuard(
      globalRank, numRanks, localRank, bootstrap_.get());
  cudaStream_t stream;
  CUDACHECK_TEST(cudaStreamCreate(&stream));

  float *sendbuff, *recvbuff;
  CUDACHECK_TEST(cudaMalloc(&sendbuff, param.count * sizeof(float)));
  CUDACHECK_TEST(cudaMalloc(&recvbuff, param.count * sizeof(float)));

  ncclInfoExt ext(
      param.algorithm, param.protocol, param.nMaxChannels, param.nWarps);

  struct ncclInfo info = {
      .coll = ncclFuncAllReduce,
      .opName = "AllReduce",
      .sendbuff = sendbuff,
      .recvbuff = recvbuff,
      .count = param.count,
      .datatype = ncclFloat,
      .op = ncclSum,
      .root = 0,
      .comm = commGuard.get(),
      .stream = stream,
      .chunkSteps = ALLREDUCE_CHUNKSTEPS,
      .sliceSteps = ALLREDUCE_SLICESTEPS,
      .ext = ext};

  ASSERT_EQ(ncclEnqueueCheck(&info), ncclSuccess);
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  // Verify algo+proto+nChannels via AlgoStats
  // Format: Baseline_{proto}_{algo}_{nChannels}
  algoStats_.verify(commGuard.get(), "AllReduce", param.expectedAlgoSubstr);

  CUDACHECK_TEST(cudaFree(sendbuff));
  CUDACHECK_TEST(cudaFree(recvbuff));
  CUDACHECK_TEST(cudaStreamDestroy(stream));
}

INSTANTIATE_TEST_SUITE_P(
    AlgoOverride,
    InfoExtAllReduceTest,
    ::testing::Values(
        // RING + SIMPLE, 8 channels
        InfoExtAllReduceParams{
            .algorithm = NCCL_ALGO_RING,
            .protocol = NCCL_PROTO_SIMPLE,
            .nMaxChannels = 8,
            .nWarps = 16,
            .count = 1024,
            .expectedAlgoSubstr = "SIMPLE_RING_8"},
        // TREE + SIMPLE, 4 channels
        InfoExtAllReduceParams{
            .algorithm = NCCL_ALGO_TREE,
            .protocol = NCCL_PROTO_SIMPLE,
            .nMaxChannels = 4,
            .nWarps = 8,
            .count = 1024,
            .expectedAlgoSubstr = "SIMPLE_TREE_4"},
        // RING + LL, 8 channels
        InfoExtAllReduceParams{
            .algorithm = NCCL_ALGO_RING,
            .protocol = NCCL_PROTO_LL,
            .nMaxChannels = 8,
            .nWarps = 8,
            .count = 256,
            .expectedAlgoSubstr = "LL_RING_8"}),
    [](const ::testing::TestParamInfo<InfoExtAllReduceParams>& info) {
      return info.param.name();
    });

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
