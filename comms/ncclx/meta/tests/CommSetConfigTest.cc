// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <folly/init/Init.h>
#include <gtest/gtest.h>

#include "comms/ncclx/meta/tests/NcclCommUtils.h"
#include "comms/ncclx/meta/tests/NcclxBaseTest.h"
#include "meta/NcclxConfig.h"
#include "nccl.h"

class CommSetConfigTest : public NcclxBaseTestFixture {};

TEST_F(CommSetConfigTest, SetSingleAlgoHint) {
  ncclx::test::NcclCommRAII comm(
      globalRank, numRanks, localRank, bootstrap_.get());
  ASSERT_NE(nullptr, comm.get());

  ncclConfig_t newConfig = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints({{"allgatherAlgo", "ctdirect"}});
  newConfig.hints = &hints;

  EXPECT_EQ(ncclSuccess, ncclx::commSetConfig(comm, &newConfig));
  EXPECT_EQ(
      NCCL_ALLGATHER_ALGO::ctdirect,
      NCCLX_CONFIG_FIELD(comm->config, allgatherAlgo));
}

TEST_F(CommSetConfigTest, SetAllAlgoHints) {
  ncclx::test::NcclCommRAII comm(
      globalRank, numRanks, localRank, bootstrap_.get());
  ASSERT_NE(nullptr, comm.get());

  ncclConfig_t newConfig = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints({
      {"allgatherAlgo", "ctring"},
      {"allreduceAlgo", "ctdirect"},
      {"sendrecvAlgo", "ctran"},
      {"alltoallvAlgo", "ctran"},
  });
  newConfig.hints = &hints;

  EXPECT_EQ(ncclSuccess, ncclx::commSetConfig(comm, &newConfig));
  EXPECT_EQ(
      NCCL_ALLGATHER_ALGO::ctring,
      NCCLX_CONFIG_FIELD(comm->config, allgatherAlgo));
  EXPECT_EQ(
      NCCL_ALLREDUCE_ALGO::ctdirect,
      NCCLX_CONFIG_FIELD(comm->config, allreduceAlgo));
  EXPECT_EQ(
      NCCL_SENDRECV_ALGO::ctran,
      NCCLX_CONFIG_FIELD(comm->config, sendrecvAlgo));
  EXPECT_EQ(
      NCCL_ALLTOALLV_ALGO::ctran,
      NCCLX_CONFIG_FIELD(comm->config, alltoallvAlgo));
}

TEST_F(CommSetConfigTest, RejectImmutableHintKey) {
  ncclx::test::NcclCommRAII comm(
      globalRank, numRanks, localRank, bootstrap_.get());
  ASSERT_NE(nullptr, comm.get());

  ncclConfig_t newConfig = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints({{"useCtran", "1"}});
  newConfig.hints = &hints;

  EXPECT_EQ(ncclInvalidUsage, ncclx::commSetConfig(comm, &newConfig));
}

TEST_F(CommSetConfigTest, RejectImmutableHintKeyNcclBuffSize) {
  ncclx::test::NcclCommRAII comm(
      globalRank, numRanks, localRank, bootstrap_.get());
  ASSERT_NE(nullptr, comm.get());

  ncclConfig_t newConfig = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints({{"ncclBuffSize", "1048576"}});
  newConfig.hints = &hints;

  EXPECT_EQ(ncclInvalidUsage, ncclx::commSetConfig(comm, &newConfig));
}

TEST_F(CommSetConfigTest, RejectFlatFieldBlocking) {
  ncclx::test::NcclCommRAII comm(
      globalRank, numRanks, localRank, bootstrap_.get());
  ASSERT_NE(nullptr, comm.get());

  ncclConfig_t newConfig = NCCL_CONFIG_INITIALIZER;
  newConfig.blocking = 1;

  EXPECT_EQ(ncclInvalidUsage, ncclx::commSetConfig(comm, &newConfig));
}

TEST_F(CommSetConfigTest, RejectFlatFieldMinCTAs) {
  ncclx::test::NcclCommRAII comm(
      globalRank, numRanks, localRank, bootstrap_.get());
  ASSERT_NE(nullptr, comm.get());

  ncclConfig_t newConfig = NCCL_CONFIG_INITIALIZER;
  newConfig.minCTAs = 4;

  EXPECT_EQ(ncclInvalidUsage, ncclx::commSetConfig(comm, &newConfig));
}

TEST_F(CommSetConfigTest, RejectNcclxFlatFieldCommDesc) {
  ncclx::test::NcclCommRAII comm(
      globalRank, numRanks, localRank, bootstrap_.get());
  ASSERT_NE(nullptr, comm.get());

  ncclConfig_t newConfig = NCCL_CONFIG_INITIALIZER;
  newConfig.commDesc = "test";

  EXPECT_EQ(ncclInvalidUsage, ncclx::commSetConfig(comm, &newConfig));
}

TEST_F(CommSetConfigTest, RejectUninitializedConfig) {
  ncclx::test::NcclCommRAII comm(
      globalRank, numRanks, localRank, bootstrap_.get());
  ASSERT_NE(nullptr, comm.get());

  ncclConfig_t newConfig{};

  EXPECT_EQ(ncclInvalidArgument, ncclx::commSetConfig(comm, &newConfig));
}

TEST_F(CommSetConfigTest, RejectNullConfig) {
  ncclx::test::NcclCommRAII comm(
      globalRank, numRanks, localRank, bootstrap_.get());
  ASSERT_NE(nullptr, comm.get());

  EXPECT_EQ(ncclInvalidArgument, ncclx::commSetConfig(comm, nullptr));
}

TEST_F(CommSetConfigTest, NoOpWithNoHints) {
  ncclx::test::NcclCommRAII comm(
      globalRank, numRanks, localRank, bootstrap_.get());
  ASSERT_NE(nullptr, comm.get());

  const auto origAlgo = NCCLX_CONFIG_FIELD(comm->config, allgatherAlgo);

  ncclConfig_t newConfig = NCCL_CONFIG_INITIALIZER;
  EXPECT_EQ(ncclSuccess, ncclx::commSetConfig(comm, &newConfig));
  EXPECT_EQ(origAlgo, NCCLX_CONFIG_FIELD(comm->config, allgatherAlgo));
}

TEST_F(CommSetConfigTest, MultipleSequentialUpdates) {
  ncclx::test::NcclCommRAII comm(
      globalRank, numRanks, localRank, bootstrap_.get());
  ASSERT_NE(nullptr, comm.get());

  {
    ncclConfig_t newConfig = NCCL_CONFIG_INITIALIZER;
    ncclx::Hints hints({{"allgatherAlgo", "ctdirect"}});
    newConfig.hints = &hints;
    EXPECT_EQ(ncclSuccess, ncclx::commSetConfig(comm, &newConfig));
    EXPECT_EQ(
        NCCL_ALLGATHER_ALGO::ctdirect,
        NCCLX_CONFIG_FIELD(comm->config, allgatherAlgo));
  }

  {
    ncclConfig_t newConfig = NCCL_CONFIG_INITIALIZER;
    ncclx::Hints hints({{"allgatherAlgo", "ctring"}});
    newConfig.hints = &hints;
    EXPECT_EQ(ncclSuccess, ncclx::commSetConfig(comm, &newConfig));
    EXPECT_EQ(
        NCCL_ALLGATHER_ALGO::ctring,
        NCCLX_CONFIG_FIELD(comm->config, allgatherAlgo));
  }
}

TEST_F(CommSetConfigTest, RejectMixedMutableAndImmutableHints) {
  ncclx::test::NcclCommRAII comm(
      globalRank, numRanks, localRank, bootstrap_.get());
  ASSERT_NE(nullptr, comm.get());

  const auto origAlgo = NCCLX_CONFIG_FIELD(comm->config, allgatherAlgo);

  ncclConfig_t newConfig = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints({
      {"allgatherAlgo", "ctdirect"},
      {"useCtran", "1"},
  });
  newConfig.hints = &hints;

  EXPECT_EQ(ncclInvalidUsage, ncclx::commSetConfig(comm, &newConfig));
  EXPECT_EQ(origAlgo, NCCLX_CONFIG_FIELD(comm->config, allgatherAlgo));
}

TEST_F(CommSetConfigTest, RejectInsideGroupCall) {
  ncclx::test::NcclCommRAII comm(
      globalRank, numRanks, localRank, bootstrap_.get());
  ASSERT_NE(nullptr, comm.get());

  ncclGroupStart();

  ncclConfig_t newConfig = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints({{"allgatherAlgo", "ctdirect"}});
  newConfig.hints = &hints;

  EXPECT_EQ(ncclInvalidUsage, ncclx::commSetConfig(comm, &newConfig));

  ncclGroupEnd();
}

TEST_F(CommSetConfigTest, RejectWhenAsyncOpInProgress) {
  ncclx::test::NcclCommRAII comm(
      globalRank, numRanks, localRank, bootstrap_.get());
  ASSERT_NE(nullptr, comm.get());

  ASSERT_EQ(ncclSuccess, ncclCommSetAsyncError(comm, ncclInProgress));

  ncclConfig_t newConfig = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints({{"allgatherAlgo", "ctdirect"}});
  newConfig.hints = &hints;

  EXPECT_NE(ncclSuccess, ncclx::commSetConfig(comm, &newConfig));

  ASSERT_EQ(ncclSuccess, ncclCommSetAsyncError(comm, ncclSuccess));
}

TEST_F(CommSetConfigTest, RejectAfterCommAbort) {
  ncclComm_t rawComm = ncclx::test::createNcclComm(
      globalRank, numRanks, localRank, bootstrap_.get());
  ASSERT_NE(nullptr, rawComm);

  ASSERT_EQ(ncclSuccess, ncclCommAbort(rawComm));

  ncclConfig_t newConfig = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints({{"allgatherAlgo", "ctdirect"}});
  newConfig.hints = &hints;

  EXPECT_NE(ncclSuccess, ncclx::commSetConfig(rawComm, &newConfig));
}

TEST_F(CommSetConfigTest, SetConfigAfterCollective) {
  ncclx::test::NcclCommRAII comm(
      globalRank, numRanks, localRank, bootstrap_.get());
  ASSERT_NE(nullptr, comm.get());

  const size_t count = 1024;
  void* buf = nullptr;
  ASSERT_EQ(cudaSuccess, cudaMalloc(&buf, count * sizeof(float)));

  cudaStream_t stream;
  ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

  ASSERT_EQ(
      ncclSuccess,
      ncclAllReduce(buf, buf, count, ncclFloat, ncclSum, comm, stream));
  ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

  ncclConfig_t newConfig = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints({{"allgatherAlgo", "ctdirect"}});
  newConfig.hints = &hints;

  EXPECT_EQ(ncclSuccess, ncclx::commSetConfig(comm, &newConfig));
  EXPECT_EQ(
      NCCL_ALLGATHER_ALGO::ctdirect,
      NCCLX_CONFIG_FIELD(comm->config, allgatherAlgo));

  cudaStreamDestroy(stream);
  cudaFree(buf);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
