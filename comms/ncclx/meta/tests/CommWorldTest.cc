// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <stdlib.h>

#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/init/Init.h>

#include "comm.h"
#include "comms/ncclx/meta/tests/NcclCommUtils.h"
#include "comms/ncclx/meta/tests/NcclxBaseTest.h"
#include "comms/testinfra/TestUtils.h"
#include "nccl.h"

class CommWorldTestFixture : public NcclxBaseTestFixture,
                             public ::testing::WithParamInterface<NcclxEnvs> {
 protected:
  void SetUp() override {
    NcclxBaseTestFixture::SetUp(GetParam());
    // Set NCCL_FIRST_COMM_AS_WORLD as the default value
    setenv("NCCL_FIRST_COMM_AS_WORLD", "false", 0);
  }
};

TEST_P(CommWorldTestFixture, FirstCommAsWorld) {
  EnvRAII env(NCCL_FIRST_COMM_AS_WORLD, true);
  NCCL_COMM_WORLD = NULL;
  EXPECT_EQ(NCCL_COMM_WORLD, nullptr);

  ncclComm_t comm = ncclx::test::createNcclComm(
      globalRank, numRanks, localRank, bootstrap_.get());
  ASSERT_NE(comm, nullptr);

  printf("NCCL_FIRST_COMM_AS_WORLD: %d\n", NCCL_FIRST_COMM_AS_WORLD);
  ASSERT_NE(NCCL_COMM_WORLD, nullptr);
  EXPECT_EQ(NCCL_COMM_WORLD->commHash, comm->commHash);
  printf("NCCL_COMM_WORLD is correctly assigned\n");
  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

TEST_P(CommWorldTestFixture, DefaultCommWorld) {
  NCCL_COMM_WORLD = NULL;
  EXPECT_EQ(NCCL_COMM_WORLD, nullptr);

  ncclComm_t comm = ncclx::test::createNcclComm(
      globalRank, numRanks, localRank, bootstrap_.get());
  ASSERT_NE(comm, nullptr);

  printf("NCCL_FIRST_COMM_AS_WORLD: %d\n", NCCL_FIRST_COMM_AS_WORLD);
  EXPECT_EQ(NCCL_COMM_WORLD, nullptr);
  printf("NCCL_COMM_WORLD is nullptr as expected\n");

  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

INSTANTIATE_TEST_SUITE_P(
    CommWorld,
    CommWorldTestFixture,
    testing::Values(
        NcclxEnvs({{"NCCL_FASTINIT_MODE", "none"}}),
        NcclxEnvs({{"NCCL_FASTINIT_MODE", "ring_hybrid"}})),
    [](const testing::TestParamInfo<CommWorldTestFixture::ParamType>& info) {
      return "fastinit_" + info.param.at(0).second;
    });

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
