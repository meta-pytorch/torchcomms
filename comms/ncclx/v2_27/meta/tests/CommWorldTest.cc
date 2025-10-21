// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <stdlib.h>

#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/init/Init.h>

#include "checks.h"
#include "comm.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "nccl.h"

class CommWorldTest : public NcclxBaseTest {
 public:
  void SetUp() override {
    NcclxBaseTest::SetUp();
    // Set NCCL_FIRST_COMM_AS_WORLD as the default value
    setenv("NCCL_FIRST_COMM_AS_WORLD", "false", 0);
  }

  void TearDown() override {
    NcclxBaseTest::TearDown();
  }
};

TEST_F(CommWorldTest, FirstCommAsWorld) {
  EnvRAII env(NCCL_FIRST_COMM_AS_WORLD, true);
  NCCL_COMM_WORLD = NULL;
  EXPECT_EQ(NCCL_COMM_WORLD, nullptr);

  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);
  ASSERT_NE(comm, nullptr);

  printf("NCCL_FIRST_COMM_AS_WORLD: %d\n", NCCL_FIRST_COMM_AS_WORLD);
  ASSERT_NE(NCCL_COMM_WORLD, nullptr);
  EXPECT_EQ(NCCL_COMM_WORLD->commHash, comm->commHash);
  printf("NCCL_COMM_WORLD is correctly assigned\n");
  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

TEST_F(CommWorldTest, DefaultCommWorld) {
  NCCL_COMM_WORLD = NULL;
  EXPECT_EQ(NCCL_COMM_WORLD, nullptr);

  ncclComm_t comm =
      createNcclComm(this->globalRank, this->numRanks, this->localRank);
  ASSERT_NE(comm, nullptr);

  printf("NCCL_FIRST_COMM_AS_WORLD: %d\n", NCCL_FIRST_COMM_AS_WORLD);
  EXPECT_EQ(NCCL_COMM_WORLD, nullptr);
  printf("NCCL_COMM_WORLD is nullptr as expected\n");

  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
