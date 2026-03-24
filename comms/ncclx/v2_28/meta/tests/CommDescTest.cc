// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstdlib>

#include <gtest/gtest.h>

#include <folly/init/Init.h>

#include "checks.h"
#include "comm.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "meta/NcclxConfig.h"
#include "nccl.h"

#include <gmock/gmock.h>

#include "comms/ncclx/meta/tests/NcclCommUtils.h"

class commDescTest : public NcclxBaseTest {
 public:
  commDescTest() = default;

 protected:
  void SetUp() override {
    NcclxBaseTest::SetUp();
  }

  void TearDown() override {
    NcclxBaseTest::TearDown();
  }
};

TEST_F(commDescTest, getUndefinedCommDesc) {
  ncclx::test::NcclCommRAII comm(
      globalRank, numRanks, localRank, bootstrap_.get());
  ASSERT_NE(nullptr, static_cast<ncclComm_t>(comm));

  EXPECT_THAT(
      NCCLX_CONFIG_FIELD(comm->config, commDesc),
      testing::StartsWith("nccl_ut_"));
}

TEST_F(commDescTest, getDefinedCommDesc) {
  ncclUniqueId ncclId;
  if (globalRank == 0) {
    NCCLCHECK_TEST(ncclGetUniqueId(&ncclId));
  }
  oobBroadcast(&ncclId, 1, 0);
  CUDACHECK_TEST(cudaSetDevice(this->localRank));

  ncclComm_t comm;
  ncclConfig_t inputConfig = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints({{"commDesc", "test_description"}});
  inputConfig.hints = &hints;

  NCCLCHECK_TEST(ncclCommInitRankConfig(
      &comm, numRanks, ncclId, globalRank, &inputConfig));
  ASSERT_NE(nullptr, comm);

  EXPECT_NE(NCCLX_CONFIG_FIELD(comm->config, commDesc), "undefined");
  EXPECT_EQ(NCCLX_CONFIG_FIELD(comm->config, commDesc), "test_description");

  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

TEST_F(commDescTest, InvalidPointerAccess) {
  ncclUniqueId ncclId;
  if (globalRank == 0) {
    NCCLCHECK_TEST(ncclGetUniqueId(&ncclId));
  }
  oobBroadcast(&ncclId, 1, 0);
  CUDACHECK_TEST(cudaSetDevice(this->localRank));

  ncclComm_t comm;
  ncclConfig_t inputConfig = NCCL_CONFIG_INITIALIZER;
  const char* commDescConst = "test_description";
  char* commDesc = strdup(commDescConst);
  ncclx::Hints hints({{"commDesc", commDesc}});
  inputConfig.hints = &hints;

  NCCLCHECK_TEST(ncclCommInitRankConfig(
      &comm, numRanks, ncclId, globalRank, &inputConfig));
  ASSERT_NE(nullptr, comm);

  free(commDesc);

  EXPECT_NE(NCCLX_CONFIG_FIELD(comm->config, commDesc), "undefined");
  EXPECT_EQ(NCCLX_CONFIG_FIELD(comm->config, commDesc), commDescConst);

  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
