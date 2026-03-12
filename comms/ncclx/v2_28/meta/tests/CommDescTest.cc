// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstdlib>

#include <gtest/gtest.h>

#include <folly/init/Init.h>

#include "checks.h"
#include "comm.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "nccl.h"

class commDescTest : public ::testing::Test {
 public:
  commDescTest() = default;

 protected:
  void SetUp() override {
    std::tie(this->localRank, this->globalRank, this->numRanks) = getMpiInfo();
  }

  void TearDown() override {}

  int localRank{0};
  int globalRank{0};
  int numRanks{0};
};

TEST_F(commDescTest, getUndefinedCommDesc) {
  NcclCommRAII comm(this->globalRank, this->numRanks, this->localRank);
  ASSERT_NE(nullptr, static_cast<ncclComm_t>(comm));

  EXPECT_STREQ(comm->config.commDesc, "nccl_ut");
}

TEST_F(commDescTest, getDefinedCommDesc) {
  ncclUniqueId ncclId;
  if (globalRank == 0) {
    NCCLCHECK_TEST(ncclGetUniqueId(&ncclId));
  }
  MPICHECK_TEST(
      MPI_Bcast((void*)&ncclId, sizeof(ncclId), MPI_BYTE, 0, MPI_COMM_WORLD));
  CUDACHECK_TEST(cudaSetDevice(this->localRank));

  ncclComm_t comm;
  ncclConfig_t inputConfig = NCCL_CONFIG_INITIALIZER;
  inputConfig.commDesc = "test_description";

  NCCLCHECK_TEST(ncclCommInitRankConfig(
      &comm, numRanks, ncclId, globalRank, &inputConfig));
  ASSERT_NE(nullptr, comm);

  EXPECT_STRNE(comm->config.commDesc, "undefined");
  EXPECT_STREQ(comm->config.commDesc, inputConfig.commDesc);

  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

TEST_F(commDescTest, InvalidPointerAccess) {
  ncclUniqueId ncclId;
  if (globalRank == 0) {
    NCCLCHECK_TEST(ncclGetUniqueId(&ncclId));
  }
  MPICHECK_TEST(
      MPI_Bcast((void*)&ncclId, sizeof(ncclId), MPI_BYTE, 0, MPI_COMM_WORLD));
  CUDACHECK_TEST(cudaSetDevice(this->localRank));

  ncclComm_t comm;
  ncclConfig_t inputConfig = NCCL_CONFIG_INITIALIZER;
  const char* commDescConst = "test_description";
  char* commDesc = strdup(commDescConst);
  inputConfig.commDesc = commDesc;

  NCCLCHECK_TEST(ncclCommInitRankConfig(
      &comm, numRanks, ncclId, globalRank, &inputConfig));
  ASSERT_NE(nullptr, comm);

  free(commDesc);

  EXPECT_STRNE(comm->config.commDesc, "undefined");
  EXPECT_STREQ(comm->config.commDesc, commDescConst);

  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
