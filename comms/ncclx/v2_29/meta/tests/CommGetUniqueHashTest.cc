// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <unordered_map>

#include <folly/init/Init.h>
#include <gtest/gtest.h>

#include <nccl.h>
#include "comm.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"

class CommGetUniqueHashTest : public ::testing::Test {
 public:
  CommGetUniqueHashTest() = default;

  void SetUp() override {
    std::tie(this->localRank, this->globalRank, this->numRanks) = getMpiInfo();
  }

  void TearDown() override {}

  int localRank{0};
  int globalRank{0};
  int numRanks{0};
};

TEST_F(CommGetUniqueHashTest, DefaultComm) {
  auto res = ncclSuccess;

  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  uint64_t commHash = 0;
  res = ncclCommGetUniqueHash(comm, &commHash);
  ASSERT_EQ(res, ncclSuccess);

  EXPECT_EQ(commHash, comm->commHash);

  // check all ranks have the same commHash
  uint64_t commHashMin = 0, commHashMax = 0;
  MPI_Allreduce(
      &commHash, &commHashMin, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(
      &commHash, &commHashMax, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
  EXPECT_TRUE(commHashMin == commHashMax && commHashMax == commHash);
}

TEST_F(CommGetUniqueHashTest, ParentChildComm) {
  auto res = ncclSuccess;

  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};

  // Split into two groups, one with odd ranks and one with even ranks
  ncclComm_t childComm = NCCL_COMM_NULL;
  NCCLCHECK_TEST(ncclCommSplit(
      comm, this->globalRank % 2, this->globalRank, &childComm, nullptr));
  EXPECT_NE(childComm, (ncclComm_t)(NCCL_COMM_NULL));

  uint64_t commHash = 0, childCommHash = 0;
  res = ncclCommGetUniqueHash(comm, &commHash);
  ASSERT_EQ(res, ncclSuccess);

  res = ncclCommGetUniqueHash(childComm, &childCommHash);
  ASSERT_EQ(res, ncclSuccess);

  EXPECT_EQ(childCommHash, childComm->commHash);
  EXPECT_NE(childCommHash, commHash);

  NCCLCHECK_TEST(ncclCommDestroy(childComm));
}

TEST_F(CommGetUniqueHashTest, ParentChildCommSplitGroup) {
  auto res = ncclSuccess;

  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};

  ncclConfig_t inputConfig = NCCL_CONFIG_INITIALIZER;
  inputConfig.splitGroupSize = this->numRanks / 2;
  if (this->globalRank % 2 == 0) {
    /* for the group with even ranks, if the total number of ranks is
     * odd, increment the group size by 1 */
    inputConfig.splitGroupSize += this->numRanks % 2;
  }

  inputConfig.splitGroupRanks = new int[inputConfig.splitGroupSize];
  int idx = 0;
  for (int i = 0; i < this->numRanks; i++) {
    if (i % 2 == this->globalRank % 2) {
      inputConfig.splitGroupRanks[idx++] = i;
    }
  }

  // Split into two groups, one with odd ranks and one with even ranks
  ncclComm_t childComm = NCCL_COMM_NULL;
  NCCLCHECK_TEST(ncclCommSplit(
      comm, this->globalRank % 2, this->globalRank, &childComm, &inputConfig));
  EXPECT_NE(childComm, (ncclComm_t)(NCCL_COMM_NULL));

  delete[] inputConfig.splitGroupRanks;

  uint64_t commHash = 0, childCommHash = 0;
  res = ncclCommGetUniqueHash(comm, &commHash);
  ASSERT_EQ(res, ncclSuccess);

  res = ncclCommGetUniqueHash(childComm, &childCommHash);
  ASSERT_EQ(res, ncclSuccess);

  EXPECT_EQ(childCommHash, childComm->commHash);
  EXPECT_NE(childCommHash, commHash);

  NCCLCHECK_TEST(ncclCommDestroy(childComm));
}

TEST_F(CommGetUniqueHashTest, InvalidComm) {
  auto res = ncclSuccess;

  ncclComm_t comm = NCCL_COMM_NULL;
  uint64_t commHash = 0;
  res = ncclCommGetUniqueHash(comm, &commHash);
  ASSERT_EQ(res, ncclInvalidArgument);
}

TEST_F(CommGetUniqueHashTest, GetHashAfteCommDestroy) {
  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};
  // Split into two groups, one with odd ranks and one with even ranks
  ncclComm_t childComm = NCCL_COMM_NULL;
  NCCLCHECK_TEST(ncclCommSplit(
      comm, this->globalRank % 2, this->globalRank, &childComm, nullptr));
  EXPECT_NE(childComm, (ncclComm_t)(NCCL_COMM_NULL));

  auto res = ncclSuccess;
  uint64_t commHash = 0;
  res = ncclCommGetUniqueHash(childComm, &commHash);
  ASSERT_EQ(res, ncclSuccess);

  NCCLCHECK_TEST(ncclCommDestroy(childComm));

  res = ncclCommGetUniqueHash(childComm, &commHash);
  ASSERT_EQ(res, ncclInvalidArgument);
}

TEST_F(CommGetUniqueHashTest, DISABLED_TwoChildCommsSameColor) {
  auto res = ncclSuccess;

  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};

  // Make two child comms from commSplit with same color, compare commHash
  // between them
  ncclComm_t childComms[2] = {NCCL_COMM_NULL, NCCL_COMM_NULL};
  for (int i = 0; i < 2; i++) {
    NCCLCHECK_TEST(ncclCommSplit(
        comm, this->globalRank % 2, this->globalRank, &childComms[i], nullptr));
    EXPECT_NE(childComms[i], (ncclComm_t)(NCCL_COMM_NULL));
  }

  uint64_t childCommHashs[2] = {0, 0};
  for (int i = 0; i < 2; i++) {
    res = ncclCommGetUniqueHash(childComms[i], &childCommHashs[i]);
    ASSERT_EQ(res, ncclSuccess);

    EXPECT_EQ(childCommHashs[i], childComms[i]->commHash);
  }

  EXPECT_NE(childCommHashs[0], childCommHashs[1]);

  for (int i = 0; i < 2; i++) {
    NCCLCHECK_TEST(ncclCommDestroy(childComms[i]));
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
