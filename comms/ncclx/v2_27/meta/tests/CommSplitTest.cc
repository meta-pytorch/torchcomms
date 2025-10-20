// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstdlib>
#include <memory>

#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "checks.h"
#include "comm.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "nccl.h"

class CommSplitTest : public ::testing::Test {
 public:
  CommSplitTest() = default;

  void SetUp() override {
    std::tie(this->localRank, this->globalRank, this->numRanks) = getMpiInfo();
    this->comm =
        createNcclComm(this->globalRank, this->numRanks, this->localRank);
    CUDACHECK_TEST(cudaStreamCreate(&stream));

    // Prepare data for sanity check after commSplit
    CUDACHECK_TEST(cudaMalloc(&this->dataBuf, sizeof(int) * this->dataCount));
  }

  void initData(int myRank) {
    std::vector<int> initVals(this->dataCount);
    for (int i = 0; i < this->dataCount; i++) {
      initVals[i] = i * myRank;
    }
    CUDACHECK_TEST(cudaMemcpy(
        this->dataBuf,
        initVals.data(),
        sizeof(int) * this->dataCount,
        cudaMemcpyHostToDevice));
  }

  int checkAllReduceResult(int numRanks) {
    std::vector<int> observedVals(this->dataCount, -1);
    CUDACHECK_TEST(cudaMemcpy(
        observedVals.data(),
        this->dataBuf,
        this->dataCount * sizeof(int),
        cudaMemcpyDefault));

    const int sumRanks = numRanks * (numRanks - 1) / 2;
    int errs = 0;
    // Use manual print rather than EXPECT_THAT to print failing location
    for (auto i = 0; i < this->dataCount; i++) {
      int expVal = i * sumRanks;
      if (observedVals[i] != expVal) {
        if (errs < 10) {
          printf(
              "[%d] observedVals[%d] = %d, expectedVal = %d\n",
              this->globalRank,
              i,
              observedVals[i],
              expVal);
        }
        errs++;
      }
    }
    return errs;
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaFree(this->dataBuf));
    CUDACHECK_TEST(cudaStreamDestroy(this->stream));
    NCCLCHECK_TEST(ncclCommDestroy(this->comm));
  }

  int localRank{0};
  int globalRank{0};
  int numRanks{0};
  int* dataBuf{nullptr};
  const int dataCount{65536};
  ncclComm_t comm;
  cudaStream_t stream;
};

TEST_F(CommSplitTest, NoColor) {
  ncclComm_t newcomm = NCCL_COMM_NULL;
  auto res = ncclSuccess;

  // Only odd ranks create subcomm
  if (this->globalRank % 2 == 0) {
    res = ncclCommSplit(
        this->comm, NCCL_SPLIT_NOCOLOR, this->globalRank, &newcomm, nullptr);
    ASSERT_EQ(res, ncclSuccess);
    EXPECT_EQ(newcomm, (ncclComm_t)(NCCL_COMM_NULL));
  } else {
    res = ncclCommSplit(this->comm, 1, this->globalRank, &newcomm, nullptr);
    ASSERT_EQ(res, ncclSuccess);
    ncclUniqueId id;
    memset(id.internal, 0, NCCL_UNIQUE_ID_BYTES);
    uint64_t hash = getHash(id.internal, NCCL_UNIQUE_ID_BYTES);
    ASSERT_NE(newcomm->commHash, hash);

    int numRanks, myRank;

    EXPECT_NE(newcomm, (ncclComm_t)(NCCL_COMM_NULL));
    NCCLCHECK_TEST(ncclCommCount(newcomm, &numRanks));
    EXPECT_EQ(numRanks, this->numRanks / 2);

    NCCLCHECK_TEST(ncclCommUserRank(newcomm, &myRank));
    EXPECT_EQ(myRank, this->globalRank / 2);

    this->initData(myRank);
    ncclAllReduce(
        this->dataBuf,
        this->dataBuf,
        this->dataCount,
        ncclInt,
        ncclSum,
        newcomm,
        this->stream);
    CUDACHECK_TEST(cudaStreamSynchronize(this->stream));

    int errs = this->checkAllReduceResult(numRanks);
    EXPECT_EQ(errs, 0);
  }

  res = ncclCommDestroy(newcomm);
  ASSERT_EQ(res, ncclSuccess);
}

TEST_F(CommSplitTest, OddEven) {
  ncclComm_t newcomm = NCCL_COMM_NULL;
  auto res = ncclSuccess;

  // Split into two groups, one with odd ranks and one with even ranks
  res = ncclCommSplit(
      this->comm, this->globalRank % 2, this->globalRank, &newcomm, nullptr);
  ASSERT_EQ(res, ncclSuccess);
  EXPECT_NE(newcomm, (ncclComm_t)(NCCL_COMM_NULL));

  int numRanks, myRank;
  NCCLCHECK_TEST(ncclCommCount(newcomm, &numRanks));
  // even group with odd num of global ranks contains one more rank (e.g., 0, 2,
  // 4 from 5 global ranks)
  const int expNumRanks = this->numRanks % 2 && this->globalRank % 2 == 0
      ? this->numRanks / 2 + 1
      : this->numRanks / 2;
  EXPECT_EQ(numRanks, expNumRanks);

  NCCLCHECK_TEST(ncclCommUserRank(newcomm, &myRank));
  EXPECT_EQ(myRank, this->globalRank / 2);

  this->initData(myRank);
  ncclAllReduce(
      this->dataBuf,
      this->dataBuf,
      this->dataCount,
      ncclInt,
      ncclSum,
      newcomm,
      this->stream);
  CUDACHECK_TEST(cudaStreamSynchronize(this->stream));

  int errs = this->checkAllReduceResult(numRanks);
  EXPECT_EQ(errs, 0);

  res = ncclCommDestroy(newcomm);
  ASSERT_EQ(res, ncclSuccess);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
