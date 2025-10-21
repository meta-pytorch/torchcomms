// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include "comms/ctran/interfaces/ICtran.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/tests_common.cuh"
#include "comms/utils/cvars/nccl_cvars.h"
#include "nccl.h"

class SimpleCtranInitTest : public ::testing::Test {
 public:
  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE", "1", 1);
    setenv("NCCL_COLLTRACE", "trace", 1);
    setenv("NCCL_USE_MEM_CACHE", "1", 1);

    ncclCvarInit();
    NCCLCHECK_TEST(ncclCudaLibraryInit());
    CUDACHECK_TEST(cudaSetDevice(0));
  }
};

TEST_F(SimpleCtranInitTest, VerifyCtranCommStructures) {
  ncclComm_t comm = nullptr;
  ncclUniqueId commId;

  NCCLCHECK_TEST(ncclGetUniqueId(&commId));
  NCCLCHECK_TEST(ncclCommInitRank(&comm, 1, commId, 0));
  ASSERT_NE(nullptr, comm);

  EXPECT_EQ(comm->rank, 0);
  EXPECT_EQ(comm->nRanks, 1);
  EXPECT_EQ(comm->cudaDev, 0);

  ASSERT_NE(nullptr, comm->ctranComm_);
  ASSERT_NE(nullptr, comm->ctranComm_->statex_);
  ASSERT_NE(nullptr, comm->ctranComm_->bootstrap_);
  ASSERT_NE(nullptr, comm->ctranComm_->memCache_);
  ASSERT_NE(nullptr, comm->ctranComm_->collTrace_);
  ASSERT_NE(nullptr, comm->ctranComm_->ctran_);
  EXPECT_TRUE(ctranInitialized(comm->ctranComm_.get()));
  EXPECT_EQ(comm->commHash, comm->ctranComm_->statex_->commHash());

  NCCLCHECK_TEST(ncclCommDestroy(comm));
}
