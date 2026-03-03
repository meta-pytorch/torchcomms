// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <stdlib.h>
#include "checks.h"
#include "comm.h"
#include "comms/ctran/utils/SkipDestroyUtil.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "nccl.h"

class CommAbortTest : public NcclxBaseTest {
 public:
  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE", "1", 0);
    setenv("NCCL_HEALTH_WATCHER_ENABLE", "True", 0);
    NcclxBaseTest::SetUp();
  }

  void TearDown() override {
    NcclxBaseTest::TearDown();
  };
};

using CommAbortDeathTest = CommAbortTest;

TEST_F(CommAbortTest, CommScope) {
  ncclComm_t comm = createNcclComm(globalRank, numRanks, localRank);

  ASSERT_NE(nullptr, comm);
  ASSERT_NE(nullptr, comm->ctranComm_->ctran_);

  ASSERT_EQ(ncclCommAbort(comm), ncclSuccess);
  EXPECT_FALSE(ctran::utils::getSkipDestroyCtran());
}

TEST_F(CommAbortTest, NoneScope) {
  ncclComm_t comm = createNcclComm(globalRank, numRanks, localRank);
  EnvRAII env(NCCL_COMM_ABORT_SCOPE, NCCL_COMM_ABORT_SCOPE::none);

  ASSERT_NE(nullptr, comm);
  ASSERT_NE(nullptr, comm->ctranComm_->ctran_);

  ASSERT_EQ(ncclCommAbort(comm), ncclSuccess);
  EXPECT_TRUE(ctran::utils::getSkipDestroyCtran());
}

// TODO: need a safe way to test JobScope that will exit(1) without hanging the
// test

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
