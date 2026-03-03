// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "comm.h"
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/utils/Exception.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "meta/wrapper/MetaFactory.h"
#include "nccl.h"

class CommAsyncErrorTest : public NcclxBaseTest {
 public:
  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE", "1", 0);
    NcclxBaseTest::SetUp();
  }

  void TearDown() override {
    NcclxBaseTest::TearDown();
  }
};

TEST_F(CommAsyncErrorTest, NcclErrorOnly) {
  ncclComm_t comm = createNcclComm(globalRank, numRanks, localRank);

  ASSERT_NE(nullptr, comm);

  const auto ncclErr = ncclInvalidUsage;
  ncclCommSetAsyncError(comm, ncclErr);

  // Except NCCL async error is propagated
  auto asyncError = ncclSuccess;
  ASSERT_EQ(ncclCommGetAsyncError(comm, &asyncError), ncclSuccess);
  ASSERT_EQ(asyncError, ncclErr);

  ASSERT_EQ(ncclCommAbort(comm), ncclSuccess);
}

TEST_F(CommAsyncErrorTest, CtranErrorOnly) {
  ncclComm_t comm = createNcclComm(globalRank, numRanks, localRank);

  ASSERT_NE(nullptr, comm);
  ASSERT_TRUE(ctranInitialized(comm->ctranComm_.get()));

  const auto ctranErr = commInvalidArgument;
  auto e = ctran::utils::Exception("test", ctranErr);
  comm->ctranComm_->setAsyncException(e);

  // Except Ctran async error is propagated
  auto asyncError = ncclSuccess;
  ASSERT_EQ(ncclCommGetAsyncError(comm, &asyncError), ncclSuccess);
  ASSERT_EQ(asyncError, metaCommToNccl(ctranErr));

  ASSERT_EQ(ncclCommAbort(comm), ncclSuccess);
}

TEST_F(CommAsyncErrorTest, NcclCtranErrorTogether) {
  ncclComm_t comm = createNcclComm(globalRank, numRanks, localRank);

  ASSERT_NE(nullptr, comm);
  ASSERT_TRUE(ctranInitialized(comm->ctranComm_.get()));

  const auto ncclErr = ncclInvalidUsage;
  ncclCommSetAsyncError(comm, ncclErr);

  const auto ctranErr = commInvalidArgument;
  auto e = ctran::utils::Exception("test", ctranErr);
  comm->ctranComm_->setAsyncException(e);

  // Except NCCL error is propagated, and Ctran async error doesn't overwrite
  auto asyncError = ncclSuccess;
  ASSERT_EQ(ncclCommGetAsyncError(comm, &asyncError), ncclSuccess);
  ASSERT_EQ(asyncError, ncclErr);

  ASSERT_EQ(ncclCommAbort(comm), ncclSuccess);
}

TEST_F(CommAsyncErrorTest, NcclInProgressCtranError) {
  ncclComm_t comm = createNcclComm(globalRank, numRanks, localRank);

  ASSERT_NE(nullptr, comm);
  ASSERT_TRUE(ctranInitialized(comm->ctranComm_.get()));

  const auto ncclErr = ncclInProgress;
  ncclCommSetAsyncError(comm, ncclErr);

  const auto ctranErr = commInvalidArgument;
  auto e = ctran::utils::Exception("test", ctranErr);
  comm->ctranComm_->setAsyncException(e);

  // Except Ctran async error overwrites
  auto asyncError = ncclSuccess;
  ASSERT_EQ(ncclCommGetAsyncError(comm, &asyncError), ncclSuccess);
  ASSERT_EQ(asyncError, metaCommToNccl(ctranErr));

  ASSERT_EQ(ncclCommAbort(comm), ncclSuccess);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
