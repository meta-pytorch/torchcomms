// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <stdlib.h>

#include <folly/init/Init.h>

#include <nccl.h>
#include "comms/ctran/CtranEx.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "meta/wrapper/CtranExComm.h"

using namespace ctran;

class CtranExCommFailureTest : public CtranExBaseTest {
 public:
  void SetUp() override {
    CtranExBaseTest::SetUp();
  }

 protected:
  const std::string defaultDesc_{"CtranExCommFailureTest"};
  std::array<int, 8192> buf{};
  void* regHdl_{nullptr};
  void* peerRemoteBuf_{nullptr};
  uint32_t peerRemoteKey_{0};
  void injectFailure(
      const int rank,
      const std::string& ibverb,
      const int seq,
      FailureType type = FailureType::API_ERROR) {
    setFailureInjection(ibverb.c_str(), seq, rank, type);
  }
};

enum class TestWaitType {
  WAIT,
  TEST,
};

class CtranExCommFailureParamFixture
    : public CtranExCommFailureTest,
      public ::testing::WithParamInterface<TestWaitType> {};

TEST_P(CtranExCommFailureParamFixture, BcastErr) {
  const auto& waitType = GetParam();
  auto ctranExComm = std::make_unique<CtranExComm>(ncclComm_, defaultDesc_);

  ASSERT_NE(ctranExComm, nullptr);
  EXPECT_TRUE(ctranExComm->isInitialized());
  EXPECT_TRUE(ctranExComm->supportBroadcast());

  void* bufHdl = nullptr;
  ASSERT_EQ(
      ctranExComm->regMem(
          buf.data(),
          buf.size() * sizeof(int),
          &bufHdl,
          true /* forceRegister */),
      commSuccess);
  ASSERT_NE(bufHdl, nullptr);

  const auto myRank = ncclComm_->rank;
  const auto nRanks = ncclComm_->nRanks;
  // Rank 0 assigns the input data
  if (myRank == 0) {
    for (auto i = 0; i < buf.size(); i++) {
      buf[i] = i + 1;
    }
  }

  // inject the 1st ibv_post_send failure on root rank
  // and inject timeout on all other ranks.
  // Set largest rank as root so that other ranks can connect to its bootstrap
  // listen thread (always smaller rank connects to larger rank in CtranIB) even
  // the root rank has already hit failure and its GPE thread terminated
  const int root = nRanks - 1;
  if (myRank == root) {
    injectFailure(root, "ibv_post_send", 0);
  } else {
    injectFailure(myRank, "ibv_poll_cq", 0, FailureType::WC_TIMEOUT);
  }

  CtranExRequest* reqPtr = nullptr;
  ASSERT_EQ(
      ctranExComm->broadcast(
          buf.data(), buf.data(), buf.size(), ncclInt, root, &reqPtr),
      commSuccess);

  ASSERT_NE(reqPtr, nullptr);
  auto req = std::unique_ptr<CtranExRequest>(reqPtr);

  // Wait completion of the bcast
  auto res = ncclSuccess;
  if (waitType == TestWaitType::WAIT) {
    res = metaCommToNccl(req->wait());
  } else {
    bool complete = false;
    while (!complete && (res == ncclSuccess || res == ncclInProgress)) {
      res = metaCommToNccl(req->test(complete));
    }
    // Check the bcast is not completed
    EXPECT_FALSE(complete);
  }

  // Check error is reported
  EXPECT_NE(res, ncclSuccess);
  auto errStr = ctranExComm->getAsyncErrorString();
  EXPECT_NE(errStr, "");

  std::cout << " === Test reads Error string: " << errStr << std::endl;

  // Ensure all ranks are done before deregistering buffer
  barrier();
  ASSERT_EQ(ctranExComm->deregMem(bufHdl), ncclSuccess);
}

INSTANTIATE_TEST_SUITE_P(
    CtranExCommFailureTest,
    CtranExCommFailureParamFixture,
    ::testing::Values(TestWaitType::TEST, TestWaitType::WAIT),
    [&](const ::testing::TestParamInfo<
        CtranExCommFailureParamFixture::ParamType>& info) {
      const std::string waitType =
          info.param == TestWaitType::TEST ? "Test" : "Wait";
      return waitType;
    });

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
