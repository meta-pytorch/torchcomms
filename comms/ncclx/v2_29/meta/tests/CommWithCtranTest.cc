// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <memory>

#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comm.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "meta/hints/GlobalHints.h" // @manual
#include "nccl.h"

class CommWithCtranTest : public ::testing::Test {
 public:
  CommWithCtranTest() = default;

  void SetUp() override {
    // Init NCCL env so that creating communicator in each test case will not
    // initialize CVAR again, and we can override.
    initEnv();
    std::tie(this->localRank, this->globalRank, this->numRanks) = getMpiInfo();
  }

  void TearDown() override {}

  int localRank{0};
  int globalRank{0};
  int numRanks{0};
};

TEST_F(CommWithCtranTest, CtranEnable) {
  EnvRAII env(NCCL_CTRAN_ENABLE, true);
  ncclComm_t comm = createNcclComm(globalRank, numRanks, localRank);
  ASSERT_NE(comm, nullptr);
  ASSERT_NE(comm->ctranComm_.get(), nullptr);
  ASSERT_TRUE(ctranInitialized(comm->ctranComm_.get()));
  ASSERT_EQ(ncclCommDestroy(comm), ncclSuccess);
}

TEST_F(CommWithCtranTest, CtranDisable) {
  EnvRAII env(NCCL_CTRAN_ENABLE, false);
  ncclComm_t comm = createNcclComm(globalRank, numRanks, localRank);
  ASSERT_NE(comm, nullptr);

  // FIXME: currently ctranComm is used also for other modules, we should remove
  // the dependency, and ensure it is nullptr when ctran is disabled
  // ASSERT_EQ(comm->ctranComm_.get(), nullptr);

  ASSERT_FALSE(ctranInitialized(comm->ctranComm_.get()));
  ASSERT_EQ(ncclCommDestroy(comm), ncclSuccess);
}

namespace {
enum class TestCommCreateMode { kDefault, kSplit };
}
class CommWithCtranTestParam : public CommWithCtranTest,
                               public ::testing::WithParamInterface<
                                   std::tuple<TestCommCreateMode, bool>> {};

TEST_P(CommWithCtranTestParam, CtranEnableByHint) {
  const auto& [createMode, blockingInit] = GetParam();

  EnvRAII env(NCCL_CTRAN_ENABLE, false);
  // Default disabled
  ncclComm_t comm1 = createNcclComm(globalRank, numRanks, localRank);
  ASSERT_NE(comm1, nullptr);
  ASSERT_FALSE(ctranInitialized(comm1->ctranComm_.get()));

  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.blocking = blockingInit ? 1 : 0;
  const auto commDescStr = fmt::format("{}-{}", kNcclUtCommDesc, "useCtran");
  config.commDesc = commDescStr.c_str();

  // Enable by hint
  ASSERT_EQ(
      ncclx::setGlobalHint(std::string(ncclx::HintKeys::kCommUseCtran), "1"),
      ncclSuccess);
  ncclComm_t comm2;
  if (createMode == TestCommCreateMode::kDefault) {
    comm2 = createNcclComm(globalRank, numRanks, localRank, false, &config);
  } else {
    ASSERT_EQ(
        ncclCommSplit(comm1, 1, this->globalRank, &comm2, &config),
        ncclSuccess);
  }
  ASSERT_NE(comm2, nullptr);

  // If nonblocking init, wait till async init is done
  if (!blockingInit) {
    auto commStatus = ncclInProgress;
    do {
      ASSERT_EQ(ncclCommGetAsyncError(comm2, &commStatus), ncclSuccess);

      if (commStatus == ncclInProgress) {
        sched_yield();
      }
    } while (commStatus == ncclInProgress);
  }

  ASSERT_TRUE(ctranInitialized(comm2->ctranComm_.get()));
  ASSERT_TRUE(
      ncclx::resetGlobalHint(std::string(ncclx::HintKeys::kCommUseCtran)));

  // Now it should be disabled again after hint reset
  ncclComm_t comm3 = createNcclComm(globalRank, numRanks, localRank);
  ASSERT_NE(comm3, nullptr);
  ASSERT_FALSE(ctranInitialized(comm3->ctranComm_.get()));

  ASSERT_EQ(ncclCommDestroy(comm3), ncclSuccess);
  ASSERT_EQ(ncclCommDestroy(comm2), ncclSuccess);
  ASSERT_EQ(ncclCommDestroy(comm1), ncclSuccess);
}

INSTANTIATE_TEST_SUITE_P(
    CommWithCtranTestInstance,
    CommWithCtranTestParam,
    ::testing::Combine(
        ::testing::Values(
            TestCommCreateMode::kDefault,
            TestCommCreateMode::kSplit),
        ::testing::Values(true, false)),
    [&](const testing::TestParamInfo<CommWithCtranTestParam::ParamType>& info) {
      return fmt::format(
          "{}_{}",
          std::get<0>(info.param) == TestCommCreateMode::kDefault ? "default"
                                                                  : "split",
          std::get<1>(info.param) ? "blockingInit" : "nonblockingInit");
    });

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
