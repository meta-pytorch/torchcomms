// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <memory>
#include <optional>

#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comm.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "meta/hints/CommHintConfig.h" // @manual
#include "meta/hints/GlobalHints.h" // @manual
#include "nccl.h"

class CommWithPatAvgTest : public ::testing::Test {
 public:
  CommWithPatAvgTest() = default;

  void SetUp() override {
    initEnv();
    std::tie(this->localRank, this->globalRank, this->numRanks) = getMpiInfo();
  }

  void TearDown() override {
    // Only reset our hint value, don't unregister all keys
    ncclx::resetGlobalHint(
        std::string(ncclx::HintKeys::kCommAlgoReduceScatter));
  }

  int localRank{0};
  int globalRank{0};
  int numRanks{0};
};

TEST_F(CommWithPatAvgTest, PatAvgDisabledByDefault) {
  EnvRAII cvarEnv(NCCL_REDUCESCATTER_PAT_AVG_ENABLE, false);
  NcclCommRAII comm{globalRank, numRanks, localRank};
  ASSERT_NE(comm.get(), nullptr);
  EXPECT_FALSE(comm->usePatAvg_);
}

TEST_F(CommWithPatAvgTest, PatAvgEnableByCvar) {
  EnvRAII cvarEnv(NCCL_REDUCESCATTER_PAT_AVG_ENABLE, true);
  NcclCommRAII comm{globalRank, numRanks, localRank};
  ASSERT_NE(comm.get(), nullptr);
  EXPECT_TRUE(comm->usePatAvg_);
}

TEST_F(CommWithPatAvgTest, PatAvgNotEnabledForOtherValues) {
  EnvRAII cvarEnv(NCCL_REDUCESCATTER_PAT_AVG_ENABLE, false);

  // Set hint to a different value
  ASSERT_EQ(
      ncclx::setGlobalHint(
          std::string(ncclx::HintKeys::kCommAlgoReduceScatter), "sum:ring"),
      ncclSuccess);

  NcclCommRAII comm{globalRank, numRanks, localRank};
  ASSERT_NE(comm.get(), nullptr);
  EXPECT_FALSE(comm->usePatAvg_);
}

namespace {
enum class TestCommCreateMode { kDefault, kSplit };
} // namespace

class CommWithPatAvgTestParam : public CommWithPatAvgTest,
                                public ::testing::WithParamInterface<
                                    std::tuple<TestCommCreateMode, bool>> {};

TEST_P(CommWithPatAvgTestParam, PatAvgEnableByHintWithModes) {
  const auto& [createMode, blockingInit] = GetParam();

  EnvRAII cvarEnv(NCCL_REDUCESCATTER_PAT_AVG_ENABLE, false);

  // Default disabled - use NcclCommRAII for base comm
  NcclCommRAII comm1{globalRank, numRanks, localRank};
  ASSERT_NE(comm1.get(), nullptr);
  EXPECT_FALSE(comm1->usePatAvg_);

  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.blocking = blockingInit ? 1 : 0;
  const auto commDescStr = fmt::format("{}-{}", kNcclUtCommDesc, "usePatAvg");
  config.commDesc = commDescStr.c_str();

  // Enable by hint
  ASSERT_EQ(
      ncclx::setGlobalHint(
          std::string(ncclx::HintKeys::kCommAlgoReduceScatter), "avg:patavg"),
      ncclSuccess);

  // Use appropriate RAII wrapper based on creation mode
  std::optional<NcclCommRAII> comm2Default;
  std::optional<NcclCommSplitRAII> comm2Split;
  ncclComm_t comm2;
  if (createMode == TestCommCreateMode::kDefault) {
    comm2Default.emplace(globalRank, numRanks, localRank, false, &config);
    comm2 = comm2Default->get();
  } else {
    comm2Split.emplace(comm1.get(), 1, this->globalRank, &config);
    comm2 = comm2Split->get();
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

  EXPECT_TRUE(comm2->usePatAvg_);

  ASSERT_TRUE(
      ncclx::resetGlobalHint(
          std::string(ncclx::HintKeys::kCommAlgoReduceScatter)));

  // Now disabled again
  {
    NcclCommRAII comm3{globalRank, numRanks, localRank};
    ASSERT_NE(comm3.get(), nullptr);
    EXPECT_FALSE(comm3->usePatAvg_);
  }
}

INSTANTIATE_TEST_SUITE_P(
    CommWithPatAvgTestInstance,
    CommWithPatAvgTestParam,
    ::testing::Combine(
        ::testing::Values(
            TestCommCreateMode::kDefault,
            TestCommCreateMode::kSplit),
        ::testing::Values(true, false)),
    [&](const testing::TestParamInfo<CommWithPatAvgTestParam::ParamType>&
            info) {
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
