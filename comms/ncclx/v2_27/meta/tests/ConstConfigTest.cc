// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cstring>

#include <folly/init/Init.h>
#include <gtest/gtest.h>

#include "comms/testinfra/DistTestBase.h"
#include "comms/testinfra/TestUtils.h"
#include "nccl.h"

using meta::comms::DistBaseTest;
using meta::comms::DistEnvironmentBase;

class ConstConfigTest : public ::testing::Test, protected DistBaseTest {
 public:
  ConstConfigTest() = default;

  void SetUp() override {
    distSetUp();
    CUDACHECK_TEST(cudaSetDevice(this->localRank));

    // Broadcast NCCL unique ID from rank 0 to all ranks
    if (globalRank == 0) {
      NCCLCHECK_TEST(ncclGetUniqueId(&commId));
    }
    oobBroadcast(&commId, 1);
  }

  void TearDown() override {
    distTearDown();
  }

  ncclUniqueId commId;
};

TEST_F(ConstConfigTest, InitRankConfigDefault) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;

  ncclConfig_t configCopy;
  std::memcpy(&configCopy, &config, sizeof(ncclConfig_t));

  ncclComm_t comm = nullptr;
  NCCLCHECK_TEST(
      ncclCommInitRankConfig(&comm, numRanks, commId, globalRank, &config));
  ASSERT_NE(nullptr, comm);

  EXPECT_EQ(0, std::memcmp(&config, &configCopy, sizeof(ncclConfig_t)));

  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

TEST_F(ConstConfigTest, InitRankConfigWithHints) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints({{"commDesc", "const_config_test"}});
  config.hints = &hints;

  ncclConfig_t configCopy;
  std::memcpy(&configCopy, &config, sizeof(ncclConfig_t));

  ncclComm_t comm = nullptr;
  NCCLCHECK_TEST(
      ncclCommInitRankConfig(&comm, numRanks, commId, globalRank, &config));
  ASSERT_NE(nullptr, comm);

  EXPECT_EQ(0, std::memcmp(&config, &configCopy, sizeof(ncclConfig_t)));

  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

TEST_F(ConstConfigTest, InitRankConfigWithBlocking) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.blocking = 1;

  ncclConfig_t configCopy;
  std::memcpy(&configCopy, &config, sizeof(ncclConfig_t));

  ncclComm_t comm = nullptr;
  NCCLCHECK_TEST(
      ncclCommInitRankConfig(&comm, numRanks, commId, globalRank, &config));
  ASSERT_NE(nullptr, comm);

  EXPECT_EQ(0, std::memcmp(&config, &configCopy, sizeof(ncclConfig_t)));

  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

TEST_F(ConstConfigTest, InitRankConfigWithSplitShare) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.splitShare = 1;

  ncclConfig_t configCopy;
  std::memcpy(&configCopy, &config, sizeof(ncclConfig_t));

  ncclComm_t comm = nullptr;
  NCCLCHECK_TEST(
      ncclCommInitRankConfig(&comm, numRanks, commId, globalRank, &config));
  ASSERT_NE(nullptr, comm);

  EXPECT_EQ(0, std::memcmp(&config, &configCopy, sizeof(ncclConfig_t)));

  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

TEST_F(ConstConfigTest, InitRankConfigWithMultipleFields) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.blocking = 1;
  config.minCTAs = 1;
  config.maxCTAs = 32;
  config.splitShare = 1;
  ncclx::Hints hints({{"commDesc", "const_config_multi_test"}});
  config.hints = &hints;

  ncclConfig_t configCopy;
  std::memcpy(&configCopy, &config, sizeof(ncclConfig_t));

  ncclComm_t comm = nullptr;
  NCCLCHECK_TEST(
      ncclCommInitRankConfig(&comm, numRanks, commId, globalRank, &config));
  ASSERT_NE(nullptr, comm);

  EXPECT_EQ(0, std::memcmp(&config, &configCopy, sizeof(ncclConfig_t)));

  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

TEST_F(ConstConfigTest, CommSplitDefault) {
  ncclComm_t rootComm = nullptr;
  NCCLCHECK_TEST(
      ncclCommInitRankConfig(&rootComm, numRanks, commId, globalRank, nullptr));
  ASSERT_NE(nullptr, rootComm);

  ncclConfig_t splitConfig = NCCL_CONFIG_INITIALIZER;

  ncclConfig_t splitConfigCopy;
  std::memcpy(&splitConfigCopy, &splitConfig, sizeof(ncclConfig_t));

  ncclComm_t childComm = nullptr;
  int color = globalRank % 2;
  NCCLCHECK_TEST(
      ncclCommSplit(rootComm, color, globalRank, &childComm, &splitConfig));
  ASSERT_NE(nullptr, childComm);

  EXPECT_EQ(
      0, std::memcmp(&splitConfig, &splitConfigCopy, sizeof(ncclConfig_t)));

  NCCLCHECK_TEST(ncclCommDestroy(childComm));
  NCCLCHECK_TEST(ncclCommDestroy(rootComm));
}

TEST_F(ConstConfigTest, CommSplitWithHints) {
  ncclComm_t rootComm = nullptr;
  NCCLCHECK_TEST(
      ncclCommInitRankConfig(&rootComm, numRanks, commId, globalRank, nullptr));
  ASSERT_NE(nullptr, rootComm);

  ncclConfig_t splitConfig = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints splitHints({{"commDesc", "split_const_config_test"}});
  splitConfig.hints = &splitHints;

  ncclConfig_t splitConfigCopy;
  std::memcpy(&splitConfigCopy, &splitConfig, sizeof(ncclConfig_t));

  ncclComm_t childComm = nullptr;
  int color = globalRank % 2;
  NCCLCHECK_TEST(
      ncclCommSplit(rootComm, color, globalRank, &childComm, &splitConfig));
  ASSERT_NE(nullptr, childComm);

  EXPECT_EQ(
      0, std::memcmp(&splitConfig, &splitConfigCopy, sizeof(ncclConfig_t)));

  NCCLCHECK_TEST(ncclCommDestroy(childComm));
  NCCLCHECK_TEST(ncclCommDestroy(rootComm));
}

TEST_F(ConstConfigTest, CommSplitWithSplitShare) {
  ncclComm_t rootComm = nullptr;
  NCCLCHECK_TEST(
      ncclCommInitRankConfig(&rootComm, numRanks, commId, globalRank, nullptr));
  ASSERT_NE(nullptr, rootComm);

  ncclConfig_t splitConfig = NCCL_CONFIG_INITIALIZER;
  splitConfig.splitShare = 1;

  ncclConfig_t splitConfigCopy;
  std::memcpy(&splitConfigCopy, &splitConfig, sizeof(ncclConfig_t));

  ncclComm_t childComm = nullptr;
  int color = globalRank % 2;
  NCCLCHECK_TEST(
      ncclCommSplit(rootComm, color, globalRank, &childComm, &splitConfig));
  ASSERT_NE(nullptr, childComm);

  EXPECT_EQ(
      0, std::memcmp(&splitConfig, &splitConfigCopy, sizeof(ncclConfig_t)));

  NCCLCHECK_TEST(ncclCommDestroy(childComm));
  NCCLCHECK_TEST(ncclCommDestroy(rootComm));
}

TEST_F(ConstConfigTest, CommSplitWithMultipleFields) {
  ncclComm_t rootComm = nullptr;
  NCCLCHECK_TEST(
      ncclCommInitRankConfig(&rootComm, numRanks, commId, globalRank, nullptr));
  ASSERT_NE(nullptr, rootComm);

  ncclConfig_t splitConfig = NCCL_CONFIG_INITIALIZER;
  splitConfig.blocking = 1;
  splitConfig.splitShare = 1;
  splitConfig.minCTAs = 1;
  splitConfig.maxCTAs = 32;
  ncclx::Hints splitHints({{"commDesc", "split_const_config_multi_test"}});
  splitConfig.hints = &splitHints;

  ncclConfig_t splitConfigCopy;
  std::memcpy(&splitConfigCopy, &splitConfig, sizeof(ncclConfig_t));

  ncclComm_t childComm = nullptr;
  int color = globalRank % 2;
  NCCLCHECK_TEST(
      ncclCommSplit(rootComm, color, globalRank, &childComm, &splitConfig));
  ASSERT_NE(nullptr, childComm);

  EXPECT_EQ(
      0, std::memcmp(&splitConfig, &splitConfigCopy, sizeof(ncclConfig_t)));

  NCCLCHECK_TEST(ncclCommDestroy(childComm));
  NCCLCHECK_TEST(ncclCommDestroy(rootComm));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
