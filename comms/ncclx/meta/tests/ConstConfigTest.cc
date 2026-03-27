// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cstring>

#include <folly/init/Init.h>
#include <gtest/gtest.h>

#include "comms/ncclx/meta/tests/NcclCommUtils.h"
#include "comms/ncclx/meta/tests/NcclxBaseTest.h"
#include "nccl.h"

class ConstConfigTest : public NcclxBaseTestFixture {};

TEST_F(ConstConfigTest, InitRankConfigDefault) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;

  ncclConfig_t configCopy;
  std::memcpy(&configCopy, &config, sizeof(ncclConfig_t));

  ncclx::test::NcclCommRAII comm(
      globalRank, numRanks, localRank, bootstrap_.get(), false, &config);
  ASSERT_NE(nullptr, comm.get());

  EXPECT_EQ(0, std::memcmp(&config, &configCopy, sizeof(ncclConfig_t)));
}

TEST_F(ConstConfigTest, InitRankConfigWithHints) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints({{"commDesc", "const_config_test"}});
  config.hints = &hints;

  ncclConfig_t configCopy;
  std::memcpy(&configCopy, &config, sizeof(ncclConfig_t));

  ncclx::test::NcclCommRAII comm(
      globalRank, numRanks, localRank, bootstrap_.get(), false, &config);
  ASSERT_NE(nullptr, comm.get());

  EXPECT_EQ(0, std::memcmp(&config, &configCopy, sizeof(ncclConfig_t)));
}

TEST_F(ConstConfigTest, InitRankConfigWithBlocking) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.blocking = 1;

  ncclConfig_t configCopy;
  std::memcpy(&configCopy, &config, sizeof(ncclConfig_t));

  ncclx::test::NcclCommRAII comm(
      globalRank, numRanks, localRank, bootstrap_.get(), false, &config);
  ASSERT_NE(nullptr, comm.get());

  EXPECT_EQ(0, std::memcmp(&config, &configCopy, sizeof(ncclConfig_t)));
}

TEST_F(ConstConfigTest, InitRankConfigWithSplitShare) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.splitShare = 1;

  ncclConfig_t configCopy;
  std::memcpy(&configCopy, &config, sizeof(ncclConfig_t));

  ncclx::test::NcclCommRAII comm(
      globalRank, numRanks, localRank, bootstrap_.get(), false, &config);
  ASSERT_NE(nullptr, comm.get());

  EXPECT_EQ(0, std::memcmp(&config, &configCopy, sizeof(ncclConfig_t)));
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

  ncclx::test::NcclCommRAII comm(
      globalRank, numRanks, localRank, bootstrap_.get(), false, &config);
  ASSERT_NE(nullptr, comm.get());

  EXPECT_EQ(0, std::memcmp(&config, &configCopy, sizeof(ncclConfig_t)));
}

TEST_F(ConstConfigTest, CommSplitDefault) {
  ncclx::test::NcclCommRAII rootComm(
      globalRank, numRanks, localRank, bootstrap_.get());
  ASSERT_NE(nullptr, rootComm.get());

  ncclConfig_t splitConfig = NCCL_CONFIG_INITIALIZER;

  ncclConfig_t splitConfigCopy;
  std::memcpy(&splitConfigCopy, &splitConfig, sizeof(ncclConfig_t));

  const int color = globalRank % 2;
  ncclx::test::NcclCommSplitRAII childComm(
      rootComm, color, globalRank, &splitConfig);
  ASSERT_NE(nullptr, childComm.get());

  EXPECT_EQ(
      0, std::memcmp(&splitConfig, &splitConfigCopy, sizeof(ncclConfig_t)));
}

TEST_F(ConstConfigTest, CommSplitWithHints) {
  ncclx::test::NcclCommRAII rootComm(
      globalRank, numRanks, localRank, bootstrap_.get());
  ASSERT_NE(nullptr, rootComm.get());

  ncclConfig_t splitConfig = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints splitHints({{"commDesc", "split_const_config_test"}});
  splitConfig.hints = &splitHints;

  ncclConfig_t splitConfigCopy;
  std::memcpy(&splitConfigCopy, &splitConfig, sizeof(ncclConfig_t));

  const int color = globalRank % 2;
  ncclx::test::NcclCommSplitRAII childComm(
      rootComm, color, globalRank, &splitConfig);
  ASSERT_NE(nullptr, childComm.get());

  EXPECT_EQ(
      0, std::memcmp(&splitConfig, &splitConfigCopy, sizeof(ncclConfig_t)));
}

TEST_F(ConstConfigTest, CommSplitWithSplitShare) {
  ncclx::test::NcclCommRAII rootComm(
      globalRank, numRanks, localRank, bootstrap_.get());
  ASSERT_NE(nullptr, rootComm.get());

  ncclConfig_t splitConfig = NCCL_CONFIG_INITIALIZER;
  splitConfig.splitShare = 1;

  ncclConfig_t splitConfigCopy;
  std::memcpy(&splitConfigCopy, &splitConfig, sizeof(ncclConfig_t));

  const int color = globalRank % 2;
  ncclx::test::NcclCommSplitRAII childComm(
      rootComm, color, globalRank, &splitConfig);
  ASSERT_NE(nullptr, childComm.get());

  EXPECT_EQ(
      0, std::memcmp(&splitConfig, &splitConfigCopy, sizeof(ncclConfig_t)));
}

TEST_F(ConstConfigTest, CommSplitWithMultipleFields) {
  ncclx::test::NcclCommRAII rootComm(
      globalRank, numRanks, localRank, bootstrap_.get());
  ASSERT_NE(nullptr, rootComm.get());

  ncclConfig_t splitConfig = NCCL_CONFIG_INITIALIZER;
  splitConfig.blocking = 1;
  splitConfig.splitShare = 1;
  splitConfig.minCTAs = 1;
  splitConfig.maxCTAs = 32;
  ncclx::Hints splitHints({{"commDesc", "split_const_config_multi_test"}});
  splitConfig.hints = &splitHints;

  ncclConfig_t splitConfigCopy;
  std::memcpy(&splitConfigCopy, &splitConfig, sizeof(ncclConfig_t));

  const int color = globalRank % 2;
  ncclx::test::NcclCommSplitRAII childComm(
      rootComm, color, globalRank, &splitConfig);
  ASSERT_NE(nullptr, childComm.get());

  EXPECT_EQ(
      0, std::memcmp(&splitConfig, &splitConfigCopy, sizeof(ncclConfig_t)));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
