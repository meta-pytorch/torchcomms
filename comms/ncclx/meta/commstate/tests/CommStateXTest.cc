// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <string>

#include <fmt/format.h>
#include <folly/init/Init.h>
#include <gtest/gtest.h>

#include "comm.h"
#include "comms/ctran/tests/VerifyCommStateXUtil.h"
#include "comms/ncclx/meta/tests/NcclCommUtils.h"
#include "comms/ncclx/meta/tests/NcclxBaseTest.h"
#include "nccl.h"

using ctran::testing::VerifyCommStateXHelper;
using RankIdentity = VerifyCommStateXHelper::RankIdentity;

class CommStateXDistTest : public NcclxBaseTestFixture {
 public:
  VerifyCommStateXHelper verifyHelper_;

  void SetUp() override {
    NcclxBaseTestFixture::SetUp();
  }

  void TearDown() override {
    NcclxBaseTestFixture::TearDown();
  }

  void gatherRankIdentities(std::vector<RankIdentity>& allRankIds) {
    allRankIds.resize(numRanks);
    allRankIds[globalRank] = RankIdentity::local();
    oobAllGather(allRankIds, sizeof(RankIdentity));
  }
};

TEST_F(CommStateXDistTest, CreateFromNcclComm) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints{{"useCtran", "1"}};
  config.hints = &hints;
  ncclx::test::NcclCommRAII comm{
      globalRank, numRanks, localRank, bootstrap_.get(), false, &config};
  ASSERT_NE(comm.get(), nullptr);
  ASSERT_TRUE(ctranInitialized(comm->ctranComm_.get()));

  const auto* statex = comm->ctranComm_->statex_.get();
  ASSERT_NE(statex, nullptr);

  EXPECT_EQ(statex->rank(), globalRank);
  EXPECT_EQ(statex->nRanks(), numRanks);
  EXPECT_EQ(statex->nLocalRanks(), numRanks);
  EXPECT_EQ(statex->nNodes(), 1);

  std::vector<RankIdentity> allRankIds;
  gatherRankIdentities(allRankIds);
  verifyHelper_.verifyAllHosts(statex, allRankIds);
  verifyHelper_.verifyAllGPids(statex, allRankIds);
}

TEST_F(CommStateXDistTest, CreateNoLocalFromNcclComm) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints{{"useCtran", "1"}, {"noLocal", "1"}};
  config.hints = &hints;
  ncclx::test::NcclCommRAII comm{
      globalRank, numRanks, localRank, bootstrap_.get(), false, &config};
  ASSERT_NE(comm.get(), nullptr);
  ASSERT_TRUE(ctranInitialized(comm->ctranComm_.get()));

  const auto* statex = comm->ctranComm_->statex_.get();
  ASSERT_NE(statex, nullptr);

  EXPECT_EQ(statex->nLocalRanks(), 1);
  EXPECT_EQ(statex->nNodes(), numRanks);
  for (int r = 0; r < numRanks; r++) {
    EXPECT_EQ(statex->node(r), r);
  }

  // noLocal uses initRankTopologyNolocal which sets fake host/pid;
  // real host/pid preservation is validated after the follow-up
  // initRankTopologyFrom fix lands
}

TEST_F(CommStateXDistTest, CreateVCliqueSizeFromNcclComm) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints({{"vCliqueSize", "2"}, {"useCtran", "1"}});
  config.hints = &hints;

  ncclx::test::NcclCommRAII comm{
      globalRank, numRanks, localRank, bootstrap_.get(), false, &config};
  ASSERT_NE(comm.get(), nullptr);
  ASSERT_TRUE(ctranInitialized(comm->ctranComm_.get()));

  const auto* statex = comm->ctranComm_->statex_.get();
  ASSERT_NE(statex, nullptr);

  EXPECT_EQ(statex->nLocalRanks(), 2);
  EXPECT_EQ(statex->nNodes(), numRanks / 2);

  std::vector<RankIdentity> allRankIds;
  gatherRankIdentities(allRankIds);
  verifyHelper_.verifyAllHosts(statex, allRankIds);
  verifyHelper_.verifyAllGPids(statex, allRankIds);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
