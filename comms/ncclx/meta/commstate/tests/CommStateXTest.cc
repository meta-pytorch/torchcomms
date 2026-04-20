// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <set>
#include <string>

#include <folly/init/Init.h>
#include <gtest/gtest.h>

#include "comm.h"
#include "comms/ncclx/meta/tests/NcclCommUtils.h"
#include "comms/ncclx/meta/tests/NcclxBaseTest.h"
#include "meta/hints/GlobalHints.h" // @manual
#include "nccl.h"

class CommStateXDistTest : public NcclxBaseTestFixture {
 public:
  void SetUp() override {
    NcclxBaseTestFixture::SetUp();
    ASSERT_EQ(
        ncclx::setGlobalHint(std::string(ncclx::HintKeys::kCommUseCtran), "1"),
        ncclSuccess);
  }

  void TearDown() override {
    ncclx::resetGlobalHint(std::string(ncclx::HintKeys::kCommUseCtran));
    ncclx::resetGlobalHint(std::string(ncclx::HintKeys::kCommNoLocal));
    NcclxBaseTestFixture::TearDown();
  }
};

TEST_F(CommStateXDistTest, CreateFromNcclComm) {
  ncclx::test::NcclCommRAII comm{
      globalRank, numRanks, localRank, bootstrap_.get()};
  ASSERT_NE(comm.get(), nullptr);
  ASSERT_TRUE(ctranInitialized(comm->ctranComm_.get()));

  const auto* statex = comm->ctranComm_->statex_.get();
  ASSERT_NE(statex, nullptr);

  EXPECT_EQ(statex->rank(), globalRank);
  EXPECT_EQ(statex->nRanks(), numRanks);
  EXPECT_EQ(statex->nLocalRanks(), numRanks);
  EXPECT_EQ(statex->nNodes(), 1);

  // Single node: all ranks share the same hostname
  const std::string myHost = statex->host(globalRank);
  EXPECT_FALSE(myHost.empty());
  for (int r = 0; r < numRanks; r++) {
    EXPECT_EQ(statex->host(r), myHost)
        << "rank " << r << " should have same host on single node";
  }

  // gPid (<host>:<pid>) should be unique across all ranks, verifying
  // that each rank has a distinct PID (separate processes on same host)
  std::set<std::string> gPids;
  for (int r = 0; r < numRanks; r++) {
    EXPECT_FALSE(statex->gPid(r).empty());
    auto [it, inserted] = gPids.insert(statex->gPid(r));
    EXPECT_TRUE(inserted) << "gPid collision at rank " << r << ": "
                          << statex->gPid(r);
  }
}

TEST_F(CommStateXDistTest, CreateNoLocalFromNcclComm) {
  // FIXME: replace with per-comm ncclx::Hints config once noLocal is
  // supported as a per-comm hint field (global hints will be deprecated)
  ASSERT_EQ(
      ncclx::setGlobalHint(std::string(ncclx::HintKeys::kCommNoLocal), "1"),
      ncclSuccess);

  ncclx::test::NcclCommRAII comm{
      globalRank, numRanks, localRank, bootstrap_.get()};
  ASSERT_NE(comm.get(), nullptr);
  ASSERT_TRUE(ctranInitialized(comm->ctranComm_.get()));

  const auto* statex = comm->ctranComm_->statex_.get();
  ASSERT_NE(statex, nullptr);

  EXPECT_EQ(statex->nLocalRanks(), 1);
  EXPECT_EQ(statex->nNodes(), numRanks);
  for (int r = 0; r < numRanks; r++) {
    EXPECT_EQ(statex->node(r), r);
  }

  std::set<std::string> gPids;
  for (int r = 0; r < numRanks; r++) {
    auto [it, inserted] = gPids.insert(statex->gPid(r));
    EXPECT_TRUE(inserted) << "gPid collision at rank " << r << ": "
                          << statex->gPid(r);
  }
}

TEST_F(CommStateXDistTest, DISABLED_CreateVCliqueSizeFromNcclComm) {
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints({{"vCliqueSize", "2"}});
  config.hints = &hints;

  ncclx::test::NcclCommRAII comm{
      globalRank, numRanks, localRank, bootstrap_.get(), false, &config};
  ASSERT_NE(comm.get(), nullptr);
  ASSERT_TRUE(ctranInitialized(comm->ctranComm_.get()));

  const auto* statex = comm->ctranComm_->statex_.get();
  ASSERT_NE(statex, nullptr);

  EXPECT_EQ(statex->nLocalRanks(), 2);
  EXPECT_EQ(statex->nNodes(), numRanks / 2);

  std::set<std::string> gPids;
  for (int r = 0; r < numRanks; r++) {
    auto [it, inserted] = gPids.insert(statex->gPid(r));
    EXPECT_TRUE(inserted) << "gPid collision at rank " << r << ": "
                          << statex->gPid(r);
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
