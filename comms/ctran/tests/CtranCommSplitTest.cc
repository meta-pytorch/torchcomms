// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <memory>
#include <utility>
#include <vector>

#include <folly/ScopeGuard.h>
#include <folly/init/Init.h>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/CtranCommSplit.h"
#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/ctran/window/CtranWin.h"
#include "comms/testinfra/TestUtils.h"

namespace {

class CtranCommSplitTest : public ctran::CtranDistTestFixture {
 public:
  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE", "1", 0);
    ctran::CtranDistTestFixture::SetUp();
    parentComm_ = makeCtranComm();
  }

 protected:
  std::unique_ptr<CtranComm> parentComm_;
};

TEST_F(CtranCommSplitTest, LocalNvlSplitMapsRanksAndBootstrap) {
  std::shared_ptr<CtranComm> childComm;
  COMMCHECK_TEST(ctranCommSplitLocalNvl(parentComm_.get(), &childComm));
  ASSERT_NE(nullptr, childComm);

  auto* parentStatex = parentComm_->statex_.get();
  auto* childStatex = childComm->statex_.get();
  const auto parentRanks = parentStatex->localRankToRanks();

  EXPECT_TRUE(childComm->isSplitShare());
  EXPECT_EQ(parentComm_.get(), childComm->resourceComm());
  EXPECT_FALSE(ctranInitialized(childComm.get()));
  EXPECT_EQ(nullptr, childComm->ctran_);
  EXPECT_FALSE(
      ctranAllGatherSupport(childComm.get(), NCCL_ALLGATHER_ALGO::ctdirect));

  EXPECT_EQ(parentStatex->localRank(), childStatex->rank());
  EXPECT_EQ(parentStatex->nLocalRanks(), childStatex->nRanks());
  EXPECT_EQ(parentRanks, childComm->parentRanks());
  // init_none leaves the parent without a world-rank map (gRank() would abort
  // via CHECK_RANKMAP_SET); mirror buildSplitShareChild()'s fallback to the
  // parent comm rank when the parent has no world-rank map.
  const bool parentHasWorldRanks =
      !parentStatex->commRanksToWorldRanksRef().empty();
  for (int rank = 0; rank < childStatex->nRanks(); ++rank) {
    const int parentRank = parentRanks[rank];
    const int expectedWorldRank =
        parentHasWorldRanks ? parentStatex->gRank(parentRank) : parentRank;
    EXPECT_EQ(expectedWorldRank, childStatex->gRank(rank));
  }

  std::vector<int> gathered(childStatex->nRanks(), -1);
  gathered[childStatex->rank()] = parentStatex->rank();
  auto allGather = childComm->bootstrap_->allGather(
      gathered.data(), sizeof(int), childStatex->rank(), childStatex->nRanks());
  COMMCHECK_TEST(static_cast<commResult_t>(std::move(allGather).get()));
  EXPECT_EQ(parentRanks, gathered);
}

TEST_F(CtranCommSplitTest, LocalNvlSplitSupportsWindowRegistration) {
  std::shared_ptr<CtranComm> childComm;
  COMMCHECK_TEST(ctranCommSplitLocalNvl(parentComm_.get(), &childComm));

  constexpr size_t kBufferBytes = 8192;
  void* buffer = nullptr;
  CUDACHECK_TEST(cudaMalloc(&buffer, kBufferBytes));
  auto bufferGuard = folly::makeGuard([&]() {
    if (buffer != nullptr) {
      cudaFree(buffer);
    }
  });

  ctran::CtranWin* win = nullptr;
  COMMCHECK_TEST(
      ctran::ctranWinRegister(buffer, kBufferBytes, childComm.get(), &win));
  auto winGuard = folly::makeGuard([&]() {
    if (win != nullptr) {
      ctran::ctranWinFree(win);
    }
  });

  ASSERT_NE(nullptr, win);
  EXPECT_TRUE(win->comm->isSplitShare());
  EXPECT_EQ(parentComm_.get(), win->comm->resourceComm());
  ASSERT_EQ(win->remWinInfo.size(), childComm->statex_->nRanks());
  const int childRank = childComm->statex_->rank();
  for (int rank = 0; rank < static_cast<int>(win->remWinInfo.size()); ++rank) {
    EXPECT_NE(nullptr, win->remWinInfo[rank].dataAddr);
    if (rank != childRank) {
      EXPECT_NE(
          CtranMapperBackend::UNSET, win->remWinInfo[rank].dataRkey.backend);
    }
  }

  COMMCHECK_TEST(ctran::ctranWinFree(win));
  win = nullptr;
}

} // namespace

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new ctran::CtranDistEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
