// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/ScopeGuard.h>
#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <cstddef>
#include <vector>

#include "comm.h"
#include "comms/ctran/Ctran.h"
#include "comms/ncclx/meta/tests/NcclCommUtils.h"
#include "comms/ncclx/meta/tests/NcclxBaseTest.h"
#include "comms/ncclx/meta/tests/VerifyAlgoStatsUtil.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsCuUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/algoconf/AlgoStrConv.h"

// Verifies ctwin's automatic fallback behavior at the top-level ncclAllGather
// entry point. In comms/ncclx/v2_30/src/collectives.cc, ncclAllGather runs
// CTRAN's window-based AllGatherP path only when recvbuff lives inside a
// registered symmetric window; otherwise it falls back to the baseline
// algorithm. AlgoStats (recorded for both paths under the same per-comm map)
// is used to prove which algorithm actually ran.
class AllGatherCtwinTest : public NcclxBaseTestFixture {
 public:
  AllGatherCtwinTest() = default;

  void SetUp() override {
    NcclxEnvs envs = {
        {"NCCL_CTRAN_ENABLE", "1"},
        {"NCCL_CTRAN_IPC_REGCACHE_ENABLE_ASYNC_SOCKET", "1"},
    };
    NcclxBaseTestFixture::SetUp(envs);
    algoStats_.enable();

    CUDACHECK_TEST(cudaSetDevice(localRank));
    CUDACHECK_TEST(cudaStreamCreate(&stream));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaStreamDestroy(stream));
    NcclxBaseTestFixture::TearDown();
  }

 protected:
  ncclComm_t comm{};
  cudaStream_t stream{};
  ncclx::test::VerifyAlgoStatsHelper algoStats_;
};

// With allgatherAlgo=ctwin but a plain recvbuf (no window registered), ctwin
// is not supported for this recvbuff, so ncclAllGather falls back to the
// baseline algorithm. Verify correctness AND (via AlgoStats) that the
// baseline path ran, not the CTRAN window path.
TEST_F(AllGatherCtwinTest, AllGatherCtwinFallbackWhenNoWindow) {
  constexpr size_t count = 1024;

  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints(
      {{"allgatherAlgo",
        ncclx::algoconf::algoValToStr(NCCL_ALLGATHER_ALGO::ctwin)}});
  config.hints = &hints;
  ncclx::test::NcclCommRAII commRaii(
      globalRank, numRanks, localRank, bootstrap_.get(), false, &config);
  this->comm = commRaii.get();
  ASSERT_NE(this->comm, nullptr);

  std::vector<TestMemSegment> sendBufSegs, recvBufSegs;
  int* recvBuf = reinterpret_cast<int*>(testAllocBuf(
      count * numRanks * sizeof(int), kMemCudaMalloc, recvBufSegs));
  ASSERT_NE(recvBuf, nullptr);
  int* sendBuf = reinterpret_cast<int*>(
      testAllocBuf(count * sizeof(int), kMemCudaMalloc, sendBufSegs));
  ASSERT_NE(sendBuf, nullptr);

  assignChunkValue(recvBuf, count * numRanks, -1);
  assignChunkValue(sendBuf, count, globalRank + 1);

  ASSERT_EQ(
      ncclAllGather(sendBuf, recvBuf, count, ncclInt, comm, stream),
      ncclSuccess);
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  for (int r = 0; r < numRanks; r++) {
    int expectedVal = r + 1;
    int errs = checkChunkValue(recvBuf + r * count, count, expectedVal);
    EXPECT_EQ(errs, 0) << "rank " << globalRank << " checked chunk " << r
                       << " at " << recvBuf + r * count << " with " << errs
                       << " errors";
  }

  // ctwin has no window to run against here, so dispatch must fall back to
  // the baseline algorithm rather than the CTRAN window path.
  algoStats_.verify(comm, "AllGather", "Baseline");
  algoStats_.verifyNot(comm, "AllGather", "CtranAllGatherP");

  testFreeBuf(sendBuf, count * sizeof(int), kMemCudaMalloc);
  testFreeBuf(recvBuf, count * numRanks * sizeof(int), kMemCudaMalloc);
}

// With allgatherAlgo=ctwin and recvbuff living inside a registered symmetric
// window, ctwin's CTRAN AllGatherP path is selected. Verify correctness AND
// (via AlgoStats) that the CtranAllGatherP path ran.
TEST_F(AllGatherCtwinTest, AllGatherCtwinUsedWhenWindowRegistered) {
  if (!ncclIsCuMemSupported()) {
    GTEST_SKIP() << "CuMem not supported, skip test";
  }

  constexpr size_t count = 1024;
  const size_t totalBytes = count * numRanks * sizeof(int);

  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints(
      {{"allgatherAlgo",
        ncclx::algoconf::algoValToStr(NCCL_ALLGATHER_ALGO::ctwin)},
       {"win_register_symmetric", "1"},
       {"win_register_ipc_only", "1"}});
  config.hints = &hints;
  ncclx::test::NcclCommRAII commRaii(
      globalRank, numRanks, localRank, bootstrap_.get(), false, &config);
  this->comm = commRaii.get();
  ASSERT_NE(this->comm, nullptr);

  if (comm->ctranComm_->ctran_->mapper->ctranIbPtr() == nullptr) {
    GTEST_SKIP() << "No IB Backend found, skip test";
  }

  std::vector<TestMemSegment> winSegs;
  void* winBase = testAllocBuf(totalBytes, kMemNcclMemAlloc, winSegs);
  ASSERT_NE(winBase, nullptr);
  // Mimic the CCA allocator hook so the window's acquireScopedRegister finds
  // the buffer's segment cached.
  NCCLCHECK_TEST(ncclGlobalRegisterWithPtr(winBase, totalBytes));
  auto regGuard = folly::makeGuard([&]() {
    NCCLCHECK_TEST(ncclGlobalDeregisterWithPtr(winBase, totalBytes));
  });

  ncclWindow_t win = nullptr;
  ASSERT_EQ(
      ncclCommWindowRegister(
          comm, winBase, totalBytes, &win, NCCL_WIN_COLL_SYMMETRIC),
      ncclSuccess);
  ASSERT_NE(win, nullptr);
  regGuard.dismiss();

  // In-place layout required by ctwin: this rank's send chunk lives at its
  // own offset within the recvbuf/window.
  int* recvBuf = reinterpret_cast<int*>(winBase);
  int* sendBuf = recvBuf + count * globalRank;

  assignChunkValue(recvBuf, count * numRanks, -1);
  assignChunkValue(sendBuf, count, globalRank + 1);

  ASSERT_EQ(
      ncclAllGather(sendBuf, recvBuf, count, ncclInt, comm, stream),
      ncclSuccess);
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  for (int r = 0; r < numRanks; r++) {
    int expectedVal = r + 1;
    int errs = checkChunkValue(recvBuf + r * count, count, expectedVal);
    EXPECT_EQ(errs, 0) << "rank " << globalRank << " checked chunk " << r
                       << " at " << recvBuf + r * count << " with " << errs
                       << " errors";
  }

  algoStats_.verify(comm, "AllGather", "CtranAllGatherP");

  EXPECT_EQ(ncclCommWindowDeregister(comm, win), ncclSuccess);
  NCCLCHECK_TEST(ncclGlobalDeregisterWithPtr(winBase, totalBytes));
  testFreeBuf(winBase, totalBytes, kMemNcclMemAlloc);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
