// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/ScopeGuard.h>
#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>
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

// Same as above but with win_register_multicast=1: the window sets up an NVL
// CE-multicast overlay in CtranWin::exchange, and ctwin's AllGatherP fans out
// through the multicast VA (nvlCeBcast) instead of N-1 unicast copies. On HW
// without multicast/fabric support the overlay setup declines and the path
// falls back to unicast, so the result must be correct either way; either way
// ctwin (CtranAllGatherP) must run.
TEST_F(AllGatherCtwinTest, AllGatherCtwinMulticastWhenWindowRegistered) {
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
       {"win_register_ipc_only", "1"},
       {"win_register_multicast", "1"}});
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

// Standalone timed benchmark of ctwin AllGatherP over a symmetric window:
// warmup, then a cudaEvent-timed loop of ncclAllGather, reporting busbw. Env
// knobs (one binary covers the size sweep + MC-vs-unicast, no recompile):
//   NCCLX_BENCH_BYTES     total recvbuf bytes (default 8 MiB)
//   NCCLX_BENCH_ITERS     timed iterations (default 200)
//   NCCLX_BENCH_WARMUP    warmup iterations (default 50)
//   NCCLX_BENCH_MULTICAST "1" (default) sets win_register_multicast; "0" = the
//                         ctwin-unicast control (isolates the multicast win)
TEST_F(AllGatherCtwinTest, AllGatherCtwinPerf) {
  if (!ncclIsCuMemSupported()) {
    GTEST_SKIP() << "CuMem not supported, skip test";
  }
  auto envOr = [](const char* k, unsigned long long d) -> unsigned long long {
    const char* v = getenv(k);
    return (v != nullptr && *v != '\0') ? strtoull(v, nullptr, 10) : d;
  };
  const size_t totalBytes = envOr("NCCLX_BENCH_BYTES", 8ull << 20);
  const int iters = static_cast<int>(envOr("NCCLX_BENCH_ITERS", 200));
  const int warmup = static_cast<int>(envOr("NCCLX_BENCH_WARMUP", 50));
  const char* mcEnv = getenv("NCCLX_BENCH_MULTICAST");
  const std::string mc = (mcEnv != nullptr && *mcEnv != '\0') ? mcEnv : "1";
  const size_t count = totalBytes / (numRanks * sizeof(int)); // per-rank chunk
  ASSERT_GT(count, 0u);
  const size_t winBytes = count * numRanks * sizeof(int);

  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  ncclx::Hints hints(
      {{"allgatherAlgo",
        ncclx::algoconf::algoValToStr(NCCL_ALLGATHER_ALGO::ctwin)},
       {"win_register_symmetric", "1"},
       {"win_register_ipc_only", "1"},
       {"win_register_multicast", mc}});
  config.hints = &hints;
  ncclx::test::NcclCommRAII commRaii(
      globalRank, numRanks, localRank, bootstrap_.get(), false, &config);
  this->comm = commRaii.get();
  ASSERT_NE(this->comm, nullptr);
  if (comm->ctranComm_->ctran_->mapper->ctranIbPtr() == nullptr) {
    GTEST_SKIP() << "No IB Backend found, skip test";
  }

  std::vector<TestMemSegment> winSegs;
  void* winBase = testAllocBuf(winBytes, kMemNcclMemAlloc, winSegs);
  ASSERT_NE(winBase, nullptr);
  NCCLCHECK_TEST(ncclGlobalRegisterWithPtr(winBase, winBytes));
  auto regGuard = folly::makeGuard([&]() {
    NCCLCHECK_TEST(ncclGlobalDeregisterWithPtr(winBase, winBytes));
  });
  ncclWindow_t win = nullptr;
  ASSERT_EQ(
      ncclCommWindowRegister(
          comm, winBase, winBytes, &win, NCCL_WIN_COLL_SYMMETRIC),
      ncclSuccess);
  ASSERT_NE(win, nullptr);
  regGuard.dismiss();

  int* recvBuf = reinterpret_cast<int*>(winBase);
  int* sendBuf = recvBuf + count * globalRank; // in-place layout ctwin requires
  assignChunkValue(recvBuf, count * numRanks, -1);
  assignChunkValue(sendBuf, count, globalRank + 1);

  for (int i = 0; i < warmup; i++) {
    ASSERT_EQ(
        ncclAllGather(sendBuf, recvBuf, count, ncclInt, comm, stream),
        ncclSuccess);
  }
  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  cudaEvent_t start, stop;
  CUDACHECK_TEST(cudaEventCreate(&start));
  CUDACHECK_TEST(cudaEventCreate(&stop));
  CUDACHECK_TEST(cudaEventRecord(start, stream));
  for (int i = 0; i < iters; i++) {
    ASSERT_EQ(
        ncclAllGather(sendBuf, recvBuf, count, ncclInt, comm, stream),
        ncclSuccess);
  }
  CUDACHECK_TEST(cudaEventRecord(stop, stream));
  CUDACHECK_TEST(cudaStreamSynchronize(stream));
  float ms = 0.f;
  CUDACHECK_TEST(cudaEventElapsedTime(&ms, start, stop));

  const double secPerIter = (ms / 1e3) / iters;
  const double algbw = (double)winBytes / 1e9 / secPerIter; // GB/s
  const double busbw = algbw * (double)(numRanks - 1) / (double)numRanks;

  int errs = 0;
  for (int r = 0; r < numRanks; r++) {
    errs += checkChunkValue(recvBuf + r * count, count, r + 1);
  }
  EXPECT_EQ(errs, 0);

  if (globalRank == 0) {
    printf(
        "# AGCTWIN-BENCH multicast=%s ranks=%d bytes=%zu iters=%d  time=%.2f us/op  algbw=%.2f GB/s  busbw=%.2f GB/s  errs=%d\n",
        mc.c_str(),
        numRanks,
        winBytes,
        iters,
        secPerIter * 1e6,
        algbw,
        busbw,
        errs);
    fflush(stdout);
  }
  algoStats_.verify(comm, "AllGather", "CtranAllGatherP");

  CUDACHECK_TEST(cudaEventDestroy(start));
  CUDACHECK_TEST(cudaEventDestroy(stop));
  EXPECT_EQ(ncclCommWindowDeregister(comm, win), ncclSuccess);
  NCCLCHECK_TEST(ncclGlobalDeregisterWithPtr(winBase, winBytes));
  testFreeBuf(winBase, winBytes, kMemNcclMemAlloc);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
