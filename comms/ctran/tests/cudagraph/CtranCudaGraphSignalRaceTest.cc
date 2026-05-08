// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <gtest/gtest.h>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/tests/cudagraph/CtranCudaGraphTestBase.h"
#include "comms/ctran/tests/cudagraph/CudaGraphTestBuilder.h"
#include "comms/utils/CudaRAII.h"

using namespace ctran;

class CtranCudaGraphSignalRaceTest : public CtranCudaGraphTestBase {
 protected:
  void SetUp() override {
    CtranCudaGraphTestBase::SetUp();
    CUDACHECK_TEST(cudaSetDevice(this->localRank));
  }
};

// ---------------------------------------------------------------------------
// CounterNotIncrementedDuringCapture
//
// Verifies that the per-peer signal/wait counters are NOT advanced during
// CUDA graph capture. The capture path skips the host-side counter
// increment; the kernel atomically increments the counter at replay time
// instead.
// ---------------------------------------------------------------------------
TEST_F(CtranCudaGraphSignalRaceTest, CounterNotIncrementedDuringCapture) {
  auto comm = makeCtranComm();
  ASSERT_NE(comm, nullptr);

  const int rank = globalRank;
  const int nRanks = numRanks;
  if (nRanks < 2) {
    GTEST_SKIP() << "Need at least 2 ranks";
  }

  const int nextPeer = (rank + 1) % nRanks;
  const int prevPeer = (rank + nRanks - 1) % nRanks;

  constexpr size_t kElements = 64;
  size_t sizeBytes = kElements * sizeof(int32_t) * nRanks;

  CtranWin* win = nullptr;
  void* winBase = nullptr;
  meta::comms::Hints hints;
  ASSERT_EQ(
      ctranWinAllocate(sizeBytes, comm.get(), &winBase, &win, hints),
      commSuccess);
  ASSERT_NE(win, nullptr);

  int32_t* sendBuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&sendBuf, kElements * sizeof(int32_t)));
  std::vector<int32_t> hostBuf(kElements, rank);
  CUDACHECK_TEST(cudaMemcpy(
      sendBuf,
      hostBuf.data(),
      kElements * sizeof(int32_t),
      cudaMemcpyHostToDevice));
  ASSERT_EQ(
      ctran::globalRegisterWithPtr(sendBuf, kElements * sizeof(int32_t)),
      commSuccess);

  meta::comms::CudaStream stream(cudaStreamNonBlocking);

  CUDACHECK_TEST(cudaDeviceSynchronize());
  oobBarrier();

  // Phase 1: N eager iterations to establish baseline counter values.
  constexpr int kEagerIters = 5;
  for (int i = 0; i < kEagerIters; ++i) {
    ASSERT_EQ(
        ctranPutSignal(
            sendBuf,
            kElements,
            commInt32,
            nextPeer,
            kElements * rank,
            win,
            stream.get(),
            true),
        commSuccess);
    ASSERT_EQ(ctranWaitSignal(prevPeer, win, stream.get()), commSuccess);
  }
  CUDACHECK_TEST(cudaStreamSynchronize(stream.get()));
  oobBarrier();

  // Snapshot counters before capture.
  const auto signalBefore =
      __atomic_load_n(&win->signalCounters[nextPeer], __ATOMIC_RELAXED);
  const auto waitBefore =
      __atomic_load_n(&win->waitCounters[prevPeer], __ATOMIC_RELAXED);

  // Phase 2: Capture a graph (no replay needed — just capture).
  meta::comms::CudaStream captureStream(cudaStreamNonBlocking);
  cudaGraph_t graph = nullptr;
  ASSERT_EQ(
      cudaStreamBeginCapture(captureStream.get(), cudaStreamCaptureModeRelaxed),
      cudaSuccess);
  ASSERT_EQ(
      ctranPutSignal(
          sendBuf,
          kElements,
          commInt32,
          nextPeer,
          kElements * rank,
          win,
          captureStream.get(),
          true),
      commSuccess);
  ASSERT_EQ(ctranWaitSignal(prevPeer, win, captureStream.get()), commSuccess);
  ASSERT_EQ(cudaStreamEndCapture(captureStream.get(), &graph), cudaSuccess);
  ASSERT_NE(graph, nullptr);

  // The counters must NOT have been incremented by graph capture.
  EXPECT_EQ(
      __atomic_load_n(&win->signalCounters[nextPeer], __ATOMIC_RELAXED),
      signalBefore)
      << "signalCounter was polluted during graph capture";
  EXPECT_EQ(
      __atomic_load_n(&win->waitCounters[prevPeer], __ATOMIC_RELAXED),
      waitBefore)
      << "waitCounter was polluted during graph capture";

  // Cleanup
  cudaGraphDestroy(graph);
  ASSERT_EQ(
      ctran::globalDeregisterWithPtr(sendBuf, kElements * sizeof(int32_t)),
      commSuccess);
  CUDACHECK_TEST(cudaFree(sendBuf));
  ASSERT_EQ(ctranWinFree(win), commSuccess);
}

// ---------------------------------------------------------------------------
// MonotonicSignalValues
//
// Verifies that signal values are monotonically increasing across eager
// iterations followed by graph replays. Both paths share a single signal
// buffer and per-peer counters, so the values must form a continuous
// ascending sequence (no resets, no gaps).
// ---------------------------------------------------------------------------
TEST_F(CtranCudaGraphSignalRaceTest, MonotonicSignalValues) {
  auto comm = makeCtranComm();
  ASSERT_NE(comm, nullptr);

  const int rank = globalRank;
  const int nRanks = numRanks;
  if (nRanks < 2) {
    GTEST_SKIP() << "Need at least 2 ranks";
  }

  const int nextPeer = (rank + 1) % nRanks;
  const int prevPeer = (rank + nRanks - 1) % nRanks;

  constexpr size_t kElements = 64;
  size_t sizeBytes = kElements * sizeof(int32_t) * nRanks;

  CtranWin* win = nullptr;
  void* winBase = nullptr;
  meta::comms::Hints hints;
  ASSERT_EQ(
      ctranWinAllocate(sizeBytes, comm.get(), &winBase, &win, hints),
      commSuccess);
  ASSERT_NE(win, nullptr);

  int32_t* sendBuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&sendBuf, kElements * sizeof(int32_t)));
  std::vector<int32_t> hostBuf(kElements, rank);
  CUDACHECK_TEST(cudaMemcpy(
      sendBuf,
      hostBuf.data(),
      kElements * sizeof(int32_t),
      cudaMemcpyHostToDevice));
  ASSERT_EQ(
      ctran::globalRegisterWithPtr(sendBuf, kElements * sizeof(int32_t)),
      commSuccess);

  meta::comms::CudaStream stream(cudaStreamNonBlocking);

  CUDACHECK_TEST(cudaDeviceSynchronize());
  oobBarrier();

  // Phase 1: Eager iterations.
  constexpr int kEagerIters = 5;
  for (int i = 0; i < kEagerIters; ++i) {
    ASSERT_EQ(
        ctranPutSignal(
            sendBuf,
            kElements,
            commInt32,
            nextPeer,
            kElements * rank,
            win,
            stream.get(),
            true),
        commSuccess);
    ASSERT_EQ(ctranWaitSignal(prevPeer, win, stream.get()), commSuccess);
  }
  CUDACHECK_TEST(cudaStreamSynchronize(stream.get()));
  oobBarrier();

  // Read signal buffer after eager phase.
  uint64_t signalAfterEager = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &signalAfterEager,
      win->winSignalPtr + prevPeer,
      sizeof(uint64_t),
      cudaMemcpyDeviceToHost));
  ASSERT_EQ(signalAfterEager, static_cast<uint64_t>(kEagerIters))
      << "Signal value should equal eager iteration count";

  // Phase 2: Graph replays — values should continue from where eager left off.
  constexpr int kGraphReplays = 3;
  ctran::testing::CtranGraphTestBuilder(comm.get(), rank, nRanks)
      .withNumReplays(kGraphReplays)
      .addCapture([&](ctran::testing::CaptureContext& ctx) {
        ASSERT_EQ(
            ctranPutSignal(
                sendBuf,
                kElements,
                commInt32,
                nextPeer,
                kElements * rank,
                win,
                ctx.stream,
                true),
            commSuccess);
        ASSERT_EQ(ctranWaitSignal(prevPeer, win, ctx.stream), commSuccess);
      })
      .withResourcePoolCheck(false)
      .run();

  oobBarrier();

  uint64_t signalAfterGraph = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &signalAfterGraph,
      win->winSignalPtr + prevPeer,
      sizeof(uint64_t),
      cudaMemcpyDeviceToHost));
  EXPECT_EQ(
      signalAfterGraph, static_cast<uint64_t>(kEagerIters + kGraphReplays))
      << "Signal values should be monotonically increasing across "
         "eager and graph replay";

  // Cleanup
  ASSERT_EQ(
      ctran::globalDeregisterWithPtr(sendBuf, kElements * sizeof(int32_t)),
      commSuccess);
  CUDACHECK_TEST(cudaFree(sendBuf));
  ASSERT_EQ(ctranWinFree(win), commSuccess);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new CtranCudaGraphEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
