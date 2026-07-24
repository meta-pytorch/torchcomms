// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/regcache/IpcRegCache.h"
#include "comms/ctran/regcache/RegCache.h"
#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/ctran/tests/CtranTestUtils.h"
#include "comms/ctran/tests/VerifyAlgoStatsUtil.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/window/CtranWin.h"
#include "comms/testinfra/TestsCuUtils.h"

using namespace ctran;

// Window-based persistent alltoall (ctwin). Registers a symmetric window over a
// cumem recvbuf, runs alltoall with algo=ctwin, and verifies the result against
// the alltoall reference (rank r's recv chunk i holds peer i's send chunk for
// rank r). Also verifies that repeated calls over the same recvbuf sub-range
// reuse one cached persistent request, that a distinct sub-range gets its own
// request, and that ctranWinFree tears the cached requests down without leaking
// imports.
class CtranAllToAllCtwinTest : public ctran::CtranDistTestFixture {
 public:
  CtranAllToAllCtwinTest() = default;

  commDataType_t dt = commInt32;
  cudaStream_t stream = 0;
  std::unique_ptr<CtranComm> ctranComm;
  std::vector<TestMemSegment> segments;
  ctran::test::VerifyAlgoStatsHelper algoStats_;

  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE", "1", 0);
    algoStats_.enable();
    ctran::CtranDistTestFixture::SetUp();
    CUDACHECK_TEST(cudaStreamCreate(&stream));
    ctranComm = makeCtranComm();
  }

  void TearDown() override {
    // Every ctwin test frees its window(s) before returning; verify no NVL IPC
    // imports leaked.
    const auto ipcRegCache = ctran::IpcRegCache::getInstance();
    // EXPECT (not ASSERT) so stream teardown below always runs.
    EXPECT_NE(ipcRegCache, nullptr);
    if (ipcRegCache != nullptr) {
      EXPECT_EQ(ipcRegCache->maxRemRegRefCount(), 0)
          << "IpcRegCache still holds live NVL IPC imports after test teardown";
    }
    CUDACHECK_TEST(cudaStreamDestroy(stream));
    ctran::CtranDistTestFixture::TearDown();
  }

  // Register a symmetric window over a freshly cumem-allocated buffer.
  // Symmetric because every rank allocates the same size at the same offset.
  // When ipcOnly is true, inter-node IB rkeys are deferred to the first ctwin
  // exec; when false, the full window exchange (including IB rkeys) happens at
  // window creation.
  void*
  createSymmetricWindow(size_t bytes, CtranWin** winOut, bool ipcOnly = true) {
    void* buf = commMemAlloc(bytes, MemAllocType::kMemCuMemAlloc, segments);
    EXPECT_NE(buf, nullptr);
    // Mimic the CCA allocator hook so the window's acquireScopedRegister finds
    // the buffer's segment cached.
    COMMCHECK_TEST(ctran::RegCache::getInstance()->globalRegister(buf, bytes));
    meta::comms::Hints hints;
    EXPECT_EQ(hints.set("win_register_symmetric", "1"), commSuccess);
    if (ipcOnly) {
      EXPECT_EQ(hints.set("win_register_ipc_only", "1"), commSuccess);
    }
    COMMCHECK_TEST(
        ctranWinRegister(buf, bytes, ctranComm.get(), winOut, hints));
    return buf;
  }

  void freeSymmetricWindow(CtranWin* win, void* buf, size_t bytes) {
    COMMCHECK_TEST(ctranWinFree(win));
    COMMCHECK_TEST(
        ctran::RegCache::getInstance()->globalDeregister(buf, bytes));
    commMemFree(buf, bytes, MemAllocType::kMemCuMemAlloc);
    segments.erase(
        std::remove_if(
            segments.begin(),
            segments.end(),
            [buf](const TestMemSegment& seg) { return seg.ptr == buf; }),
        segments.end());
  }

  // Value this rank sends to `dst` on iteration `iter`. Distinct per
  // (srcRank, dst, iter) so a mis-routed chunk is caught.
  static int sendVal(int srcRank, int dst, int iter) {
    return srcRank * 100 + dst + 1 + iter * 100000;
  }

  // Fill a device sendbuf (numRanks chunks of `count`) with this rank's per-dst
  // send values for iteration `iter`.
  void fillSendBuf(void* sendbuf, size_t count, int iter) {
    std::vector<int> host(count * numRanks);
    for (int dst = 0; dst < numRanks; ++dst) {
      std::fill(
          host.begin() + dst * count,
          host.begin() + (dst + 1) * count,
          sendVal(globalRank, dst, iter));
    }
    CUDACHECK_TEST(cudaMemcpy(
        sendbuf,
        host.data(),
        count * numRanks * commTypeSize(dt),
        cudaMemcpyDefault));
  }

  // Verify a device recvbuf: chunk i must equal peer i's send value for this
  // rank on iteration `iter`.
  void verifyRecvBuf(void* recvbuf, size_t count, int iter) {
    const size_t chunkBytes = count * commTypeSize(dt);
    for (int peer = 0; peer < numRanks; ++peer) {
      std::vector<int> observed(count, -1);
      CUDACHECK_TEST(cudaMemcpy(
          observed.data(),
          static_cast<char*>(recvbuf) + peer * chunkBytes,
          chunkBytes,
          cudaMemcpyDefault));
      const std::vector<int> expected(count, sendVal(peer, globalRank, iter));
      EXPECT_EQ(observed, expected) << "at rank " << globalRank << " iter "
                                    << iter << " chunk from peer " << peer;
    }
  }

  // Run one out-of-place ctwin alltoall over the window sub-range at byteOffset
  // on execStream and verify every received chunk.
  void runAllToAllOnStream(
      void* winBase,
      size_t byteOffset,
      size_t count,
      int iter,
      cudaStream_t execStream,
      enum NCCL_ALLTOALL_ALGO algo = NCCL_ALLTOALL_ALGO::ctwin) {
    const size_t typeSize = commTypeSize(dt);
    const size_t totalBytes = count * numRanks * typeSize;
    void* recvbuf = static_cast<char*>(winBase) + byteOffset;

    // sendbuf is a plain (non-window) buffer: alltoall is out-of-place.
    void* sendbuf = nullptr;
    CUDACHECK_TEST(cudaMalloc(&sendbuf, totalBytes));

    fillSendBuf(sendbuf, count, iter);
    CUDACHECK_TEST(cudaMemset(recvbuf, 0xEE, totalBytes));
    CUDACHECK_TEST(cudaDeviceSynchronize());
    oobBarrier();

    ASSERT_EQ(
        ctranAllToAll(
            sendbuf, recvbuf, count, dt, ctranComm.get(), execStream, algo),
        commSuccess);
    ASSERT_EQ(cudaStreamSynchronize(execStream), cudaSuccess);

    verifyRecvBuf(recvbuf, count, iter);
    oobBarrier();

    CUDACHECK_TEST(cudaFree(sendbuf));
  }

  // Convenience overload that runs on the fixture's default stream.
  void runAllToAll(
      void* winBase,
      size_t byteOffset,
      size_t count,
      int iter,
      enum NCCL_ALLTOALL_ALGO algo = NCCL_ALLTOALL_ALGO::ctwin) {
    runAllToAllOnStream(winBase, byteOffset, count, iter, stream, algo);
  }
};

TEST_F(CtranAllToAllCtwinTest, FullAndSubsetReuseAndFree) {
  if (!ncclIsCuMemSupported()) {
    GTEST_SKIP() << "CuMem not supported, skipping ctwin test";
  }
  if (ctranComm->ctran_->mapper->ctranIbPtr() == nullptr) {
    GTEST_SKIP() << "No IB Backend found, skip test";
  }

  const size_t typeSize = commTypeSize(dt);
  const size_t fullCount = 8192;
  const size_t subCount = 2048;
  const size_t fullTotalBytes = fullCount * numRanks * typeSize;
  const size_t subTotalBytes = subCount * numRanks * typeSize;
  // Window large enough for the full alltoall (at offset 0) plus a distinct
  // subset alltoall placed after it.
  const size_t windowBytes = fullTotalBytes + subTotalBytes;

  CtranWin* win = nullptr;
  void* winBase = createSymmetricWindow(windowBytes, &win);
  ASSERT_NE(win, nullptr);
  EXPECT_TRUE(win->isSymmetric());

  // ctwin reports supported for a recvbuf that lives inside this window.
  EXPECT_TRUE(ctranAllToAllSupport(
      fullCount,
      dt,
      ctranComm.get(),
      NCCL_ALLTOALL_ALGO::ctwin,
      stream,
      winBase));

  // Full-window alltoall run twice: the second call must reuse the single
  // cached persistent request rather than building a new one.
  runAllToAll(winBase, /*byteOffset=*/0, fullCount, /*iter=*/0);
  runAllToAll(winBase, /*byteOffset=*/0, fullCount, /*iter=*/1);
  EXPECT_EQ(win->numPersistentRequests(), 1u);

  // A distinct sub-range gets its own cached request; repeating it reuses it.
  runAllToAll(winBase, fullTotalBytes, subCount, /*iter=*/2);
  EXPECT_EQ(win->numPersistentRequests(), 2u);
  runAllToAll(winBase, fullTotalBytes, subCount, /*iter=*/3);
  EXPECT_EQ(win->numPersistentRequests(), 2u);

  oobBarrier();

  // ctwin executes via the persistent AllToAllP machinery, so the recorded algo
  // name is "CtranAllToAllP" (not "CtranAllToAllCtwin"), confirming the ctwin
  // path (not a fallback/other algo) executed.
  algoStats_.verify(ctranComm.get(), "AllToAll", "CtranAllToAllP");

  // Free must tear down the cached persistent requests and release all imports.
  freeSymmetricWindow(win, winBase, windowBytes);
}

// A symmetric window registered WITHOUT ipc_only does the full window exchange
// at creation -- including the inter-node IB rkey exchange -- so ctwin reuses
// those rkeys (ibKeysExchanged=true) and skips the first-exec IB exchange.
// Verifies a correct alltoall (with reuse) and that the ctwin (AllToAllP) path
// ran.
TEST_F(CtranAllToAllCtwinTest, FullExchangeSymmetricWindow) {
  if (!ncclIsCuMemSupported()) {
    GTEST_SKIP() << "CuMem not supported, skipping ctwin test";
  }
  if (ctranComm->ctran_->mapper->ctranIbPtr() == nullptr) {
    GTEST_SKIP() << "No IB Backend found, skip test";
  }

  const size_t typeSize = commTypeSize(dt);
  const size_t count = 8192;
  const size_t windowBytes = count * numRanks * typeSize;

  CtranWin* win = nullptr;
  // Full-exchange symmetric window (NOT ipc_only): IB rkeys are exchanged at
  // window creation.
  void* winBase = createSymmetricWindow(windowBytes, &win, /*ipcOnly=*/false);
  ASSERT_NE(win, nullptr);
  EXPECT_TRUE(win->isSymmetric());
  EXPECT_FALSE(win->isIpcOnly());

  EXPECT_TRUE(ctranAllToAllSupport(
      count, dt, ctranComm.get(), NCCL_ALLTOALL_ALGO::ctwin, stream, winBase));

  // Run twice: the second call reuses the single cached persistent request.
  runAllToAll(winBase, /*byteOffset=*/0, count, /*iter=*/0);
  runAllToAll(winBase, /*byteOffset=*/0, count, /*iter=*/1);
  EXPECT_EQ(win->numPersistentRequests(), 1u);

  // Confirm the ctwin (AllToAllP) path actually ran.
  algoStats_.verify(ctranComm.get(), "AllToAll", "CtranAllToAllP");

  oobBarrier();
  freeSymmetricWindow(win, winBase, windowBytes);
}

// Captures a ctwin alltoall over the symmetric window into a CUDA graph and
// replays it several times, verifying the result each replay. Because ctwin's
// persistent request is window-owned (not graph-owned), the request is built
// once during capture and reused across replays; the graph is destroyed before
// the window is freed (the required lifetime order).
TEST_F(CtranAllToAllCtwinTest, GraphCaptureReplayReuseAndFree) {
  if (!ncclIsCuMemSupported()) {
    GTEST_SKIP() << "CuMem not supported, skipping ctwin graph test";
  }
  if (ctranComm->ctran_->mapper->ctranIbPtr() == nullptr) {
    GTEST_SKIP() << "No IB Backend found, skip test";
  }

  const size_t typeSize = commTypeSize(dt);
  const size_t count = 8192;
  const size_t totalBytes = count * numRanks * typeSize;

  CtranWin* win = nullptr;
  void* winBase = createSymmetricWindow(totalBytes, &win);
  ASSERT_NE(win, nullptr);
  EXPECT_TRUE(win->isSymmetric());

  // recvbuf lives in the window; sendbuf is a persistent (non-window) buffer
  // captured by the graph and updated in place between replays.
  void* recvbuf = winBase;
  void* sendbuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&sendbuf, totalBytes));

  cudaStream_t captureStream;
  CUDACHECK_TEST(
      cudaStreamCreateWithFlags(&captureStream, cudaStreamNonBlocking));

  // Seed send/recv so the capture-time exec sees valid data (capture-time
  // results are discarded; only replays are verified).
  fillSendBuf(sendbuf, count, /*iter=*/0);
  CUDACHECK_TEST(cudaMemset(recvbuf, 0xEE, totalBytes));
  CUDACHECK_TEST(cudaDeviceSynchronize());
  oobBarrier();

  // Capture the ctwin alltoall. request->stream == captureStream, so exec
  // submits directly onto the captured stream.
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graphExec = nullptr;
  ASSERT_EQ(
      cudaStreamBeginCapture(captureStream, cudaStreamCaptureModeGlobal),
      cudaSuccess);
  ASSERT_EQ(
      ctranAllToAll(
          sendbuf,
          recvbuf,
          count,
          dt,
          ctranComm.get(),
          captureStream,
          NCCL_ALLTOALL_ALGO::ctwin),
      commSuccess);
  ASSERT_EQ(cudaStreamEndCapture(captureStream, &graph), cudaSuccess);
  ASSERT_NE(graph, nullptr);
  ASSERT_EQ(cudaGraphInstantiate(&graphExec, graph, 0), cudaSuccess);

  // The persistent request built during capture is cached on the window.
  EXPECT_EQ(win->numPersistentRequests(), 1u);

  constexpr int kReplays = 3;
  for (int r = 0; r < kReplays; ++r) {
    const int iter = r + 1;
    fillSendBuf(sendbuf, count, iter);
    CUDACHECK_TEST(cudaMemset(recvbuf, 0xEE, totalBytes));
    CUDACHECK_TEST(cudaDeviceSynchronize());
    oobBarrier();

    ASSERT_EQ(cudaGraphLaunch(graphExec, captureStream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(captureStream), cudaSuccess);

    verifyRecvBuf(recvbuf, count, iter);
    // No new request is created across replays.
    EXPECT_EQ(win->numPersistentRequests(), 1u);
    oobBarrier();
  }

  // Correct lifetime order: destroy the graph BEFORE freeing the window (the
  // window owns the request the graph captured).
  ASSERT_EQ(cudaGraphExecDestroy(graphExec), cudaSuccess);
  ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
  CUDACHECK_TEST(cudaFree(sendbuf));
  CUDACHECK_TEST(cudaStreamDestroy(captureStream));

  oobBarrier();
  freeSymmetricWindow(win, winBase, totalBytes);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new ctran::CtranDistEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
