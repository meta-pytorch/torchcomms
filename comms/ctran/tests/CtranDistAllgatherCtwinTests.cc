// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <unordered_map>
#include <vector>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/regcache/IpcRegCache.h"
#include "comms/ctran/regcache/RegCache.h"
#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/ctran/tests/CtranTestUtils.h"
#include "comms/ctran/tests/VerifyAlgoStatsUtil.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/window/CtranWin.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/testinfra/TestsCuUtils.h"

using namespace ctran;

// Window-based persistent allgather (ctwin). Registers a symmetric ipc_only
// window over a cumem recvbuf, runs allgather with algo=ctwin (in-place, so the
// window IS the recvbuf), and verifies the gathered result against a reference.
// Also verifies that repeated calls over the same recvbuf sub-range reuse one
// cached persistent request, that a distinct sub-range gets its own request,
// and that ctranWinFree tears the cached requests down without leaking imports.
class CtranAllgatherCtwinTest : public ctran::CtranDistTestFixture {
 public:
  CtranAllgatherCtwinTest() = default;

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
    // EXPECT (not ASSERT) so stream/base teardown below always runs.
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

  // Run one in-place allgather over the window sub-range at byteOffset on
  // execStream using the given ctwin-family algo (default forces the persistent
  // pipeline) and verify every peer chunk equals that peer's rank+iter-specific
  // pattern.
  void runGatherOnStream(
      CtranWin* win,
      void* winBase,
      size_t byteOffset,
      size_t sendCount,
      int iter,
      cudaStream_t execStream,
      enum NCCL_ALLGATHER_ALGO algo = NCCL_ALLGATHER_ALGO::ctwin_pipeline) {
    const size_t typeSize = commTypeSize(dt);
    const size_t chunkBytes = sendCount * typeSize;
    void* recvbuf = static_cast<char*>(winBase) + byteOffset;
    // In-place: this rank's send data lives in its own chunk of the recvbuf.
    void* sendbuf = static_cast<char*>(recvbuf) + globalRank * chunkBytes;

    const int myVal = globalRank + iter * 100;
    const std::vector<int> myChunk(sendCount, myVal);
    CUDACHECK_TEST(cudaMemset(recvbuf, 0xEE, sendCount * numRanks * typeSize));
    CUDACHECK_TEST(
        cudaMemcpy(sendbuf, myChunk.data(), chunkBytes, cudaMemcpyDefault));
    CUDACHECK_TEST(cudaDeviceSynchronize());
    oobBarrier();

    ASSERT_EQ(
        ctranAllGather(
            sendbuf, recvbuf, sendCount, dt, ctranComm.get(), execStream, algo),
        commSuccess);
    ASSERT_EQ(cudaStreamSynchronize(execStream), cudaSuccess);

    for (int peer = 0; peer < numRanks; ++peer) {
      std::vector<int> observed(sendCount, -1);
      CUDACHECK_TEST(cudaMemcpy(
          observed.data(),
          static_cast<char*>(recvbuf) + peer * chunkBytes,
          chunkBytes,
          cudaMemcpyDefault));
      const std::vector<int> expected(sendCount, peer + iter * 100);
      EXPECT_EQ(observed, expected)
          << "at rank " << globalRank << " iter " << iter << " byteOffset "
          << byteOffset << " chunk from peer " << peer;
    }
    oobBarrier();
  }

  // Convenience overload that runs on the fixture's default stream.
  void runGather(
      CtranWin* win,
      void* winBase,
      size_t byteOffset,
      size_t sendCount,
      int iter,
      enum NCCL_ALLGATHER_ALGO algo = NCCL_ALLGATHER_ALGO::ctwin_pipeline) {
    runGatherOnStream(win, winBase, byteOffset, sendCount, iter, stream, algo);
  }
};

TEST_F(CtranAllgatherCtwinTest, FullAndSubsetReuseAndFree) {
  if (!ncclIsCuMemSupported()) {
    GTEST_SKIP() << "CuMem not supported, skipping ctwin test";
  }
  if (ctranComm->ctran_->mapper->ctranIbPtr() == nullptr) {
    GTEST_SKIP() << "No IB Backend found, skip test";
  }

  const size_t typeSize = commTypeSize(dt);
  const size_t fullSendCount = 8192;
  const size_t subSendCount = 2048;
  const size_t fullTotalBytes = fullSendCount * numRanks * typeSize;
  const size_t subTotalBytes = subSendCount * numRanks * typeSize;
  // Window large enough for the full gather (at offset 0) plus a distinct
  // subset gather placed after it.
  const size_t windowBytes = fullTotalBytes + subTotalBytes;

  CtranWin* win = nullptr;
  void* winBase = createSymmetricWindow(windowBytes, &win);
  ASSERT_NE(win, nullptr);
  EXPECT_TRUE(win->isSymmetric());

  // ctwin reports supported for a recvbuf that lives inside this window.
  EXPECT_TRUE(ctranAllGatherSupport(
      ctranComm.get(),
      NCCL_ALLGATHER_ALGO::ctwin,
      stream,
      winBase,
      fullTotalBytes));

  // Full-window gather run twice: the second call must reuse the single cached
  // persistent request rather than building a new one.
  runGather(win, winBase, /*byteOffset=*/0, fullSendCount, /*iter=*/0);
  runGather(win, winBase, /*byteOffset=*/0, fullSendCount, /*iter=*/1);
  EXPECT_EQ(win->numPersistentRequests(), 1u);

  // A distinct sub-range gets its own cached request; repeating it reuses it.
  runGather(win, winBase, fullTotalBytes, subSendCount, /*iter=*/2);
  EXPECT_EQ(win->numPersistentRequests(), 2u);
  runGather(win, winBase, fullTotalBytes, subSendCount, /*iter=*/3);
  EXPECT_EQ(win->numPersistentRequests(), 2u);

  oobBarrier();

  // ctwin executes via the persistent AllGatherP machinery, so the recorded
  // algo name is "CtranAllGatherP<variant>" (not "CtranAllGatherWin"). The
  // "CtranAllGatherP" prefix matches whichever AGP variant runs and confirms
  // the ctwin path (not a fallback/other algo) executed.
  algoStats_.verify(ctranComm.get(), "AllGather", "CtranAllGatherP");

  // Free must tear down the cached persistent requests and release all imports.
  freeSymmetricWindow(win, winBase, windowBytes);
}

// A symmetric window registered WITHOUT ipc_only does the full window exchange
// at creation -- including the inter-node IB rkey exchange -- so ctwin reuses
// those rkeys (ibKeysExchanged=true) and skips the first-exec IB exchange. This
// path is exercised meaningfully on the nolocal/vnode configs (all-inter-node),
// where the IB rkeys carried in the window are actually used. Verifies a
// correct gather (with reuse) and that the ctwin (AllGatherP) path ran.
TEST_F(CtranAllgatherCtwinTest, FullExchangeSymmetricWindow) {
  if (!ncclIsCuMemSupported()) {
    GTEST_SKIP() << "CuMem not supported, skipping ctwin test";
  }
  if (ctranComm->ctran_->mapper->ctranIbPtr() == nullptr) {
    GTEST_SKIP() << "No IB Backend found, skip test";
  }

  const size_t typeSize = commTypeSize(dt);
  const size_t sendCount = 8192;
  const size_t windowBytes = sendCount * numRanks * typeSize;

  CtranWin* win = nullptr;
  // Full-exchange symmetric window (NOT ipc_only): IB rkeys are exchanged at
  // window creation.
  void* winBase = createSymmetricWindow(windowBytes, &win, /*ipcOnly=*/false);
  ASSERT_NE(win, nullptr);
  EXPECT_TRUE(win->isSymmetric());
  EXPECT_FALSE(win->isIpcOnly());

  EXPECT_TRUE(ctranAllGatherSupport(
      ctranComm.get(),
      NCCL_ALLGATHER_ALGO::ctwin,
      stream,
      winBase,
      windowBytes));

  // Run twice: the second call reuses the single cached persistent request.
  runGather(win, winBase, /*byteOffset=*/0, sendCount, /*iter=*/0);
  runGather(win, winBase, /*byteOffset=*/0, sendCount, /*iter=*/1);
  EXPECT_EQ(win->numPersistentRequests(), 1u);

  // Confirm the ctwin (AllGatherP) path actually ran.
  algoStats_.verify(ctranComm.get(), "AllGather", "CtranAllGatherP");

  oobBarrier();
  freeSymmetricWindow(win, winBase, windowBytes);
}

// Captures a ctwin allgather over the symmetric window into a CUDA graph and
// replays it several times, verifying the gathered result each replay. Because
// ctwin's persistent request is window-owned (not graph-owned), the request is
// built once during capture and reused across replays and the graph is
// destroyed before the window is freed (the required lifetime order).
TEST_F(CtranAllgatherCtwinTest, GraphCaptureReplayReuseAndFree) {
  if (!ncclIsCuMemSupported()) {
    GTEST_SKIP() << "CuMem not supported, skipping ctwin graph test";
  }
  if (ctranComm->ctran_->mapper->ctranIbPtr() == nullptr) {
    GTEST_SKIP() << "No IB Backend found, skip test";
  }

  const size_t typeSize = commTypeSize(dt);
  const size_t sendCount = 8192;
  const size_t chunkBytes = sendCount * typeSize;
  const size_t totalBytes = sendCount * numRanks * typeSize;

  CtranWin* win = nullptr;
  void* winBase = createSymmetricWindow(totalBytes, &win);
  ASSERT_NE(win, nullptr);
  EXPECT_TRUE(win->isSymmetric());

  // In-place, full-window gather: this rank's send data lives in its own chunk.
  void* recvbuf = winBase;
  void* sendbuf = static_cast<char*>(recvbuf) + globalRank * chunkBytes;

  cudaStream_t captureStream;
  CUDACHECK_TEST(
      cudaStreamCreateWithFlags(&captureStream, cudaStreamNonBlocking));

  // Seed this rank's chunk so the capture-time exec sees valid data.
  {
    const std::vector<int> seed(sendCount, globalRank);
    CUDACHECK_TEST(cudaMemset(recvbuf, 0xEE, totalBytes));
    CUDACHECK_TEST(
        cudaMemcpy(sendbuf, seed.data(), chunkBytes, cudaMemcpyDefault));
    CUDACHECK_TEST(cudaDeviceSynchronize());
  }
  oobBarrier();

  // Capture the ctwin allgather. request->stream == captureStream, so exec
  // submits directly onto the captured stream (no fork/join needed).
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graphExec = nullptr;
  ASSERT_EQ(
      cudaStreamBeginCapture(captureStream, cudaStreamCaptureModeGlobal),
      cudaSuccess);
  ASSERT_EQ(
      ctranAllGather(
          sendbuf,
          recvbuf,
          sendCount,
          dt,
          ctranComm.get(),
          captureStream,
          NCCL_ALLGATHER_ALGO::ctwin_pipeline),
      commSuccess);
  ASSERT_EQ(cudaStreamEndCapture(captureStream, &graph), cudaSuccess);
  ASSERT_NE(graph, nullptr);
  ASSERT_EQ(cudaGraphInstantiate(&graphExec, graph, 0), cudaSuccess);

  // The persistent request built during capture is cached on the window.
  EXPECT_EQ(win->numPersistentRequests(), 1u);

  constexpr int kReplays = 3;
  for (int r = 0; r < kReplays; ++r) {
    const int myVal = globalRank + r * 100;
    const std::vector<int> myChunk(sendCount, myVal);
    CUDACHECK_TEST(cudaMemset(recvbuf, 0xEE, totalBytes));
    CUDACHECK_TEST(
        cudaMemcpy(sendbuf, myChunk.data(), chunkBytes, cudaMemcpyDefault));
    CUDACHECK_TEST(cudaDeviceSynchronize());
    oobBarrier();

    ASSERT_EQ(cudaGraphLaunch(graphExec, captureStream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(captureStream), cudaSuccess);

    for (int peer = 0; peer < numRanks; ++peer) {
      std::vector<int> observed(sendCount, -1);
      CUDACHECK_TEST(cudaMemcpy(
          observed.data(),
          static_cast<char*>(recvbuf) + peer * chunkBytes,
          chunkBytes,
          cudaMemcpyDefault));
      const std::vector<int> expected(sendCount, peer + r * 100);
      EXPECT_EQ(observed, expected) << "at rank " << globalRank << " replay "
                                    << r << " chunk from peer " << peer;
    }
    // No new request is created across replays.
    EXPECT_EQ(win->numPersistentRequests(), 1u);
    oobBarrier();
  }

  // Correct lifetime order: destroy the graph BEFORE freeing the window (the
  // window owns the request that the graph captured).
  ASSERT_EQ(cudaGraphExecDestroy(graphExec), cudaSuccess);
  ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
  CUDACHECK_TEST(cudaStreamDestroy(captureStream));

  oobBarrier();
  freeSymmetricWindow(win, winBase, totalBytes);
}

// Warms up a ctwin allgather eagerly on one stream, then captures a CUDA graph
// on a DIFFERENT stream that runs two ctwin allgathers over the same window:
// region A reuses the eager warmup's range but on the capture stream, and
// region B is a distinct range. Because a window-owned persistent request binds
// its stream at creation, the cache is keyed by <offset, len, stream>: region A
// on the capture stream must get its own request (distinct from the eager one)
// so its work is captured on the capture stream instead of escaping to the
// eager stream. Each in-graph allgather is followed by a captured device
// copy-out into its own staging buffer, so every gather's result is verified
// independently on every replay.
TEST_F(CtranAllgatherCtwinTest, EagerThenGraphMultiGatherSharedWindow) {
  if (!ncclIsCuMemSupported()) {
    GTEST_SKIP() << "CuMem not supported, skipping ctwin graph test";
  }
  if (ctranComm->ctran_->mapper->ctranIbPtr() == nullptr) {
    GTEST_SKIP() << "No IB Backend found, skip test";
  }

  const size_t typeSize = commTypeSize(dt);
  const size_t sendCount = 8192;
  const size_t chunkBytes = sendCount * typeSize;
  const size_t regionBytes = sendCount * numRanks * typeSize;
  // Two distinct full-gather regions (A then B) inside a single window.
  const size_t offsetA = 0;
  const size_t offsetB = regionBytes;
  const size_t windowBytes = 2 * regionBytes;

  CtranWin* win = nullptr;
  void* winBase = createSymmetricWindow(windowBytes, &win);
  ASSERT_NE(win, nullptr);
  EXPECT_TRUE(win->isSymmetric());

  void* regionA = static_cast<char*>(winBase) + offsetA;
  void* regionB = static_cast<char*>(winBase) + offsetB;
  void* sendA = static_cast<char*>(regionA) + globalRank * chunkBytes;
  void* sendB = static_cast<char*>(regionB) + globalRank * chunkBytes;

  // Eager execution stream, distinct from the graph capture stream below.
  cudaStream_t eagerStream;
  CUDACHECK_TEST(
      cudaStreamCreateWithFlags(&eagerStream, cudaStreamNonBlocking));

  // Eager warmup over region A on eagerStream. Running it twice reuses the one
  // request cached for <offsetA, regionBytes, eagerStream> (same range + same
  // stream).
  runGatherOnStream(win, winBase, offsetA, sendCount, /*iter=*/0, eagerStream);
  runGatherOnStream(win, winBase, offsetA, sendCount, /*iter=*/1, eagerStream);
  EXPECT_EQ(win->numPersistentRequests(), 1u);

  // Per-gather device staging buffers so each in-graph gather's result is
  // captured before a later op could overwrite the window.
  void* stagingA = nullptr;
  void* stagingB = nullptr;
  CUDACHECK_TEST(cudaMalloc(&stagingA, regionBytes));
  CUDACHECK_TEST(cudaMalloc(&stagingB, regionBytes));

  // Seed both regions' send chunks so the capture-time execs see valid data
  // (capture-time results are discarded; only replays are verified).
  {
    const std::vector<int> seed(sendCount, globalRank);
    CUDACHECK_TEST(cudaMemset(regionA, 0xEE, regionBytes));
    CUDACHECK_TEST(cudaMemset(regionB, 0xEE, regionBytes));
    CUDACHECK_TEST(
        cudaMemcpy(sendA, seed.data(), chunkBytes, cudaMemcpyDefault));
    CUDACHECK_TEST(
        cudaMemcpy(sendB, seed.data(), chunkBytes, cudaMemcpyDefault));
    CUDACHECK_TEST(cudaDeviceSynchronize());
  }
  oobBarrier();

  // Capture two ctwin allgathers over the shared window on a DEDICATED capture
  // stream. Region A reuses the eager range but on captureStream, so a distinct
  // request bound to captureStream is built during capture. Region B is a
  // distinct range thus separate request.
  cudaStream_t captureStream;
  CUDACHECK_TEST(
      cudaStreamCreateWithFlags(&captureStream, cudaStreamNonBlocking));

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graphExec = nullptr;
  ASSERT_EQ(
      cudaStreamBeginCapture(captureStream, cudaStreamCaptureModeGlobal),
      cudaSuccess);
  ASSERT_EQ(
      ctranAllGather(
          sendA,
          regionA,
          sendCount,
          dt,
          ctranComm.get(),
          captureStream,
          NCCL_ALLGATHER_ALGO::ctwin_pipeline),
      commSuccess);
  CUDACHECK_TEST(cudaMemcpyAsync(
      stagingA, regionA, regionBytes, cudaMemcpyDeviceToDevice, captureStream));
  ASSERT_EQ(
      ctranAllGather(
          sendB,
          regionB,
          sendCount,
          dt,
          ctranComm.get(),
          captureStream,
          NCCL_ALLGATHER_ALGO::ctwin_pipeline),
      commSuccess);
  CUDACHECK_TEST(cudaMemcpyAsync(
      stagingB, regionB, regionBytes, cudaMemcpyDeviceToDevice, captureStream));
  ASSERT_EQ(cudaStreamEndCapture(captureStream, &graph), cudaSuccess);
  ASSERT_NE(graph, nullptr);
  ASSERT_EQ(cudaGraphInstantiate(&graphExec, graph, 0), cudaSuccess);

  // eagerStream's region-A request, plus captureStream's region-A and region-B
  // requests: three distinct <offset, len, stream> entries.
  EXPECT_EQ(win->numPersistentRequests(), 3u);

  constexpr int kReplays = 3;
  // Distinct additive bias for region B so a region mix-up is caught.
  constexpr int kRegionBBias = 50;
  for (int r = 0; r < kReplays; ++r) {
    const std::vector<int> chunkA(sendCount, globalRank + r * 100);
    const std::vector<int> chunkB(
        sendCount, globalRank + r * 100 + kRegionBBias);
    CUDACHECK_TEST(cudaMemset(regionA, 0xEE, regionBytes));
    CUDACHECK_TEST(cudaMemset(regionB, 0xEE, regionBytes));
    CUDACHECK_TEST(
        cudaMemcpy(sendA, chunkA.data(), chunkBytes, cudaMemcpyDefault));
    CUDACHECK_TEST(
        cudaMemcpy(sendB, chunkB.data(), chunkBytes, cudaMemcpyDefault));
    CUDACHECK_TEST(cudaDeviceSynchronize());
    oobBarrier();

    ASSERT_EQ(cudaGraphLaunch(graphExec, captureStream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(captureStream), cudaSuccess);

    for (int peer = 0; peer < numRanks; ++peer) {
      std::vector<int> observedA(sendCount, -1);
      std::vector<int> observedB(sendCount, -1);
      CUDACHECK_TEST(cudaMemcpy(
          observedA.data(),
          static_cast<char*>(stagingA) + peer * chunkBytes,
          chunkBytes,
          cudaMemcpyDefault));
      CUDACHECK_TEST(cudaMemcpy(
          observedB.data(),
          static_cast<char*>(stagingB) + peer * chunkBytes,
          chunkBytes,
          cudaMemcpyDefault));
      const std::vector<int> expectedA(sendCount, peer + r * 100);
      const std::vector<int> expectedB(
          sendCount, peer + r * 100 + kRegionBBias);
      EXPECT_EQ(observedA, expectedA) << "region A at rank " << globalRank
                                      << " replay " << r << " peer " << peer;
      EXPECT_EQ(observedB, expectedB) << "region B at rank " << globalRank
                                      << " replay " << r << " peer " << peer;
    }
    // No new requests are created across replays.
    EXPECT_EQ(win->numPersistentRequests(), 3u);
    oobBarrier();
  }

  // Correct lifetime order: destroy the graph BEFORE freeing the window (the
  // window owns the requests the graph captured).
  ASSERT_EQ(cudaGraphExecDestroy(graphExec), cudaSuccess);
  ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
  CUDACHECK_TEST(cudaFree(stagingA));
  CUDACHECK_TEST(cudaFree(stagingB));
  CUDACHECK_TEST(cudaStreamDestroy(captureStream));
  CUDACHECK_TEST(cudaStreamDestroy(eagerStream));

  oobBarrier();
  freeSymmetricWindow(win, winBase, windowBytes);
}

// Plain `ctwin` auto-selects by topology: at nLocalRanks>1 it uses the
// persistent AGP path (which caches a window request); at nLocalRanks==1 it
// routes to the dedicated ring/streamed-RD (no persistent request cached).
// Result correctness is checked by runGatherOnStream in both cases.
TEST_F(CtranAllgatherCtwinTest, AutoSelectByTopology) {
  if (!ncclIsCuMemSupported()) {
    GTEST_SKIP() << "CuMem not supported, skipping ctwin test";
  }
  if (ctranComm->ctran_->mapper->ctranIbPtr() == nullptr) {
    GTEST_SKIP() << "No IB Backend found, skip test";
  }

  const size_t typeSize = commTypeSize(dt);
  const size_t sendCount = 8192;
  const size_t windowBytes = sendCount * numRanks * typeSize;

  CtranWin* win = nullptr;
  void* winBase = createSymmetricWindow(windowBytes, &win);
  ASSERT_NE(win, nullptr);
  EXPECT_TRUE(win->isSymmetric());

  runGatherOnStream(
      win,
      winBase,
      /*byteOffset=*/0,
      sendCount,
      /*iter=*/0,
      stream,
      NCCL_ALLGATHER_ALGO::ctwin);

  if (ctranComm->statex_->nLocalRanks() > 1) {
    // Persistent AGP path caches exactly one request for this range/stream.
    EXPECT_EQ(win->numPersistentRequests(), 1u);
    algoStats_.verify(ctranComm.get(), "AllGather", "CtranAllGatherP");
  } else {
    // Dedicated path caches no persistent request; small message + power-of-2
    // nRanks selects streamed recursive-doubling (ctsrd).
    EXPECT_EQ(win->numPersistentRequests(), 0u);
    algoStats_.verify(ctranComm.get(), "AllGather", "CtranAllGatherStreamedRd");
  }

  oobBarrier();
  freeSymmetricWindow(win, winBase, windowBytes);
}

// Forced ctwin_* variants: the two persistent ones (ctwin_pipeline,
// ctwin_rdpipeline) each cache a window request; the two dedicated ones
// (ctwin_ring, ctwin_srd, valid only at nLocalRanks==1) cache none. Each call
// verifies its gather result via runGatherOnStream.
TEST_F(CtranAllgatherCtwinTest, ForceVariants) {
  if (!ncclIsCuMemSupported()) {
    GTEST_SKIP() << "CuMem not supported, skipping ctwin test";
  }
  if (ctranComm->ctran_->mapper->ctranIbPtr() == nullptr) {
    GTEST_SKIP() << "No IB Backend found, skip test";
  }

  const size_t typeSize = commTypeSize(dt);
  const size_t sendCount = 8192;
  const size_t regionBytes = sendCount * numRanks * typeSize;
  const size_t windowBytes = 4 * regionBytes;

  CtranWin* win = nullptr;
  void* winBase = createSymmetricWindow(windowBytes, &win);
  ASSERT_NE(win, nullptr);
  EXPECT_TRUE(win->isSymmetric());

  // Persistent forced variants: each caches a distinct window request.
  runGatherOnStream(
      win,
      winBase,
      /*byteOffset=*/0,
      sendCount,
      /*iter=*/0,
      stream,
      NCCL_ALLGATHER_ALGO::ctwin_pipeline);
  runGatherOnStream(
      win,
      winBase,
      /*byteOffset=*/regionBytes,
      sendCount,
      /*iter=*/1,
      stream,
      NCCL_ALLGATHER_ALGO::ctwin_rdpipeline);
  EXPECT_EQ(win->numPersistentRequests(), 2u);

  // Dedicated forced variants are valid only at nLocalRanks==1; they cache no
  // persistent request.
  if (ctranComm->statex_->nLocalRanks() == 1) {
    runGatherOnStream(
        win,
        winBase,
        /*byteOffset=*/2 * regionBytes,
        sendCount,
        /*iter=*/2,
        stream,
        NCCL_ALLGATHER_ALGO::ctwin_ring);
    runGatherOnStream(
        win,
        winBase,
        /*byteOffset=*/3 * regionBytes,
        sendCount,
        /*iter=*/3,
        stream,
        NCCL_ALLGATHER_ALGO::ctwin_srd);
    EXPECT_EQ(win->numPersistentRequests(), 2u);
  }

  oobBarrier();
  freeSymmetricWindow(win, winBase, windowBytes);
}

// A/B for NCCL_CTRAN_AGP_SKIP_MC_INTRA_BARRIER, both settings in one job so a
// single launch covers the knob on and off.
//
// The knob gates only the per-step intra-node NVL broadcast barrier, and only
// where the multicast write is engaged. Reaching that branch needs BOTH:
//   - nLocalRanks > 1, else nvlCeBcast is never called at all; and
//   - nNodes > 1, else the loops holding the gated call run zero iterations
//     (ctpipeline is bounded by nNodes-1, ctsrdpipeline by log2(nNodes)).
// A single-node config therefore cannot exercise it, so skip loudly rather than
// pass vacuously. Reachable topologies include the vnode configs (which fake
// nNodes>1 on one host) and real multinode runs.
class CtranAllgatherCtwinMcBarrierTest
    : public CtranAllgatherCtwinTest,
      public ::testing::WithParamInterface<bool> {};

TEST_P(CtranAllgatherCtwinMcBarrierTest, PerStepBarrierSkip) {
  if (!ncclIsCuMemSupported()) {
    GTEST_SKIP() << "CuMem not supported, skipping ctwin test";
  }
  if (ctranComm->ctran_->mapper->ctranIbPtr() == nullptr) {
    GTEST_SKIP() << "No IB Backend found, skip test";
  }
  const auto nLocalRanks = ctranComm->statex_->nLocalRanks();
  const auto nNodes = ctranComm->statex_->nNodes();
  if (nLocalRanks < 2 || nNodes < 2) {
    GTEST_SKIP() << "the gated per-step barrier needs nLocalRanks>1 and "
                 << "nNodes>1 to be reached; got nLocalRanks=" << nLocalRanks
                 << " nNodes=" << nNodes;
  }

  // Read fresh inside each exec, so flipping it here applies to the gathers
  // below even though the window caches the AGP request across calls. Every
  // rank runs the same parameterized case in the same order, so the local ranks
  // agree on whether the device-side barrier is emitted -- they must, or a rank
  // that still emits it would wait on peers that skipped it.
  EnvRAII<bool> skipBarrier(NCCL_CTRAN_AGP_SKIP_MC_INTRA_BARRIER, GetParam());

  const size_t typeSize = commTypeSize(dt);
  const size_t sendCount = 8192;
  const size_t regionBytes = sendCount * numRanks * typeSize;
  const size_t windowBytes = 2 * regionBytes;

  CtranWin* win = nullptr;
  void* winBase = createSymmetricWindow(windowBytes, &win);
  ASSERT_NE(win, nullptr);
  EXPECT_TRUE(win->isSymmetric());

  // Both variants that gate the barrier on this cvar, on disjoint sub-ranges so
  // each gets its own cached persistent request.
  runGatherOnStream(
      win,
      winBase,
      /*byteOffset=*/0,
      sendCount,
      /*iter=*/0,
      stream,
      NCCL_ALLGATHER_ALGO::ctwin_pipeline);
  runGatherOnStream(
      win,
      winBase,
      /*byteOffset=*/regionBytes,
      sendCount,
      /*iter=*/1,
      stream,
      NCCL_ALLGATHER_ALGO::ctwin_rdpipeline);
  EXPECT_EQ(win->numPersistentRequests(), 2u);

  oobBarrier();
  freeSymmetricWindow(win, winBase, windowBytes);
}

INSTANTIATE_TEST_SUITE_P(
    CtranTest,
    CtranAllgatherCtwinMcBarrierTest,
    ::testing::Bool(),
    [](const ::testing::TestParamInfo<bool>& info) {
      return info.param ? "SkipMcIntraBarrier" : "KeepMcIntraBarrier";
    });

// Asserts the recorded topology of the captured GPE HOST nodes with and without
// NCCL_CTRAN_GPE_HOST_NODE_SIDE_STREAM. The knob's whole effect is a change in
// where the HOST node is recorded, so the assertions are on graph edges rather
// than on numerics:
//
//   inline (knob off): every HOST is a pass-through link (out-degree 1) and no
//                      HOST depends on another -> the driver is free to spread
//                      the graph's internal streams across hardware channels.
//   spine  (knob on):  HOST[i] depends on HOST[i-1] AND parents the
//   collective's
//                      first node, so out-degree reaches 2 and there are
//                      exactly N-1 HOST -> HOST edges.
//
// The (N-1) HOST -> HOST count is the direct fingerprint of the serial spine.
// Note the spine's cudaEventRecord is ABSORBED by capture into a plain
// dependency edge rather than becoming a standalone EVENT_RECORD node, so the
// edge is HOST -> HOST directly (measured; this is the same absorption that
// mixing=0's fence relies on).
class CtranAllgatherCtwinHostSpineTest
    : public CtranAllgatherCtwinTest,
      public ::testing::WithParamInterface<bool> {
 public:
  void SetUp() override {
    // Must be set before the comm is constructed: CtranGpe::Impl caches both
    // knobs in its constructor.
    setenv("NCCL_CTRAN_GRAPH_MIXING_SUPPORT", "0", 1);
    setenv("NCCL_CTRAN_GPE_HOST_NODE_SIDE_STREAM", GetParam() ? "1" : "0", 1);
    CtranAllgatherCtwinTest::SetUp();
  }

  void TearDown() override {
    unsetenv("NCCL_CTRAN_GPE_HOST_NODE_SIDE_STREAM");
    unsetenv("NCCL_CTRAN_GRAPH_MIXING_SUPPORT");
    CtranAllgatherCtwinTest::TearDown();
  }
};

TEST_P(CtranAllgatherCtwinHostSpineTest, HostNodeWiring) {
  if (!ncclIsCuMemSupported()) {
    GTEST_SKIP() << "CuMem not supported, skipping ctwin graph test";
  }
  if (ctranComm->ctran_->mapper->ctranIbPtr() == nullptr) {
    GTEST_SKIP() << "No IB Backend found, skip test";
  }
  // The GPE host node comes from the PipeStart submit, which ctpipeline makes
  // only when nNodes > 1; a single-node config emits no host node at all, so
  // both arms would assert on an empty set. Skip loudly rather than vacuously.
  const auto nNodes = ctranComm->statex_->nNodes();
  if (nNodes < 2) {
    GTEST_SKIP() << "the captured GPE host node needs nNodes>1 to be emitted; "
                 << "got nNodes=" << nNodes;
  }
  const bool spine = GetParam();

  const size_t typeSize = commTypeSize(dt);
  const size_t sendCount = 8192;
  const size_t chunkBytes = sendCount * typeSize;
  const size_t totalBytes = sendCount * numRanks * typeSize;

  CtranWin* win = nullptr;
  void* winBase = createSymmetricWindow(totalBytes, &win);
  ASSERT_NE(win, nullptr);

  void* recvbuf = winBase;
  void* sendbuf = static_cast<char*>(recvbuf) + globalRank * chunkBytes;

  cudaStream_t captureStream;
  CUDACHECK_TEST(
      cudaStreamCreateWithFlags(&captureStream, cudaStreamNonBlocking));
  {
    const std::vector<int> seed(sendCount, globalRank);
    CUDACHECK_TEST(cudaMemset(recvbuf, 0xEE, totalBytes));
    CUDACHECK_TEST(
        cudaMemcpy(sendbuf, seed.data(), chunkBytes, cudaMemcpyDefault));
    CUDACHECK_TEST(cudaDeviceSynchronize());
  }
  oobBarrier();

  // Several collectives back to back: the spine only becomes visible across
  // collectives (HOST[i] <- HOST[i-1]), so a single-collective capture cannot
  // distinguish the two shapes.
  constexpr int kNumCollectives = 4;
  cudaGraph_t graph = nullptr;
  ASSERT_EQ(
      cudaStreamBeginCapture(captureStream, cudaStreamCaptureModeGlobal),
      cudaSuccess);
  for (int i = 0; i < kNumCollectives; ++i) {
    ASSERT_EQ(
        ctranAllGather(
            sendbuf,
            recvbuf,
            sendCount,
            dt,
            ctranComm.get(),
            captureStream,
            NCCL_ALLGATHER_ALGO::ctwin_pipeline),
        commSuccess);
  }
  ASSERT_EQ(cudaStreamEndCapture(captureStream, &graph), cudaSuccess);
  ASSERT_NE(graph, nullptr);

  // Walk the recorded graph.
  size_t numNodes = 0;
  ASSERT_EQ(cudaGraphGetNodes(graph, nullptr, &numNodes), cudaSuccess);
  std::vector<cudaGraphNode_t> nodes(numNodes);
  ASSERT_EQ(cudaGraphGetNodes(graph, nodes.data(), &numNodes), cudaSuccess);

  size_t numEdges = 0;
  ASSERT_EQ(cudaGraphGetEdges(graph, nullptr, nullptr, &numEdges), cudaSuccess);
  std::vector<cudaGraphNode_t> from(numEdges), to(numEdges);
  ASSERT_EQ(
      cudaGraphGetEdges(graph, from.data(), to.data(), &numEdges), cudaSuccess);

  std::unordered_map<cudaGraphNode_t, cudaGraphNodeType> kind;
  for (cudaGraphNode_t n : nodes) {
    cudaGraphNodeType t{};
    ASSERT_EQ(cudaGraphNodeGetType(n, &t), cudaSuccess);
    kind[n] = t;
  }
  std::unordered_map<cudaGraphNode_t, int> outDeg, inDeg;
  int hostToRecord = 0, recordToHost = 0, hostToHost = 0;
  for (size_t e = 0; e < numEdges; ++e) {
    outDeg[from[e]]++;
    inDeg[to[e]]++;
    const bool fromHost = kind[from[e]] == cudaGraphNodeTypeHost;
    const bool toHost = kind[to[e]] == cudaGraphNodeTypeHost;
    const bool fromRec = kind[from[e]] == cudaGraphNodeTypeEventRecord;
    const bool toRec = kind[to[e]] == cudaGraphNodeTypeEventRecord;
    if (fromHost && toRec) {
      hostToRecord++;
    }
    if (fromRec && toHost) {
      recordToHost++;
    }
    if (fromHost && toHost) {
      hostToHost++;
    }
  }
  std::vector<cudaGraphNode_t> hosts;
  for (cudaGraphNode_t n : nodes) {
    if (kind[n] == cudaGraphNodeTypeHost) {
      hosts.push_back(n);
    }
  }

  // One GPE host node per collective either way -- the knob moves the node, it
  // does not add or remove any.
  ASSERT_EQ(hosts.size(), static_cast<size_t>(kNumCollectives));
  int minIn = 1 << 30, maxIn = 0, minOut = 1 << 30, maxOut = 0;
  for (cudaGraphNode_t h : hosts) {
    minIn = std::min(minIn, inDeg[h]);
    maxIn = std::max(maxIn, inDeg[h]);
    minOut = std::min(minOut, outDeg[h]);
    maxOut = std::max(maxOut, outDeg[h]);
  }
  if (globalRank == 0) {
    fprintf(
        stderr,
        "[hostspine] spine=%d nodes=%zu edges=%zu hosts=%zu "
        "inDeg=[%d,%d] outDeg=[%d,%d] host->rec=%d rec->host=%d host->host=%d\n",
        spine ? 1 : 0,
        numNodes,
        numEdges,
        hosts.size(),
        minIn,
        maxIn,
        minOut,
        maxOut,
        hostToRecord,
        recordToHost,
        hostToHost);
  }
  if (spine) {
    // Each HOST feeds its own EVENT_RECORD (the spine tip) and the collective's
    // first node; every HOST but the first is fed by the previous tip.
    EXPECT_EQ(hostToHost, kNumCollectives - 1)
        << "serial host spine: HOST[i] depends on HOST[i-1]";
    EXPECT_EQ(maxOut, 2)
        << "a spine HOST feeds both the next HOST and its collective";
  } else {
    EXPECT_EQ(hostToHost, 0) << "inline: host nodes are mutually independent";
    EXPECT_EQ(maxOut, 1) << "inline HOST is a pass-through link";
  }

  // The graph must still replay correctly in both shapes.
  cudaGraphExec_t graphExec = nullptr;
  ASSERT_EQ(cudaGraphInstantiate(&graphExec, graph, 0), cudaSuccess);
  const std::vector<int> myChunk(sendCount, globalRank + 7);
  CUDACHECK_TEST(cudaMemset(recvbuf, 0xEE, totalBytes));
  CUDACHECK_TEST(
      cudaMemcpy(sendbuf, myChunk.data(), chunkBytes, cudaMemcpyDefault));
  CUDACHECK_TEST(cudaDeviceSynchronize());
  oobBarrier();
  ASSERT_EQ(cudaGraphLaunch(graphExec, captureStream), cudaSuccess);
  ASSERT_EQ(cudaStreamSynchronize(captureStream), cudaSuccess);
  for (int peer = 0; peer < numRanks; ++peer) {
    std::vector<int> observed(sendCount, -1);
    CUDACHECK_TEST(cudaMemcpy(
        observed.data(),
        static_cast<char*>(recvbuf) + peer * chunkBytes,
        chunkBytes,
        cudaMemcpyDefault));
    EXPECT_EQ(observed, std::vector<int>(sendCount, peer + 7))
        << "chunk from peer " << peer;
  }

  ASSERT_EQ(cudaGraphExecDestroy(graphExec), cudaSuccess);
  ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
  CUDACHECK_TEST(cudaStreamDestroy(captureStream));
  oobBarrier();
  freeSymmetricWindow(win, winBase, totalBytes);
}

INSTANTIATE_TEST_SUITE_P(
    CtranTest,
    CtranAllgatherCtwinHostSpineTest,
    ::testing::Bool(),
    [](const ::testing::TestParamInfo<bool>& info) {
      return info.param ? "HostSpine" : "HostInline";
    });

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new ctran::CtranDistEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
