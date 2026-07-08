// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <stdlib.h>

#include "CtranUtUtils.h"
#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/RMA/WaitSignalImpl.h"
#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsCuUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

using namespace ctran;

class CtranRMATest : public ctran::CtranDistTestFixture, public CtranBaseTest {
 public:
  CtranRMATest() = default;

 protected:
  std::unique_ptr<CtranComm> ctranComm{nullptr};

  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE", "1", 0);
    setenv("NCCL_CTRAN_IB_EPOCH_LOCK_ENFORCE_CHECK", "true", 0);
#ifdef CTRAN_TEST_IB_ONLY_BACKEND
    setenv("NCCL_CTRAN_BACKENDS", "ib", 1);
#endif
    ctran::CtranDistTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(this->localRank));
    ctranComm = makeCtranComm();
    ASSERT_NE(ctranComm.get(), nullptr);
  }
  void TearDown() override {
    if (ctranComm) {
      ctran::waitForCollTraceDrain(ctranComm.get());
    }
    ctranComm.reset();
    ctran::CtranDistTestFixture::TearDown();
    // Check that all allocated memory segments have been freed
    EXPECT_TRUE(segments.empty()) << "Not all memory segments were freed";
  }

  void barrier() {
    CUDACHECK_TEST(cudaDeviceSynchronize());
    oobBarrier();
  }

  void createWin(
      bool isUserBuf,
      MemAllocType bufType,
      void** winBasePtr,
      CtranWin** winPtr,
      size_t sizeBytes,
      meta::comms::Hints hints) {
    auto res = commSuccess;
    // If userBuf is true, allocate buffer and use ctranWinRegister API
    if (isUserBuf) {
      *winBasePtr = commMemAlloc(sizeBytes, bufType, segments);
      res = ctranWinRegister(
          (void*)*winBasePtr, sizeBytes, ctranComm.get(), winPtr, hints);

    } else {
      hints.set(
          "window_buffer_location",
          bufType == MemAllocType::kMemHostManaged ||
                  bufType == MemAllocType::kMemHostUnregistered
              ? "cpu"
              : "gpu");
      res = ctranWinAllocate(
          sizeBytes, ctranComm.get(), (void**)winBasePtr, winPtr, hints);
    }
    ASSERT_EQ(res, commSuccess);
    ASSERT_NE(*winBasePtr, nullptr);
  }

  void
  freeWinBuf(bool isUserBuf, void* ptr, size_t size, MemAllocType bufType) {
    if (isUserBuf) {
      commMemFree(ptr, size, bufType);
      if (bufType == MemAllocType::kCuMemAllocDisjoint) {
        // Disjoint allocations create multiple sub-segment entries in the
        // tracking vector. Clear all since commMemFree handles the actual free.
        segments.clear();
      } else {
        segments.erase(
            std::remove_if(
                segments.begin(),
                segments.end(),
                [ptr](const TestMemSegment& seg) { return seg.ptr == ptr; }),
            segments.end());
      }
    }
  }
  std::vector<TestMemSegment> segments;
};

class MultiWindowTestParam
    : public CtranRMATest,
      public ::testing::WithParamInterface<std::tuple<size_t, size_t>> {};

TEST_P(MultiWindowTestParam, multiWindow) {
  const auto& [kMaxNumElements, kNumIters] = GetParam();
  EXPECT_GE(kMaxNumElements, 1);
  EXPECT_GE(kNumIters, 1);

  auto statex = ctranComm->statex_.get();
  ASSERT_NE(statex, nullptr);
  EXPECT_EQ(statex->nRanks(), this->numRanks);

  for (size_t numElements = 1; numElements <= kMaxNumElements;
       numElements = numElements * 2) {
    cudaStream_t put_stream, wait_stream;
    CUDACHECK_TEST(
        cudaStreamCreateWithFlags(&put_stream, cudaStreamNonBlocking));
    CUDACHECK_TEST(
        cudaStreamCreateWithFlags(&wait_stream, cudaStreamNonBlocking));

    size_t sizeBytes = numElements * sizeof(int) * statex->nRanks();
    CtranWin* win = nullptr;
    void* winBase = nullptr;
    auto res = ctranWinAllocate(sizeBytes, ctranComm.get(), &winBase, &win);
    ASSERT_EQ(res, commSuccess);
    ASSERT_NE(winBase, nullptr);
    int* localbuf = reinterpret_cast<int*>(winBase);

    EXPECT_THAT(win, ::testing::NotNull());
    // localbuf: range from [myrank, myrank + numElements * numRanks]
    assignChunkValue(
        localbuf, numElements * statex->nRanks(), statex->rank(), 1);
    // Barrier to ensure all peers have finished creation
    barrier();

    int nextPeer = (this->globalRank + 1) % this->numRanks;
    int prevPeer = (this->globalRank + this->numRanks - 1) % this->numRanks;

    for (auto iter = 0; iter < kNumIters; iter++) {
      COMMCHECK_TEST(ctranPutSignal(
          localbuf + numElements * statex->rank(),
          numElements,
          commInt32,
          nextPeer,
          numElements * statex->rank(),
          win,
          put_stream,
          true));
      COMMCHECK_TEST(ctranWaitSignal(prevPeer, win, wait_stream));
    }
    // Barrier to ensure all peers have finished put
    barrier();

    // Check results
    size_t errs = checkChunkValue(
        localbuf + numElements * prevPeer,
        numElements,
        prevPeer + (int)numElements * prevPeer,
        1,
        this->globalRank,
        wait_stream);
    EXPECT_EQ(errs, 0);

    res = ctranWinFree(win);
    EXPECT_EQ(res, commSuccess);

    CUDACHECK_TEST(cudaStreamDestroy(put_stream));
    CUDACHECK_TEST(cudaStreamDestroy(wait_stream));
  }
}

class CtranRMATestParam
    : public CtranRMATest,
      public ::testing::WithParamInterface<
          std::tuple<size_t, size_t, bool, MemAllocType, bool>> {
 protected:
  void SetUp() override {
#ifdef CTRAN_TEST_IB_ONLY_BACKEND
    const auto ctranAllReduce = std::get<2>(GetParam());
    if (ctranAllReduce) {
      GTEST_SKIP() << "IB-only mode: NVL not available for ctranAllReduce";
    }
#endif
    CtranRMATest::SetUp();
  }
};

TEST_P(CtranRMATestParam, winPutWait) {
  const auto& [kNumElements, kNumIters, ctranAllReduce, bufType, userBuf] =
      GetParam();
  EXPECT_GE(kNumElements, 8192);
  EXPECT_GE(kNumIters, 1);
  if (!ctranComm->ctran_->mapper->hasBackend(
          globalRank, CtranMapperBackend::NVL) &&
      ctranAllReduce) {
    GTEST_SKIP()
        << "NVL not enabled, which is required for ctranAllReduce. Skip test";
  }

  auto statex = ctranComm->statex_.get();
  ASSERT_NE(statex, nullptr);
  EXPECT_EQ(statex->nRanks(), this->numRanks);

  cudaStream_t put_stream, wait_stream;
  cudaEvent_t start_event, end_event;
  CUDACHECK_TEST(cudaStreamCreateWithFlags(&put_stream, cudaStreamNonBlocking));
  CUDACHECK_TEST(
      cudaStreamCreateWithFlags(&wait_stream, cudaStreamNonBlocking));
  CUDACHECK_TEST(cudaEventCreate(&start_event));
  CUDACHECK_TEST(cudaEventCreate(&end_event));

  size_t sizeBytes = kNumElements * sizeof(int) * statex->nRanks();

  meta::comms::Hints hints;
  CtranWin* win = nullptr;
  void* winBase = nullptr;
  createWin(userBuf, bufType, &winBase, &win, sizeBytes, hints);

  // Allocate localBuf from GPU mem for allreduce usage
  int* localBuf = nullptr;
  size_t localBufBytes = kNumElements * sizeof(int);
  CUDACHECK_TEST(cudaMalloc(&localBuf, localBufBytes));
  COMMCHECK_TEST(ctran::globalRegisterWithPtr(localBuf, localBufBytes));

  EXPECT_THAT(win, ::testing::NotNull());

  for (int peer = 0; peer < this->numRanks; peer++) {
    void* remoteAddr = nullptr;
    auto res = ctranWinSharedQuery(peer, win, &remoteAddr);
    EXPECT_EQ(res, commSuccess);
    if (peer == statex->rank()) {
      EXPECT_EQ(remoteAddr, winBase);
    } else if (!win->nvlEnabled(peer)) {
      EXPECT_THAT(remoteAddr, ::testing::IsNull());
    } else {
      EXPECT_THAT(remoteAddr, ::testing::NotNull());
    }
  }

  assignChunkValue((int*)winBase, kNumElements * statex->nRanks(), -1, 0);
  assignChunkValue(localBuf, kNumElements, statex->rank(), 1);
  // Barrier to ensure all peers have finished value assignment
  barrier();

  int nextPeer = (this->globalRank + 1) % this->numRanks;
  int prevPeer = (this->globalRank + this->numRanks - 1) % this->numRanks;

  for (auto iter = 0; iter < kNumIters; iter++) {
    COMMCHECK_TEST(ctranPutSignal(
        localBuf,
        kNumElements,
        commInt32,
        nextPeer,
        kNumElements * statex->rank(),
        win,
        put_stream,
        true));
    COMMCHECK_TEST(ctranWaitSignal(prevPeer, win, wait_stream));
    if (iter == 0) {
      // Skip first iteration to avoid any warmup overhead
      CUDACHECK_TEST(cudaEventRecord(start_event, put_stream));
    }
  }
  CUDACHECK_TEST(cudaEventRecord(end_event, put_stream));

  size_t errs = checkChunkValue(
      (int*)winBase + kNumElements * prevPeer,
      kNumElements,
      prevPeer,
      1,
      this->globalRank,
      wait_stream);

  CUDACHECK_TEST(cudaStreamSynchronize(put_stream));
  CUDACHECK_TEST(cudaStreamSynchronize(wait_stream));

  size_t chunkBytes = kNumElements * sizeof(int);
  if (chunkBytes > 0) {
    float elapsed_time_ms = -1.0;
    CUDACHECK_TEST(
        cudaEventElapsedTime(&elapsed_time_ms, start_event, end_event));
    // time captured with kNumIters - 1 iterations
    float achieved_bw = chunkBytes / elapsed_time_ms / 1e6 * (kNumIters - 1);
    XLOGF(
        INFO,
        "[%d] elapsed time %.2f ms for %zu bytes * %ld iterations (%.2f GB/s), on %s\n",
        statex->rank(),
        elapsed_time_ms,
        chunkBytes,
        kNumIters,
        achieved_bw,
        testMemAllocTypeToStr(bufType));
  }

  auto res = ctranWinFree(win);
  EXPECT_EQ(res, commSuccess);

  COMMCHECK_TEST(ctran::globalDeregisterWithPtr(localBuf, localBufBytes));
  CUDACHECK_TEST(cudaFree(localBuf));
  freeWinBuf(userBuf, winBase, sizeBytes, bufType);

  CUDACHECK_TEST(cudaEventDestroy(start_event));
  CUDACHECK_TEST(cudaEventDestroy(end_event));
  CUDACHECK_TEST(cudaStreamDestroy(put_stream));
  CUDACHECK_TEST(cudaStreamDestroy(wait_stream));

  EXPECT_EQ(errs, 0);
}

TEST_P(CtranRMATestParam, winWaitPut) {
  const auto& [kNumElements, kNumIters, ctranAllReduce, bufType, userBuf] =
      GetParam();
  EXPECT_GE(kNumElements, 8192);
  EXPECT_GE(kNumIters, 1);
  if (bufType != MemAllocType::kMemCuMemAlloc &&
      bufType != MemAllocType::kMemCudaMalloc) {
    GTEST_SKIP() << "Only support GPU memory for winWaitPut";
  }

  auto statex = ctranComm->statex_.get();
  ASSERT_NE(statex, nullptr);
  EXPECT_EQ(statex->nRanks(), this->numRanks);

  cudaStream_t put_stream, wait_stream;
  cudaEvent_t start_event, end_event;
  CUDACHECK_TEST(cudaStreamCreateWithFlags(&put_stream, cudaStreamNonBlocking));
  CUDACHECK_TEST(
      cudaStreamCreateWithFlags(&wait_stream, cudaStreamNonBlocking));
  CUDACHECK_TEST(cudaEventCreate(&start_event));
  CUDACHECK_TEST(cudaEventCreate(&end_event));

  size_t sizeBytes = kNumElements * sizeof(int) * statex->nRanks();

  meta::comms::Hints hints;
  CtranWin* win = nullptr;
  void* winBase = nullptr;
  createWin(userBuf, bufType, &winBase, &win, sizeBytes, hints);

  // Allocate localBuf from GPU mem for allreduce usage
  int* localBuf = nullptr;
  size_t localBufBytes = kNumElements * sizeof(int);
  CUDACHECK_TEST(cudaMalloc(&localBuf, localBufBytes));
  COMMCHECK_TEST(ctran::globalRegisterWithPtr(localBuf, localBufBytes));

  EXPECT_THAT(win, ::testing::NotNull());

  for (int peer = 0; peer < this->numRanks; peer++) {
    void* remoteAddr = nullptr;
    auto res = ctranWinSharedQuery(peer, win, &remoteAddr);
    EXPECT_EQ(res, commSuccess);
    if (peer == statex->rank()) {
      EXPECT_EQ(remoteAddr, winBase);
    } else if (!win->nvlEnabled(peer)) {
      EXPECT_THAT(remoteAddr, ::testing::IsNull());
    } else {
      EXPECT_THAT(remoteAddr, ::testing::NotNull());
    }
  }

  assignChunkValue((int*)winBase, kNumElements * statex->nRanks(), -1, 0);
  assignChunkValue(localBuf, kNumElements, statex->rank(), 1);
  // Barrier to ensure all peers have finished value assignment
  barrier();

  int nextPeer = (this->globalRank + 1) % this->numRanks;
  int prevPeer = (this->globalRank + this->numRanks - 1) % this->numRanks;

  for (auto iter = 0; iter < kNumIters; iter++) {
    COMMCHECK_TEST(waitSignalSpinningKernel(
        prevPeer,
        win,
        wait_stream,
        win->updateOpCount(prevPeer, window::OpCountType::kWaitSignal)));
    COMMCHECK_TEST(ctranPutSignal(
        localBuf,
        kNumElements,
        commInt32,
        nextPeer,
        kNumElements * statex->rank(),
        win,
        put_stream,
        true));
    if (iter == 0) {
      // Skip first iteration to avoid any warmup overhead
      CUDACHECK_TEST(cudaEventRecord(start_event, put_stream));
    }
  }
  CUDACHECK_TEST(cudaEventRecord(end_event, put_stream));

  size_t errs = checkChunkValue(
      (int*)winBase + kNumElements * prevPeer,
      kNumElements,
      prevPeer,
      1,
      this->globalRank,
      wait_stream);

  CUDACHECK_TEST(cudaStreamSynchronize(put_stream));
  CUDACHECK_TEST(cudaStreamSynchronize(wait_stream));

  size_t chunkBytes = kNumElements * sizeof(int);
  if (chunkBytes > 0) {
    float elapsed_time_ms = -1.0;
    CUDACHECK_TEST(
        cudaEventElapsedTime(&elapsed_time_ms, start_event, end_event));
    // time captured with kNumIters - 1 iterations
    float achieved_bw = chunkBytes / elapsed_time_ms / 1e6 * (kNumIters - 1);
    XLOGF(
        INFO,
        "[%d] elapsed time %.2f ms for %zu bytes * %ld iterations (%.2f GB/s), on %s\n",
        statex->rank(),
        elapsed_time_ms,
        chunkBytes,
        kNumIters,
        achieved_bw,
        testMemAllocTypeToStr(bufType));
  }

  auto res = ctranWinFree(win);
  EXPECT_EQ(res, commSuccess);

  COMMCHECK_TEST(ctran::globalDeregisterWithPtr(localBuf, localBufBytes));
  CUDACHECK_TEST(cudaFree(localBuf));
  freeWinBuf(userBuf, winBase, sizeBytes, bufType);

  CUDACHECK_TEST(cudaEventDestroy(start_event));
  CUDACHECK_TEST(cudaEventDestroy(end_event));
  CUDACHECK_TEST(cudaStreamDestroy(put_stream));
  CUDACHECK_TEST(cudaStreamDestroy(wait_stream));

  EXPECT_EQ(errs, 0);
}

TEST_P(CtranRMATestParam, winPutOnly) {
  const auto& [kNumElements, kNumIters, ctranAllReduce, bufType, userBuf] =
      GetParam();
  EXPECT_GE(kNumElements, 8192);
  EXPECT_GE(kNumIters, 1);
  if (!ctranComm->ctran_->mapper->hasBackend(
          globalRank, CtranMapperBackend::NVL) &&
      ctranAllReduce) {
    GTEST_SKIP()
        << "NVL not enabled, which is required for ctranAllReduce. Skip test";
  }

  auto statex = ctranComm->statex_.get();
  ASSERT_NE(statex, nullptr);
  EXPECT_EQ(statex->nRanks(), this->numRanks);

  cudaStream_t put_stream;
  CUDACHECK_TEST(cudaStreamCreateWithFlags(&put_stream, cudaStreamNonBlocking));

  size_t sizeBytes = kNumElements * sizeof(int) * statex->nRanks();

  meta::comms::Hints hints;
  CtranWin* win = nullptr;
  void* winBase = nullptr;
  createWin(userBuf, bufType, &winBase, &win, sizeBytes, hints);

  // Allocate localBuf from GPU mem
  int* localBuf = nullptr;
  size_t localBufBytes = kNumElements * sizeof(int);
  CUDACHECK_TEST(cudaMalloc(&localBuf, localBufBytes));
  COMMCHECK_TEST(ctran::globalRegisterWithPtr(localBuf, localBufBytes));

  assignChunkValue((int*)winBase, kNumElements * statex->nRanks(), -1, 0);
  assignChunkValue(localBuf, kNumElements, statex->rank(), 1);
  // Barrier to ensure all peers have finished value assignment
  barrier();

  const auto rank = statex->rank();
  const auto numRanks = statex->nRanks();
  int nextPeer = (rank + 1) % numRanks;
  int prevPeer = (rank + numRanks - 1) % numRanks;

  for (auto iter = 0; iter < kNumIters; iter++) {
    // Put data to next peer at offset of kNumElements * rank
    COMMCHECK_TEST(ctranPutSignal(
        localBuf,
        kNumElements,
        commInt32,
        nextPeer,
        kNumElements * statex->rank(),
        win,
        put_stream,
        false));
  }
  CUDACHECK_TEST(cudaStreamSynchronize(put_stream));
  barrier();

  // barrier ensures all remote puts have finished
  cudaStream_t default_stream = 0;
  size_t errs = checkChunkValue(
      (int*)winBase + kNumElements * prevPeer,
      kNumElements,
      prevPeer,
      1,
      this->globalRank,
      default_stream);

  auto res = ctranWinFree(win);
  EXPECT_EQ(res, commSuccess);

  COMMCHECK_TEST(ctran::globalDeregisterWithPtr(localBuf, localBufBytes));
  CUDACHECK_TEST(cudaFree(localBuf));
  freeWinBuf(userBuf, winBase, sizeBytes, bufType);

  CUDACHECK_TEST(cudaStreamDestroy(put_stream));
  EXPECT_EQ(errs, 0);
}

TEST_P(CtranRMATestParam, winGet) {
  const auto& [kNumElements, kNumIters, ctranAllReduce, bufType, userBuf] =
      GetParam();
  EXPECT_GE(kNumElements, 8192);
  EXPECT_GE(kNumIters, 1);
  if (!ctranComm->ctran_->mapper->hasBackend(
          globalRank, CtranMapperBackend::NVL) &&
      ctranAllReduce) {
    GTEST_SKIP()
        << "NVL not enabled, which is required for ctranAllReduce. Skip test";
  }

  auto statex = ctranComm->statex_.get();
  ASSERT_NE(statex, nullptr);
  EXPECT_EQ(statex->nRanks(), this->numRanks);

  cudaStream_t get_stream;
  CUDACHECK_TEST(cudaStreamCreateWithFlags(&get_stream, cudaStreamNonBlocking));

  size_t sizeBytes = kNumElements * sizeof(int) * statex->nRanks();

  meta::comms::Hints hints;

  CtranWin* win = nullptr;
  void* winBase = nullptr;
  createWin(userBuf, bufType, &winBase, &win, sizeBytes, hints);

  // Allocate localBuf from GPU mem
  int* localBuf = nullptr;
  size_t localBufBytes = kNumElements * sizeof(int);
  CUDACHECK_TEST(cudaMalloc(&localBuf, localBufBytes));
  COMMCHECK_TEST(ctran::globalRegisterWithPtr(localBuf, localBufBytes));

  const auto rank = statex->rank();
  const auto numRanks = statex->nRanks();

  assignChunkValue((int*)winBase, kNumElements * statex->nRanks(), rank, 0);
  assignChunkValue(localBuf, kNumElements, statex->rank(), -1);
  // Barrier to ensure all peers have finished value assignment
  barrier();

  int nextPeer = (rank + 1) % numRanks;

  for (auto iter = 0; iter < kNumIters; iter++) {
    // Get data from next peer at offset of kNumElements * rank
    COMMCHECK_TEST(ctranGet(
        localBuf,
        kNumElements * statex->rank(),
        kNumElements,
        commInt32,
        nextPeer,
        win,
        win->comm,
        get_stream));
  }
  CUDACHECK_TEST(cudaStreamSynchronize(get_stream));
  barrier();

  // barrier ensures all remote gets have finished
  cudaStream_t default_stream = 0;
  size_t errs = checkChunkValue(
      (int*)localBuf,
      kNumElements,
      nextPeer,
      0,
      this->globalRank,
      default_stream);

  auto res = ctranWinFree(win);
  EXPECT_EQ(res, commSuccess);

  COMMCHECK_TEST(ctran::globalDeregisterWithPtr(localBuf, localBufBytes));
  CUDACHECK_TEST(cudaFree(localBuf));
  freeWinBuf(userBuf, winBase, sizeBytes, bufType);

  CUDACHECK_TEST(cudaStreamDestroy(get_stream));
  EXPECT_EQ(errs, 0);
}

class RMATestSignalParam : public CtranRMATest,
                           public ::testing::WithParamInterface<
                               std::tuple<size_t, MemAllocType, bool, bool>> {};

TEST_P(RMATestSignalParam, winSignalWait) {
  const auto& [kNumIters, bufType, signalOnly, userBuf] = GetParam();

  auto statex = ctranComm->statex_.get();
  ASSERT_NE(statex, nullptr);
  EXPECT_EQ(statex->nRanks(), this->numRanks);

  cudaStream_t sig_stream, wait_stream;
  CUDACHECK_TEST(cudaStreamCreateWithFlags(&sig_stream, cudaStreamNonBlocking));
  CUDACHECK_TEST(
      cudaStreamCreateWithFlags(&wait_stream, cudaStreamNonBlocking));

  meta::comms::Hints hints;
  CtranWin* win = nullptr;
  int* winBase = nullptr;

  // allocte sizeof(int) * statex->nRanks() Bytes buffer size
  size_t sizeBytes = sizeof(int) * statex->nRanks();
  createWin(userBuf, bufType, (void**)&winBase, &win, sizeBytes, hints);
  uint64_t* signalBase = win->winSignalPtr;
  ASSERT_NE(signalBase, nullptr);

  // barrier to ensure all peers have finished value assignment
  barrier();

  const auto myrank = statex->rank();
  const auto numRanks = statex->nRanks();

  for (auto iter = 0; iter < kNumIters; iter++) {
    // In iterations, each rank signals the continuous memory region of other
    // ranks, with its rank id as the signal value
    for (auto peerRank = 0; peerRank < numRanks; peerRank++) {
      if (peerRank == myrank) {
        continue;
      }
      COMMCHECK_TEST(ctranSignal(peerRank, win, sig_stream));
    }
    if (!signalOnly) {
      for (auto peerRank = 0; peerRank < numRanks; peerRank++) {
        if (peerRank == myrank) {
          continue;
        }
        COMMCHECK_TEST(ctranWaitSignal(peerRank, win, wait_stream));
      }
    }
  }

  if (signalOnly) {
    // barrier to ensure all peers have finished signal, and check values
    CUDACHECK_TEST(cudaStreamSynchronize(sig_stream));
    barrier();
  } else {
    CUDACHECK_TEST(cudaStreamSynchronize(wait_stream));
  }

  int errs = 0;
  for (auto peerRank = 0; peerRank < numRanks; peerRank++) {
    if (peerRank == myrank) {
      continue;
    }
    errs += checkChunkValue(
        signalBase + peerRank,
        1,
        (uint64_t)kNumIters,
        0UL,
        this->globalRank,
        wait_stream);
  }
  EXPECT_EQ(errs, 0);
  // For signalWait, this is necessary since we must ensure all ctranSignal
  // ops finish for other peers
  CUDACHECK_TEST(cudaStreamSynchronize(sig_stream));

  auto res = ctranWinFree(win);
  EXPECT_EQ(res, commSuccess);
  freeWinBuf(userBuf, winBase, sizeBytes, bufType);

  CUDACHECK_TEST(cudaStreamDestroy(sig_stream));
  CUDACHECK_TEST(cudaStreamDestroy(wait_stream));
}

class RMAAtomicTestParam
    : public CtranRMATest,
      public ::testing::WithParamInterface<std::tuple<size_t, MemAllocType>> {};

// Basic fetchAdd: each rank atomically increments a counter on the next peer.
// Verifies both the final counter value and that fetched old values are
// monotonically increasing (single writer per slot).
TEST_P(RMAAtomicTestParam, winFetchAdd) {
  const auto& [kNumIters, bufType] = GetParam();
  const bool userBuf = true;

  auto statex = ctranComm->statex_.get();
  ASSERT_NE(statex, nullptr);
  EXPECT_EQ(statex->nRanks(), this->numRanks);

  cudaStream_t atomic_stream;
  CUDACHECK_TEST(
      cudaStreamCreateWithFlags(&atomic_stream, cudaStreamNonBlocking));

  const auto rank = statex->rank();
  const auto numRanks = statex->nRanks();

  // Allocate window with one uint64_t counter per rank
  size_t sizeBytes = numRanks * sizeof(uint64_t);
  meta::comms::Hints hints;
  CtranWin* win = nullptr;
  void* winBase = nullptr;
  createWin(userBuf, bufType, &winBase, &win, sizeBytes, hints);

  CUDACHECK_TEST(cudaMemset(winBase, 0, sizeBytes));
  barrier();

  // Collect fetched old values to verify monotonicity
  uint64_t* resultBuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&resultBuf, kNumIters * sizeof(uint64_t)));

  int nextPeer = (rank + 1) % numRanks;

  // Each rank does kNumIters fetchAdd(1) on the next peer's counter[rank]
  for (size_t iter = 0; iter < kNumIters; iter++) {
    COMMCHECK_TEST(ctranFetchAdd(
        resultBuf + iter,
        1,
        rank,
        nextPeer,
        win,
        ctranComm.get(),
        atomic_stream));
  }
  CUDACHECK_TEST(cudaStreamSynchronize(atomic_stream));
  barrier();

  // Verify final counter value on local window
  int prevPeer = (rank + numRanks - 1) % numRanks;
  uint64_t hostVal = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &hostVal,
      reinterpret_cast<uint64_t*>(winBase) + prevPeer,
      sizeof(uint64_t),
      cudaMemcpyDeviceToHost));
  EXPECT_EQ(hostVal, kNumIters)
      << "Rank " << rank << ": counter at slot[" << prevPeer << "] expected "
      << kNumIters << " got " << hostVal;

  // Verify fetched old values are monotonically increasing: 0, 1, 2, ...
  // Since this rank is the only writer to slot[rank] on nextPeer, the
  // returned sequence must be consecutive.
  std::vector<uint64_t> hostResults(kNumIters);
  CUDACHECK_TEST(cudaMemcpy(
      hostResults.data(),
      resultBuf,
      kNumIters * sizeof(uint64_t),
      cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < kNumIters; i++) {
    EXPECT_EQ(hostResults[i], i)
        << "Rank " << rank << ": fetchAdd iter " << i << " returned "
        << hostResults[i] << " expected " << i;
  }

  CUDACHECK_TEST(cudaFree(resultBuf));
  auto res = ctranWinFree(win);
  EXPECT_EQ(res, commSuccess);
  freeWinBuf(userBuf, winBase, sizeBytes, bufType);
  CUDACHECK_TEST(cudaStreamDestroy(atomic_stream));
}

// Multi-rank contention: all ranks atomically add to a single counter on
// rank 0's window at slot[0]. Verifies the final counter equals
// (numRanks - 1) * kNumIters.
TEST_P(RMAAtomicTestParam, winFetchAddContention) {
  const auto& [kNumIters, bufType] = GetParam();
  const bool userBuf = true;

  auto statex = ctranComm->statex_.get();
  ASSERT_NE(statex, nullptr);
  EXPECT_EQ(statex->nRanks(), this->numRanks);

  cudaStream_t atomic_stream;
  CUDACHECK_TEST(
      cudaStreamCreateWithFlags(&atomic_stream, cudaStreamNonBlocking));

  const auto rank = statex->rank();
  const auto numRanks = statex->nRanks();

  size_t sizeBytes = sizeof(uint64_t);
  CtranWin* win = nullptr;
  void* winBase = nullptr;
  createWin(userBuf, bufType, &winBase, &win, sizeBytes, {});

  CUDACHECK_TEST(cudaMemset(winBase, 0, sizeBytes));
  barrier();

  uint64_t* resultBuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&resultBuf, sizeof(uint64_t)));

  // All non-zero ranks atomically add to rank 0's counter
  if (rank != 0) {
    for (size_t iter = 0; iter < kNumIters; iter++) {
      COMMCHECK_TEST(ctranFetchAdd(
          resultBuf, 1, 0, 0, win, ctranComm.get(), atomic_stream));
    }
  }
  CUDACHECK_TEST(cudaStreamSynchronize(atomic_stream));
  barrier();

  // Rank 0 verifies the final value
  if (rank == 0) {
    uint64_t hostVal = 0;
    CUDACHECK_TEST(cudaMemcpy(
        &hostVal, winBase, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    uint64_t expected = static_cast<uint64_t>(numRanks - 1) * kNumIters;
    EXPECT_EQ(hostVal, expected) << "Rank 0: contention counter expected "
                                 << expected << " got " << hostVal;
  }

  CUDACHECK_TEST(cudaFree(resultBuf));
  auto res = ctranWinFree(win);
  EXPECT_EQ(res, commSuccess);
  freeWinBuf(userBuf, winBase, sizeBytes, bufType);
  CUDACHECK_TEST(cudaStreamDestroy(atomic_stream));
}

INSTANTIATE_TEST_SUITE_P(
    RMAAtomicTestInstance,
    RMAAtomicTestParam,
    ::testing::Combine(
        ::testing::Values(10, 100),
        ::testing::Values(
            MemAllocType::kMemCuMemAlloc,
            MemAllocType::kMemCudaMalloc)),
    [](const ::testing::TestParamInfo<RMAAtomicTestParam::ParamType>& info) {
      const auto kNumIters = std::get<0>(info.param);
      const auto bufType = std::get<1>(info.param);
      std::string name = fmt::format(
          "numIters{}_{}", kNumIters, testMemAllocTypeToStr(bufType));
      return name;
    });

INSTANTIATE_TEST_SUITE_P(
    RMATestInstance,
    CtranRMATestParam,
    ::testing::Combine(
        // kNumElements, kNumIters, ctranAllReduce, cpuWin, userBuf
        ::testing::Values(8192, 8 * 1024 * 1024),
        ::testing::Values(50),
        ::testing::Values(true, false),
        ::testing::Values(
            MemAllocType::kMemCuMemAlloc,
            MemAllocType::kCuMemAllocDisjoint,
            MemAllocType::kMemCudaMalloc,
            MemAllocType::kMemHostManaged,
            MemAllocType::kMemHostUnregistered),
        ::testing::Values(true, false)),
    [](const ::testing::TestParamInfo<CtranRMATestParam::ParamType>& info) {
      const auto kNumElements = std::get<0>(info.param);
      const auto kNumIters = std::get<1>(info.param);
      const auto ctranAllReduce = std::get<2>(info.param);
      const auto bufType = std::get<3>(info.param);
      const auto userBuf = std::get<4>(info.param);
      std::string name = fmt::format(
          "numElem{}_numIters{}_{}_{}_{}",
          kNumElements,
          kNumIters,
          ctranAllReduce ? "ctranAR" : "ncclAR",
          testMemAllocTypeToStr(bufType),
          userBuf ? "userBuf" : "allocBuf");
      return name;
    });

INSTANTIATE_TEST_SUITE_P(
    MultiWindowTestInstance,
    MultiWindowTestParam,
    ::testing::Values(std::make_tuple(8, 10)));

INSTANTIATE_TEST_SUITE_P(
    RMATestSignalInstance,
    RMATestSignalParam,
    ::testing::Combine(
        // kNumElements, cpuWin, signalOnly, userBuf
        ::testing::Values(100),
        ::testing::Values(
            MemAllocType::kMemCuMemAlloc,
            MemAllocType::kMemCudaMalloc,
            MemAllocType::kMemHostManaged,
            MemAllocType::kMemHostUnregistered),
        ::testing::Values(true, false),
        ::testing::Values(true, false)),
    [](const ::testing::TestParamInfo<RMATestSignalParam::ParamType>& info) {
      const auto kNumIters = std::get<0>(info.param);
      const auto bufType = std::get<1>(info.param);
      const auto signalOnly = std::get<2>(info.param);
      const auto userBuf = std::get<3>(info.param);
      std::string name = fmt::format(
          "numIters{}_{}_{}_{}",
          kNumIters,
          testMemAllocTypeToStr(bufType),
          signalOnly ? "signalOnly" : "signalWait",
          userBuf ? "userBuf" : "allocBuf");
      return name;
    });

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new ctran::CtranDistEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
