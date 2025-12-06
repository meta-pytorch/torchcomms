// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <comm.h>
#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>

#include "CtranUtUtils.h"
#include "comms/ctran/Ctran.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsCuUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

using namespace ctran;

class CtranRMATest : public CtranDistBaseTest {
 public:
  CtranRMATest() = default;
  ncclComm_t comm{nullptr};

 protected:
  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE", "1", 0);
    setenv("NCCL_CTRAN_IB_EPOCH_LOCK_ENFORCE_CHECK", "true", 0);
#ifdef CTRAN_TEST_IB_ONLY_BACKEND
    setenv("NCCL_CTRAN_BACKENDS", "ib", 1);
#endif
    CtranDistBaseTest::SetUp();
    CUDACHECK_TEST(cudaSetDevice(this->localRank));
    comm = commWorld;
    ASSERT_NE(comm, nullptr);
  }
  void TearDown() override {
    CtranDistBaseTest::TearDown();
  }

  void barrier(ncclComm_t comm, cudaStream_t stream) {
    // simple Allreduce as barrier before get data from other ranks
    void* buf;
    CUDACHECK_TEST(cudaMalloc(&buf, sizeof(char)));
    NCCLCHECK_TEST(ncclAllReduce(buf, buf, 1, ncclChar, ncclSum, comm, stream));
    CUDACHECK_TEST(cudaDeviceSynchronize());
    CUDACHECK_TEST(cudaFree(buf));
  }

  template <typename T>
  void assignChunkValue(T* buf, size_t count, T seed, T inc) {
    std::vector<T> expectedVals(count, 0);
    for (size_t i = 0; i < count; ++i) {
      expectedVals[i] = seed + i * inc;
    }
    CUDACHECK_TEST(cudaMemcpy(
        buf, expectedVals.data(), count * sizeof(T), cudaMemcpyDefault));
  }

  template <typename T>
  int checkChunkValue(
      T* buf,
      size_t count,
      T seed,
      T inc,
      cudaStream_t stream) {
    std::vector<T> observedVals(count, -1);
    CUDACHECK_TEST(cudaMemcpyAsync(
        observedVals.data(),
        buf,
        count * sizeof(T),
        cudaMemcpyDefault,
        stream));
    CUDACHECK_TEST(cudaStreamSynchronize(stream));
    int errs = 0;
    // Use manual print rather than EXPECT_THAT to print failing location.
    for (auto i = 0; i < count; ++i) {
      T val = seed + inc * i;
      if (observedVals[i] != val) {
        if (errs < 10) {
          // avoid using formatted string since we don't know the value type
          std::cout << "[" << this->globalRank << "] observedVals[" << i
                    << "] = " << observedVals[i] << ", expectedVal = " << val
                    << std::endl;
        }
        errs++;
      }
    }
    return errs;
  }

  void createWin(
      bool isUserBuf,
      bool isCpuWin,
      void** winBasePtr,
      CtranWin** winPtr,
      size_t sizeBytes,
      meta::comms::Hints hints) {
    auto res = commSuccess;
    // If userBuf is true, allocate buffer and use ctranWinRegister API
    if (isUserBuf) {
      if (isCpuWin) {
        ASSERT_EQ(cudaMallocHost((void**)winBasePtr, sizeBytes), cudaSuccess);
      } else {
        ASSERT_EQ(ncclMemAlloc((void**)winBasePtr, sizeBytes), ncclSuccess);
      }
      res = ctranWinRegister(
          (void*)*winBasePtr, sizeBytes, comm->ctranComm_.get(), winPtr, hints);

    } else {
      res = ctranWinAllocate(
          sizeBytes, comm->ctranComm_.get(), (void**)winBasePtr, winPtr, hints);
    }
    ASSERT_EQ(res, commSuccess);
    ASSERT_NE(*winBasePtr, nullptr);
  }

  void freeWinBuf(bool isUserBuf, bool isCpuWin, void* winBase) {
    if (isUserBuf) {
      if (isCpuWin) {
        ASSERT_EQ(cudaFreeHost(winBase), cudaSuccess);
      } else {
        ASSERT_EQ(ncclMemFree(winBase), ncclSuccess);
      }
    }
  }
};

class MultiWindowTestParam
    : public CtranRMATest,
      public ::testing::WithParamInterface<std::tuple<size_t, size_t>> {};

TEST_P(MultiWindowTestParam, multiWindow) {
  const auto& [kMaxNumElements, kNumIters] = GetParam();
  EXPECT_GE(kMaxNumElements, 1);
  EXPECT_GE(kNumIters, 1);

  auto comm = this->comm;
  auto statex = comm->ctranComm_->statex_.get();
  ASSERT_NE(statex, nullptr);
  EXPECT_EQ(statex->nRanks(), this->numRanks);

  for (size_t numElements = 1; numElements <= kMaxNumElements;
       numElements = numElements * 2) {
    cudaStream_t main_stream = 0;
    cudaStream_t put_stream, wait_stream;
    CUDACHECK_TEST(
        cudaStreamCreateWithFlags(&put_stream, cudaStreamNonBlocking));
    CUDACHECK_TEST(
        cudaStreamCreateWithFlags(&wait_stream, cudaStreamNonBlocking));

    size_t sizeBytes = numElements * sizeof(int) * statex->nRanks();
    CtranWin* win = nullptr;
    void* winBase = nullptr;
    auto res =
        ctranWinAllocate(sizeBytes, comm->ctranComm_.get(), &winBase, &win);
    ASSERT_EQ(res, ncclSuccess);
    ASSERT_NE(winBase, nullptr);
    int* localbuf = reinterpret_cast<int*>(winBase);

    EXPECT_THAT(win, testing::NotNull());
    // localbuf: range from [myrank, myrank + numElements * numRanks]
    assignChunkValue(
        localbuf, numElements * statex->nRanks(), statex->rank(), 1);
    // Barrier to ensure all peers have finished creation
    this->barrier(comm, main_stream);

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
    this->barrier(comm, main_stream);

    // Check results
    int errs = checkChunkValue(
        localbuf + numElements * prevPeer,
        numElements,
        prevPeer + (int)numElements * prevPeer,
        1,
        wait_stream);
    EXPECT_EQ(errs, 0);

    res = ctranWinFree(win);
    EXPECT_EQ(res, ncclSuccess);

    CUDACHECK_TEST(cudaStreamDestroy(put_stream));
    CUDACHECK_TEST(cudaStreamDestroy(wait_stream));
  }
}

class CtranRMATestParam : public CtranRMATest,
                          public ::testing::WithParamInterface<
                              std::tuple<size_t, size_t, bool, bool, bool>> {};

TEST_P(CtranRMATestParam, winPutWait) {
  const auto& [kNumElements, kNumIters, ctranAllReduce, cpuWin, userBuf] =
      GetParam();
  EXPECT_GE(kNumElements, 8192);
  EXPECT_GE(kNumIters, 1);
  if (!comm->ctranComm_->ctran_->mapper->hasBackend(
          globalRank, CtranMapperBackend::NVL) &&
      ctranAllReduce) {
    GTEST_SKIP()
        << "NVL not enabled, which is required for ctranAllReduce. Skip test";
  }

  // Enable ctran for all-reduce
  auto envGuard = EnvRAII(
      NCCL_ALLREDUCE_ALGO,
      ctranAllReduce ? NCCL_ALLREDUCE_ALGO::ctdirect
                     : NCCL_ALLREDUCE_ALGO::orig);

  auto statex = comm->ctranComm_->statex_.get();
  ASSERT_NE(statex, nullptr);
  EXPECT_EQ(statex->nRanks(), this->numRanks);

  cudaStream_t main_stream = 0;
  cudaStream_t put_stream, wait_stream;
  cudaEvent_t start_event, end_event;
  CUDACHECK_TEST(cudaStreamCreateWithFlags(&put_stream, cudaStreamNonBlocking));
  CUDACHECK_TEST(
      cudaStreamCreateWithFlags(&wait_stream, cudaStreamNonBlocking));
  CUDACHECK_TEST(cudaEventCreate(&start_event));
  CUDACHECK_TEST(cudaEventCreate(&end_event));

  size_t sizeBytes = kNumElements * sizeof(int) * statex->nRanks();

  meta::comms::Hints hints;
  hints.set("window_buffer_location", cpuWin ? "cpu" : "gpu");
  CtranWin* win = nullptr;
  void* winBase = nullptr;
  createWin(userBuf, cpuWin, &winBase, &win, sizeBytes, hints);

  // Always allocate localBuf from GPU mem so it can be used in ctranAllReduce
  int* localBuf = nullptr;
  void* localHdl = nullptr;
  ASSERT_EQ(
      ncclMemAlloc((void**)&localBuf, kNumElements * sizeof(int)), ncclSuccess);
  ASSERT_EQ(
      ncclCommRegister(comm, localBuf, kNumElements * sizeof(int), &localHdl),
      ncclSuccess);

  EXPECT_THAT(win, testing::NotNull());

  for (int peer = 0; peer < this->numRanks; peer++) {
    void* remoteAddr = nullptr;
    auto res = ctranWinSharedQuery(peer, win, &remoteAddr);
    EXPECT_EQ(res, ncclSuccess);
    if (peer == statex->rank()) {
      EXPECT_EQ(remoteAddr, winBase);
    } else if (!win->nvlEnabled(peer)) {
      EXPECT_THAT(remoteAddr, testing::IsNull());
    } else {
      EXPECT_THAT(remoteAddr, testing::NotNull());
    }
  }

  assignChunkValue((int*)winBase, kNumElements * statex->nRanks(), -1, 0);
  assignChunkValue(localBuf, kNumElements, statex->rank(), 1);
  // Barrier to ensure all peers have finished value assignment
  this->barrier(comm, main_stream);

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

  // A couple of all-reduce after RMA tests
  // waitSignal on wait_stream should ensure all remote puts have finished
  for (auto iter = 0; iter < kNumIters; iter++) {
    NCCLCHECK_TEST(ncclAllReduce(
        localBuf,
        localBuf,
        kNumElements,
        ncclInt32,
        ncclSum,
        comm,
        wait_stream));
  }

  int errs = checkChunkValue(
      (int*)winBase + kNumElements * prevPeer,
      kNumElements,
      prevPeer,
      1,
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
    printf(
        "[%d] elapsed time %.2f ms for %zu bytes * %ld iterations (%.2f GB/s), on %s\n",
        statex->rank(),
        elapsed_time_ms,
        chunkBytes,
        kNumIters,
        achieved_bw,
        cpuWin ? "cpuWin" : "gpuWin");
  }

  auto res = ctranWinFree(win);
  EXPECT_EQ(res, ncclSuccess);

  ASSERT_EQ(ncclCommDeregister(comm, localHdl), ncclSuccess);
  ASSERT_EQ(ncclMemFree(localBuf), ncclSuccess);
  freeWinBuf(userBuf, cpuWin, winBase);

  CUDACHECK_TEST(cudaEventDestroy(start_event));
  CUDACHECK_TEST(cudaEventDestroy(end_event));
  CUDACHECK_TEST(cudaStreamDestroy(put_stream));
  CUDACHECK_TEST(cudaStreamDestroy(wait_stream));

  EXPECT_EQ(errs, 0);
}

TEST_P(CtranRMATestParam, winPutOnly) {
  const auto& [kNumElements, kNumIters, ctranAllReduce, cpuWin, userBuf] =
      GetParam();
  EXPECT_GE(kNumElements, 8192);
  EXPECT_GE(kNumIters, 1);
  if (!comm->ctranComm_->ctran_->mapper->hasBackend(
          globalRank, CtranMapperBackend::NVL) &&
      ctranAllReduce) {
    GTEST_SKIP()
        << "NVL not enabled, which is required for ctranAllReduce. Skip test";
  }

  // Enable ctran for all-reduce
  auto envGuard = EnvRAII(
      NCCL_ALLREDUCE_ALGO,
      ctranAllReduce ? NCCL_ALLREDUCE_ALGO::ctdirect
                     : NCCL_ALLREDUCE_ALGO::orig);

  auto statex = comm->ctranComm_->statex_.get();
  ASSERT_NE(statex, nullptr);
  EXPECT_EQ(statex->nRanks(), this->numRanks);

  cudaStream_t main_stream = 0, put_stream;
  CUDACHECK_TEST(cudaStreamCreateWithFlags(&put_stream, cudaStreamNonBlocking));

  size_t sizeBytes = kNumElements * sizeof(int) * statex->nRanks();

  meta::comms::Hints hints;
  hints.set("window_buffer_location", cpuWin ? "cpu" : "gpu");
  CtranWin* win = nullptr;
  void* winBase = nullptr;
  createWin(userBuf, cpuWin, &winBase, &win, sizeBytes, hints);

  // Always allocate localBuf from GPU mem
  int* localBuf = nullptr;
  void* localHdl = nullptr;
  ASSERT_EQ(
      ncclMemAlloc((void**)&localBuf, kNumElements * sizeof(int)), ncclSuccess);
  ASSERT_EQ(
      ncclCommRegister(comm, localBuf, kNumElements * sizeof(int), &localHdl),
      ncclSuccess);

  assignChunkValue((int*)winBase, kNumElements * statex->nRanks(), -1, 0);
  assignChunkValue(localBuf, kNumElements, statex->rank(), 1);
  // Barrier to ensure all peers have finished value assignment
  this->barrier(comm, main_stream);

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
  this->barrier(comm, main_stream);

  // allreduce ensures all remote puts have finished
  int errs = checkChunkValue(
      (int*)winBase + kNumElements * prevPeer,
      kNumElements,
      prevPeer,
      1,
      main_stream);

  auto res = ctranWinFree(win);
  EXPECT_EQ(res, ncclSuccess);

  ASSERT_EQ(ncclCommDeregister(comm, localHdl), ncclSuccess);
  ASSERT_EQ(ncclMemFree(localBuf), ncclSuccess);
  freeWinBuf(userBuf, cpuWin, winBase);

  CUDACHECK_TEST(cudaStreamDestroy(put_stream));
  EXPECT_EQ(errs, 0);
}

TEST_P(CtranRMATestParam, winGet) {
  const auto& [kNumElements, kNumIters, ctranAllReduce, cpuWin, userBuf] =
      GetParam();
  EXPECT_GE(kNumElements, 8192);
  EXPECT_GE(kNumIters, 1);
  if (!comm->ctranComm_->ctran_->mapper->hasBackend(
          globalRank, CtranMapperBackend::NVL) &&
      ctranAllReduce) {
    GTEST_SKIP()
        << "NVL not enabled, which is required for ctranAllReduce. Skip test";
  }

  // Enable ctran for all-reduce
  auto envGuard = EnvRAII(
      NCCL_ALLREDUCE_ALGO,
      ctranAllReduce ? NCCL_ALLREDUCE_ALGO::ctdirect
                     : NCCL_ALLREDUCE_ALGO::orig);

  auto statex = comm->ctranComm_->statex_.get();
  ASSERT_NE(statex, nullptr);
  EXPECT_EQ(statex->nRanks(), this->numRanks);

  cudaStream_t main_stream = 0, get_stream;
  CUDACHECK_TEST(cudaStreamCreateWithFlags(&get_stream, cudaStreamNonBlocking));

  size_t sizeBytes = kNumElements * sizeof(int) * statex->nRanks();

  meta::comms::Hints hints;
  hints.set("window_buffer_location", cpuWin ? "cpu" : "gpu");

  CtranWin* win = nullptr;
  void* winBase = nullptr;
  createWin(userBuf, cpuWin, &winBase, &win, sizeBytes, hints);

  // Always allocate localBuf from GPU mem
  int* localBuf = nullptr;
  void* localHdl = nullptr;
  ASSERT_EQ(
      ncclMemAlloc((void**)&localBuf, kNumElements * sizeof(int)), ncclSuccess);
  ASSERT_EQ(
      ncclCommRegister(comm, localBuf, kNumElements * sizeof(int), &localHdl),
      ncclSuccess);

  const auto rank = statex->rank();
  const auto numRanks = statex->nRanks();

  assignChunkValue((int*)winBase, kNumElements * statex->nRanks(), rank, 0);
  assignChunkValue(localBuf, kNumElements, statex->rank(), -1);
  // Barrier to ensure all peers have finished value assignment
  this->barrier(comm, main_stream);

  int nextPeer = (rank + 1) % numRanks;

  for (auto iter = 0; iter < kNumIters; iter++) {
    // Put data to next peer at offset of kNumElements * rank
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
  this->barrier(comm, main_stream);

  // allreduce ensures all remote puts have finished
  int errs =
      checkChunkValue((int*)localBuf, kNumElements, nextPeer, 0, main_stream);

  auto res = ctranWinFree(win);
  EXPECT_EQ(res, ncclSuccess);

  ASSERT_EQ(ncclCommDeregister(comm, localHdl), ncclSuccess);
  ASSERT_EQ(ncclMemFree(localBuf), ncclSuccess);
  freeWinBuf(userBuf, cpuWin, winBase);

  CUDACHECK_TEST(cudaStreamDestroy(get_stream));
  EXPECT_EQ(errs, 0);
}

TEST_P(CtranRMATestParam, winPutWait_v2) {
  const auto& [kNumElements, kNumIters, ctranAllReduce, cpuWin, userBuf] =
      GetParam();
  EXPECT_GE(kNumElements, 8192);
  EXPECT_GE(kNumIters, 1);
  if (!comm->ctranComm_->ctran_->mapper->hasBackend(
          globalRank, CtranMapperBackend::NVL) &&
      ctranAllReduce) {
    GTEST_SKIP()
        << "NVL not enabled, which is required for ctranAllReduce. Skip test";
  }

  // Enable ctran for all-reduce
  auto envGuard = EnvRAII(
      NCCL_ALLREDUCE_ALGO,
      ctranAllReduce ? NCCL_ALLREDUCE_ALGO::ctdirect
                     : NCCL_ALLREDUCE_ALGO::orig);

  auto statex = comm->ctranComm_->statex_.get();
  ASSERT_NE(statex, nullptr);
  EXPECT_EQ(statex->nRanks(), this->numRanks);

  cudaStream_t main_stream = 0;
  cudaStream_t put_stream, wait_stream;
  cudaEvent_t start_event, end_event;
  CUDACHECK_TEST(cudaStreamCreateWithFlags(&put_stream, cudaStreamNonBlocking));
  CUDACHECK_TEST(
      cudaStreamCreateWithFlags(&wait_stream, cudaStreamNonBlocking));
  CUDACHECK_TEST(cudaEventCreate(&start_event));
  CUDACHECK_TEST(cudaEventCreate(&end_event));

  size_t sizeBytes = kNumElements * sizeof(int) * statex->nRanks();

  meta::comms::Hints hints;
  hints.set("window_buffer_location", cpuWin ? "cpu" : "gpu");
  hints.set("window_signal_bufsize", std::to_string(statex->nRanks()));
  CtranWin* win = nullptr;
  int* winBase = nullptr;
  createWin(userBuf, cpuWin, (void**)&winBase, &win, sizeBytes, hints);

  uint64_t* signalBase = win->winSignalPtr;
  ASSERT_NE(signalBase, nullptr);

  // Always allocate localBuf from GPU mem so it can be used in ctranAllReduce
  int* localBuf = nullptr;
  void* localHdl = nullptr;
  ASSERT_EQ(
      ncclMemAlloc((void**)&localBuf, kNumElements * sizeof(int)), ncclSuccess);
  ASSERT_EQ(
      ncclCommRegister(comm, localBuf, kNumElements * sizeof(int), &localHdl),
      ncclSuccess);

  EXPECT_THAT(win, testing::NotNull());

  for (int peer = 0; peer < this->numRanks; peer++) {
    void* remoteAddr = nullptr;
    auto res = ctranWinSharedQuery(peer, win, &remoteAddr);
    EXPECT_EQ(res, ncclSuccess);
    if (peer == statex->rank()) {
      EXPECT_EQ(remoteAddr, (void*)winBase);
    } else if (!win->nvlEnabled(peer)) {
      EXPECT_THAT(remoteAddr, testing::IsNull());
    } else {
      EXPECT_THAT(remoteAddr, testing::NotNull());
    }
  }

  // initialize signals: 0UL
  assignChunkValue(signalBase, statex->nRanks(), 0UL, 0UL);
  // initialize data bufs
  assignChunkValue((int*)winBase, kNumElements * statex->nRanks(), -1, 0);
  assignChunkValue(localBuf, kNumElements, statex->rank(), 1);
  // Barrier to ensure all peers have finished value assignment
  this->barrier(comm, main_stream);

  int nextPeer = (this->globalRank + 1) % this->numRanks;
  int prevPeer = (this->globalRank + this->numRanks - 1) % this->numRanks;

  for (auto iter = 0; iter < kNumIters; iter++) {
    COMMCHECK_TEST(ctranPutSignal_v2(
        localBuf,
        kNumElements * statex->rank(),
        kNumElements,
        commInt32,
        statex->rank(),
        (uint64_t)(iter + 1),
        nextPeer,
        win,
        put_stream,
        true));
    if (iter == 0) {
      // Skip first iteration to avoid any warmup overhead
      CUDACHECK_TEST(cudaEventRecord(start_event, put_stream));
    }
  }
  COMMCHECK_TEST(ctranWaitSignal_v2(
      prevPeer, (uint64_t)kNumIters, commCmpEQ, win, wait_stream));
  CUDACHECK_TEST(cudaEventRecord(end_event, put_stream));

  int errs = checkChunkValue(
      (int*)winBase + kNumElements * prevPeer,
      kNumElements,
      prevPeer,
      1,
      wait_stream);
  EXPECT_EQ(errs, 0);
  CUDACHECK_TEST(cudaStreamSynchronize(put_stream));
  CUDACHECK_TEST(cudaStreamSynchronize(wait_stream));

  size_t chunkBytes = kNumElements * sizeof(int);
  if (chunkBytes > 0) {
    float elapsed_time_ms = -1.0;
    CUDACHECK_TEST(
        cudaEventElapsedTime(&elapsed_time_ms, start_event, end_event));
    // time captured with kNumIters - 1 iterations
    float achieved_bw = chunkBytes / elapsed_time_ms / 1e6 * (kNumIters - 1);
    printf(
        "[%d] elapsed time %.2f ms for %zu bytes * %ld iterations (%.2f GB/s), on %s\n",
        statex->rank(),
        elapsed_time_ms,
        chunkBytes,
        kNumIters,
        achieved_bw,
        cpuWin ? "cpuWin" : "gpuWin");
  }

  auto res = ctranWinFree(win);
  EXPECT_EQ(res, ncclSuccess);

  ASSERT_EQ(ncclCommDeregister(comm, localHdl), ncclSuccess);
  ASSERT_EQ(ncclMemFree(localBuf), ncclSuccess);
  freeWinBuf(userBuf, cpuWin, winBase);

  CUDACHECK_TEST(cudaEventDestroy(start_event));
  CUDACHECK_TEST(cudaEventDestroy(end_event));
  CUDACHECK_TEST(cudaStreamDestroy(put_stream));
}

class RMATestSignalParam : public CtranRMATest,
                           public ::testing::WithParamInterface<
                               std::tuple<size_t, bool, bool, bool>> {};

TEST_P(RMATestSignalParam, winSignalWait) {
  const auto& [kNumIters, cpuWin, signalOnly, userBuf] = GetParam();
  auto envGuard = EnvRAII(NCCL_ALLREDUCE_ALGO, NCCL_ALLREDUCE_ALGO::orig);

  auto comm = this->comm;
  auto statex = comm->ctranComm_->statex_.get();
  ASSERT_NE(statex, nullptr);
  EXPECT_EQ(statex->nRanks(), this->numRanks);

  cudaStream_t main_stream = 0;
  cudaStream_t sig_stream, wait_stream;
  CUDACHECK_TEST(cudaStreamCreateWithFlags(&sig_stream, cudaStreamNonBlocking));
  CUDACHECK_TEST(
      cudaStreamCreateWithFlags(&wait_stream, cudaStreamNonBlocking));

  meta::comms::Hints hints;
  CtranWin* win = nullptr;
  int* winBase = nullptr;

  // allocte sizeof(int) * statex->nRanks() Bytes buffer size
  size_t sizeBytes = sizeof(int) * statex->nRanks();
  createWin(userBuf, cpuWin, (void**)&winBase, &win, sizeBytes, hints);
  uint64_t* signalBase = win->winSignalPtr;
  ASSERT_NE(signalBase, nullptr);

  // barrier to ensure all peers have finished value assignment
  this->barrier(comm, main_stream);

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
    this->barrier(comm, sig_stream);
    CUDACHECK_TEST(cudaStreamSynchronize(sig_stream));
  } else {
    CUDACHECK_TEST(cudaStreamSynchronize(wait_stream));
  }

  int errs = 0;
  for (auto peerRank = 0; peerRank < numRanks; peerRank++) {
    if (peerRank == myrank) {
      continue;
    }
    errs += checkChunkValue(
        signalBase + peerRank, 1, (uint64_t)kNumIters, 0UL, wait_stream);
  }
  EXPECT_EQ(errs, 0);
  // For signalWait, this is necessary since we must ensure all ctranSignal
  // ops finish for other peers
  CUDACHECK_TEST(cudaStreamSynchronize(sig_stream));

  auto res = ctranWinFree(win);
  EXPECT_EQ(res, ncclSuccess);
  freeWinBuf(userBuf, cpuWin, winBase);

  CUDACHECK_TEST(cudaStreamDestroy(sig_stream));
  CUDACHECK_TEST(cudaStreamDestroy(wait_stream));
}

INSTANTIATE_TEST_SUITE_P(
    RMATestInstance,
    CtranRMATestParam,
    ::testing::Combine(
        // kNumElements, kNumIters, ctranAllReduce, cpuWin, userBuf
        ::testing::Values(8192, 8 * 1024 * 1024),
        ::testing::Values(500),
        ::testing::Values(true, false),
        ::testing::Values(true, false),
        ::testing::Values(true, false)),
    [](const testing::TestParamInfo<CtranRMATestParam::ParamType>& info) {
      const auto kNumElements = std::get<0>(info.param);
      const auto kNumIters = std::get<1>(info.param);
      const auto ctranAllReduce = std::get<2>(info.param);
      const auto cpuWin = std::get<3>(info.param);
      const auto userBuf = std::get<4>(info.param);
      std::string name = fmt::format(
          "numElem{}_numIters{}_{}_{}_{}",
          kNumElements,
          kNumIters,
          ctranAllReduce ? "ctranAR" : "ncclAR",
          cpuWin ? "cpuWin" : "gpuWin",
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
        ::testing::Values(true, false),
        ::testing::Values(true, false),
        ::testing::Values(true, false)),
    [](const testing::TestParamInfo<RMATestSignalParam::ParamType>& info) {
      const auto kNumIters = std::get<0>(info.param);
      const auto cpuWin = std::get<1>(info.param);
      const auto signalOnly = std::get<2>(info.param);
      const auto userBuf = std::get<3>(info.param);
      std::string name = fmt::format(
          "numIters{}_{}_{}_{}",
          kNumIters,
          cpuWin ? "cpuWin" : "gpuWin",
          signalOnly ? "signalOnly" : "signalWait",
          userBuf ? "userBuf" : "allocBuf");
      return name;
    });

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
