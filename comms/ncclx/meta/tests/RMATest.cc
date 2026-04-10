// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/init/Init.h>

#include <nccl.h>
#include <cstddef>
#include "comm.h"
#include "comms/ncclx/meta/tests/NcclCommUtils.h"
#include "comms/ncclx/meta/tests/NcclxBaseTest.h"
#include "comms/testinfra/AlgoTestUtils.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

using testinfra::AlgoRAII;

// In v2.29+, RMA functions moved to ncclx:: namespace.
// Bring them into scope so the same code compiles across versions.
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)
using ncclx::ncclGet;
using ncclx::ncclPut;
using ncclx::ncclPutSignal;
using ncclx::ncclWaitSignal;
#endif

class RMATest : public NcclxBaseTestFixture {
 public:
  RMATest() = default;

 protected:
  void SetUp() override {
    NcclxBaseTestFixture::SetUp();

    this->comm = ncclx::test::createNcclComm(
        globalRank, numRanks, localRank, bootstrap_.get());
    ASSERT_NE(this->comm, nullptr);

    CUDACHECK_TEST(cudaSetDevice(localRank));
    CUDACHECK_TEST(cudaStreamCreate(&stream));
  }
  void TearDown() override {
    NCCLCHECK_TEST(ncclCommDestroy(this->comm));
    CUDACHECK_TEST(cudaStreamDestroy(stream));
    NcclxBaseTestFixture::TearDown();
  }

  void barrier(ncclComm_t ncclComm, cudaStream_t s) {
    void* buf;
    CUDACHECK_TEST(cudaMalloc(&buf, sizeof(char)));
    NCCLCHECK_TEST(ncclAllReduce(buf, buf, 1, ncclChar, ncclSum, ncclComm, s));
    CUDACHECK_TEST(cudaDeviceSynchronize());
    CUDACHECK_TEST(cudaFree(buf));
  }

  void createWin(
      MemAllocType bufType,
      void** winBasePtr,
      ncclWindow_t* winPtr,
      size_t sizeBytes) {
    *winBasePtr = testAllocBuf(sizeBytes, bufType, segments);
    ASSERT_NE(*winBasePtr, nullptr);
    auto res = ncclCommWindowRegister(
        comm, *winBasePtr, sizeBytes, winPtr, NCCL_WIN_DEFAULT);
    ASSERT_EQ(res, ncclSuccess);
  }

  void freeWinBuf(void* ptr, size_t size, MemAllocType bufType) {
    testFreeBuf(ptr, size, bufType);
    segments.erase(
        std::remove_if(
            segments.begin(),
            segments.end(),
            [ptr](const TestMemSegment& seg) { return seg.ptr == ptr; }),
        segments.end());
  }

  ncclComm_t comm{nullptr};
  cudaStream_t stream{};
  std::vector<TestMemSegment> segments;
};

TEST_F(RMATest, winPutOnly) {
  const size_t kNumElements = 8192;
  const size_t kNumIters = 10;

  size_t sizeBytes = kNumElements * sizeof(int) * numRanks;

  void* winBase = nullptr;
  ncclWindow_t win = nullptr;
  ASSERT_EQ(ncclMemAlloc(&winBase, sizeBytes), ncclSuccess);
  ASSERT_NE(winBase, nullptr);
  auto res =
      ncclCommWindowRegister(comm, winBase, sizeBytes, &win, NCCL_WIN_DEFAULT);
  ASSERT_EQ(res, ncclSuccess);

  int* localBuf = nullptr;
  void* localHdl = nullptr;
  ASSERT_EQ(
      ncclMemAlloc((void**)&localBuf, kNumElements * sizeof(int)), ncclSuccess);
  ASSERT_EQ(
      ncclCommRegister(comm, localBuf, kNumElements * sizeof(int), &localHdl),
      ncclSuccess);

  assignChunkValue((int*)winBase, kNumElements * numRanks, -1, 0);
  assignChunkValue(localBuf, kNumElements, globalRank, 1);
  this->barrier(comm, stream);

  int nextPeer = (globalRank + 1) % numRanks;
  int prevPeer = (globalRank + numRanks - 1) % numRanks;

  for (size_t iter = 0; iter < kNumIters; iter++) {
    NCCLCHECK_TEST(ncclPut(
        localBuf,
        kNumElements,
        ncclInt32,
        nextPeer,
        kNumElements * globalRank,
        win,
        stream));
  }

  NCCLCHECK_TEST(ncclAllReduce(
      localBuf, localBuf, kNumElements, ncclInt32, ncclSum, comm, stream));

  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  size_t errs = checkChunkValue(
      (int*)winBase + kNumElements * prevPeer,
      kNumElements,
      prevPeer,
      1,
      globalRank,
      stream);

  res = ncclCommWindowDeregister(comm, win);
  EXPECT_EQ(res, ncclSuccess);

  ASSERT_EQ(ncclCommDeregister(comm, localHdl), ncclSuccess);
  ASSERT_EQ(ncclMemFree(localBuf), ncclSuccess);
  ASSERT_EQ(ncclMemFree(winBase), ncclSuccess);

  EXPECT_EQ(errs, 0u);
}

TEST_F(RMATest, winAllGather) {
  const size_t kNumElements = 8192;
  const size_t kNumIters = 10;

  size_t sizeBytes = kNumElements * sizeof(int) * numRanks;

  void* winBase = nullptr;
  ASSERT_EQ(ncclMemAlloc(&winBase, sizeBytes), ncclSuccess);
  ASSERT_NE(winBase, nullptr);

  ncclWindow_t win = nullptr;
  auto res =
      ncclCommWindowRegister(comm, winBase, sizeBytes, &win, NCCL_WIN_DEFAULT);
  ASSERT_EQ(res, ncclSuccess);

  bool supported = false;
  ASSERT_EQ(ncclx::winAllGatherPSupported(win, &supported), ncclSuccess);
  if (!supported) {
    GTEST_SKIP() << "Window AllGatherP not supported";
  }

  void* agpRequest = nullptr;
  ASSERT_EQ(
      ncclx::winAllGatherInit(win, comm, stream, &agpRequest), ncclSuccess);
  ASSERT_NE(agpRequest, nullptr);

  // Fill each rank's chunk with its rank value
  int* winBuf = reinterpret_cast<int*>(winBase);
  assignChunkValue(
      winBuf + kNumElements * globalRank, kNumElements, globalRank + 1, 0);

  for (size_t iter = 0; iter < kNumIters; iter++) {
    ASSERT_EQ(
        ncclx::winAllGatherExec(
            winBuf + kNumElements * globalRank,
            kNumElements,
            ncclInt32,
            agpRequest),
        ncclSuccess);
  }

  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  // Verify all chunks
  for (int r = 0; r < numRanks; r++) {
    int errs = checkChunkValue(winBuf + kNumElements * r, kNumElements, r + 1);
    EXPECT_EQ(errs, 0) << "rank " << globalRank << " checked chunk " << r
                       << " with " << errs << " errors";
  }

  ASSERT_EQ(ncclx::winAllGatherDestroy(agpRequest), ncclSuccess);

  res = ncclWinFree(comm, win);
  EXPECT_EQ(res, ncclSuccess);
  ASSERT_EQ(ncclMemFree(winBase), ncclSuccess);
}

// Parameterized test: (kNumElements, ctranAllReduce, bufType)
class RMATestParam : public RMATest,
                     public ::testing::WithParamInterface<
                         std::tuple<size_t, bool, MemAllocType>> {};

TEST_P(RMATestParam, winPut) {
  const auto& [kNumElements, ctranAllReduce, bufType] = GetParam();
  const size_t kNumIters = 10;

  auto envGuard = AlgoRAII(
      NCCL_ALLREDUCE_ALGO,
      ctranAllReduce ? NCCL_ALLREDUCE_ALGO::ctdirect
                     : NCCL_ALLREDUCE_ALGO::orig);

  cudaStream_t put_stream, wait_stream;
  CUDACHECK_TEST(cudaStreamCreateWithFlags(&put_stream, cudaStreamNonBlocking));
  CUDACHECK_TEST(
      cudaStreamCreateWithFlags(&wait_stream, cudaStreamNonBlocking));

  size_t sizeBytes = kNumElements * sizeof(int) * numRanks;

  ncclWindow_t win = nullptr;
  void* winBase = nullptr;
  createWin(bufType, &winBase, &win, sizeBytes);

  int* localBuf = nullptr;
  void* localHdl = nullptr;
  ASSERT_EQ(
      ncclMemAlloc((void**)&localBuf, kNumElements * sizeof(int)), ncclSuccess);
  ASSERT_EQ(
      ncclCommRegister(comm, localBuf, kNumElements * sizeof(int), &localHdl),
      ncclSuccess);

  assignChunkValue((int*)winBase, kNumElements * numRanks, -1, 0);
  assignChunkValue(localBuf, kNumElements, globalRank, 1);
  this->barrier(comm, stream);

  int nextPeer = (globalRank + 1) % numRanks;
  int prevPeer = (globalRank + numRanks - 1) % numRanks;

  // PutSignal + WaitSignal path
  for (size_t iter = 0; iter < kNumIters; iter++) {
    NCCLCHECK_TEST(ncclPutSignal(
        localBuf,
        kNumElements,
        ncclInt32,
        nextPeer,
        kNumElements * globalRank,
        win,
        put_stream));
    NCCLCHECK_TEST(ncclWaitSignal(prevPeer, win, wait_stream));
  }

  // AllReduce after RMA to stress-test stream coexistence
  for (size_t iter = 0; iter < kNumIters; iter++) {
    NCCLCHECK_TEST(ncclAllReduce(
        localBuf,
        localBuf,
        kNumElements,
        ncclInt32,
        ncclSum,
        comm,
        put_stream));
  }

  size_t errs = checkChunkValue(
      (int*)winBase + kNumElements * prevPeer,
      kNumElements,
      prevPeer,
      1,
      globalRank,
      wait_stream);

  CUDACHECK_TEST(cudaStreamSynchronize(put_stream));
  CUDACHECK_TEST(cudaStreamSynchronize(wait_stream));

  auto res = ncclCommWindowDeregister(comm, win);
  EXPECT_EQ(res, ncclSuccess);

  ASSERT_EQ(ncclCommDeregister(comm, localHdl), ncclSuccess);
  ASSERT_EQ(ncclMemFree(localBuf), ncclSuccess);
  freeWinBuf(winBase, sizeBytes, bufType);

  CUDACHECK_TEST(cudaStreamDestroy(put_stream));
  CUDACHECK_TEST(cudaStreamDestroy(wait_stream));

  EXPECT_EQ(errs, 0u);
}

TEST_P(RMATestParam, winGet) {
  const auto& [kNumElements, ctranAllReduce, bufType] = GetParam();
  const size_t kNumIters = 10;

  auto envGuard = AlgoRAII(
      NCCL_ALLREDUCE_ALGO,
      ctranAllReduce ? NCCL_ALLREDUCE_ALGO::ctdirect
                     : NCCL_ALLREDUCE_ALGO::orig);

  cudaStream_t get_stream;
  CUDACHECK_TEST(cudaStreamCreateWithFlags(&get_stream, cudaStreamNonBlocking));

  size_t sizeBytes = kNumElements * sizeof(int) * numRanks;

  ncclWindow_t win = nullptr;
  void* winBase = nullptr;
  createWin(bufType, &winBase, &win, sizeBytes);

  int* localBuf = nullptr;
  int* arBuf = nullptr;
  void* localHdl = nullptr;
  void* arHdl = nullptr;
  ASSERT_EQ(
      ncclMemAlloc((void**)&localBuf, kNumElements * sizeof(int)), ncclSuccess);
  ASSERT_EQ(
      ncclMemAlloc((void**)&arBuf, kNumElements * sizeof(int)), ncclSuccess);
  ASSERT_EQ(
      ncclCommRegister(comm, localBuf, kNumElements * sizeof(int), &localHdl),
      ncclSuccess);
  ASSERT_EQ(
      ncclCommRegister(comm, arBuf, kNumElements * sizeof(int), &arHdl),
      ncclSuccess);

  assignChunkValue((int*)winBase, kNumElements * numRanks, globalRank, 0);
  assignChunkValue(localBuf, kNumElements, globalRank, -1);
  this->barrier(comm, stream);

  int nextPeer = (globalRank + 1) % numRanks;

  for (size_t iter = 0; iter < kNumIters; iter++) {
    NCCLCHECK_TEST(ncclGet(
        localBuf,
        kNumElements * globalRank,
        kNumElements,
        ncclInt32,
        nextPeer,
        win,
        get_stream));
  }

  // AllReduce after RMA to stress-test stream coexistence
  for (size_t iter = 0; iter < kNumIters; iter++) {
    NCCLCHECK_TEST(ncclAllReduce(
        arBuf, arBuf, kNumElements, ncclInt32, ncclSum, comm, get_stream));
  }
  CUDACHECK_TEST(cudaStreamSynchronize(get_stream));
  this->barrier(comm, stream);

  size_t errs = checkChunkValue(
      localBuf, kNumElements, nextPeer, 0, globalRank, get_stream);

  auto res = ncclCommWindowDeregister(comm, win);
  EXPECT_EQ(res, ncclSuccess);

  ASSERT_EQ(ncclCommDeregister(comm, localHdl), ncclSuccess);
  ASSERT_EQ(ncclCommDeregister(comm, arHdl), ncclSuccess);
  ASSERT_EQ(ncclMemFree(localBuf), ncclSuccess);
  ASSERT_EQ(ncclMemFree(arBuf), ncclSuccess);
  freeWinBuf(winBase, sizeBytes, bufType);

  CUDACHECK_TEST(cudaStreamDestroy(get_stream));
  EXPECT_EQ(errs, 0u);
}

INSTANTIATE_TEST_SUITE_P(
    RMATestInstance,
    RMATestParam,
    ::testing::Combine(
        ::testing::Values(8192, 8 * 1024 * 1024), // kNumElements
        ::testing::Values(true, false), // ctranAllReduce
        ::testing::Values(
            MemAllocType::kMemNcclMemAlloc,
            MemAllocType::kMemCudaMalloc,
            MemAllocType::kMemHostManaged,
            MemAllocType::kMemHostUnregistered)),
    [](const testing::TestParamInfo<RMATestParam::ParamType>& info) {
      const auto kNumElements = std::get<0>(info.param);
      const auto ctranAllReduce = std::get<1>(info.param);
      const auto bufType = std::get<2>(info.param);
      return fmt::format(
          "numElem{}_{}_{}",
          kNumElements,
          ctranAllReduce ? "ctranAR" : "ncclAR",
          testMemAllocTypeToStr(bufType));
    });

// MultiWindow test: PutSignal + WaitSignal with varying element counts
class MultiWindowTestParam
    : public RMATest,
      public ::testing::WithParamInterface<std::tuple<size_t, size_t>> {};

TEST_P(MultiWindowTestParam, multiWindow) {
  const auto& [kMaxNumElements, kNumIters] = GetParam();
  EXPECT_GE(kMaxNumElements, 1);
  EXPECT_GE(kNumIters, 1);

  for (size_t numElements = 1; numElements <= kMaxNumElements;
       numElements = numElements * 2) {
    cudaStream_t put_stream, wait_stream;
    CUDACHECK_TEST(
        cudaStreamCreateWithFlags(&put_stream, cudaStreamNonBlocking));
    CUDACHECK_TEST(
        cudaStreamCreateWithFlags(&wait_stream, cudaStreamNonBlocking));

    size_t sizeBytes = numElements * sizeof(int) * numRanks;

    void* winBase = nullptr;
    ASSERT_EQ(ncclMemAlloc(&winBase, sizeBytes), ncclSuccess);
    ASSERT_NE(winBase, nullptr);

    ncclWindow_t win = nullptr;
    auto res = ncclCommWindowRegister(
        comm, winBase, sizeBytes, &win, NCCL_WIN_DEFAULT);
    ASSERT_EQ(res, ncclSuccess);

    int* localbuf = reinterpret_cast<int*>(winBase);

    EXPECT_THAT(win, testing::NotNull());
    assignChunkValue(localbuf, numElements * numRanks, globalRank, 1);
    this->barrier(comm, stream);

    int nextPeer = (globalRank + 1) % numRanks;
    int prevPeer = (globalRank + numRanks - 1) % numRanks;

    for (size_t iter = 0; iter < kNumIters; iter++) {
      NCCLCHECK_TEST(ncclPutSignal(
          localbuf + numElements * globalRank,
          numElements,
          ncclInt32,
          nextPeer,
          numElements * globalRank,
          win,
          put_stream));
      NCCLCHECK_TEST(ncclWaitSignal(prevPeer, win, wait_stream));
    }
    this->barrier(comm, stream);

    size_t errs = checkChunkValue(
        localbuf + numElements * prevPeer,
        numElements,
        prevPeer + static_cast<int>(numElements) * prevPeer,
        1,
        globalRank,
        wait_stream);
    EXPECT_EQ(errs, 0u);

    res = ncclCommWindowDeregister(comm, win);
    EXPECT_EQ(res, ncclSuccess);
    ASSERT_EQ(ncclMemFree(winBase), ncclSuccess);

    CUDACHECK_TEST(cudaStreamDestroy(put_stream));
    CUDACHECK_TEST(cudaStreamDestroy(wait_stream));
  }
}

INSTANTIATE_TEST_SUITE_P(
    MultiWindowTestInstance,
    MultiWindowTestParam,
    ::testing::Values(std::make_tuple(8, 2)));

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
