// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/init/Init.h>

#include <nccl.h>
#include <cstddef>
#include "comm.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"

class RMATest : public ::testing::Test {
 public:
  RMATest() = default;
  ncclComm_t comm{nullptr};

 protected:
  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE", "1", 0); // enable ctran
    setenv("NCCL_CTRAN_IB_EPOCH_LOCK_ENFORCE_CHECK", "true", 0);

    std::tie(this->localRank, this->globalRank, this->numRanks) = getMpiInfo();

    CUDACHECK_TEST(cudaSetDevice(this->localRank));
    this->comm =
        createNcclComm(this->globalRank, this->numRanks, this->localRank);
    ASSERT_NE(this->comm, nullptr);
  }
  void TearDown() override {
    // Destroy the communicator at the end
    NCCLCHECK_TEST(ncclCommDestroy(this->comm));
    // Check that all allocated memory segments have been freed
    EXPECT_TRUE(segments.empty()) << "Not all memory segments were freed";
  }

  void barrier(ncclComm_t ncclComm, cudaStream_t stream) {
    // simple Allreduce as barrier before get data from other ranks
    void* buf;
    CUDACHECK_TEST(cudaMalloc(&buf, sizeof(char)));
    NCCLCHECK_TEST(
        ncclAllReduce(buf, buf, 1, ncclChar, ncclSum, ncclComm, stream));
    CUDACHECK_TEST(cudaDeviceSynchronize());
    CUDACHECK_TEST(cudaFree(buf));
  }

  void createWin(
      bool isUserBuf,
      MemAllocType bufType,
      void** winBasePtr,
      ncclWindow_t* winPtr,
      size_t sizeBytes,
      std::vector<int>& buf) {
    auto res = ncclSuccess;
    ncclx::Hints hints;

    // If userBuf is true, allocate buffer and use ctranWinRegister API
    if (isUserBuf) {
      *winBasePtr = testAllocBuf(sizeBytes, bufType, segments);
      res = ncclCommWindowRegister(
          comm,
          const_cast<void*>(*winBasePtr),
          sizeBytes,
          winPtr,
          NCCL_WIN_DEFAULT);

    } else {
      hints.set(
          "window_buffer_location",
          bufType == MemAllocType::kMemHostManaged ||
                  bufType == MemAllocType::kMemHostUnregistered
              ? "cpu"
              : "gpu");
      res = ncclWinAllocate(sizeBytes, comm, winBasePtr, winPtr, hints);
    }
    ASSERT_EQ(res, ncclSuccess);
    ASSERT_NE(*winBasePtr, nullptr);
  }

  void
  freeWinBuf(bool isUserBuf, void* ptr, size_t size, MemAllocType bufType) {
    if (isUserBuf) {
      testFreeBuf(ptr, size, bufType);
      // Remove the segment from the tracking vector
      segments.erase(
          std::remove_if(
              segments.begin(),
              segments.end(),
              [ptr](const TestMemSegment& seg) { return seg.ptr == ptr; }),
          segments.end());
    }
  }

  int localRank{0};
  int globalRank{0};
  int numRanks{0};
  std::vector<TestMemSegment> segments;
};

class MultiWindowTestParam
    : public RMATest,
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
    ncclWindow_t win = nullptr;
    void* winBase = nullptr;
    auto res = ncclWinAllocate(sizeBytes, comm, &winBase, &win);
    ASSERT_EQ(res, ncclSuccess);
    ASSERT_NE(winBase, nullptr);
    int* localbuf = reinterpret_cast<int*>(winBase);

    EXPECT_THAT(win, testing::NotNull());
    assignChunkValue(
        localbuf, numElements * statex->nRanks(), statex->rank(), 1);
    // Barrier to ensure all peers have finished creation
    this->barrier(comm, main_stream);

    int nextPeer = (this->globalRank + 1) % this->numRanks;
    int prevPeer = (this->globalRank + this->numRanks - 1) % this->numRanks;

    for (auto iter = 0; iter < kNumIters; iter++) {
      NCCLCHECK_TEST(ncclPutSignal(
          localbuf + numElements * statex->rank(),
          numElements,
          ncclInt32,
          nextPeer,
          numElements * statex->rank(),
          win,
          put_stream));
      NCCLCHECK_TEST(ncclWaitSignal(prevPeer, win, wait_stream));
    }
    // Barrier to ensure all peers have finished put
    this->barrier(comm, main_stream);

    // Check results
    int errs = checkChunkValue(
        localbuf + numElements * prevPeer,
        numElements,
        prevPeer + (int)numElements * prevPeer,
        1,
        this->globalRank,
        wait_stream);
    EXPECT_EQ(errs, 0);

    res = ncclWinFree(comm, win);
    EXPECT_EQ(res, ncclSuccess);

    CUDACHECK_TEST(cudaStreamDestroy(put_stream));
    CUDACHECK_TEST(cudaStreamDestroy(wait_stream));
  }
}

class RMATestParam : public RMATest,
                     public ::testing::WithParamInterface<
                         std::tuple<size_t, size_t, bool, MemAllocType, bool>> {
};

TEST_P(RMATestParam, winPutWait) {
  const auto& [kNumElements, kNumIters, ctranAllReduce, bufType, userBuf] =
      GetParam();
  EXPECT_GE(kNumElements, 8192);
  EXPECT_GE(kNumIters, 1);

  // Enable ctran for all-reduce
  auto envGuard = EnvRAII(
      NCCL_ALLREDUCE_ALGO,
      ctranAllReduce ? NCCL_ALLREDUCE_ALGO::ctdirect
                     : NCCL_ALLREDUCE_ALGO::orig);

  auto comm = this->comm;
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

  ncclWindow_t win = nullptr;
  void* winBase = nullptr;
  std::vector<int> buf(kNumElements * statex->nRanks(), 0);

  createWin(userBuf, bufType, &winBase, &win, sizeBytes, buf);

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
    auto res = ncclWinSharedQuery(peer, comm, win, &remoteAddr);
    EXPECT_EQ(res, ncclSuccess);
    if (peer == statex->rank()) {
      EXPECT_EQ(remoteAddr, winBase);
    } else if (
        bufType == MemAllocType::kMemHostManaged ||
        bufType == MemAllocType::kMemHostUnregistered ||
        statex->node() != statex->node(peer)) {
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
    NCCLCHECK_TEST(ncclPutSignal(
        localBuf,
        kNumElements,
        ncclInt32,
        nextPeer,
        kNumElements * statex->rank(),
        win,
        put_stream));
    NCCLCHECK_TEST(ncclWaitSignal(prevPeer, win, wait_stream));
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

  auto res = ncclWinFree(comm, win);
  EXPECT_EQ(res, ncclSuccess);

  ASSERT_EQ(ncclCommDeregister(comm, localHdl), ncclSuccess);
  ASSERT_EQ(ncclMemFree(localBuf), ncclSuccess);
  freeWinBuf(userBuf, winBase, sizeBytes, bufType);

  CUDACHECK_TEST(cudaEventDestroy(start_event));
  CUDACHECK_TEST(cudaEventDestroy(end_event));
  CUDACHECK_TEST(cudaStreamDestroy(put_stream));
  CUDACHECK_TEST(cudaStreamDestroy(wait_stream));

  EXPECT_EQ(errs, 0);
}

TEST_P(RMATestParam, winPutOnly) {
  const auto& [kNumElements, kNumIters, ctranAllReduce, bufType, userBuf] =
      GetParam();
  EXPECT_GE(kNumElements, 8192);
  EXPECT_GE(kNumIters, 1);

  // Enable ctran for all-reduce
  auto envGuard = EnvRAII(
      NCCL_ALLREDUCE_ALGO,
      ctranAllReduce ? NCCL_ALLREDUCE_ALGO::ctdirect
                     : NCCL_ALLREDUCE_ALGO::orig);

  auto comm = this->comm;
  auto statex = comm->ctranComm_->statex_.get();
  ASSERT_NE(statex, nullptr);
  EXPECT_EQ(statex->nRanks(), this->numRanks);

  cudaStream_t main_stream = 0, put_stream;
  CUDACHECK_TEST(cudaStreamCreateWithFlags(&put_stream, cudaStreamNonBlocking));

  size_t sizeBytes = kNumElements * sizeof(int) * statex->nRanks();

  ncclWindow_t win = nullptr;
  void* winBase = nullptr;

  std::vector<int> buf(kNumElements * statex->nRanks(), 0);

  createWin(userBuf, bufType, &winBase, &win, sizeBytes, buf);

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
    NCCLCHECK_TEST(ncclPut(
        localBuf,
        kNumElements,
        ncclInt32,
        nextPeer,
        kNumElements * rank,
        win,
        put_stream));
  }

  // A couple of all-reduce after RMA tests
  for (auto iter = 0; iter < kNumIters; iter++) {
    NCCLCHECK_TEST(ncclAllReduce(
        localBuf,
        localBuf,
        kNumElements,
        ncclInt32,
        ncclSum,
        comm,
        put_stream));
  }
  // allreduce ensures all remote puts have finished
  int errs = checkChunkValue(
      (int*)winBase + kNumElements * prevPeer,
      kNumElements,
      prevPeer,
      1,
      this->globalRank,
      put_stream);

  auto res = ncclWinFree(comm, win);
  EXPECT_EQ(res, ncclSuccess);

  ASSERT_EQ(ncclCommDeregister(comm, localHdl), ncclSuccess);
  ASSERT_EQ(ncclMemFree(localBuf), ncclSuccess);
  freeWinBuf(userBuf, winBase, sizeBytes, bufType);

  CUDACHECK_TEST(cudaStreamDestroy(put_stream));
  EXPECT_EQ(errs, 0);
}

TEST_P(RMATestParam, winGet) {
  const auto& [kNumElements, kNumIters, ctranAllReduce, bufType, userBuf] =
      GetParam();
  EXPECT_GE(kNumElements, 8192);
  EXPECT_GE(kNumIters, 1);

  // Enable ctran for all-reduce
  auto envGuard = EnvRAII(
      NCCL_ALLREDUCE_ALGO,
      ctranAllReduce ? NCCL_ALLREDUCE_ALGO::ctdirect
                     : NCCL_ALLREDUCE_ALGO::orig);

  auto comm = this->comm;
  auto statex = comm->ctranComm_->statex_.get();
  ASSERT_NE(statex, nullptr);
  EXPECT_EQ(statex->nRanks(), this->numRanks);

  cudaStream_t main_stream = 0, get_stream;
  CUDACHECK_TEST(cudaStreamCreateWithFlags(&get_stream, cudaStreamNonBlocking));

  size_t sizeBytes = kNumElements * sizeof(int) * statex->nRanks();

  ncclWindow_t win = nullptr;
  void* winBase = nullptr;
  std::vector<int> buf(kNumElements * statex->nRanks(), 0);

  createWin(userBuf, bufType, &winBase, &win, sizeBytes, buf);

  // Always allocate localBuf from GPU mem
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

  const auto rank = statex->rank();
  const auto numRanks = statex->nRanks();

  assignChunkValue((int*)winBase, kNumElements * statex->nRanks(), rank, 0);
  assignChunkValue(localBuf, kNumElements, statex->rank(), -1);
  // Barrier to ensure all peers have finished value assignment
  this->barrier(comm, main_stream);

  int nextPeer = (rank + 1) % numRanks;

  for (auto iter = 0; iter < kNumIters; iter++) {
    // Put data to next peer at offset of kNumElements * rank
    NCCLCHECK_TEST(ncclGet(
        localBuf,
        kNumElements * rank,
        kNumElements,
        ncclInt32,
        nextPeer,
        win,
        get_stream));
  }

  // A couple of all-reduce after RMA tests
  for (auto iter = 0; iter < kNumIters; iter++) {
    NCCLCHECK_TEST(ncclAllReduce(
        arBuf, arBuf, kNumElements, ncclInt32, ncclSum, comm, get_stream));
  }
  CUDACHECK_TEST(cudaStreamSynchronize(get_stream));
  this->barrier(comm, main_stream);

  // allreduce ensures all remote puts have finished
  int errs = checkChunkValue(
      (int*)localBuf, kNumElements, nextPeer, 0, this->globalRank, get_stream);

  auto res = ncclWinFree(comm, win);
  EXPECT_EQ(res, ncclSuccess);

  ASSERT_EQ(ncclCommDeregister(comm, localHdl), ncclSuccess);
  ASSERT_EQ(ncclMemFree(localBuf), ncclSuccess);
  freeWinBuf(userBuf, winBase, sizeBytes, bufType);

  CUDACHECK_TEST(cudaStreamDestroy(get_stream));
  EXPECT_EQ(errs, 0);
}

INSTANTIATE_TEST_SUITE_P(
    RMATestInstance,
    RMATestParam,
    ::testing::Combine(
        // kNumElements, kNumIters, ctranAllReduce, bufType, userBuf
        ::testing::Values(8192, 8 * 1024 * 1024),
        ::testing::Values(500),
        ::testing::Values(true, false),
        ::testing::Values(
            MemAllocType::kMemNcclMemAlloc,
            MemAllocType::kMemCudaMalloc,
            MemAllocType::kMemHostManaged,
            MemAllocType::kMemHostUnregistered),
        ::testing::Values(true, false)),
    [](const testing::TestParamInfo<RMATestParam::ParamType>& info) {
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

class NvlEnabledTestParam
    : public RMATest,
      public ::testing::WithParamInterface<
          std::tuple<std::vector<enum NCCL_CTRAN_BACKENDS>, bool>> {};

TEST_P(NvlEnabledTestParam, ncclWinGetAttributes) {
  const auto& [backends, expectNvlEnabled] = GetParam();
  EnvRAII envBackend(NCCL_CTRAN_BACKENDS, backends);

  // create a test comm using the provided backends
  ncclComm_t comm = createNcclComm(
      this->globalRank, this->numRanks, this->localRank, false, nullptr);
  ASSERT_NE(comm, nullptr);

  auto statex = comm->ctranComm_->statex_.get();
  ASSERT_NE(statex, nullptr);
  EXPECT_EQ(statex->nRanks(), this->numRanks);

  size_t sizeBytes = 8192 * sizeof(int);
  ncclWindow_t win = nullptr;
  void* winBase = nullptr;
  ASSERT_EQ(ncclWinAllocate(sizeBytes, comm, &winBase, &win), ncclSuccess);
  ASSERT_NE(winBase, nullptr);

  EXPECT_THAT(win, testing::NotNull());

  for (int peer = 0; peer < this->numRanks; peer++) {
    ncclWinAttr_t winAttr;
    EXPECT_EQ(ncclWinGetAttributes(peer, win, &winAttr), ncclSuccess);

    // NVL is only expected to be enabled for local peers when NVL backend is
    // configured
    bool expectedEnabled =
        expectNvlEnabled && (statex->node() == statex->node(peer));
    auto expectedType = expectedEnabled
        ? ncclWinAccessType::ncclWinAccessUnified
        : ncclWinAccessType::ncclWinAccessSeparate;
    EXPECT_EQ(winAttr->accessType, expectedType)
        << "Peer " << peer << ": expected accessType=" << expectedType
        << ", got " << winAttr->accessType;
    delete winAttr;
  }

  EXPECT_EQ(ncclWinFree(comm, win), ncclSuccess);

  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

INSTANTIATE_TEST_SUITE_P(
    NvlEnabledTestInstance,
    NvlEnabledTestParam,
    ::testing::Values(
        std::make_tuple(
            std::vector<enum NCCL_CTRAN_BACKENDS>(
                {NCCL_CTRAN_BACKENDS::nvl, NCCL_CTRAN_BACKENDS::ib}),
            true),
        std::make_tuple(
            std::vector<enum NCCL_CTRAN_BACKENDS>({NCCL_CTRAN_BACKENDS::ib}),
            false)));

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
