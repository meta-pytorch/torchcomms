// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstddef>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/init/Init.h>

#include "comm.h"
#include "comms/ctran/Ctran.h"
#include "comms/ctran/regcache/IpcRegCache.h"
#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/ctran/tests/CtranTestUtils.h"
#include "comms/ctran/utils/Checks.h"
#include "comms/ctran/window/CtranWin.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "nccl.h"

using namespace ctran;

class CtranWinTest : public NcclxBaseTest {
 public:
  CtranWinTest() = default;

 protected:
  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE", "1", 0);
    setenv("NCCL_CTRAN_IB_EPOCH_LOCK_ENFORCE_CHECK", "true", 0);
    NcclxBaseTest::SetUp();
    CUDACHECK_TEST(cudaSetDevice(this->localRank));
  }
  void TearDown() override {
    // Check that all allocated memory segments have been freed
    EXPECT_TRUE(segments.empty()) << "Not all memory segments were freed";
  }

  void barrier(ncclComm_t comm, cudaStream_t stream) {
    // simple Allreduce as barrier before get data from other ranks
    void* buf;
    CUDACHECK_TEST(cudaMalloc(&buf, sizeof(char)));
    NCCLCHECK_TEST(ncclAllReduce(buf, buf, 1, ncclChar, ncclSum, comm, stream));
    CUDACHECK_TEST(cudaDeviceSynchronize());
    CUDACHECK_TEST(cudaFree(buf));
  }

  void createWin(
      ncclComm_t comm,
      bool isUserBuf,
      MemAllocType bufType,
      void** winBasePtr,
      CtranWin** winPtr,
      size_t sizeBytes) {
    meta::comms::Hints hints;
    auto res = commSuccess;
    // If userBuf is true, allocate buffer and use ctranWinRegister API
    if (isUserBuf) {
      *winBasePtr = commMemAlloc(sizeBytes, bufType, segments);

      res = ctranWinRegister(
          (void*)*winBasePtr, sizeBytes, comm->ctranComm_.get(), winPtr, hints);

    } else {
      hints.set(
          "window_buffer_location",
          bufType == MemAllocType::kMemHostManaged ||
                  bufType == MemAllocType::kMemHostUnregistered
              ? "cpu"
              : "gpu");
      res = ctranWinAllocate(
          sizeBytes, comm->ctranComm_.get(), (void**)winBasePtr, winPtr, hints);
    }
    ASSERT_EQ(res, commSuccess);
    ASSERT_NE(*winBasePtr, nullptr);
  }

  void
  freeWinBuf(bool isUserBuf, void* ptr, size_t size, MemAllocType bufType) {
    if (isUserBuf) {
      commMemFree(ptr, size, bufType);
      // Remove the segment from the tracking vector
      segments.erase(
          std::remove_if(
              segments.begin(),
              segments.end(),
              [ptr](const TestMemSegment& seg) { return seg.ptr == ptr; }),
          segments.end());
    }
  }
  std::vector<TestMemSegment> segments;
};

class CtranWinTestParam
    : public CtranWinTest,
      public ::testing::WithParamInterface<std::tuple<MemAllocType, bool>> {};

TEST_P(CtranWinTestParam, winAllocCreate) {
  auto [bufType, userBuf] = GetParam();

  ncclComm_t comm = createNcclComm(
      this->globalRank,
      this->numRanks,
      this->localRank,
      false,
      nullptr,
      server.get());

  ASSERT_NE(comm, nullptr);

  auto statex = comm->ctranComm_->statex_.get();
  ASSERT_NE(statex, nullptr);

  cudaStream_t stream = 0;
  CtranWin* win = nullptr;
  size_t sizeBytes = 8192 * sizeof(int);
  void* winBase = nullptr;
  createWin(comm, userBuf, bufType, &winBase, &win, sizeBytes);

  EXPECT_THAT(win, ::testing::NotNull());

  // Expect window allocation would trigger internal buffer registration export
  const auto dump0 = ctran::IpcRegCache::getInstance()->dumpExportRegCache();
  EXPECT_GE(dump0.size(), 0);

  for (int peer = 0; peer < this->numRanks; peer++) {
    void* remoteAddr = nullptr;
    auto res = ctranWinSharedQuery(peer, win, &remoteAddr);
    EXPECT_EQ(res, commSuccess);
    if (peer == statex->rank()) {
      EXPECT_EQ(remoteAddr, winBase);
      // For CPU window or peers on remote node, remote address is null
    } else if (!(statex->node(peer) == statex->node() &&
                 win->nvlEnabled(peer))) {
      EXPECT_THAT(remoteAddr, ::testing::IsNull());
    } else {
      // Do actual copy to validate remote address is accessible
      FB_CUDACHECKIGNORE(
          cudaMemcpy(remoteAddr, winBase, sizeBytes, cudaMemcpyDefault));
      EXPECT_THAT(remoteAddr, ::testing::NotNull());
    }
  }

  int next_peer = (this->globalRank + 1) % this->numRanks;
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(win->updateOpCount(next_peer), i);
    EXPECT_EQ(win->updateOpCount(next_peer, window::OpCountType::kPut), i);
    EXPECT_EQ(
        win->updateOpCount(next_peer, window::OpCountType::kWaitSignal), i);
  }

  // Barrier to ensure all peers have finished creation and query
  this->barrier(comm, stream);

  auto res = ctranWinFree(win);
  EXPECT_EQ(res, commSuccess);

  // This test only exported buffers in window, thus expect all exported cache
  // is freed upon window free
  const auto dump1 = ctran::IpcRegCache::getInstance()->dumpExportRegCache();
  EXPECT_EQ(dump1.size(), 0);

  freeWinBuf(userBuf, winBase, sizeBytes, bufType);

  finalizeNcclComm(globalRank, server.get());
  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

TEST_F(CtranWinTest, directCopy) {
  ncclComm_t comm = createNcclComm(
      this->globalRank,
      this->numRanks,
      this->localRank,
      false,
      nullptr,
      server.get());
  ASSERT_NE(comm, nullptr);

  auto statex = comm->ctranComm_->statex_.get();
  ASSERT_NE(statex, nullptr);

  if (statex->nLocalRanks() == 1) {
    NCCLCHECK_TEST(ncclCommDestroy(comm));
    GTEST_SKIP() << "Host needs to have at least 2 GPUs to run this test";
  }

  cudaStream_t stream = 0;
  CtranWin* win = nullptr;
  size_t count = 8192;
  void* winBase = nullptr;
  auto res = ctranWinAllocate(
      count * sizeof(int), comm->ctranComm_.get(), &winBase, &win);
  EXPECT_EQ(res, commSuccess);
  ASSERT_NE(winBase, nullptr);

  int* localWinAddr = reinterpret_cast<int*>(winBase);
  int seed = this->globalRank * count;
  assignChunkValue(localWinAddr, count, seed, 1);

  // Barrier to ensure remote GPU has finished data write to window
  this->barrier(comm, stream);

  srand(time(NULL));
  std::vector<int> remoteData_host(count, rand());
  for (int peer = 0; peer < this->numRanks; ++peer) {
    // Direct remote memory access is only allowed for local GPUs
    if ((peer != this->globalRank) && (statex->node() == statex->node(peer))) {
      void* remoteWinBase = nullptr;
      res = ctranWinSharedQuery(peer, win, &remoteWinBase);
      EXPECT_EQ(res, commSuccess);
      EXPECT_THAT(remoteWinBase, ::testing::NotNull());

      FB_CUDACHECKIGNORE(cudaMemcpy(
          remoteData_host.data(),
          remoteWinBase,
          count * sizeof(int),
          cudaMemcpyDefault));
      for (size_t i = 0; i < count; ++i) {
        int seed = peer * count;
        EXPECT_EQ(remoteData_host[i], seed + i);
      }
    }
  }

  // Barrier to ensure all peers have completed remote data access
  this->barrier(comm, stream);

  res = ctranWinFree(win);
  EXPECT_EQ(res, commSuccess);

  finalizeNcclComm(globalRank, server.get());
  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

TEST_F(CtranWinTest, nvlDisabled) {
  EnvRAII env1(
      NCCL_CTRAN_BACKENDS,
      std::vector<enum NCCL_CTRAN_BACKENDS>{NCCL_CTRAN_BACKENDS::ib});
  ncclComm_t comm = createNcclComm(
      this->globalRank,
      this->numRanks,
      this->localRank,
      false,
      nullptr,
      server.get());

  ASSERT_NE(comm, nullptr);

  auto statex = comm->ctranComm_->statex_.get();
  ASSERT_NE(statex, nullptr);

  CtranWin* win = nullptr;
  size_t sizeBytes = 8192 * sizeof(int);
  void* winBase = nullptr;
  auto res =
      ctranWinAllocate(sizeBytes, comm->ctranComm_.get(), &winBase, &win);
  ASSERT_EQ(res, commSuccess);
  ASSERT_NE(winBase, nullptr);

  EXPECT_THAT(win, ::testing::NotNull());

  for (int peer = 0; peer < this->numRanks; peer++) {
    ASSERT_EQ(win->nvlEnabled(peer), false);
    void* remoteAddr = nullptr;
    ASSERT_EQ(ctranWinSharedQuery(peer, win, &remoteAddr), commSuccess);
    // Expect can only directly access local GPU's window
    if (peer == statex->rank()) {
      EXPECT_EQ(remoteAddr, winBase);
    } else {
      EXPECT_THAT(remoteAddr, ::testing::IsNull());
    }
  }

  res = ctranWinFree(win);
  EXPECT_EQ(res, commSuccess);

  finalizeNcclComm(globalRank, server.get());
  NCCLCHECK_TEST(ncclCommDestroy(comm));
}

// Test fixture using CtranDistTestFixture (without NCCL dependency)
class CtranWinDistTest : public ctran::CtranDistTestFixture {
 public:
  CtranWinDistTest() = default;

 protected:
  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE", "1", 0);
    setenv("NCCL_CTRAN_IB_EPOCH_LOCK_ENFORCE_CHECK", "true", 0);
    CtranDistTestFixture::SetUp();
  }

  void TearDown() override {
    CtranDistTestFixture::TearDown();
  }

  void barrier(CtranComm* comm) {
    auto resFuture = comm->bootstrap_->barrier(
        comm->statex_->rank(), comm->statex_->nRanks());
    ASSERT_EQ(
        static_cast<commResult_t>(std::move(resFuture).get()), commSuccess);
  }
};

// Test asymmetric window allocation where each rank allocates different sizes
TEST_F(CtranWinDistTest, asymmetricWindowAllocation) {
  auto comm = makeCtranComm();
  ASSERT_NE(comm, nullptr);

  auto statex = comm->statex_.get();
  ASSERT_NE(statex, nullptr);

  // Define arbitrary sizes for each rank (not following any formula)
  const std::vector<size_t> rankSizes = {
      4096, // rank 0: 4KB
      16384, // rank 1: 16KB
      8192, // rank 2: 8KB
      32768, // rank 3: 32KB
      2048, // rank 4: 2KB
      65536, // rank 5: 64KB
      12288, // rank 6: 12KB
      24576, // rank 7: 24KB
  };

  // Each rank picks its size from the vector
  size_t localSizeBytes =
      rankSizes[this->globalRank % rankSizes.size()] * sizeof(int);

  CtranWin* win = nullptr;
  void* winBase = nullptr;
  auto res = ctranWinAllocate(localSizeBytes, comm.get(), &winBase, &win);
  ASSERT_EQ(res, commSuccess);
  ASSERT_NE(winBase, nullptr);
  EXPECT_THAT(win, ::testing::NotNull());

  // Verify that remWinInfo has correct sizes for all ranks
  ASSERT_EQ(win->remWinInfo.size(), static_cast<size_t>(this->numRanks));

  for (int peer = 0; peer < this->numRanks; peer++) {
    // Expected size for each peer from the predefined vector
    size_t expectedSize = rankSizes[peer % rankSizes.size()] * sizeof(int);
    // Account for minimum size and 8-byte alignment applied internally by
    // ctranWinAllocate (see window.cc) to ensure signal buffer alignment
    if (expectedSize < CTRAN_MIN_REGISTRATION_SIZE) {
      expectedSize = CTRAN_MIN_REGISTRATION_SIZE;
    }
    // Round up to 8-byte alignment to match window.cc internal behavior
    expectedSize = (expectedSize + 7) & ~7;

    EXPECT_EQ(win->remWinInfo[peer].dataBytes, expectedSize)
        << "Mismatch for peer " << peer << ": expected " << expectedSize
        << " but got " << win->remWinInfo[peer].dataBytes;

    // Verify getDataSize returns correct value
    EXPECT_EQ(win->getDataSize(peer), expectedSize)
        << "getDataSize mismatch for peer " << peer;
  }

  // Verify local size matches what we allocated
  size_t myExpectedSize = localSizeBytes;
  if (myExpectedSize < CTRAN_MIN_REGISTRATION_SIZE) {
    myExpectedSize = CTRAN_MIN_REGISTRATION_SIZE;
  }
  myExpectedSize = (myExpectedSize + 7) & ~7;
  EXPECT_EQ(win->dataBytes, myExpectedSize);

  // Verify remote addresses are valid for local peers
  for (int peer = 0; peer < this->numRanks; peer++) {
    void* remoteAddr = nullptr;
    res = ctranWinSharedQuery(peer, win, &remoteAddr);
    EXPECT_EQ(res, commSuccess);

    if (peer == statex->rank()) {
      EXPECT_EQ(remoteAddr, winBase);
    } else if (statex->node(peer) == statex->node() && win->nvlEnabled(peer)) {
      // For local peers with NVL, address should be non-null
      EXPECT_THAT(remoteAddr, ::testing::NotNull());
    }
  }

  res = ctranWinFree(win);
  EXPECT_EQ(res, commSuccess);
}

// Test asymmetric window with direct memory copy between ranks
TEST_F(CtranWinDistTest, asymmetricWindowDirectCopy) {
  auto comm = makeCtranComm();
  ASSERT_NE(comm, nullptr);

  auto statex = comm->statex_.get();
  ASSERT_NE(statex, nullptr);

  if (statex->nLocalRanks() == 1) {
    GTEST_SKIP() << "Host needs to have at least 2 GPUs to run this test";
  }

  // Define arbitrary element counts for each rank (not following any formula)
  const std::vector<size_t> rankCounts = {
      1024, // rank 0
      4096, // rank 1
      2048, // rank 2
      8192, // rank 3
      512, // rank 4
      16384, // rank 5
      3072, // rank 6
      6144, // rank 7
  };

  // Each rank picks its count from the vector
  size_t localCount = rankCounts[this->globalRank % rankCounts.size()];
  size_t localSizeBytes = localCount * sizeof(int);

  CtranWin* win = nullptr;
  void* winBase = nullptr;
  auto res = ctranWinAllocate(localSizeBytes, comm.get(), &winBase, &win);
  EXPECT_EQ(res, commSuccess);
  ASSERT_NE(winBase, nullptr);

  // Initialize local window with rank-specific data
  int* localWinAddr = reinterpret_cast<int*>(winBase);
  int seed = this->globalRank * 10000;
  assignChunkValue(localWinAddr, localCount, seed, 1);

  // Barrier to ensure all ranks have initialized their windows
  this->barrier(comm.get());

  // Read from other local ranks and verify data
  for (int peer = 0; peer < this->numRanks; ++peer) {
    if ((peer != this->globalRank) && (statex->node() == statex->node(peer))) {
      void* remoteWinBase = nullptr;
      res = ctranWinSharedQuery(peer, win, &remoteWinBase);
      EXPECT_EQ(res, commSuccess);

      if (remoteWinBase != nullptr) {
        // Peer's count from the predefined vector
        size_t peerCount = rankCounts[peer % rankCounts.size()];
        std::vector<int> remoteData_host(peerCount, 0);

        FB_CUDACHECKIGNORE(cudaMemcpy(
            remoteData_host.data(),
            remoteWinBase,
            peerCount * sizeof(int),
            cudaMemcpyDefault));

        // Verify the data matches what peer wrote
        int peerSeed = peer * 10000;
        for (size_t i = 0; i < peerCount; ++i) {
          EXPECT_EQ(remoteData_host[i], peerSeed + static_cast<int>(i))
              << "Data mismatch at index " << i << " from peer " << peer;
        }
      }
    }
  }

  // Barrier before cleanup
  this->barrier(comm.get());

  res = ctranWinFree(win);
  EXPECT_EQ(res, commSuccess);
}

// Test asymmetric window with PUT and GET operations between ranks
TEST_F(CtranWinDistTest, asymmetricWindowPutGet) {
  auto comm = makeCtranComm();
  ASSERT_NE(comm, nullptr);

  auto statex = comm->statex_.get();
  ASSERT_NE(statex, nullptr);

  if (statex->nRanks() < 2) {
    GTEST_SKIP() << "Need at least 2 ranks to run this test";
  }

  // Define arbitrary element counts for each rank
  const std::vector<size_t> rankCounts = {
      1024, // rank 0
      2048, // rank 1
      4096, // rank 2
      8192, // rank 3
      512, // rank 4
      16384, // rank 5
      3072, // rank 6
      6144, // rank 7
  };

  const int rank = statex->rank();
  const int numRanks = statex->nRanks();
  const int nextPeer = (rank + 1) % numRanks;
  const int prevPeer = (rank + numRanks - 1) % numRanks;

  // Each rank allocates window buffer sized to hold data from all ranks
  size_t maxCount = *std::max_element(rankCounts.begin(), rankCounts.end());
  size_t winSizeBytes = maxCount * sizeof(int) * numRanks;

  CtranWin* win = nullptr;
  void* winBase = nullptr;
  auto res = ctranWinAllocate(winSizeBytes, comm.get(), &winBase, &win);
  ASSERT_EQ(res, commSuccess);
  ASSERT_NE(winBase, nullptr);

  size_t localCount = rankCounts[rank % rankCounts.size()];
  size_t prevPeerCount = rankCounts[prevPeer % rankCounts.size()];

  // Local buffers for PUT and GET
  int* putBuf = nullptr;
  int* getBuf = nullptr;
  CUDACHECK_TEST(cudaMalloc(&putBuf, localCount * sizeof(int)));
  CUDACHECK_TEST(cudaMalloc(&getBuf, localCount * sizeof(int)));

  // Initialize window with -1, PUT buffer with rank data, GET buffer with -1
  int* winBufInt = reinterpret_cast<int*>(winBase);
  CUDACHECK_TEST(cudaMemset(winBase, -1, winSizeBytes));
  assignChunkValue(putBuf, localCount, rank * 1000, 1);
  CUDACHECK_TEST(cudaMemset(getBuf, -1, localCount * sizeof(int)));

  cudaStream_t putStream, getStream, waitStream;
  CUDACHECK_TEST(cudaStreamCreateWithFlags(&putStream, cudaStreamNonBlocking));
  CUDACHECK_TEST(cudaStreamCreateWithFlags(&getStream, cudaStreamNonBlocking));
  CUDACHECK_TEST(cudaStreamCreateWithFlags(&waitStream, cudaStreamNonBlocking));

  this->barrier(comm.get());

  // === PUT test: each rank PUTs to next peer ===
  const int kNumIters = 10;
  for (int iter = 0; iter < kNumIters; iter++) {
    COMMCHECK_TEST(ctranPutSignal(
        putBuf,
        localCount,
        commInt32,
        nextPeer,
        maxCount * rank,
        win,
        putStream,
        true));
    COMMCHECK_TEST(ctranWaitSignal(prevPeer, win, waitStream));
  }

  CUDACHECK_TEST(cudaStreamSynchronize(putStream));
  CUDACHECK_TEST(cudaStreamSynchronize(waitStream));

  // Verify PUT: check data received from previous peer
  std::vector<int> putRecvData(prevPeerCount, -1);
  CUDACHECK_TEST(cudaMemcpy(
      putRecvData.data(),
      winBufInt + maxCount * prevPeer,
      prevPeerCount * sizeof(int),
      cudaMemcpyDefault));

  int putErrs = 0;
  for (size_t i = 0; i < prevPeerCount; ++i) {
    int expected = prevPeer * 1000 + static_cast<int>(i);
    if (putRecvData[i] != expected && putErrs++ < 10) {
      XLOG(ERR) << "PUT: Rank " << rank << ": data[" << i
                << "] = " << putRecvData[i] << ", expected = " << expected;
    }
  }
  EXPECT_EQ(putErrs, 0) << "PUT verification failed";

  this->barrier(comm.get());

  // === GET test: each rank GETs from next peer ===
  // After PUT phase, nextPeer's window has data from rank at offset
  // maxCount*rank So we GET from nextPeer's window at offset maxCount*rank (our
  // own data)
  for (int iter = 0; iter < kNumIters; iter++) {
    COMMCHECK_TEST(ctranGet(
        getBuf,
        maxCount * rank,
        localCount,
        commInt32,
        nextPeer,
        win,
        comm.get(),
        getStream));
  }

  CUDACHECK_TEST(cudaStreamSynchronize(getStream));
  this->barrier(comm.get());

  // Verify GET: we should get back our own data that we PUT to nextPeer
  std::vector<int> getRecvData(localCount, -1);
  CUDACHECK_TEST(cudaMemcpy(
      getRecvData.data(), getBuf, localCount * sizeof(int), cudaMemcpyDefault));

  int getErrs = 0;
  for (size_t i = 0; i < localCount; ++i) {
    int expected = rank * 1000 + static_cast<int>(i);
    if (getRecvData[i] != expected && getErrs++ < 10) {
      XLOG(ERR) << "GET: Rank " << rank << ": data[" << i
                << "] = " << getRecvData[i] << ", expected = " << expected;
    }
  }
  EXPECT_EQ(getErrs, 0) << "GET verification failed";

  CUDACHECK_TEST(cudaStreamDestroy(putStream));
  CUDACHECK_TEST(cudaStreamDestroy(getStream));
  CUDACHECK_TEST(cudaStreamDestroy(waitStream));
  CUDACHECK_TEST(cudaFree(putBuf));
  CUDACHECK_TEST(cudaFree(getBuf));

  res = ctranWinFree(win);
  EXPECT_EQ(res, commSuccess);
}

INSTANTIATE_TEST_SUITE_P(
    CtranWinInstance,
    CtranWinTestParam,
    ::testing::Combine(
        // bufType, userBuf
        ::testing::Values(
            MemAllocType::kMemCuMemAlloc,
            MemAllocType::kMemCudaMalloc,
            MemAllocType::kMemHostManaged,
            MemAllocType::kMemHostUnregistered),
        ::testing::Values(true, false)));

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
