// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstddef>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/init/Init.h>

#include "comm.h"
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

  template <typename T>
  void assignChunkValue(T* buf, size_t count, T seed, T inc) {
    std::vector<T> expectedVals(count, 0);
    for (size_t i = 0; i < count; ++i) {
      expectedVals[i] = seed + i * inc;
    }
    CUDACHECK_TEST(cudaMemcpy(
        buf, expectedVals.data(), count * sizeof(T), cudaMemcpyDefault));
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
  const auto dump0 = comm->ctranComm_->ctran_->mapper->dumpExportRegCache();
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
  const auto dump1 = comm->ctranComm_->ctran_->mapper->dumpExportRegCache();
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
