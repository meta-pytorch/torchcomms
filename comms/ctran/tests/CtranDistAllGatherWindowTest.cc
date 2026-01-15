// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <comm.h>
#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nccl.h>
#include <stdlib.h>

#include "CtranUtUtils.h"
#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/AllGather/AllGatherImpl.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestsCuUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

using namespace ctran;

class CtranAllGatherWindowTest : public CtranDistBaseTest {
 public:
  CtranAllGatherWindowTest() = default;
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
    for (auto i = 0; i < count; ++i) {
      T val = seed + inc * i;
      if (observedVals[i] != val) {
        if (errs < 10) {
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
      MemAllocType bufType,
      void** winBasePtr,
      CtranWin** winPtr,
      size_t sizeBytes,
      meta::comms::Hints hints) {
    auto res = commSuccess;
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

class AllGatherWindowTestParam
    : public CtranAllGatherWindowTest,
      public ::testing::WithParamInterface<
          std::tuple<size_t, size_t, MemAllocType, bool>> {};

/**
 * Test: AllGatherWindow
 *
 * Tests the window-based allgather implementation that uses RMA put+signal.
 *
 * Each rank:
 * 1. Allocates a window for the recv buffer
 * 2. Initializes its send buffer with rank-specific values
 * 3. Calls ctranAllGatherWindow to gather data from all ranks
 * 4. Verifies that each slot in the recv buffer contains the correct data
 *
 * Expected recv buffer layout after allgather:
 * [rank0_data | rank1_data | ... | rankN-1_data]
 * where rank_i_data = [i, i+1, i+2, ..., i+numElements-1]
 */
TEST_P(AllGatherWindowTestParam, AllGatherWindow) {
  const auto& [kNumElements, kNumIters, bufType, userBuf] = GetParam();
  EXPECT_GE(kNumElements, 1);
  EXPECT_GE(kNumIters, 1);

  auto comm = this->comm;
  auto statex = comm->ctranComm_->statex_.get();
  ASSERT_NE(statex, nullptr);
  EXPECT_EQ(statex->nRanks(), this->numRanks);

  const int rank = statex->rank();
  const int nRanks = statex->nRanks();

  cudaStream_t main_stream = 0;
  cudaStream_t ag_stream;
  CUDACHECK_TEST(cudaStreamCreateWithFlags(&ag_stream, cudaStreamNonBlocking));

  // Window size: sendcount * nRanks * sizeof(int)
  size_t sendBytes = kNumElements * sizeof(int);
  size_t sizeBytes = sendBytes * nRanks;

  meta::comms::Hints hints;
  CtranWin* win = nullptr;
  void* winBase = nullptr;
  createWin(userBuf, bufType, &winBase, &win, sizeBytes, hints);
  int* recvBuf = reinterpret_cast<int*>(winBase);

  // Allocate send buffer
  int* sendBuf = nullptr;
  CUDACHECK_TEST(cudaMalloc((void**)&sendBuf, sendBytes));

  EXPECT_THAT(win, ::testing::NotNull());

  // Initialize recv buffer with -1
  assignChunkValue(recvBuf, kNumElements * nRanks, -1, 0);

  // Initialize send buffer with rank-specific values: [rank, rank+1, rank+2,
  // ...]
  assignChunkValue(sendBuf, kNumElements, rank, 1);

  // Barrier to ensure all peers have finished initialization
  this->barrier(comm, main_stream);

  for (auto iter = 0; iter < kNumIters; iter++) {
    // Reset recv buffer for each iteration (except first)
    if (iter > 0) {
      // Barrier to ensure all peers finished previous iteration before reset
      this->barrier(comm, main_stream);
      assignChunkValue(recvBuf, kNumElements * nRanks, -1, 0);
      CUDACHECK_TEST(cudaDeviceSynchronize());
      // Barrier after reset to ensure all ranks finished resetting before
      // any rank starts the next iteration's puts
      this->barrier(comm, main_stream);
    }

    // Call window-based allgather
    COMMCHECK_TEST(ctranAllGatherWindow(
        sendBuf, // send buffer
        kNumElements, // count
        commInt32, // datatype
        win, // window (recv buffer is win->winDataPtr)
        ag_stream)); // stream

    // Sync to ensure this iteration completes
    CUDACHECK_TEST(cudaStreamSynchronize(ag_stream));
  }

  // Wait for completion
  CUDACHECK_TEST(cudaStreamSynchronize(ag_stream));

  // Barrier to ensure all peers have finished
  this->barrier(comm, main_stream);

  // Verify results: each slot should contain [peer, peer+1, peer+2, ...]
  int totalErrs = 0;
  for (int peer = 0; peer < nRanks; peer++) {
    int errs = checkChunkValue(
        recvBuf + kNumElements * peer,
        kNumElements,
        peer, // seed: peer rank
        1, // increment
        main_stream);
    if (errs > 0) {
      std::cout << "[" << rank << "] Errors in slot for peer " << peer << ": "
                << errs << std::endl;
    }
    totalErrs += errs;
  }
  EXPECT_EQ(totalErrs, 0);

  // Cleanup
  auto res = ctranWinFree(win);
  EXPECT_EQ(res, ncclSuccess);

  CUDACHECK_TEST(cudaFree(sendBuf));
  freeWinBuf(userBuf, winBase, sizeBytes, bufType);
  CUDACHECK_TEST(cudaStreamDestroy(ag_stream));
}

/**
 * Test: AllGatherWindowInPlace
 *
 * Tests the window-based allgather with in-place send (send buffer == recv
 * slot).
 */
TEST_P(AllGatherWindowTestParam, AllGatherWindowInPlace) {
  const auto& [kNumElements, kNumIters, bufType, userBuf] = GetParam();
  EXPECT_GE(kNumElements, 1);
  EXPECT_GE(kNumIters, 1);

  auto comm = this->comm;
  auto statex = comm->ctranComm_->statex_.get();
  ASSERT_NE(statex, nullptr);
  EXPECT_EQ(statex->nRanks(), this->numRanks);

  const int rank = statex->rank();
  const int nRanks = statex->nRanks();

  cudaStream_t main_stream = 0;
  cudaStream_t ag_stream;
  CUDACHECK_TEST(cudaStreamCreateWithFlags(&ag_stream, cudaStreamNonBlocking));

  size_t sendBytes = kNumElements * sizeof(int);
  size_t sizeBytes = sendBytes * nRanks;

  meta::comms::Hints hints;
  CtranWin* win = nullptr;
  void* winBase = nullptr;
  createWin(userBuf, bufType, &winBase, &win, sizeBytes, hints);
  int* recvBuf = reinterpret_cast<int*>(winBase);

  EXPECT_THAT(win, ::testing::NotNull());

  // Initialize recv buffer with -1
  assignChunkValue(recvBuf, kNumElements * nRanks, -1, 0);

  // For in-place: send buffer is the rank's slot in recv buffer
  int* sendBuf = recvBuf + kNumElements * rank;
  assignChunkValue(sendBuf, kNumElements, rank, 1);

  // Barrier to ensure all peers have finished initialization
  this->barrier(comm, main_stream);

  for (auto iter = 0; iter < kNumIters; iter++) {
    // Reset non-local slots for each iteration (except first)
    if (iter > 0) {
      // Barrier to ensure all peers finished previous iteration before reset
      this->barrier(comm, main_stream);
      for (int p = 0; p < nRanks; p++) {
        if (p != rank) {
          assignChunkValue(recvBuf + kNumElements * p, kNumElements, -1, 0);
        }
      }
      CUDACHECK_TEST(cudaDeviceSynchronize());
      // Barrier after reset to ensure all ranks finished resetting before
      // any rank starts the next iteration's puts
      this->barrier(comm, main_stream);
    }

    // Call window-based allgather (in-place: sendBuf points to our slot)
    COMMCHECK_TEST(
        ctranAllGatherWindow(sendBuf, kNumElements, commInt32, win, ag_stream));

    // Sync to ensure this iteration completes
    CUDACHECK_TEST(cudaStreamSynchronize(ag_stream));
  }

  // Wait for completion
  CUDACHECK_TEST(cudaStreamSynchronize(ag_stream));
  this->barrier(comm, main_stream);

  // Verify results
  int totalErrs = 0;
  for (int peer = 0; peer < nRanks; peer++) {
    int errs = checkChunkValue(
        recvBuf + kNumElements * peer, kNumElements, peer, 1, main_stream);
    if (errs > 0) {
      std::cout << "[" << rank << "] Errors in slot for peer " << peer << ": "
                << errs << std::endl;
    }
    totalErrs += errs;
  }
  EXPECT_EQ(totalErrs, 0);

  // Cleanup
  auto res = ctranWinFree(win);
  EXPECT_EQ(res, ncclSuccess);
  freeWinBuf(userBuf, winBase, sizeBytes, bufType);
  CUDACHECK_TEST(cudaStreamDestroy(ag_stream));
}

INSTANTIATE_TEST_SUITE_P(
    AllGatherWindowTestInstance,
    AllGatherWindowTestParam,
    ::testing::Combine(
        // kNumElements, kNumIters, bufType, userBuf
        ::testing::Values(8192, 1024 * 1024),
        ::testing::Values(1, 10),
        ::testing::Values(
            MemAllocType::kMemCuMemAlloc,
            MemAllocType::kMemCudaMalloc),
        ::testing::Values(true, false)),
    [](const ::testing::TestParamInfo<AllGatherWindowTestParam::ParamType>&
           info) {
      const auto kNumElements = std::get<0>(info.param);
      const auto kNumIters = std::get<1>(info.param);
      const auto bufType = std::get<2>(info.param);
      const auto userBuf = std::get<3>(info.param);
      std::string name = fmt::format(
          "numElem{}_numIters{}_{}_{}",
          kNumElements,
          kNumIters,
          testMemAllocTypeToStr(bufType),
          userBuf ? "userBuf" : "allocBuf");
      return name;
    });

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
