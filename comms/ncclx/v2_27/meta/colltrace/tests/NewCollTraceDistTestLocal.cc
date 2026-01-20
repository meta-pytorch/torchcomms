// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <stdlib.h>
#include <cstddef>
#include <iostream>

#include <folly/ScopeGuard.h>
#include <folly/Synchronized.h>
#include <folly/init/Init.h>
#include <folly/json/json.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <rfe/scubadata/ScubaData.h>

#include "comm.h" // @manual
#include "nccl.h" // @manual

#include "comms/ctran/Ctran.h"
#include "comms/testinfra/TestUtils.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/testinfra/TestsDistUtils.h"
#include "meta/commDump.h"

class CollTraceTestLocal : public NcclxBaseTest {
 public:
  CollTraceTestLocal() = default;
  void SetUp() override {
    // Set up dummy values for environment variables for Scuba test
    setenv("WORLD_SIZE", "4", 0);
    setenv("HPC_JOB_NAME", "CollTraceUT", 0);
    setenv("HPC_JOB_VERSION", "1", 0);
    setenv("HPC_JOB_ATTEMPT_INDEX", "2", 0);
    setenv(
        "NCCL_HPC_JOB_IDS",
        "HPC_JOB_NAME,HPC_JOB_VERSION,HPC_JOB_ATTEMPT_INDEX",
        0);
    setenv("NCCL_COLLTRACE", "trace", 0);
    setenv("NCCL_COLLTRACE_USE_NEW_COLLTRACE", "1", 0);

    // Enable Ctran
    setenv("NCCL_CTRAN_ENABLE", "1", 0);
    setenv("NCCL_CTRAN_IB_EPOCH_LOCK_ENFORCE_CHECK", "true", 0);

    NcclxBaseTest::SetUp();
    std::tie(this->localRank, this->globalRank, this->numRanks) = getMpiInfo();

    CUDACHECK_TEST(cudaSetDevice(this->localRank));
    CUDACHECK_TEST(cudaStreamCreate(&this->stream));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaStreamDestroy(this->stream));
    // cudaFree in case test case doesn't free
    if (sendBuf) {
      CUDACHECK_TEST(cudaFree(sendBuf));
    }
    if (recvBuf) {
      CUDACHECK_TEST(cudaFree(recvBuf));
    }
  }

  // Use MPI to ensure that we don't see additional all reduce for that nccl
  // communicator
  void barrier() {
    CUDACHECK_TEST(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
  }

 protected:
  int localRank{0};
  int globalRank{0};
  int numRanks{0};
  int* sendBuf{nullptr};
  int* recvBuf{nullptr};
  void* sendHandle{nullptr};
  void* recvHandle{nullptr};
  cudaStream_t stream{nullptr};
};

TEST_F(CollTraceTestLocal, winSignalAllToAll) {
  const int kNumElements = 16;

  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};

  auto statex = comm->ctranComm_->statex_.get();
  ASSERT_NE(statex, nullptr);
  EXPECT_EQ(statex->nRanks(), this->numRanks);

  cudaStream_t sig_stream, wait_stream;
  CUDACHECK_TEST(cudaStreamCreateWithFlags(&sig_stream, cudaStreamNonBlocking));
  CUDACHECK_TEST(
      cudaStreamCreateWithFlags(&wait_stream, cudaStreamNonBlocking));

  meta::comms::Hints hints;
  hints.set(
      "window_signal_bufsize", std::to_string(kNumElements * statex->nRanks()));
  ::ctran::CtranWin* win = nullptr;
  int* winBase = nullptr;
  // Allocate the window with data buffer size of 0 since this test only uses
  // the signal buffer for inter-rank signaling
  auto res = ctranWinAllocate(
      0, comm->ctranComm_.get(), (void**)&winBase, &win, hints);
  ASSERT_EQ(res, commSuccess);
  ASSERT_NE(winBase, nullptr);
  uint64_t* signalBase = win->winSignalPtr;
  ASSERT_NE(signalBase, nullptr);

  // initialize with 1000UL, a random number
  assignChunkValue(signalBase, kNumElements * statex->nRanks(), 1000UL, 0UL);
  // barrier to ensure all peers have finished value assignment
  this->barrier();

  const auto myrank = statex->rank();
  const auto numRanks = statex->nRanks();

  for (auto iter = 0; iter < kNumElements; iter++) {
    // In iterations, each rank signals the continuous memory region of other
    // ranks, with its rank id as the signal value
    for (auto peerRank = 0; peerRank < numRanks; peerRank++) {
      if (peerRank == myrank) {
        continue;
      }
      COMMCHECK_TEST(ctranSignal(peerRank, win, sig_stream));
      COMMCHECK_TEST(ctranWaitSignal((uint64_t)peerRank, win, wait_stream));
    }
  }

  CUDACHECK_TEST(cudaStreamSynchronize(wait_stream));

  int errs = 0;
  for (auto peerRank = 0; peerRank < numRanks; peerRank++) {
    if (peerRank == myrank) {
      continue;
    }
    errs += checkChunkValue(
        signalBase + kNumElements * peerRank,
        kNumElements,
        (uint64_t)peerRank,
        0UL,
        this->globalRank,
        wait_stream);
  }
  EXPECT_EQ(errs, 0);
  // For signalWait, this is necessary since we must ensure all ctranSignal
  // ops finish for other peers
  CUDACHECK_TEST(cudaStreamSynchronize(sig_stream));

  res = ctranWinFree(win);
  EXPECT_EQ(res, ncclSuccess);

  CUDACHECK_TEST(cudaDeviceSynchronize());
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  auto dumpMap =
      meta::comms::ncclx::dumpNewCollTrace(*comm->ctranComm_->colltraceNew_);
  EXPECT_NE(dumpMap["CT_pastColls"], "[]");
  EXPECT_EQ(dumpMap["CT_pendingColls"], "[]");
  EXPECT_EQ(dumpMap["CT_currentColl"], "null");

  CUDACHECK_TEST(cudaStreamDestroy(sig_stream));
  CUDACHECK_TEST(cudaStreamDestroy(wait_stream));
}

TEST_F(CollTraceTestLocal, winPutOnly) {
  const int kNumElements = 8192;
  const int kNumIters = 500;

  NcclCommRAII comm{this->globalRank, this->numRanks, this->localRank};

  EXPECT_GE(kNumElements, 8192);
  EXPECT_GE(kNumIters, 1);
  if (!comm->ctranComm_->ctran_->mapper->hasBackend(
          globalRank, CtranMapperBackend::NVL)) {
    GTEST_SKIP() << "NVL not enabled, skip test";
  }

  auto statex = comm->ctranComm_->statex_.get();
  ASSERT_NE(statex, nullptr);
  EXPECT_EQ(statex->nRanks(), this->numRanks);

  cudaStream_t main_stream = 0, put_stream;
  CUDACHECK_TEST(cudaStreamCreateWithFlags(&put_stream, cudaStreamNonBlocking));

  size_t sizeBytes = kNumElements * sizeof(int) * statex->nRanks();

  meta::comms::Hints hints;
  ::ctran::CtranWin* win = nullptr;
  void* winBase = nullptr;
  auto res = ctranWinAllocate(
      sizeBytes, comm->ctranComm_.get(), &winBase, &win, hints);
  ASSERT_EQ(res, ncclSuccess);
  ASSERT_NE(winBase, nullptr);

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
  this->barrier();

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
  this->barrier();

  // allreduce ensures all remote puts have finished
  size_t errs = checkChunkValue(
      (int*)winBase + kNumElements * prevPeer,
      kNumElements,
      prevPeer,
      1,
      this->globalRank,
      main_stream);

  res = ctranWinFree(win);
  EXPECT_EQ(res, ncclSuccess);

  ASSERT_EQ(ncclCommDeregister(comm, localHdl), ncclSuccess);
  ASSERT_EQ(ncclMemFree(localBuf), ncclSuccess);

  cudaDeviceSynchronize();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  auto dumpMap =
      meta::comms::ncclx::dumpNewCollTrace(*comm->ctranComm_->colltraceNew_);
  EXPECT_NE(dumpMap["CT_pastColls"], "[]");
  EXPECT_EQ(dumpMap["CT_pendingColls"], "[]");
  EXPECT_EQ(dumpMap["CT_currentColl"], "null");

  CUDACHECK_TEST(cudaStreamDestroy(put_stream));
  EXPECT_EQ(errs, 0);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new DistEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
