// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include <vector>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/tests/CtranDistAlgoDevUTBase.h"
#include "comms/ctran/algos/tests/CtranDistAlgoDevUTKernels.h"
#include "comms/testinfra/TestUtils.h"

// Directly exercises the NVL device barrier for whatever nLocalRanks the target
// is configured with (see the ppn variants in BUCK, which include
// non-powers-of-two). A barrier that drops synchronization edges lets some rank
// exit early and observe a stale or future token, which this test detects.
TEST_F(CtranDistAlgoDevTest, NvlBarrierManyIters) {
  constexpr int kNumIters = 200;

  const int localRankId = ctranComm_->statex_->localRank();
  const int nLocalRanks = ctranComm_->statex_->nLocalRanks();

  // Each rank exposes a single NVL-visible token slot in its IPC buffer.
  initIpcBufs<int>(1);
  CUDACHECK_TEST(cudaMemset(ipcBuf_, 0, sizeof(int)));

  // Build the device array of per-rank slot pointers (self + imported peers).
  std::vector<int*> peerSlotsHost(nLocalRanks, nullptr);
  for (int peer = 0; peer < nLocalRanks; peer++) {
    peerSlotsHost[peer] = (peer == localRankId)
        ? reinterpret_cast<int*>(ipcBuf_)
        : reinterpret_cast<int*>(ipcRemMem_.at(peer)->getBase());
  }

  int** peerSlots = nullptr;
  int* errCount = nullptr;
  CUDACHECK_TEST(cudaMalloc(&peerSlots, nLocalRanks * sizeof(int*)));
  CUDACHECK_TEST(cudaMalloc(&errCount, sizeof(int)));
  CUDACHECK_TEST(cudaMemcpy(
      peerSlots,
      peerSlotsHost.data(),
      nLocalRanks * sizeof(int*),
      cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(errCount, 0, sizeof(int)));

  // Ensure every rank has initialized its slot before any peer reads it.
  barrierNvlDomain(ctranComm_.get());

  cudaStream_t stream = nullptr;
  CUDACHECK_TEST(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // Single block / single thread, mirroring the production nvlBarrier launch.
  dim3 grid = {1, 1, 1};
  dim3 blocks = {1, 1, 1};
  CUDACHECK_TEST(testKernNvlBarrierLoopWrapper(
      grid,
      blocks,
      stream,
      localRankId,
      nLocalRanks,
      kNumIters,
      reinterpret_cast<int*>(ipcBuf_),
      peerSlots,
      errCount,
      ctranComm_->ctran_->algo->getDevState()));

  CUDACHECK_TEST(cudaStreamSynchronize(stream));

  int errCountHost = -1;
  CUDACHECK_TEST(
      cudaMemcpy(&errCountHost, errCount, sizeof(int), cudaMemcpyDeviceToHost));
  EXPECT_EQ(errCountHost, 0)
      << "Rank " << globalRank << " observed " << errCountHost
      << " stale/future tokens across " << kNumIters << " iterations with "
      << nLocalRanks << " local ranks";

  CUDACHECK_TEST(cudaStreamDestroy(stream));
  CUDACHECK_TEST(cudaFree(peerSlots));
  CUDACHECK_TEST(cudaFree(errCount));
  freeIpcBufs();
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new ctran::CtranDistEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
