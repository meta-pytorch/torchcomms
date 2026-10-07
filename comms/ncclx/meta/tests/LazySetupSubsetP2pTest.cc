// Copyright (c) Meta Platforms, Inc. and affiliates.

// Runs with NCCL_LAZY_SETUP_CHANNELS=1, set by the BUCK target.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "nccl.h"

namespace {

constexpr int kNumRanks = 4;
constexpr size_t kCount = 1024;

} // namespace

// One process drives every rank and the first operation involves only ranks 0
// and 1. A hang here means channel setup waited on the idle ranks.
TEST(LazySetupSubsetP2pTest, FirstP2pOnSomeRanksCompletes) {
  int nDevs = 0;
  ASSERT_EQ(cudaGetDeviceCount(&nDevs), cudaSuccess);
  if (nDevs < kNumRanks) {
    GTEST_SKIP() << "needs " << kNumRanks << " GPUs";
  }

  ncclUniqueId id;
  ASSERT_EQ(ncclGetUniqueId(&id), ncclSuccess);
  ncclComm_t comms[kNumRanks];
  ASSERT_EQ(ncclGroupStart(), ncclSuccess);
  for (int r = 0; r < kNumRanks; ++r) {
    ASSERT_EQ(cudaSetDevice(r), cudaSuccess);
    ASSERT_EQ(ncclCommInitRank(&comms[r], kNumRanks, id, r), ncclSuccess);
  }
  ASSERT_EQ(ncclGroupEnd(), ncclSuccess);

  float* bufs[2];
  for (int r = 0; r < 2; ++r) {
    ASSERT_EQ(cudaSetDevice(r), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&bufs[r], kCount * sizeof(float)), cudaSuccess);
  }
  ASSERT_EQ(ncclGroupStart(), ncclSuccess);
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  ASSERT_EQ(
      ncclSend(bufs[0], kCount, ncclFloat, 1, comms[0], nullptr), ncclSuccess);
  ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
  ASSERT_EQ(
      ncclRecv(bufs[1], kCount, ncclFloat, 0, comms[1], nullptr), ncclSuccess);
  ASSERT_EQ(ncclGroupEnd(), ncclSuccess);
  for (int r = 0; r < 2; ++r) {
    ASSERT_EQ(cudaSetDevice(r), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaFree(bufs[r]), cudaSuccess);
  }

  ASSERT_EQ(ncclGroupStart(), ncclSuccess);
  for (int r = 0; r < kNumRanks; ++r) {
    ASSERT_EQ(ncclCommDestroy(comms[r]), ncclSuccess);
  }
  ASSERT_EQ(ncclGroupEnd(), ncclSuccess);
}
