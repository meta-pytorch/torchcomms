// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/commstate/CommStateXDev.h"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include "comms/ctran/commstate/tests/CommStateXDevTest.cuh"

using ctran::CommStateXDev;

TEST(CommStateXTest, DevQuery) {
  const int rank = 12;
  const int nRanks = 8;
  const int nNodes = 4;
  const int localRank = 2;
  const int nLocalRanks = 8;
  const uint64_t commHash = 0x1234567;
  const int queryLocalRank = 4;
  const int queryNode = 3;

  const int expNode = rank / nLocalRanks;
  const int expLocalRankToRank = nLocalRanks * queryNode + queryLocalRank;

  CommStateXDevTestOutput* output = nullptr;
  ASSERT_EQ(
      cudaHostAlloc(
          &output, sizeof(CommStateXDevTestOutput), cudaHostAllocDefault),
      cudaSuccess);

  // reset output
  new (output) CommStateXDevTestOutput();

  CommStateXDev statexD = {
      .rank_ = rank,
      .pid_ = getpid(),
      .localRank_ = localRank,
      .localRanks_ = nLocalRanks,
      .nRanks_ = nRanks,
      .nNodes_ = nNodes,
      .commHash_ = commHash,
  };

  dim3 grid = {1, 1, 1};
  dim3 block = {1, 1, 1};
  void* args[4] = {
      (void*)&statexD, (void*)&queryLocalRank, (void*)&queryNode, &output};
  void* fn = (void*)commStateXDevKernel;
  ASSERT_EQ(cudaLaunchKernel(fn, grid, block, args), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(output->rank, rank);
  EXPECT_EQ(output->nRanks, nRanks);
  EXPECT_EQ(output->node, expNode);
  EXPECT_EQ(output->nNodes, nNodes);
  EXPECT_EQ(output->localRank, localRank);
  EXPECT_EQ(output->nLocalRanks, nLocalRanks);
  EXPECT_EQ(output->commHash, commHash);
  EXPECT_EQ(output->localRankToRank, expLocalRankToRank);

  ASSERT_EQ(cudaFreeHost(output), cudaSuccess);
}
