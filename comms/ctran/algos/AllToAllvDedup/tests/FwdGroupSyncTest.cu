// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/ctran/algos/AllToAllvDedup/FwdGroupSync.h"
#include "comms/ctran/algos/AllToAllvDedup/FwdGroupSyncDev.cuh"
#include "comms/ctran/algos/CtranAlgoDev.h"

using namespace ctran::alltoallvdedup;

__global__ void fwdGroupSyncTestInitKernel(
    int numGroups,
    int numWorkers,
    FwdGroupSync* fwdGroupSync) {
  const auto warpId = threadIdx.x / kWarpSize;
  if (warpId == 0 && blockIdx.x == 0) {
    FwdGroupSyncDev::resetWarp(numGroups, numWorkers, fwdGroupSync);
  }
}

__global__ void fwdGroupSyncTestKernel(
    int numGroups,
    int numWorkers,
    int numIter,
    FwdGroupSync* fwdGroupSync,
    // output of numGroups * numWorkers * iter steps
    int* steps) {
  const auto groupId = blockIdx.x / numWorkers;
  const auto workerId = blockIdx.x & (numWorkers - 1);

  const auto outOffset = (groupId * numWorkers + workerId) * numIter;
  for (auto x = 0; x < numIter; x++) {
    // shift recvRank each time
    int recvLocalRank = x & (CTRAN_MAX_NVL_PEERS - 1);
    int step = FwdGroupSyncDev::getNextStep(
        fwdGroupSync, groupId, workerId, recvLocalRank);
    // Return steps only in check mode
    if (threadIdx.x == 0) {
      steps[outOffset + x] = step;
    }
  }
}

__global__ void fwdGroupSyncPerfBenchKernel(
    int numGroups,
    int numWorkers,
    int numIter,
    FwdGroupSync* fwdGroupSync) {
  const auto groupId = blockIdx.x / numWorkers;
  const auto workerId = blockIdx.x & (numWorkers - 1);
  for (auto x = 0; x < numIter; x++) {
    int recvLocalRank = 0;
    FwdGroupSyncDev::getNextStep(
        fwdGroupSync, groupId, workerId, recvLocalRank);
  }
}
