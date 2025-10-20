// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once
#include "comms/ctran/algos/AllToAllvDedup/FwdGroupSync.h"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/commstate/CommStateXDev.h"

namespace ctran::alltoallvdedup::FwdGroupSyncDev {

// reset via a full warp; reset must be called before or after each execution
__device__ inline void resetWarp(
    const int numGroups,
    const int numWorkers,
    FwdGroupSync* fwdGroupSync) {
  const auto laneId = threadIdx.x & (kWarpSize - 1);
  if (laneId == 0) {
    fwdGroupSync->numWorkers = numWorkers;
    fwdGroupSync->numGroups = numGroups;
  }
  const auto nLocalRanks = statex->nLocalRanks();
  for (auto r = laneId; r < nLocalRanks; r += kWarpSize) {
    fwdGroupSync->remRecvSteps[r] = 0;
  }
  for (auto i = laneId; i < numWorkers * numGroups; i += kWarpSize) {
    fwdGroupSync->remRecvStepsInGroup[i] = FwdGroupSync::Status::kUnset;
  }
}

// Get the next stepId for a given receive local ranks. The stepId is an atomic
// counter for all forwarding groups in the same forwarding rank, and the same
// value is shared by all workers in each forwarding group.
__device__ inline int getNextStep(
    FwdGroupSync* fwdGroupSync,
    const int groupId,
    const int workerId,
    const int recvLocalRank) {
  const auto numWorkers = fwdGroupSync->numWorkers;
  __shared__ int shared;

  // Thread 0 of each worker (thread block) synchronize on the next remote
  // chunkIdx
  // FIXME: any better way to share the chunkIdx with other workerIds? Maybe use
  // cooperative_groups
  if (threadIdx.x == 0) {
    int res;
    if (workerId == 0) {
      res = atomicAdd(&fwdGroupSync->remRecvSteps[recvLocalRank], 1);
      // Update to other workerIds in the same group
      for (auto w = 1; w < numWorkers; w++) {
        int rres;
        const auto idx = groupId * numWorkers + w;
        // Wait till the other worker has consumed the previous chunkIdx, then
        // update the current one
        do {
          rres = atomicCAS(
              &fwdGroupSync->remRecvStepsInGroup[idx],
              FwdGroupSync::Status::kUnset,
              res);
        } while (rres != FwdGroupSync::Status::kUnset);
      }
    } else {
      const auto idx = groupId * numWorkers + workerId;
      // other worker wait till workerId 0 got the remote chunkIdx
      do {
        res = atomicExch(
            &fwdGroupSync->remRecvStepsInGroup[idx],
            FwdGroupSync::Status::kUnset);
      } while (res == FwdGroupSync::Status::kUnset);
    }

    // Update to other threads within each worker
    shared = res;
  }

  // all threads wait till thread 0 updated shared
  __syncthreads();
  return shared;
}
} // namespace ctran::alltoallvdedup::FwdGroupSyncDev
