// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once
#include "comms/ctran/algos/AllToAllvDedup/WorkerSync.h"
#include "comms/ctran/algos/CtranAlgoDev.h"

namespace ctran::alltoallvdedup::WorkerSyncDev {

// reset via a full warp; reset must be called before or after each execution
__device__ inline void resetWarp(WorkerSync* workerSync) {
  const auto laneId = threadIdx.x & (kWarpSize - 1);
  for (auto i = laneId; i < CTRAN_ALGO_MAX_THREAD_BLOCKS; i += kWarpSize) {
    workerSync->sendGroups[i] = 0;
    workerSync->fwdGroups[i] = 0;
    workerSync->recvGroups[i] = 0;
  }
}

__device__ inline void sync(
    WorkerSync* workerSync,
    const int groupId,
    const int numWorkers,
    WorkerSync::GroupType groupType,
    int step) {
  if (threadIdx.x == 0) {
    int* flags;
    if (groupType == WorkerSync::GroupType::kSend) {
      flags = workerSync->sendGroups;
    } else if (groupType == WorkerSync::GroupType::kFwd) {
      flags = workerSync->fwdGroups;
    } else { // groupType == WorkerSync::GroupType::kRecv
      flags = workerSync->recvGroups;
    }
    auto& blockCounter = flags[groupId];
    atomicAdd(&blockCounter, 1);
    while (atomicAdd(&blockCounter, 0) < numWorkers * (step + 1)) {
      // Busy wait
    }
  }

  __syncthreads();
}
} // namespace ctran::alltoallvdedup::WorkerSyncDev
