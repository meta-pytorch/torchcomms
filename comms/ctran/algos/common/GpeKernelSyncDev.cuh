// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once
#include "comms/common/DevUtils.cuh"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/algos/common/GpeKernelSync.h"

namespace ctran::algos::GpeKernelSyncDev {
__device__ __forceinline__ void reset(GpeKernelSync* sync, const int workerId) {
  if (threadIdx.x == 0) {
    sync->postFlag[workerId] = GpeKernelSync::Status::kUnset;
    sync->completeFlag[workerId] = GpeKernelSync::Status::kUnset;
  }
}

__device__ __forceinline__ void resetWarp(
    GpeKernelSync* sync,
    const int nworkers) {
  const auto laneId = threadIdx.x & (kWarpSize - 1);
  if (laneId == 0) {
    sync->nworkers = nworkers;
  }

  for (auto i = laneId; i < nworkers; i += kWarpSize) {
    sync->postFlag[i] = GpeKernelSync::Status::kUnset;
    sync->completeFlag[i] = GpeKernelSync::Status::kUnset;
  }
  syncwarp();
}

__device__ __forceinline__ void
waitPost(GpeKernelSync* sync, const int workerId, const int step) {
  if (threadIdx.x == 0) {
    int val;
    do {
      val = comms::device::loadIntAcq(&sync->postFlag[workerId]);
    } while (step > val && !ctran::device::KernelTestHostAbort(kernelFlag));
  }
  __syncthreads();
}

__device__ __forceinline__ void
waitPostWarp(GpeKernelSync* sync, const int workerId, const int step) {
  const auto laneId = threadIdx.x & (kWarpSize - 1);
  if (laneId == 0) {
    int val;
    do {
      val = comms::device::loadIntAcq(&sync->postFlag[workerId]);
    } while (step > val && !ctran::device::KernelTestHostAbort(kernelFlag));
  }
  syncwarp();
}

__device__ __forceinline__ bool
checkPost(GpeKernelSync* sync, const int workerId, const int step) {
  __shared__ bool found;
  if (threadIdx.x == 0) {
    int val = comms::device::loadIntAcq(&sync->postFlag[workerId]);
    found = step > val ? false : true;
  }
  __syncthreads();
  return found;
}

__device__ __forceinline__ void
complete(GpeKernelSync* sync, const int workerId, const int step) {
  __syncthreads();
  if (threadIdx.x == 0) {
    comms::device::storeIntRel(&sync->completeFlag[workerId], step);
  }
}
} // namespace ctran::algos::GpeKernelSyncDev
