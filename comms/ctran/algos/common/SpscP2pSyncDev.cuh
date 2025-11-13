// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/algos/common/SpscP2pSync.h"

namespace ctran::algos::SpscP2pSyncDev {

__device__ __forceinline__ void reset(SpscP2pSync* sync) {
  sync->flag = SpscP2pSync::Status::kUnset;
}

__device__ __forceinline__ void waitReady(SpscP2pSync* sync) {
  if (threadIdx.x == 0) {
    int cur;
    do {
      cur = ctran::utils::loadIntAcq(&sync->flag);
    } while (cur != SpscP2pSync::Status::kUnset);
  }
  __syncthreads();
}

__device__ __forceinline__ bool checkReady(SpscP2pSync* sync) {
  __shared__ int ready;
  if (threadIdx.x == 0) {
    ready = ctran::utils::loadIntAcq(&sync->flag);
  }
  __syncthreads();
  return ready == SpscP2pSync::Status::kUnset;
}

__device__ __forceinline__ void post(SpscP2pSync* sync, const int step) {
  __syncthreads();
  if (threadIdx.x == 0) {
    ctran::utils::storeIntRel(&sync->flag, step);
  }
}

__device__ __forceinline__ void waitPost(SpscP2pSync* sync, const int step) {
  if (threadIdx.x == 0) {
    int cur;
    do {
      cur = ctran::utils::loadIntAcq(&sync->flag);
    } while (cur != step);
  }

  __syncthreads();
}

__device__ __forceinline__ bool checkPost(SpscP2pSync* sync, const int step) {
  __shared__ int cur;
  if (threadIdx.x == 0) {
    cur = ctran::utils::loadIntAcq(&sync->flag);
  }
  __syncthreads();
  return cur >= step;
}

__device__ __forceinline__ void complete(SpscP2pSync* sync) {
  __syncthreads();
  if (threadIdx.x == 0) {
    ctran::utils::storeIntRel(&sync->flag, SpscP2pSync::Status::kUnset);
  }
}

} // namespace ctran::algos::SpscP2pSyncDev
