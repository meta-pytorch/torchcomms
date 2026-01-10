// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once
#include "comms/common/AtomicUtils.cuh"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/algos/common/MPSCTbSync.h"

namespace ctran::algos::MPSCTbSyncDev {
// Each producer thread block waits till the flag is set to ready by consumer
// before post
__device__ __forceinline__ void waitReady(MPSCTbSync<>* sync, int workerId) {
  if (threadIdx.x == 0) {
    int cur;
    do {
      cur = comms::device::ld_acquire_sys_global(&sync->flags[workerId]);
    } while (cur != MPSCTbSync<>::Status::kUnset);
  }
  __syncthreads();
}

// Each producer thread block posts its data for consumer to use
__device__ __forceinline__ void post(MPSCTbSync<>* sync, int workerId) {
  __syncthreads();
  if (threadIdx.x == 0) {
    comms::device::st_release_sys_global(
        &sync->flags[workerId], MPSCTbSync<>::Status::kPosted);
  }
}

// Consumer thread block waits till all producer thread blocks have posted
__device__ __forceinline__ void waitPost(MPSCTbSync<>* sync) {
  // Let different threads wait on different producer flags in parallel
  // Return to all threads only after all the checking threads have finished.
  for (auto w = threadIdx.x; w < sync->numProducers; w += blockDim.x) {
    int val = MPSCTbSync<>::Status::kUnset;
    do {
      val = comms::device::ld_acquire_sys_global(&sync->flags[w]);
    } while (val != MPSCTbSync<>::Status::kPosted);
  }

  __syncthreads();
}

// Consumer thread block marks completion; producers will see it is ready to
// post after completion is updated.
__device__ __forceinline__ void complete(MPSCTbSync<>* sync) {
  __syncthreads();
  // Let different threads update the flags in parallel
  for (auto w = threadIdx.x; w < sync->numProducers; w += blockDim.x) {
    comms::device::st_release_sys_global(
        &sync->flags[w], MPSCTbSync<>::Status::kUnset);
  }
}

// Each producer thread block waits till the flag is ready by consumer
// before post. The flag is incremented by consumer after each post.
// caller need to __syncthreads() after this call.
__device__ __forceinline__ void
waitStep(MPSCTbSync<1>* sync, int threadId, int base, int step) {
  if (threadIdx.x == threadId) {
    int cur;
    do {
      cur = comms::device::ld_acquire_sys_global(&sync->flags[0]) + base;
    } while (cur < step);
  }
}

// Each producer thread block posts its data for consumer to use.
// caller need to __syncthreads() before this call.
__device__ __forceinline__ void
postStep(MPSCTbSync<1>* sync, int threadId, int step) {
  if (threadIdx.x == threadId) {
    comms::device::st_release_sys_global(&sync->flags[0], step);
  }
}

} // namespace ctran::algos::MPSCTbSyncDev
