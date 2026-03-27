// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/algos/barrier.cuh"
#include "comms/ctran/gpe/CtranGpeDev.h"

namespace ctran::allgatherwindow {

/**
 * Direct algorithm kernel
 *
 * Starts GPE to do all-to-all PUT with atomic signaling, then waits for
 * signals from all remote peers. Uses barrier for intra-node synchronization.
 */
__global__ void ncclKernelAllGatherWindowDirect(
    int* flag,
    CtranAlgoDeviceState* devState) {
  if (flag) {
    ctran::device::devLoadAbortFlags(flag, devState);
    ctran::device::KernelStartGpe(flag);
  }

  devStateLoadToShm(devState);

  const auto localRank = statex->localRank();
  const auto nLocalRanks = statex->nLocalRanks();

  // Ensure NVLink intra-node copies complete before proceeding
  barrier(localRank, nLocalRanks);

  if (flag) {
    ctran::device::KernelWaitGpeTerminate(flag);
  }
}

} // namespace ctran::allgatherwindow
