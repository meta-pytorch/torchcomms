// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/algos/barrier.cuh"
#include "comms/ctran/gpe/CtranGpeDev.h"

namespace ctran::alltoallvp {

// Stub kernel to hold stream while GPE thread does exchangeMemHdl during
// AllToAllvPInit.
__global__ void ncclKernelAllToAllvPInit(
    int* flag,
    CtranAlgoDeviceState* devState) {
  if (flag) {
    ctran::device::devLoadAbortFlags(flag, devState);
    ctran::device::KernelStartGpe(flag);

    ctran::device::KernelWaitGpeTerminate(flag);
  }
}

__global__ void ncclKernelAllToAllvP(
    int* flag,
    CtranAlgoDeviceState* devState) {
  if (flag) {
    ctran::device::devLoadAbortFlags(flag, devState);
    ctran::device::KernelStartGpe(flag);
  }

  devStateLoadToShm(devState);

  const auto localRank = statex->localRank();
  const auto nLocalRanks = statex->nLocalRanks();

  // ensure nvl intra-node comm finishes
  barrier(localRank, nLocalRanks);
  if (flag) {
    ctran::device::KernelWaitGpeTerminate(flag);
  }
}

} // namespace ctran::alltoallvp
