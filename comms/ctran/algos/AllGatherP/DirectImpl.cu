// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/algos/barrier.cuh"
#include "comms/ctran/gpe/CtranGpeDev.h"

namespace ctran::allgatherp {
__global__ void ncclKernelAllGatherPDirect(
    int* flag,
    CtranAlgoDeviceState* devState) {
  if (flag) {
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
} // namespace ctran::allgatherp
