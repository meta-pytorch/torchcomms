// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/ctran/algos/AllGatherP/Types.h"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/algos/barrier.cuh"
#include "comms/ctran/algos/common/GpeKernelSyncDev.cuh"
#include "comms/ctran/gpe/CtranGpeDev.h"

using namespace ctran::algos;
namespace ctran::allgatherp {

__global__ void ncclKernelStepDone(
    int* flag,
    CtranAlgoDeviceState* devState,
    StepDoneKernArgs args) {
  ctran::device::devLoadAbortFlags(flag, devState);
  GpeKernelSyncDev::complete(args.stepDoneSync, 0, args.stepId);
}

__global__ void ncclKernelAllGatherPPatCopyPipeEnd(
    int* flag,
    CtranAlgoDeviceState* devState,
    PatCopyPipeEndKernArgs args) {
  GpeKernelSyncDev::reset(args.pipeSync, 0);
  GpeKernelSyncDev::reset(args.stepDoneSync, 0);

  devStateLoadToShm(devState);
  const auto localRank = statex->localRank();
  const auto nLocalRanks = statex->nLocalRanks();
  barrier(localRank, nLocalRanks);
}

} // namespace ctran::allgatherp
