// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/ctran/algos/AllGatherP/Types.h"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/algos/barrier.cuh"
#include "comms/ctran/algos/common/GpeKernelSyncDev.cuh"
#include "comms/ctran/gpe/CtranGpeDev.h"

using namespace ctran::algos;
namespace ctran::allgatherp {

// Kernel to block GPE thread to start the internode transfer till allgatherP
// starts on the stream. It doesn't wait for GPE termination, so that the
// algorithm can schedule intra-node CE copies to overlap with internode
// transfer.
// Used when nLocalRanks > 1 in allgatherP pipeline algorithm. It is called once
// to start the algorithm
__global__ void ncclKernelAllGatherPPipeStart(
    int* flag,
    CtranAlgoDeviceState* devState) {
  // TODO(T243528798): remove this preload of devstate by splitting h2d/d2h
  // channels.
  shmDevState.enableCancellableWaits = devState->enableCancellableWaits;
  if (flag) {
    ctran::device::KernelStartGpeAndExit(flag);
  }
}

// Kernel to block intranode CE copies on the stream till GPE thread has
// finished the corresponding step's internode transfer.
// Used when nLocalRanks > 1 in allgatherP pipeline algorithm. It is called
// nNode - 1 times.
__global__ void ncclKernelAllGatherPPipeSync(
    int* flag,
    CtranAlgoDeviceState* devState,
    PipeSyncKernArgs args) {
  // TODO(T243528798): remove this preload of devstate by splitting h2d/d2h
  // channels.
  shmDevState.enableCancellableWaits = devState->enableCancellableWaits;
  if (threadIdx.x == 0) {
    kernelFlag = flag;
    kernelDoAbort = false;
  }
  // wait till GPE thread post the current stepId
  GpeKernelSyncDev::waitPost(args.pipeSync, 0, args.stepId);
}

// Stub kernel to hold stream while GPE thread does inter-node transfer.
// Used when nLocalRanks == 1 in allgatherP pipeline algorithm
__global__ void ncclKernelAllGatherPPipe(
    int* flag,
    CtranAlgoDeviceState* devState) {
  // TODO(T243528798): remove this preload of devstate by splitting h2d/d2h
  // channels.
  shmDevState.enableCancellableWaits = devState->enableCancellableWaits;
  if (flag) {
    ctran::device::KernelStartGpe(flag);

    ctran::device::KernelWaitGpeTerminate(flag);
  }
}

// Kernel to ensure all local ranks have finished the last round intra-node
// broadcast. Used when nLocalRanks > 1 in allgatherP pipeline algorithm. It is
// called once at the end.
__global__ void ncclKernelAllGatherPPipeEnd(
    int* flag,
    CtranAlgoDeviceState* devState,
    PipeEndKernArgs args) {
  // Reset sync flag for next GPE->kernel pipeline sync to use
  GpeKernelSyncDev::reset(args.pipeSync, 0);

  // Ensure nvl intra-node comm finishes
  devStateLoadToShm(devState);
  const auto localRank = statex->localRank();
  const auto nLocalRanks = statex->nLocalRanks();
  barrier(localRank, nLocalRanks);
}

} // namespace ctran::allgatherp
