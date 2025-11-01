// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/common/GpeKernelDev.cuh"
#include "comms/ctran/gpe/CtranGpeDev.h"

__global__ void ncclKernelAllGatherCtranRing(
    int* flag,
    CtranAlgoDeviceState* devState,
    CtranKernelAllGatherArgs args) {
  // TODO(T243528798): remove this preload of devstate by splitting h2d/d2h
  // channels.
  shmDevState.enableCancellableWaits = devState->enableCancellableWaits;
  if (flag) {
    ctran::device::KernelStartGpe(flag);
    ctran::device::KernelWaitGpeTerminate(flag);
  }
}
