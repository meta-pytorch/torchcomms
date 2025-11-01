// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/common/GpeKernelDev.cuh"
#include "comms/ctran/gpe/CtranGpeDev.h"

namespace ctran::allgatherp {
// Stub kernel to hold stream while GPE thread does exchangeMemHdl during
// AllGatherPInit.
__global__ void ncclKernelAllGatherPInit(
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
} // namespace ctran::allgatherp
