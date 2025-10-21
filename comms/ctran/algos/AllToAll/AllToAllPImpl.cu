// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/common/GpeKernelDev.cuh"

namespace ctran::alltoallp {

// Stub kernel to hold stream while GPE thread does exchangeMemHdl during
// AllToAllPInit.
__global__ void ncclKernelAllToAllPInitWait(
    int* flag,
    CtranAlgoDeviceState* devState) {
  if (flag) {
    ctran::device::KernelStartGpe(flag);

    ctran::device::KernelWaitGpeTerminate(flag);
  }
}

} // namespace ctran::alltoallp
