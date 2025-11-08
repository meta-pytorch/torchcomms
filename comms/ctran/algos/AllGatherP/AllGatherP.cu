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
  if (flag) {
    ctran::device::devLoadAbortFlags(flag, devState);
    ctran::device::KernelStartGpe(flag);

    ctran::device::KernelWaitGpeTerminate(flag);
  }
}
} // namespace ctran::allgatherp
