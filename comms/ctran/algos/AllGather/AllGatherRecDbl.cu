// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/common/GpeKernelDev.cuh"
#include "comms/ctran/gpe/CtranGpeDev.h"

__global__ void ncclKernelAllGatherCtranRecDbl(
    int* flag,
    CtranAlgoDeviceState* devState,
    CtranKernelAllGatherArgs args) {
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;
  if (flag && gtIdx == 0) {
    ctran::device::devLoadAbortFlags(flag, devState);
    ctran::device::KernelStartGpe(flag);
    ctran::device::KernelWaitGpeTerminate(flag);
  }
}
