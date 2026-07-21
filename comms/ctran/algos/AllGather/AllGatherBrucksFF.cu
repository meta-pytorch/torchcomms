// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/AllGather/Types.h"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/common/GpeKernelDev.cuh"
#include "comms/ctran/gpe/CtranGpeDev.h"

__global__ void ncclKernelAllGatherCtranBrucks(
    ctran::gpe::KernelFlagDev* f,
    CtranAlgoDeviceState* devState,
    ctran::allgather::KernelArgs args) {
  ctran::device::ColltraceEventScope colltraceScope(f);
  int* flag = f ? const_cast<int*>(f->flag_) : nullptr;
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;
  if (flag && gtIdx == 0) {
    ctran::device::devLoadAbortFlags(flag, devState);
    ctran::device::KernelStartGpe(f);
    ctran::device::KernelWaitGpeTerminate(flag);
  }
}
