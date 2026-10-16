// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/common/GpeKernelDev.cuh"
#include "comms/utils/colltrace/common/ColltraceDeviceHandle.h"

__global__ void ncclKernelGet(
    ctran::gpe::KernelFlagDev* f,
    CtranAlgoDeviceState* devState,
    meta::comms::colltrace::ColltraceDeviceHandle colltraceHdr) {
  ctran::device::ColltraceEventScope colltraceScope(colltraceHdr);
  int* flag = f ? const_cast<int*>(f->flag_) : nullptr;
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;
  if (flag && gtIdx == 0) {
    ctran::device::devLoadAbortFlags(flag, devState);
    ctran::device::KernelStartGpe(f);
    ctran::device::KernelWaitGpeTerminate(flag);
  }
}
