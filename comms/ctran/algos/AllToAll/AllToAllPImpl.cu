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
    ctran::device::devLoadAbortFlags(flag, devState);
    ctran::device::KernelStartGpe(flag);

    ctran::device::KernelWaitGpeTerminate(flag);
  }
}

} // namespace ctran::alltoallp

namespace ctran::alltoall {

// Lightweight stub kernel for AllToAll collectives when there are no local
// NVL peers (pure IB path). Synchronizes with GPE thread using 1 block /
// 1 thread to minimize SM occupancy. Self D2D copy is handled separately
// via cudaMemcpyAsync in setupKernelConfig.
__global__ void ncclKernelAllToAllStub(
    int* flag,
    CtranAlgoDeviceState* devState) {
  if (flag) {
    ctran::device::devLoadAbortFlags(flag, devState);
    ctran::device::KernelStartGpe(flag);

    ctran::device::KernelWaitGpeTerminate(flag);
  }
}

} // namespace ctran::alltoall
