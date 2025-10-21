// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef REDUCE_SCATTER_RHD_H_INCLUDED
#define REDUCE_SCATTER_RHD_H_INCLUDED

#include <stdio.h>
#include <cstddef>
#include <iostream>

#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/DevAlgoImpl.cuh"
#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/gpe/CtranGpeDev.h"

template <typename T, commRedOp_t RedOp>
__global__ void __launch_bounds__(1024, 1) ncclKernelReduceScatterRHD(
    int* flag,
    CtranAlgoDeviceState* devState,
    CtranKernelReduceScatterArgs args) {
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  if (flag && gtIdx == 0) {
    ctran::device::KernelStartGpe(flag);
  }

  devStateLoadToShm(devState);

  // Inter-node reduce
  // Reuse the same reduce elem for each block defined by host side
  int totalNSteps = args.nStepsInterReduce;
  int stepId = 0;
  while (stepId < totalNSteps && args.interReduce != nullptr) {
    ctranKernMultiReduce<T, RedOp, true /* Complete*/, false /*Free*/>(
        args.interReduce, stepId);
    stepId++;
  }

  if (flag && gtIdx == 0) {
    ctran::device::KernelWaitGpeTerminate(flag);
  }
}

#define DECL_CTRAN_REDUCESCATTERRHD_KERN(T, RedOp)    \
  template __global__ void __launch_bounds__(1024, 1) \
      ncclKernelReduceScatterRHD<T, RedOp>(           \
          int* flag,                                  \
          CtranAlgoDeviceState* devState,             \
          CtranKernelReduceScatterArgs args)

#endif
