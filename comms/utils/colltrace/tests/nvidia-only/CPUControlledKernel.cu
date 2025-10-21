// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/colltrace/tests/nvidia-only/CPUControlledKernel.cuh"

#include <cstdio>

__global__ void waitCPUKernel(KernelFlag* flag) {
  kernelSignalStart(flag);
  kernelWaitCPUSignal(flag);
  return;
}
