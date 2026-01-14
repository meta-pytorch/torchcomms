// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/common/GpeKernelSync.h"
#include "comms/ctran/algos/common/GpeKernelSyncDev.cuh"

using namespace ctran::algos;

__global__ void GpeKernelSyncResetKernel(
    GpeKernelSync* sync,
    const int nworkers) {
  const auto warpId = threadIdx.x / comms::device::kWarpSize;
  if (warpId == 0) {
    GpeKernelSyncDev::resetWarp(sync, nworkers);
  }
}

__global__ void
GpeKernelSyncKernel(GpeKernelSync* sync, int* data, int numElem, int nSteps) {
  const auto workerId = blockIdx.x;
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  for (auto i = 0; i < nSteps; i++) {
    GpeKernelSyncDev::waitPost(sync, workerId, i);
    for (auto e = gtIdx; e < numElem; e += gridDim.x * blockDim.x) {
      data[e] += i;
    }
    GpeKernelSyncDev::complete(sync, workerId, i);
  }
}
