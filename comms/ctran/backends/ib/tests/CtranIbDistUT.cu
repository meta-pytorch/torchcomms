// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda/atomic>
#include "comms/ctran/algos/common/GpeKernelSync.h"
#include "comms/ctran/algos/common/GpeKernelSyncDev.cuh"

using namespace ctran::algos;

__global__ void
waitValTestKernel(GpeKernelSync* sync, uint64_t* data, int cmpVal) {
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;
  ::cuda::atomic_ref<uint64_t, cuda::thread_scope_system> ref{*data};
  if (gtIdx == 0) {
    while (ref.load(cuda::memory_order_acquire) != cmpVal)
      ;
    GpeKernelSyncDev::complete(sync, gtIdx, 0);
  }
}
