// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda.h>
#include <cuda_runtime_api.h>

#if defined(__HIP_PLATFORM_AMD__)
#else
#include <cuda/atomic>
#endif

#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/RMA/Types.h"
#include "comms/ctran/algos/common/GpeKernelDev.cuh"

__global__ void ncclKernelFetchAdd(
    int* flag,
    CtranAlgoDeviceState* devState,
    CtranKernelFetchAddArgs args) {
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;
  if (flag && gtIdx == 0) {
    ctran::device::devLoadAbortFlags(flag, devState);
    ctran::device::KernelStartGpe(flag);
  }

  if (gtIdx == 0 && args.remoteAddr != nullptr) {
#if defined(__HIP_PLATFORM_AMD__)
    // TODO: implement atomic operations for AMD GPUs.
    __builtin_trap();
#else
    ::cuda::atomic_ref<uint64_t, cuda::thread_scope_system> ref{
        *args.remoteAddr};
    uint64_t oldVal =
        ref.fetch_add(args.addVal, cuda::std::memory_order_relaxed);
    if (args.resultAddr != nullptr) {
      *args.resultAddr = oldVal;
    }
#endif
  }

  if (flag && gtIdx == 0) {
    ctran::device::KernelWaitGpeTerminate(flag);
  }
}
