// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda.h>
#include <cuda_runtime_api.h>

#if defined(__HIP_PLATFORM_AMD__)
#else
#include <cuda/atomic>
#endif

#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/algos/RMA/Types.h"
#include "comms/ctran/gpe/CtranGpeDev.h"

__global__ void ncclKernelPutNotify(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::rma::KernelPutNotifyArgs args) {
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;
  if (flag && gtIdx == 0) {
    ctran::device::devLoadAbortFlags(flag, devState);
    ctran::device::KernelStartGpe(flag);
  }

  // Avoid copy
  const auto& statex_ = devState->statex;
  const auto localRank = statex_.localRank();

  // If not directly notify via backend, use AlgoDevice channel.
  // Should not concurrently used by other collectives using the channel (e.g.,
  // alltoall), nor RMA notify to the same peer.
  if (!args.isDirect && gtIdx == 0 && args.peerLocalRank != localRank) {
    // This kernel has only one thread, and only needs devSync from one peer,
    // thus loading the whole devState into shared memory has overheads, and it
    // is unnecessary.
    CtranAlgoDeviceSync* sync =
        devSyncGetLoc<REMOTE>(args.peerLocalRank, devState);
    devSyncSetNotify(sync, 0);
  }

  if (flag && gtIdx == 0) {
    ctran::device::KernelWaitGpeTerminate(flag);
  }
}

__global__ void ncclKernelWaitNotify(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::rma::KernelWaitNotifyArgs args) {
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;
  if (flag && gtIdx == 0) {
    ctran::device::devLoadAbortFlags(flag, devState);
    ctran::device::KernelStartGpe(flag);
  }

  const auto& statex_ = devState->statex;
  const auto localRank = statex_.localRank();

  // If not directly notify via backend, use AlgoDevice channel.
  // Should not concurrently used by other collectives using the channel (e.g.,
  // alltoall), nor RMA notify to the same peer.
  if (!args.isDirect && gtIdx == 0 && args.peerLocalRank != localRank) {
    // This kernel has only one thread, and only needs devSync from one peer,
    // thus loading the whole devState into shared memory has overheads, and it
    // is unnecessary.
    CtranAlgoDeviceSync* sync =
        devSyncGetLoc<LOCAL>(args.peerLocalRank, devState);
    devSyncWaitNotify(sync, 1);
  }

  if (flag && gtIdx == 0) {
    ctran::device::KernelWaitGpeTerminate(flag);
  }
}

__global__ void ncclKernelPutSignal(
    int* flag,
    CtranAlgoDeviceState* devState,
    CtranKernelPutSignalArgs args) {
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;
  if (flag && gtIdx == 0) {
    ctran::device::devLoadAbortFlags(flag, devState);
    ctran::device::KernelStartGpe(flag);
  }
  // just atomic store
  if (gtIdx == 0 && args.signalAddr != nullptr) {
#if defined(__HIP_PLATFORM_AMD__)
    // TODO: implement this atomic operations for AMD GPUs.
    trap();
#else
    ::cuda::atomic_ref<uint64_t, cuda::thread_scope_system> ref{
        *args.signalAddr};
    ref.store(args.signalVal, cuda::std::memory_order_release);
#endif
  }

  if (flag && gtIdx == 0) {
    ctran::device::KernelWaitGpeTerminate(flag);
  }
}

__global__ void ncclKernelPut(int* flag, CtranAlgoDeviceState* devState) {
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;
  if (flag && gtIdx == 0) {
    ctran::device::devLoadAbortFlags(flag, devState);
    ctran::device::KernelStartGpe(flag);
    ctran::device::KernelWaitGpeTerminate(flag);
  }
}

__global__ void ncclKernelWaitSignal(
    int* flag,
    CtranAlgoDeviceState* devState,
    CtranKernelWaitSignalArgs args) {
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;
  if (flag && gtIdx == 0) {
    ctran::device::devLoadAbortFlags(flag, devState);
    ctran::device::KernelStartGpe(flag);
  }

  if (args.signalAddr != nullptr && gtIdx == 0) {
#if defined(__HIP_PLATFORM_AMD__)
    // TODO: implement this atomic operations for AMD GPUs.
    trap();
#else
    ::cuda::atomic_ref<uint64_t, cuda::thread_scope_system> ref{
        *args.signalAddr};
    while (ref.load(cuda::std::memory_order_acquire) < args.cmpVal) {
    }
#endif
  }

  if (flag && gtIdx == 0) {
    ctran::device::KernelWaitGpeTerminate(flag);
  }
}

__global__ void ncclKernelSignal(
    int* flag,
    CtranAlgoDeviceState* devState,
    CtranKernelSignalArgs args) {
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;
  if (flag && gtIdx == 0) {
    ctran::device::devLoadAbortFlags(flag, devState);
    ctran::device::KernelStartGpe(flag);
  }
  if (gtIdx == 0 && args.signalAddr != nullptr) {
#if defined(__HIP_PLATFORM_AMD__)
    // TODO: implement this atomic operations for AMD GPUs.
    trap();
#else
    ::cuda::atomic_ref<uint64_t, cuda::thread_scope_system> ref{
        *args.signalAddr};
    ref.store(args.signalVal, cuda::std::memory_order_release);
#endif
  }

  if (flag && gtIdx == 0) {
    ctran::device::KernelWaitGpeTerminate(flag);
  }
}
