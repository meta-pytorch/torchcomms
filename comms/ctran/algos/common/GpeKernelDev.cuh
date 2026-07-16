// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/common/AtomicUtils.cuh"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/DevShmState.cuh"
#include "comms/ctran/algos/common/GpeKernel.h"
#include "comms/ctran/algos/common/GpeRing.h"
#include "comms/ctran/utils/DevUtils.cuh"

// This file includes only functions to manage Gpe and device kernel lifecycles.
// This file is separate from DevCommon.cuh, since there are source files that
// do not need the full set of helpers from DevCommon.cuh.

/* Gpe Kernel Synchronization functions */

namespace ctran::device {

// TODO(T243528798): remove this preload of devstate by splitting h2d/d2h
// channels.
static inline __device__ void devLoadAbortFlags(
    int* flag,
    CtranAlgoDeviceState* devState) {
  shmDevState.enableCancellableWaits = devState->enableCancellableWaits;
  kernelFlag = flag;
  kernelDoAbort = false;
}

// Publish this cmd's id to the per-comm device dispatch ring, in GPU execution
// order, when the device-ring GPE path is armed for this launch. The ring
// header (GpeKernelFlagHeader) is co-located in the KernelFlagItem immediately
// after its flag array, so it is recovered from `flag` alone — no kernel needs
// a dedicated ring parameter. Single-writer election (block 0, thread 0). A
// no-op when the ring is not armed (enabled == 0), e.g. eager launches, so it
// is safe to call unconditionally at kernel start. The GPE worker consumes the
// ring to learn which command started and in what order. The ring write is the
// HRDWRingBuffer System-scope 128b atomic path (requires sm_90+).
static __forceinline__ __device__ void KernelPublishGpeRing(
    volatile int* flag) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    // flag points at KernelFlagItem::flag_[0]; the header sits right after the
    // CTRAN_ALGO_MAX_THREAD_BLOCKS-int flag array.
    auto* hdr = reinterpret_cast<ctran::gpe::GpeKernelFlagHeader*>(
        const_cast<int*>(flag + CTRAN_ALGO_MAX_THREAD_BLOCKS));
    if (hdr->enabled) {
      hdr->ring.write(hdr->cmdId);
    }
  }
}

static inline __device__ void KernelStartGpe(volatile int* flag) {
  KernelPublishGpeRing(flag);
  comms::device::st_volatile_global(flag, KERNEL_STARTED);
}

static inline __device__ void KernelStartGpeAndExit(volatile int* flag) {
  KernelPublishGpeRing(flag);
  comms::device::st_volatile_global(flag, KERNEL_STARTED_AND_EXIT);
}

static inline __device__ bool KernelTestHostAbort(volatile int* flag) {
  // enableCancellableWaits is the feature guard. See comment on the struct
  // field.
  //
  // Host code (GPE) would set the `flag` to `KERNEL_HOST_ABORT` when the
  // CtranComm instance is in abort state. Any host algoFn errors (remote or
  // local), or user active aborts/timeouts will put CtranComm in such state, if
  // abort is enabled for CtranComm.
  return shmDevState.enableCancellableWaits &&
      (kernelDoAbort ||
       (flag && comms::device::ld_volatile_global(flag) == KERNEL_HOST_ABORT));
}

static inline __device__ bool KernelTestHostAbortBlock(volatile int* flag) {
  if (threadIdx.x == 0 && ctran::device::KernelTestHostAbort(kernelFlag)) {
    kernelDoAbort = true;
  }
  __syncthreads();
  return kernelDoAbort;
}

static inline __device__ void KernelWaitGpeTerminate(volatile int* flag) {
  int flagVal = KERNEL_STARTED;
  do {
    flagVal = comms::device::ld_volatile_global(flag);
  } while (
      flagVal != KERNEL_TERMINATE &&
      !(shmDevState.enableCancellableWaits && flagVal == KERNEL_HOST_ABORT));

  // Mark the flag as unset for reclaim
  comms::device::st_volatile_global(flag, KERNEL_UNSET);
}

} // namespace ctran::device
