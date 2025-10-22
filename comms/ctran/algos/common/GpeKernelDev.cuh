// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/algos/DevShmState.cuh"
#include "comms/ctran/algos/common/GpeKernel.h"
#include "comms/ctran/utils/DevUtils.cuh"

// This file includes only functions to manage Gpe and device kernel lifecycles.
// This file is separate from DevCommon.cuh, since there are source files that
// do not need the full set of helpers from DevCommon.cuh.

/* Gpe Kernel Synchronization functions */

namespace ctran::device {

static inline __device__ bool KernelTestHostAbort(volatile int* flag);

static inline __device__ void KernelStartGpe(volatile int* flag) {
  if (KernelTestHostAbort(flag)) {
    return;
  }
  ctran::utils::storeInt(flag, KERNEL_STARTED);
}

static inline __device__ void KernelStartGpeAndExit(volatile int* flag) {
  ctran::utils::storeInt(flag, KERNEL_STARTED_AND_EXIT);
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
      (kernelDoAbort || ctran::utils::loadInt(flag) == KERNEL_HOST_ABORT);
}

static inline __device__ bool KernelTestHostAbortBlock(volatile int* flag) {
  if (threadIdx.x == 0 && ctran::device::KernelTestHostAbort(kernelFlag)) {
    kernelDoAbort = true;
  }
  return kernelDoAbort;
}

static inline __device__ void KernelWaitGpeTerminate(volatile int* flag) {
  int flagVal = KERNEL_STARTED;
  do {
    flagVal = ctran::utils::loadInt(flag);
  } while (
      flagVal != KERNEL_TERMINATE &&
      !(shmDevState.enableCancellableWaits && flagVal == KERNEL_HOST_ABORT));

  // Mark the flag as unset for reclaim
  ctran::utils::storeInt(flag, KERNEL_UNSET);
}

} // namespace ctran::device
