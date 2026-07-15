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

// Publish this launch's cmd id to the device ring in GPU execution order.
// Single-writer election (block 0, thread 0); no-op when not armed
// (hdr.enabled == 0). The GPE worker consumes the ring to order started cmds.
// The ring write is the HRDWRingBuffer System-scope 128b atomic path.
static __forceinline__ __device__ void KernelPublishGpeRing(
    ctran::gpe::GpeKernelFlagHeader hdr) {
  if (blockIdx.x == 0 && threadIdx.x == 0 && hdr.enabled) {
    hdr.ring.write(hdr.cmdId);
  }
}

// Kernel start prologue: publish this cmd's id to the ring (block 0 only, no-op
// when not armed), then signal the GPE worker that block `bId` has started. The
// ring header is read directly from f->gpeHdr — no pointer-offset recovery.
static inline __device__ void KernelStartGpe(
    ctran::gpe::KernelFlagDev* f,
    int bId = 0) {
  KernelPublishGpeRing(f->gpeHdr);
  comms::device::st_volatile_global(&f->flag_[bId], KERNEL_STARTED);
}

static inline __device__ void KernelStartGpeAndExit(
    ctran::gpe::KernelFlagDev* f,
    int bId = 0) {
  KernelPublishGpeRing(f->gpeHdr);
  comms::device::st_volatile_global(&f->flag_[bId], KERNEL_STARTED_AND_EXIT);
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
