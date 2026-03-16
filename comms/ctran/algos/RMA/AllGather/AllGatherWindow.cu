// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/algos/barrier.cuh"
#include "comms/ctran/gpe/CtranGpeDev.h"

namespace ctran::allgatherwindow {

/**
 * Direct algorithm kernel
 *
 * Starts GPE to do all-to-all PUT with atomic signaling, then waits for
 * signals from all remote peers. Uses barrier for intra-node synchronization.
 */
__global__ void ncclKernelAllGatherWindowDirect(
    int* flag,
    CtranAlgoDeviceState* devState) {
  if (flag) {
    ctran::device::devLoadAbortFlags(flag, devState);
    ctran::device::KernelStartGpe(flag);
  }

  devStateLoadToShm(devState);

  const auto localRank = statex->localRank();
  const auto nLocalRanks = statex->nLocalRanks();

  // Ensure NVLink intra-node copies complete before proceeding
  barrier(localRank, nLocalRanks);

  if (flag) {
    ctran::device::KernelWaitGpeTerminate(flag);
  }
}

/**
 * Pipeline algorithm: PipeStart kernel
 *
 * Starts GPE and exits immediately so the stream can proceed with
 * nvlCeBcast + ctranWaitSignal for inter-rank synchronization.
 * Used when nLocalRanks > 1.
 */
__global__ void ncclKernelAllGatherWindowPipeStart(
    int* flag,
    CtranAlgoDeviceState* devState) {
  if (flag) {
    ctran::device::KernelStartGpeAndExit(flag);
  }
}

/**
 * Pipeline algorithm: PipeEnd kernel
 *
 * Barrier to ensure all local ranks finished nvlCeBcast before proceeding.
 * Used when nLocalRanks > 1, called once at the end.
 */
__global__ void ncclKernelAllGatherWindowPipeEnd(
    int* flag,
    CtranAlgoDeviceState* devState) {
  devStateLoadToShm(devState);
  const auto localRank = statex->localRank();
  const auto nLocalRanks = statex->nLocalRanks();
  barrier(localRank, nLocalRanks);
}

/**
 * Pipeline algorithm: Pipe kernel (blocking)
 *
 * Starts GPE and waits for it to terminate. Used when nLocalRanks == 1
 * (no intra-node broadcast needed, GPE does the entire ring transfer).
 */
__global__ void ncclKernelAllGatherWindowPipe(
    int* flag,
    CtranAlgoDeviceState* devState) {
  if (flag) {
    ctran::device::devLoadAbortFlags(flag, devState);
    ctran::device::KernelStartGpe(flag);
    ctran::device::KernelWaitGpeTerminate(flag);
  }
}

} // namespace ctran::allgatherwindow
