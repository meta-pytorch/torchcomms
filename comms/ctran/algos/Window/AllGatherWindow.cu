// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/algos/Window/AllGatherWindowDevTypes.h"
#include "comms/ctran/algos/barrier.cuh"
#include "comms/ctran/algos/common/GpeKernelSyncDev.cuh"
#include "comms/ctran/gpe/CtranGpeDev.h"

using namespace ctran::algos;
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
 * Pipeline start kernel
 *
 * Used when nLocalRanks > 1 in pipeline algorithm. Starts GPE thread for
 * inter-node transfer but does not wait for termination, allowing intra-node
 * CE copies to overlap with inter-node transfer.
 */
__global__ void ncclKernelAllGatherWindowPipeStart(
    int* flag,
    CtranAlgoDeviceState* devState) {
  if (flag) {
    ctran::device::KernelStartGpeAndExit(flag);
  }
}

/**
 * Pipeline sync kernel
 *
 * Blocks intra-node CE copies on the stream until GPE thread has finished
 * the corresponding step's inter-node transfer. Called nNodes-1 times during
 * pipeline algorithm when nLocalRanks > 1.
 */
__global__ void ncclKernelAllGatherWindowPipeSync(
    int* flag,
    CtranAlgoDeviceState* devState,
    PipeSyncKernArgs args) {
  ctran::device::devLoadAbortFlags(flag, devState);
  // Wait until GPE thread posts the current stepId
  GpeKernelSyncDev::waitPost(args.pipeSync, 0, args.stepId);
}

/**
 * Pipeline end kernel
 *
 * Ensures all local ranks have finished the last round intra-node broadcast.
 * Resets sync flags for next use. Called once at the end of pipeline when
 * nLocalRanks > 1.
 */
__global__ void ncclKernelAllGatherWindowPipeEnd(
    int* flag,
    CtranAlgoDeviceState* devState,
    PipeEndKernArgs args) {
  // Reset sync flag for next GPE->kernel pipeline sync
  GpeKernelSyncDev::reset(args.pipeSync, 0);

  // Ensure NVLink intra-node comm finishes
  devStateLoadToShm(devState);
  const auto localRank = statex->localRank();
  const auto nLocalRanks = statex->nLocalRanks();
  barrier(localRank, nLocalRanks);
}

/**
 * Pipeline kernel for single local rank
 *
 * Used when nLocalRanks == 1 in pipeline algorithm. Starts GPE thread and
 * waits for termination. No intra-node broadcast needed.
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
