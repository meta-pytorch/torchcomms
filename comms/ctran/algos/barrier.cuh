// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#else
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#endif
#include "comms/ctran/algos/DevCommon.cuh"

/*
 * Dissemination barrier: a logarithmic-step barrier that is correct for ANY
 * number of ranks, including non-powers-of-two.
 *
 * At step k (k = 0, 1, ..., nsteps-1, with distance d = 2^k) every rank r:
 *   - signals rank up   = (r + d)          % nranks, and
 *   - waits on rank down = (r - d + nranks) % nranks.
 * After ceil(log2(nranks)) steps the arrival of every rank has propagated to
 * every other rank, so no rank exits before all ranks have arrived. Unlike a
 * pairwise/butterfly barrier, no peer is ever skipped -- that skip is exactly
 * what dropped synchronization edges (and thus broke correctness) for
 * non-power-of-two rank counts.
 *
 * Each directed edge r -> up uses the mailbox that lives in up's memory and is
 * dedicated to the (r -> up) channel: r posts via devSyncGetLoc<REMOTE>(up) and
 * up consumes the very same slot via devSyncGetLoc<LOCAL>(r) (its down at the
 * same step is r). A given physical mailbox is therefore touched by exactly one
 * (writer, reader) pair, at exactly one step, per barrier() call.
 *
 * Reuse safety across repeated barrier() calls relies on the two-phase
 * step/RESET handshake: the writer waits for the slot to read RESET (left so by
 * the reader of the previous call, or by initialization) before posting `step`,
 * and the reader restores RESET after consuming. A stale value can never
 * satisfy a later call's wait -- the reader always resets its own slot before
 * its next call, and the writer is blocked from overwriting until the previous
 * value has been consumed.
 *
 * Each rank posts its outgoing signal before blocking on its incoming one.
 * Posting only requires the target slot to already be RESET (guaranteed by
 * initialization or the previous call, never by the current step), so every
 * rank can post before any rank blocks; this breaks the cyclic wait that would
 * otherwise deadlock the ring of edges within a step.
 */
__device__ __forceinline__ void barrier(int rank, int nranks) {
  int nsteps = 0;
  while ((1 << nsteps) < nranks) {
    nsteps++;
  }

  for (int step = 0; step < nsteps; step++) {
    const int dist = 1 << step;
    const int up = (rank + dist) % nranks;
    const int down = (rank - dist + nranks) % nranks;

    // Signal `up`: post `step` into the slot in up's memory that belongs to
    // this rank, once the previous call's reader has left it at RESET.
    CtranAlgoDeviceSync* upSync = devSyncGetLoc<REMOTE>(up);
    devSyncWaitStep(upSync, blockIdx.x, CTRAN_ALGO_STEP_RESET);
    devSyncSetStep(upSync, blockIdx.x, step);

    // Wait on `down`: consume its signal from the slot in this rank's memory,
    // then restore RESET so the next call's writer can reuse the slot.
    CtranAlgoDeviceSync* downSync = devSyncGetLoc<LOCAL>(down);
    devSyncWaitStep(downSync, blockIdx.x, step);
    devSyncSetStep(downSync, blockIdx.x, CTRAN_ALGO_STEP_RESET);
  }
}
