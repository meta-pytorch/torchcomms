// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include "comms/common/HipBf16Compat.h"
#else
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#endif
#include "comms/ctran/algos/DevCommon.cuh"

/*
 * This is a multi-step logarithmic complexity barrier algorithm that
 * is safe for non-power-of-two processes.
 *
 * Step 1: We form groups of 2.  Each group synchronizes within it.
 * Step 2: We form groups of 4.  In each group, the lower half of the
 *   processes do a pair-wise synchronization with the higher half of
 *   the processes.  At this point, everyone in the group is
 *   synchronized.
 * Steps 3 onward: We continue the same model with doubling group
 *   sizes.  After each step, all processes in the group are
 *   synchronized.
 *
 * If a process does not have a peer (because the number of ranks is
 * not a power of two), that process simply skips that step.
 *
 * For synchronization, the lower rank process sets a flag on the
 * higher rank process' mailbox.  The higher rank process on
 * receiving this flag, resets the flag and sets a flag on the lower
 * rank process' mailbox.  The lower rank flag then resets its local
 * flag.  This approach avoids race conditions where the flag could be
 * overwritten before it is read by the other process.
 */
__device__ __forceinline__ void barrier(int rank, int nranks) {
  CtranAlgoDeviceSync* sync;

  int nsteps = 0;
  while ((1 << nsteps) < nranks) {
    nsteps++;
  }

  for (int step = 0; step < nsteps; step++) {
    int groupSize = 1 << (step + 1);
    int group = rank / groupSize;
    int groupRank = rank - group * groupSize;

    if (groupRank < groupSize / 2) {
      int peer = group * groupSize + groupRank + groupSize / 2;

      if (peer >= nranks) {
        continue;
      }

      // Ping to remote peer
      sync = devSyncGetLoc<REMOTE>(peer);
      // First wait for CTRAN_ALGO_STEP_RESET ensures the completion of previous
      // step or previous collective
      devSyncWaitStep(sync, blockIdx.x, CTRAN_ALGO_STEP_RESET);
      devSyncSetStep(sync, blockIdx.x, step);

      // Pong from remote peer to local
      sync = devSyncGetLoc<LOCAL>(peer);
      devSyncWaitStep(sync, blockIdx.x, step);
      // Mark this step has been synced, thus peer can use for next step
      devSyncSetStep(sync, blockIdx.x, CTRAN_ALGO_STEP_RESET);
    } else {
      int peer = group * groupSize + groupRank - groupSize / 2;

      if (peer >= nranks) {
        continue;
      }

      // Pong from remote peer to local
      sync = devSyncGetLoc<LOCAL>(peer);
      devSyncWaitStep(sync, blockIdx.x, step);
      // Mark this step has been synced, thus peer can use for next step
      devSyncSetStep(sync, blockIdx.x, CTRAN_ALGO_STEP_RESET);

      // Ping to remote peer
      sync = devSyncGetLoc<REMOTE>(peer);
      // First wait for CTRAN_ALGO_STEP_RESET ensures the completion of previous
      // step or previous collective
      devSyncWaitStep(sync, blockIdx.x, CTRAN_ALGO_STEP_RESET);
      devSyncSetStep(sync, blockIdx.x, step);
    }
  }
}
