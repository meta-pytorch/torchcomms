// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/gpe/CtranGpeDev.h"

//------------------------------------------------------------------------------
// Test Kernel
//------------------------------------------------------------------------------

__global__ void devSyncWaitNotifyKernel(
    CtranAlgoDeviceSync* localSync,
    int nGroups) {
  // TODO(T243528798): remove this preload of devstate by splitting h2d/d2h
  // channels.
  shmDevState.enableCancellableWaits = false;
  // Receiver waits for notification - sync is pre-initialized to NOTIFY_SET
  if (blockIdx.x == 0) {
    devSyncWaitNotify(localSync, nGroups);
  }
}

__global__ void
devSyncOnStepsKernel(CtranAlgoDeviceSync* sync, bool isProducer, int nSteps) {
  // TODO(T243528798): remove this preload of devstate by splitting h2d/d2h
  // channels.
  shmDevState.enableCancellableWaits = false;
  auto groupIdx = blockIdx.x;
  for (int step = 1; step <= nSteps; step++) {
    if (isProducer) {
      devSyncWaitStep(sync, groupIdx, CTRAN_ALGO_STEP_RESET);
      devSyncSetStep(sync, groupIdx, step);
    } else {
      devSyncWaitStep(sync, groupIdx, step);
      devSyncSetStep(sync, groupIdx, CTRAN_ALGO_STEP_RESET);
    }
  }
}
