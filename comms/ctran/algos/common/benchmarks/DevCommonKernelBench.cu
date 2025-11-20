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

// Call KernelElem-related funcs to measure time consumed by KernelElem in
// putNotify.
__global__ void
KernelElemPutNotifyKernel(KernelElem* elem, int nGroups, int iters) {
  // TODO(T243528798): remove this preload of devstate by splitting h2d/d2h
  // channels.
  shmDevState.enableCancellableWaits = false;
  const auto groupIdx = blockIdx.x;
  for (int i = 0; i < iters; i++) {
    bool revoked = false;
    uint64_t recvbuffAddr =
        elemWaitPostOrRevokeByGroupForMultiPut(elem, groupIdx, &revoked);

    const char* sendbuff =
        reinterpret_cast<const char*>(elem->putNotify.sendbuff);
    size_t nbytes = elem->putNotify.nbytes;
    elemCompleteByGroup(elem, groupIdx);
  }
}

// Call KernelElem-related funcs to measure time consumed by KernelElem in
// waitNotify.
__global__ void KernelElemWaitNotifyKernel(KernelElem* elem, int iters) {
  // TODO(T243528798): remove this preload of devstate by splitting h2d/d2h
  // channels.
  shmDevState.enableCancellableWaits = false;
  const auto groupIdx = blockIdx.x;
  if (groupIdx != 0) {
    return;
  }
  for (int i = 0; i < iters; i++) {
    bool revoked = false;
    elemWaitPostOrRevokeByGroup(elem, groupIdx, &revoked);
    auto peerLocalRank = elem->waitNotify.peerLocalRank;
    peerLocalRank++;
    auto ngroups = elem->waitNotify.ngroups;
    ngroups++;
    elemCompleteByGroup(elem, groupIdx);
  }
}
