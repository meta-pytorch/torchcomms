// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/algos/common/GpeKernelSyncDev.cuh"
#include "comms/ctran/algos/tests/CtranDevWaitUTKernels.cuh"

namespace ctran::testing {

__device__ void testDeviceWait(int bId, CtranTestDeviceWaitArgs& args) {
  switch (args.h2d->fnName) {
    case FnName::elemWaitPostOrRevoke:
      elemWaitPostOrRevoke(
          &args.h2d->elem, /*groupIdx=*/bId, &(args.d2h->revoked));
      break;
    case FnName::devSyncWaitStep:
      // host side is also expecting the step value 1
      devSyncWaitStep(&args.h2d->intSync, /*groupIdx=*/bId, /*val=*/1);
      break;
    case FnName::devSyncSetNotify:
      devSyncSetNotify(&args.h2d->devSync, /*groupIdx=*/bId);
      break;
    case FnName::devSyncWaitNotify:
      devSyncWaitNotify(&args.h2d->devSync, /*groupIdx=*/bId);
      break;
    case FnName::waitPost:
      algos::GpeKernelSyncDev::waitPost(
          &args.h2d->gpeKernelSync, /*workerId=*/bId, /*step=*/1);
      break;
    case FnName::waitPostWarp:
      const auto wId = threadIdx.x / kWarpSize;
      const auto laneId = threadIdx.x & (kWarpSize - 1);
      // ensure warps operate independently
      if (wId == 1) {
        if (laneId == 0) {
          args.d2h->warpTested = true;
        }
        algos::GpeKernelSyncDev::waitPostWarp(
            &args.h2d->gpeKernelSync, /*workerId=*/bId, /*step=*/1);
      }
      break;
  }
}

} // namespace ctran::testing

__global__ void ncclKernelTestDeviceWait(
    int* flag,
    CtranAlgoDeviceState* devState,
    CtranTestDeviceWaitArgs args) {
  const auto bId = blockIdx.x;

  devStateLoadToShm(&flag[bId], devState);

  ctran::testing::testDeviceWait(bId, args);

  __syncthreads();
}
