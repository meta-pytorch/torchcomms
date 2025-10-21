// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/common/GpeKernelSync.h"
#include "comms/ctran/gpe/CtranGpeDev.h"

// Enumerate functions to test
enum class FnName {
  // KernelElem
  elemWaitPostOrRevoke,
  // CtranAlgoDeviceSync
  devSyncWaitStep,
  devSyncSetNotify,
  devSyncWaitNotify,
  // GpeKernelSync
  waitPost,
  waitPostWarp,
};

struct CtranTestDeviceWaitArgs {
  // inputs
  struct H2d {
    enum FnName fnName;
    KernelElem elem;
    int intSync;
    CtranAlgoDeviceSync devSync;
    ctran::algos::GpeKernelSync gpeKernelSync;
  };
  H2d* h2d;

  // outputs
  struct D2h {
    bool revoked;
    bool warpTested;
  };
  D2h* d2h;
};

extern __global__ void ncclKernelTestDeviceWait(
    int* flag,
    CtranAlgoDeviceState* devState,
    CtranTestDeviceWaitArgs args);
