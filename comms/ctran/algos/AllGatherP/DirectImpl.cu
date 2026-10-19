// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/algos/barrier.cuh"
#include "comms/ctran/algos/common/GpeRing.h"
#include "comms/ctran/gpe/CtranGpeDev.h"
#include "comms/utils/colltrace/ColltraceDeviceHandle.h"

namespace ctran::allgatherp {
__global__ void ncclKernelAllGatherPDirect(
    ctran::gpe::KernelFlagDev* f,
    CtranAlgoDeviceState* devState,
    meta::comms::colltrace::ColltraceDeviceHandle colltraceHdr) {
  ctran::device::ColltraceEventScope colltraceScope(colltraceHdr);
  int* flag = f ? const_cast<int*>(f->flag_) : nullptr;
  if (flag) {
    ctran::device::devLoadAbortFlags(flag, devState);
    ctran::device::KernelStartGpe(f);
  }

  devStateLoadToShm(devState);

  const auto localRank = statex->localRank();
  const auto nLocalRanks = statex->nLocalRanks();

  // ensure nvl intra-node comm finishes
  barrier(localRank, nLocalRanks);
  if (flag) {
    ctran::device::KernelWaitGpeTerminate(flag);
  }
}
} // namespace ctran::allgatherp
