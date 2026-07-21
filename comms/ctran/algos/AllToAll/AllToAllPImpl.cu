// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/algos/barrier.cuh"
#include "comms/ctran/gpe/CtranGpeDev.h"

namespace ctran::alltoallp {
// Sync-only kernel for the host-CE-copy AllToAllP path. The intra-node data is
// moved by host-issued copy-engine copies enqueued on the same stream ahead of
// this kernel; the kernel provides (a) the GPE-thread handshake the inter-node
// IB path needs and (b) a cross-rank NVL barrier so every rank's copy-engine
// writes into its peers' recvbufs are globally visible before any rank returns.
// Stream ordering guarantees the CE copies complete before this kernel runs;
// barrier() uses system-scoped release/acquire, so the CE writes are visible
// cross-GPU once the barrier completes. Mirrors AllGatherP's
// ncclKernelAllGatherPDirect.
__global__ void ncclKernelAllToAllPDirect(
    ctran::gpe::KernelFlagDev* f,
    CtranAlgoDeviceState* devState) {
  int* flag = f ? const_cast<int*>(f->flag_) : nullptr;
  ctran::device::ColltraceEventScope colltraceScope(f);
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
} // namespace ctran::alltoallp
