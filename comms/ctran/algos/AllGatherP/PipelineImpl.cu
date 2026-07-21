// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/ctran/algos/AllGatherP/Types.h"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/algos/barrier.cuh"
#include "comms/ctran/algos/common/GpeKernelSyncDev.cuh"
#include "comms/ctran/algos/common/GpeRing.h"
#include "comms/ctran/gpe/CtranGpeDev.h"

using namespace ctran::algos;
namespace ctran::allgatherp {

// Kernel to block GPE thread to start the internode transfer till allgatherP
// starts on the stream. It doesn't wait for GPE termination, so that the
// algorithm can schedule intra-node CE copies to overlap with internode
// transfer.
// Used when nLocalRanks > 1 in allgatherP pipeline algorithm. It is called once
// to start the algorithm
__global__ void ncclKernelAllGatherPPipeStart(
    ctran::gpe::KernelFlagDev* f,
    CtranAlgoDeviceState* devState) {
  ctran::device::ColltraceEventScope colltraceScope(f);
  int* flag = f ? const_cast<int*>(f->flag_) : nullptr;
  if (flag) {
    ctran::device::KernelStartGpeAndExit(f);
  }
}

// Kernel to block intranode CE copies on the stream till GPE thread has
// finished the corresponding step's internode transfer.
// Used when nLocalRanks > 1 in allgatherP pipeline algorithm. It is called
// nNode - 1 times.
__global__ void ncclKernelAllGatherPPipeSync(
    ctran::gpe::KernelFlagDev* f,
    CtranAlgoDeviceState* devState,
    PipeSyncKernArgs args) {
  ctran::device::ColltraceEventScope colltraceScope(f);
  int* flag = f ? const_cast<int*>(f->flag_) : nullptr;
  ctran::device::devLoadAbortFlags(flag, devState);
  // wait till GPE thread post the current stepId
  GpeKernelSyncDev::waitPost(args.pipeSync, 0, args.stepId);
}

// Stream-holder stub for the ctpipeline (ring) variant at nLocalRanks==1: holds
// the stream while the GPE thread runs the inter-node ring transfer.
// Ring = ctpipeline nLocalRanks==1.
__global__ void ncclKernelAllGatherPRing(
    ctran::gpe::KernelFlagDev* f,
    CtranAlgoDeviceState* devState) {
  ctran::device::ColltraceEventScope colltraceScope(f);
  int* flag = f ? const_cast<int*>(f->flag_) : nullptr;
  if (flag) {
    ctran::device::devLoadAbortFlags(flag, devState);
    ctran::device::KernelStartGpe(f);
    ctran::device::KernelWaitGpeTerminate(flag);
  }
}

// Stream-holder stub for the ctsrdpipeline (streamed recursive-doubling)
// variant at nLocalRanks==1. StreamedRd = ctsrdpipeline nLocalRanks==1.
__global__ void ncclKernelAllGatherPStreamedRd(
    ctran::gpe::KernelFlagDev* f,
    CtranAlgoDeviceState* devState) {
  ctran::device::ColltraceEventScope colltraceScope(f);
  int* flag = f ? const_cast<int*>(f->flag_) : nullptr;
  if (flag) {
    ctran::device::devLoadAbortFlags(flag, devState);
    ctran::device::KernelStartGpe(f);
    ctran::device::KernelWaitGpeTerminate(flag);
  }
}

// Kernel to ensure all local ranks have finished the last round intra-node
// broadcast. Used when nLocalRanks > 1 in allgatherP pipeline algorithm. It is
// called once at the end.
__global__ void ncclKernelAllGatherPPipeEnd(
    ctran::gpe::KernelFlagDev* f,
    CtranAlgoDeviceState* devState,
    PipeEndKernArgs args) {
  ctran::device::ColltraceEventScope colltraceScope(f);

  // Reset sync flag for next GPE->kernel pipeline sync to use
  GpeKernelSyncDev::reset(args.pipeSync, 0);

  // Ensure nvl intra-node comm finishes
  devStateLoadToShm(devState);
  const auto localRank = statex->localRank();
  const auto nLocalRanks = statex->nLocalRanks();
  barrier(localRank, nLocalRanks);
}

// The ctsrdpipeline (streamed recursive-doubling) variant's nLocalRanks > 1
// pipeline kernels. Bodies are identical to the ctpipeline PPipe* kernels
// above; they exist as separate symbols so each variant owns its own kernels.
__global__ void ncclKernelAllGatherPSrdPipeStart(
    ctran::gpe::KernelFlagDev* f,
    CtranAlgoDeviceState* devState) {
  int* flag = f ? const_cast<int*>(f->flag_) : nullptr;
  if (flag) {
    ctran::device::KernelStartGpeAndExit(f);
  }
}

__global__ void ncclKernelAllGatherPSrdPipeSync(
    ctran::gpe::KernelFlagDev* f,
    CtranAlgoDeviceState* devState,
    PipeSyncKernArgs args) {
  int* flag = f ? const_cast<int*>(f->flag_) : nullptr;
  ctran::device::devLoadAbortFlags(flag, devState);
  // wait till GPE thread post the current stepId
  GpeKernelSyncDev::waitPost(args.pipeSync, 0, args.stepId);
}

__global__ void ncclKernelAllGatherPSrdPipeEnd(
    ctran::gpe::KernelFlagDev* flag,
    CtranAlgoDeviceState* devState,
    PipeEndKernArgs args) {
  // Reset sync flag for next GPE->kernel pipeline sync to use
  GpeKernelSyncDev::reset(args.pipeSync, 0);

  // Ensure nvl intra-node comm finishes
  devStateLoadToShm(devState);
  const auto localRank = statex->localRank();
  const auto nLocalRanks = statex->nLocalRanks();
  barrier(localRank, nLocalRanks);
}

} // namespace ctran::allgatherp
