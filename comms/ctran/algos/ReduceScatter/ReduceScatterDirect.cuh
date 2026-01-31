// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef REDUCE_SCATTER_DIRECT_H_INCLUDED
#define REDUCE_SCATTER_DIRECT_H_INCLUDED

#include <stdio.h>
#include <cstddef>
#include <iostream>

#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/DevAlgoImpl.cuh"
#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/algos/ReduceScatter/Types.h"
#include "comms/ctran/gpe/CtranGpeDev.h"

template <typename T>
__device__ void prepareRedArg(
    KernelElem* elemH,
    ctran::reducescatter::KernelArgs& args,
    CtranAlgoDevReduceArg& redArg) {
  // const value for single node ReduceScatter
  redArg.dsts[0] = reinterpret_cast<void*>(args.recvbuff);
  // only single round
  redArg.isFinal = true;
  // need barrier to ensure remote local ranks have finsihed
  redArg.barrier = true;
  // no flush since no data will be read before kernel ends
  redArg.flushMem = false;
  redArg.count = args.recvcount;
  redArg.nsrcs = statex->nLocalRanks();
  redArg.ndsts = 1;

  // H2D load srcs that has to be specified by GPE thread
  loadAlgoDevVecPtrs(
      redArg.srcs,
      reinterpret_cast<const void* volatile*>(elemH->reduce.srcs),
      redArg.nsrcs);
}

template <typename T>
__device__ void prepareRedArg(
    ctran::reducescatter::KernelArgs& args,
    CtranAlgoDevReduceArg& redArg) {
  const auto localRank = statex->localRank();
  const auto nLocalRanks = statex->nLocalRanks();

  // const value for single node ReduceScatter
  redArg.dsts[0] = reinterpret_cast<void*>(args.recvbuff);
  // only single round
  redArg.isFinal = true;
  // need barrier to ensure remote local ranks have finsihed
  redArg.barrier = true;
  // no flush since no data will be read before kernel ends
  redArg.flushMem = false;
  redArg.count = args.recvcount;
  redArg.nsrcs = nLocalRanks;
  redArg.ndsts = 1;
  // Specify pre-allocated IPC buffer as src of each peer
  // NOTE: unlike direct approach where GPE thread already counts the offset
  // shift, the IPC buffer contains all data in each peer's sendbuff, thus we
  // need shift offset here.
  size_t srcOffset = localRank * args.recvcount * sizeof(T);
  for (int i = 0; i < nLocalRanks; i++) {
    redArg.srcs[i] = reinterpret_cast<char*>(devBcastBufGetLoc(i)) + srcOffset;
  }
}

template <typename T, commRedOp_t RedOp>
__device__ void reduceScatterDirectStaged(
    ctran::reducescatter::KernelArgs& args) {
  const auto localRank = statex->localRank();
  const auto nRanks = statex->nRanks();

  CtranAlgoDevReduceArg redArg;
  prepareRedArg<T>(args, redArg);

  // Copy data from local sendbuff to pre-allocated bcast IPC buffer on the
  // local GPU
  T* ipcSrc = reinterpret_cast<T*>(devBcastBufGetLoc(localRank));
  const auto groupIdx = blockIdx.x;
  const auto ngroups = gridDim.x;
  size_t sendcount = args.recvcount * nRanks;
  ctranKernCopy<T>(
      reinterpret_cast<const T*>(args.sendbuff),
      ipcSrc,
      sendcount,
      groupIdx,
      ngroups);

  // Ensure all ranks have filled the IPC buffer with local send data
  barrier(localRank, redArg.nsrcs);

  // Perform reduce via loading data from peers' IPC buffer
  ctranKernReduce<T, RedOp>(redArg, nullptr, 0);

  // Omit post-reduce barrier since it is already called within ctranKernReduce
}

template <typename T, commRedOp_t RedOp>
__device__ void reduceScatterDirect(ctran::reducescatter::KernelArgs& args) {
  // intra-node reduce from all localRanks
  if (args.intraReduce != nullptr) {
    const auto groupIdx = blockIdx.x;

    KernelElem* elemH = args.intraReduce;
    bool revoked = false;
    elemWaitPostOrRevokeByGroup(elemH, groupIdx, &revoked);

    // Skip if entire elem has revoked
    if (!revoked) {
      CtranAlgoDevReduceArg redArg;
      // Prepare redArg from host pinned-memory elemH and local args
      prepareRedArg<T>(elemH, args, redArg);

      // Perform reduce op with loaded arguments
      ctranKernReduce<T, RedOp>(redArg, elemH, 0);

      // Only thread 0 to label the finished step and the complete check will
      // ensure all groups have finished when host side sees the completion with
      // the stepId.
      if (threadIdx.x == 0) {
        elemH->stepDone = 0;
      }
      elemCompleteByGroup(elemH, groupIdx);
    }
  }
}

template <typename T, commRedOp_t RedOp>
__global__ void __launch_bounds__(1024, 1) ncclKernelReduceScatterDirect(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::reducescatter::KernelArgs args) {
  // TODO(T243528798): remove this preload of devstate by splitting h2d/d2h
  // channels.
  shmDevState.enableCancellableWaits = devState->enableCancellableWaits;
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  if (flag && gtIdx == 0) {
    ctran::device::devLoadAbortFlags(flag, devState);
    ctran::device::KernelStartGpe(flag);
  }

  devStateLoadToShm(devState);
  if (args.stageCopy) {
    reduceScatterDirectStaged<T, RedOp>(args);
  } else {
    reduceScatterDirect<T, RedOp>(args);
  }

  if (flag && gtIdx == 0) {
    ctran::device::KernelWaitGpeTerminate(flag);
  }
}

#define DECL_CTRAN_REDUCESCATTERDIRECT_KERN(T, RedOp) \
  template __global__ void __launch_bounds__(1024, 1) \
      ncclKernelReduceScatterDirect<T, RedOp>(        \
          int* flag,                                  \
          CtranAlgoDeviceState* devState,             \
          ctran::reducescatter::KernelArgs args)

#endif
