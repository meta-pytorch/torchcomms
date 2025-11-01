// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <stdio.h>
#include <cstddef>
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/DevAlgoImpl.cuh"
#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/gpe/CtranGpeDev.h"

template <typename T>
__device__ void prepareBcastArg(
    KernelElem* elemH,
    CtranKernelAllToAllDedupArgs& args,
    CtranAlgoDevBcastArg& bcastArg) {
  bcastArg.src = elemH->bcast.src;
  bcastArg.count = elemH->bcast.count;
  // need barrier to ensure all peers have finished update to the local dst
  // before kernel finishes
  bcastArg.barrier = true;
  // No host side access before kernel terminates; thus skip flush
  bcastArg.flushMem = false;
  bcastArg.nvectors = statex->nLocalRanks();

  // TODO: recv buffs are static since persistent collective, so pass in
  // pointers instead and compute offset
  //  H2D load dsts that has to be specified by GPE thread
  loadAlgoDevVecPtr(bcastArg.dsts, elemH->bcast.dsts, bcastArg.nvectors);
}

template <typename T>
static __device__ __forceinline__ void bcastOnPost(
    KernelElem* elemH,
    CtranKernelAllToAllDedupArgs& args,
    const int numBlocksPerBcast,
    int bcastBlockIdx) {
  bool revoked = false;
  elemWaitPostOrRevokeByGroup(elemH, bcastBlockIdx, &revoked);

  // Load arguments from host-pinned memory before executing bcast
  CtranAlgoDevBcastArg bcastArg;
  prepareBcastArg<T>(elemH, args, bcastArg);

  // Choose kMultiPutBcast since GPE thread ensures all ranks have joined and
  // the internal peer shift can avoid incast congestion in multiPut
  ctranKernBcast<T, kMultiPutBcast>(
      bcastArg, elemH, bcastBlockIdx, numBlocksPerBcast, 0);

  elemCompleteByGroup(elemH, bcastBlockIdx);
}

template <typename T>
__global__ void ncclKernelAllToAllDedup(
    int* flag,
    CtranAlgoDeviceState* devState,
    CtranKernelAllToAllDedupArgs args) {
  // TODO(T243528798): remove this preload of devstate by splitting h2d/d2h
  // channels.
  shmDevState.enableCancellableWaits = devState->enableCancellableWaits;
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  if (flag && gtIdx == 0) {
    ctran::device::KernelStartGpe(flag);
  }

  devStateLoadToShm(devState);

  const auto numBlocks = gridDim.x;
  const auto numBcasts = args.numIbPeers;
  const auto numBlocksPerBcast = numBlocks / numBcasts;
  const auto bcastGroup = blockIdx.x / numBlocksPerBcast;
  const auto bcastBlockIdx = blockIdx.x % numBlocksPerBcast;

  // get KernelElem associated with bcastGroup
  KernelElem* curElem = args.bcastElemList;
  for (int i = 0; i < bcastGroup; i++) {
    curElem = curElem->next;
  }

  bcastOnPost<T>(curElem, args, numBlocksPerBcast, bcastBlockIdx);

  if (flag && gtIdx == 0) {
    ctran::device::KernelWaitGpeTerminate(flag);
  }
}

#define DECL_ALLTOALLDEDUP_KERN(T)                     \
  template __global__ void ncclKernelAllToAllDedup<T>( \
      int* flag,                                       \
      CtranAlgoDeviceState* devState,                  \
      CtranKernelAllToAllDedupArgs args)

DECL_ALLTOALLDEDUP_KERN(int8_t);
DECL_ALLTOALLDEDUP_KERN(uint8_t);
DECL_ALLTOALLDEDUP_KERN(int32_t);
DECL_ALLTOALLDEDUP_KERN(uint32_t);
DECL_ALLTOALLDEDUP_KERN(int64_t);
DECL_ALLTOALLDEDUP_KERN(uint64_t);
DECL_ALLTOALLDEDUP_KERN(half);
DECL_ALLTOALLDEDUP_KERN(float);
DECL_ALLTOALLDEDUP_KERN(double);
#if defined(__CUDA_BF16_TYPES_EXIST__)
DECL_ALLTOALLDEDUP_KERN(__nv_bfloat16);
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__) && defined(NCCL_ENABLE_FP8)
DECL_ALLTOALLDEDUP_KERN(__nv_fp8_e4m3);
DECL_ALLTOALLDEDUP_KERN(__nv_fp8_e5m2);
#endif
