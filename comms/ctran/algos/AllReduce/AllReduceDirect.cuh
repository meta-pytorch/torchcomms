// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ALLREDUCE_DIRECT_H_INCLUDED
#define ALLREDUCE_DIRECT_H_INCLUDED

#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/DevAlgoImpl.cuh"
#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/algos/barrier.cuh"
#include "comms/ctran/algos/bcast.cuh"
#include "comms/ctran/algos/localReduce.cuh"
#include "comms/ctran/commstate/CommStateXDev.h"
#include "comms/ctran/gpe/CtranGpeDev.h"
#include "comms/utils/commSpecs.h"

template <typename T, commRedOp_t RedOp>
static __device__ __forceinline__ void reduceOnPost(
    CtranAlgoDeviceState* devState,
    KernelElem* elem) {
  bool revoked = false;
  const T* srcs[CTRAN_MAX_NVL_PEERS];
  T* dst;

  elemWaitPostOrRevokeByGroup(elem, blockIdx.x, &revoked);

  for (int i = 0; i < elem->localReduce.nvectors; i++) {
    srcs[i] = reinterpret_cast<const T*>(elem->localReduce.srcs[i]);
  }
  dst = reinterpret_cast<T*>(elem->localReduce.dst);

  localReduce<T, RedOp>(
      elem->localReduce.nvectors, srcs, dst, elem->localReduce.count);

  /* sync with the remaining local GPUs to tell them that we are done */
  __threadfence_system();

  const auto localRank = statex->localRank();
  const auto nLocalRanks = statex->nLocalRanks();
  barrier(localRank, nLocalRanks);

  elemCompleteByGroup(elem, blockIdx.x);
}

template <typename T>
static __device__ __forceinline__ void bcastOnPost(
    CtranAlgoDeviceState* devState,
    KernelElem* elem) {
  bool revoked = false;
  const T* src;
  T* dsts[CTRAN_MAX_NVL_PEERS];

  elemWaitPostOrRevokeByGroup(elem, blockIdx.x, &revoked);

  const auto nLocalRanks = statex->nLocalRanks();
  const auto localRank = statex->localRank();

  src = reinterpret_cast<const T*>(elem->bcast.src);
  for (int i = 0; i < nLocalRanks; i++) {
    dsts[i] = reinterpret_cast<T*>(elem->bcast.dsts[i]);
  }

  bcast<T>(nLocalRanks, src, dsts, elem->bcast.count, blockIdx.x, gridDim.x);

  /* sync with the remaining local GPUs to tell them that we are done */
  __threadfence_system();
  barrier(localRank, nLocalRanks);

  elemCompleteByGroup(elem, blockIdx.x);
}

template <typename T, commRedOp_t RedOp>
__global__ void ncclKernelAllReduceCtranDirect(
    int* flag,
    CtranAlgoDeviceState* devState,
    CtranKernelAllReduceArgs args) {
  const auto tId = threadIdx.x;
  const auto bId = blockIdx.x;

  if (flag && tId == 0) {
    ctran::device::KernelStartGpe(&flag[bId]);
  }

  devStateLoadToShm(devState);

  const auto nLocalRanks = statex->nLocalRanks();
  const auto localRank = statex->localRank();

  /* sync with the remaining local GPUs to tell them that we are done */
  __threadfence_system();
  barrier(localRank, nLocalRanks);

  // Repeat nSteps for symmetric segment (nSteps == 0 if segment count is 0)
  for (int c = 0; c < args.nSteps; c++) {
    /* Step 1: Intra-node Reduce-scatter */
    reduceOnPost<T, RedOp>(
        devState,
        args.kernelElems[static_cast<int>(
            AllReduceKernElemRole::kIntraReduceScatter)]);

    /* Step 2: Inter-node Reduce-scatter */
    // Each step handles different portion, thus apply post-sum-avg here
    ctranKernMultiStridedReduce<T, RedOp, true /* complete */, false /*free*/>(
        args.kernelElems[static_cast<int>(
            AllReduceKernElemRole::kInterReduceScatter)],
        true,
        c);

    /* Step 3: Inter-node Allgather */
    /* This does not need a kernel element; listing here for
     * completeness */

    /* Step 4: Intra-node Allgather */
    bcastOnPost<T>(
        devState,
        args.kernelElems[static_cast<int>(
            AllReduceKernElemRole::kIntraAllGather)]);
  }

  // Optional steps for remCount
  KernelElem* kRemIntraReduce = args.kernelElems[static_cast<int>(
      AllReduceKernElemRole::kRemIntraReduce)];
  if (kRemIntraReduce) {
    /* Step 5: Intra-node reduce(root=0) */
    reduceOnPost<T, RedOp>(devState, kRemIntraReduce);
  }

  /* Step 6: Intra-node Bcast */
  KernelElem* kRemIntraBcast =
      args.kernelElems[static_cast<int>(AllReduceKernElemRole::kRemIntraBcast)];
  if (kRemIntraBcast) {
    bcastOnPost<T>(devState, kRemIntraBcast);
  }

  /* Step 7: Inter-node Allreduce */
  KernelElem* kRemInterReduce = args.kernelElems[static_cast<int>(
      AllReduceKernElemRole::kRemInterReduce)];
  if (kRemInterReduce) {
    // Last step to handle remCount segment, thus apply post-sum-avg here
    ctranKernMultiStridedReduce<T, RedOp, true /* complete */, false /*free*/>(
        kRemInterReduce, true /* last step*/);
  }

  /* Complete kernel */
  if (flag && tId == 0) {
    ctran::device::KernelWaitGpeTerminate(&flag[bId]);
  }
}

#define DECL_CTRAN_ALLREDUCEDIRECT_KERN(T, RedOp)                    \
  template __global__ void ncclKernelAllReduceCtranDirect<T, RedOp>( \
      int* flag,                                                     \
      CtranAlgoDeviceState* devState,                                \
      CtranKernelAllReduceArgs args);

#endif
