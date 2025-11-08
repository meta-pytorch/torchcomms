// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <stdio.h>
#include <cstddef>
#include <iostream>
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/DevAlgoImpl.cuh"
#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/gpe/CtranGpeDev.h"

__device__ void prepareBcastArg(
    KernelElem* elemH,
    CtranKernelAllGatherArgs& args,
    CtranAlgoDevBcastArg& bcastArg) {
  bcastArg.src = args.sendbuff;
  bcastArg.count = args.count;
  // need barrier to ensure all peers have finished update to the local dst
  // before kernel finishes
  bcastArg.barrier = true;
  // No host side access before kernel terminates; thus skip flush
  bcastArg.flushMem = false;
  bcastArg.nvectors = statex->nLocalRanks();

  // H2D load dsts that has to be specified by GPE thread
  loadAlgoDevVecPtr(bcastArg.dsts, elemH->bcast.dsts, bcastArg.nvectors);
}

template <typename T>
static __device__ __forceinline__ void bcastOnPost(
    KernelElem* elemH,
    CtranKernelAllGatherArgs& args) {
  bool revoked = false;
  elemWaitPostOrRevokeByGroup(elemH, blockIdx.x, &revoked);

  // Load arguments from host-pinned memory before executing bcast
  CtranAlgoDevBcastArg bcastArg;
  prepareBcastArg(elemH, args, bcastArg);

  // Choose kMultiPutBcast since GPE thread ensures all ranks have joined and
  // the internal peer shift can avoid incast congestion in multiPut
  ctranKernBcast<T, kMultiPutBcast>(bcastArg, elemH, blockIdx.x, gridDim.x, 0);

  elemCompleteByGroup(elemH, blockIdx.x);
}

template <typename T>
__global__ void __launch_bounds__(1024, 1) ncclKernelAllGatherCtranDirect(
    int* flag,
    CtranAlgoDeviceState* devState,
    CtranKernelAllGatherArgs args) {
  const auto gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  if (flag && gtIdx == 0) {
    ctran::device::devLoadAbortFlags(flag, devState);
    ctran::device::KernelStartGpe(flag);
  }

  devStateLoadToShm(devState);

  if (args.bcastElem != nullptr) {
    bcastOnPost<T>(args.bcastElem, args);
  }

  if (flag && gtIdx == 0) {
    ctran::device::KernelWaitGpeTerminate(flag);
  }
}

#define DECL_ALLGATHER_KERN(T)                                \
  template __global__ void ncclKernelAllGatherCtranDirect<T>( \
      int* flag,                                              \
      CtranAlgoDeviceState* devState,                         \
      CtranKernelAllGatherArgs args)

DECL_ALLGATHER_KERN(int8_t);
DECL_ALLGATHER_KERN(uint8_t);
DECL_ALLGATHER_KERN(int32_t);
DECL_ALLGATHER_KERN(uint32_t);
DECL_ALLGATHER_KERN(int64_t);
DECL_ALLGATHER_KERN(uint64_t);
DECL_ALLGATHER_KERN(half);
DECL_ALLGATHER_KERN(float);
DECL_ALLGATHER_KERN(double);
#if defined(__CUDA_BF16_TYPES_EXIST__)
DECL_ALLGATHER_KERN(__nv_bfloat16);
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__) && defined(NCCL_ENABLE_FP8)
DECL_ALLGATHER_KERN(__nv_fp8_e4m3);
DECL_ALLGATHER_KERN(__nv_fp8_e5m2);
#endif
