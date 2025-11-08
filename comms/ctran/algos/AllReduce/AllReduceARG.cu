// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <stdio.h>
#include <cstddef>
#include "comms/ctran/algos/AllReduce/AllReduceARGCommonDev.h"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/DevAlgoImpl.cuh"
#include "comms/ctran/algos/DevCommon.cuh"
#include "comms/ctran/algos/barrier.cuh"
#include "comms/ctran/algos/localReduce.cuh"
#include "comms/ctran/gpe/CtranGpeDev.h"

using namespace ctran::allreduce::arg;

template <typename T>
__device__ void prepareBcastArg(
    CtranKernelAllReduceArgs& args,
    AllReduceARGContext& context,
    CtranAlgoDevBcastArg& bcastArg) {
  const auto localRank = statex->localRank();
  const auto nLocalRanks = statex->nLocalRanks();
  const auto myRank = statex->rank();
  bcastArg.count = context.stepCount;
  T* srcTmpbuff =
      reinterpret_cast<T*>(args.recvbuff) + getUserbuffOffset(context, myRank);
  bcastArg.src = srcTmpbuff;
  // need barrier to ensure all peers have finished update to the local dst
  // before kernel finishes
  bcastArg.barrier = true;
  // No host side access before kernel terminates; thus skip flush
  bcastArg.flushMem = false;
  bcastArg.nvectors = nLocalRanks;

  for (int i = 0; i < nLocalRanks; i++) {
    if (i == localRank) {
      bcastArg.dsts[i] = srcTmpbuff;
    } else {
      bcastArg.dsts[i] =
          reinterpret_cast<T*>(args.intraNodeRemoteRecvBuffs[i]) +
          getUserbuffOffset(context, myRank);
    }
  }
}

template <typename T, typename RedT, commRedOp_t RedOp>
__device__ __forceinline__ void reduceInLocal(
    CtranAlgoDeviceState* devState,
    CtranKernelAllReduceArgs& args,
    AllReduceARGContext& context) {
  bool revoked = false;
  const auto nRanks = statex->nRanks();
  const auto myRank = statex->rank();
  const auto localRank = statex->localRank();
  const auto nLocalRanks = statex->nLocalRanks();

  KernelElem* elem = args.kernelElems[kLocalReduce];
  elemWaitPostOrRevokeByGroup(elem, blockIdx.x, &revoked);
  const auto count = context.stepCount;
  const T* localSendbuff = reinterpret_cast<const T*>(args.tmpbuff);
  T* localRecvbuff =
      reinterpret_cast<T*>(args.recvbuff) + getUserbuffOffset(context, myRank);

  localReduceForDequantAllToAll<T, RedT, RedOp>(
      localSendbuff, localRecvbuff, count, myRank, nRanks);

  /* sync with the remaining local GPUs to tell them that we are done */
  __threadfence_system();
  barrier(localRank, nLocalRanks);

  elemCompleteByGroup(elem, blockIdx.x);
}

template <typename T>
__device__ __forceinline__ void intraNodeAllGather(
    CtranAlgoDeviceState* devState,
    CtranKernelAllReduceArgs& args,
    AllReduceARGContext& context) {
  bool allGatherRevoke;
  KernelElem* elem = args.kernelElems[kIntraAllGather];
  elemWaitPostOrRevokeByGroup(elem, blockIdx.x, &allGatherRevoke);

  // prepare bcast arg
  CtranAlgoDevBcastArg bcastArg;
  prepareBcastArg<T>(args, context, bcastArg);

  // Choose kMultiPutBcast since GPE thread ensures all ranks have joined and
  // the internal peer shift can avoid incast congestion in multiPut
  ctranKernBcast<T, kMultiPutBcast>(bcastArg, elem, blockIdx.x, gridDim.x, 0);

  elemCompleteByGroup(elem, blockIdx.x);
}

template <typename T>
__device__ __forceinline__ void alltoall(
    CtranAlgoDeviceState* devState,
    CtranKernelAllReduceArgs& args,
    AllReduceARGContext& context) {
  KernelElem* alltoall = args.kernelElems[kIntraAllToAll];
  const auto localRank = statex->localRank();
  const auto nLocalRanks = statex->nLocalRanks();
  const auto myRank = statex->rank();

  bool revoked = false;
  elemWaitPostOrRevokeByGroup(alltoall, blockIdx.x, &revoked);

  // copy self to tmpbuff
  ctranKernCopy<T>(
      reinterpret_cast<const T*>(args.sendbuff) +
          getUserbuffOffset(context, myRank),
      reinterpret_cast<T*>(args.tmpbuff) + getTmpbuffOffset(context, myRank),
      context.stepCount,
      blockIdx.x,
      gridDim.x);

  // Ensure each rank sends to different peer at a time to avoid alltoone P2P
  // write congestion. For example, with localRanks = 4, the following
  // schedule is used:
  // - Round0:
  // rank0: s(1)r(3); rank1: s(2)r(0); rank2: s(3)r(1); rank3: s(0)r(2)
  // - Round1:
  // rank0: s(2)r(2); rank1: s(3)r(3); rank2: s(0)r(0); rank3: s(1)r(1)
  // - Round2:
  // rank0: s(3)r(1); rank1: s(0)r(2); rank2: s(1)r(3); rank3: s(2)r(0)
  for (auto lr = 1; lr < nLocalRanks; lr++) {
    auto peer = (localRank + lr) % nLocalRanks;
    auto peerGlobalRank = statex->localRankToRank(peer);
    ctranKernCopy<T>(
        reinterpret_cast<const T*>(args.sendbuff) +
            getUserbuffOffset(context, peerGlobalRank),
        reinterpret_cast<T*>(args.intraNodeRemoteTmpRecvBuffs[peer]) +
            getTmpbuffOffset(context, myRank),
        context.stepCount,
        blockIdx.x,
        gridDim.x);
  }
  __threadfence_system();
  barrier(localRank, nLocalRanks);

  elemCompleteByGroup(alltoall, blockIdx.x);
}

template <typename T, typename RedT, commRedOp_t RedOp>
__global__ void ncclKernelAllReduceARG(
    int* flag,
    CtranAlgoDeviceState* devState,
    CtranKernelAllReduceArgs args) {
  const auto tId = threadIdx.x;
  const auto bId = blockIdx.x;
  if (flag && tId == 0) {
    ctran::device::devLoadAbortFlags(&flag[bId], devState);
    ctran::device::KernelStartGpe(&flag[bId]);
  }
  devStateLoadToShm(devState);

  AllReduceARGContext context = {
      .localRank = statex->localRank(),
      .nLocalRanks = statex->nLocalRanks(),
      .rank = statex->rank(),
      .nRanks = statex->nRanks(),
      .tmpbuffSize = args.tmpbuffSize,
      .typeSize = sizeof(T),
      .count = args.count,
  };
  prepareContext(context);

  if (bId == 0 && tId == 0) {
    CTRAN_DEV_TRACE(
        "Prepare for AllReduceARGContext with nSteps=%d, stepCount=%ld, totalStepCount=%ld, tmpbuffSize=%ld, count=%ld, nRanks=%d\n",
        context.nSteps,
        context.stepCount,
        context.totalStepCount,
        context.tmpbuffSize,
        context.count,
        context.nRanks);
  }

  for (auto i = 0; i < context.nSteps; i++) {
    // step-1 intra node alltoall
    alltoall<T>(devState, args, context);

    // step-2: inter node alltoall

    // step-3 local reduction with dequantization
    reduceInLocal<T, RedT, RedOp>(devState, args, context);

    // step-4 inter node all gather

    // step 5: intra node all gather
    intraNodeAllGather<T>(devState, args, context);

    updateContext(context);
  }

  if (context.remCount) {
    updateContextForRemainder(context);

    if (bId == 0 && tId == 0) {
      CTRAN_DEV_TRACE(
          "remCount for AllReduceARGContext with stepCount=%ld, totalStepCount=%ld, tmpbuffSize=%ld, count=%ld, nRanks=%d\n",
          context.stepCount,
          context.totalStepCount,
          context.tmpbuffSize,
          context.count,
          context.nRanks);
    }
    // step-6 remainder intra node all scatter
    alltoall<T>(devState, args, context);
    // step-7: inter node all scatter

    // step-8: remainder local reduction with dequantization
    reduceInLocal<T, RedT, RedOp>(devState, args, context);

    // step-9: remainder inter node all gather
  }

  if (flag && tId == 0) {
    ctran::device::KernelWaitGpeTerminate(&flag[bId]);
  }
}

#define DECL_ALL_REDUCE_ARG(T, RedT, RedOp)                        \
  template __global__ void ncclKernelAllReduceARG<T, RedT, RedOp>( \
      int* flag,                                                   \
      CtranAlgoDeviceState* devState,                              \
      CtranKernelAllReduceArgs args);

#define DECL_ALL_REDUCE_ARG_ALL(T, RedT)               \
  DECL_ALL_REDUCE_ARG(T, RedT, commRedOp_t::commSum);  \
  DECL_ALL_REDUCE_ARG(T, RedT, commRedOp_t::commProd); \
  DECL_ALL_REDUCE_ARG(T, RedT, commRedOp_t::commAvg);  \
  DECL_ALL_REDUCE_ARG(T, RedT, commRedOp_t::commMax);  \
  DECL_ALL_REDUCE_ARG(T, RedT, commRedOp_t::commMin);

#define DECL_ALL_REDUCE_ARG_SELF(T) DECL_ALL_REDUCE_ARG_ALL(T, T)

DECL_ALL_REDUCE_ARG_SELF(int8_t);
DECL_ALL_REDUCE_ARG_SELF(uint8_t);
DECL_ALL_REDUCE_ARG_SELF(int32_t);
DECL_ALL_REDUCE_ARG_SELF(uint32_t);
DECL_ALL_REDUCE_ARG_SELF(int64_t);
DECL_ALL_REDUCE_ARG_SELF(uint64_t);
DECL_ALL_REDUCE_ARG_SELF(half);
DECL_ALL_REDUCE_ARG_SELF(float);
DECL_ALL_REDUCE_ARG_SELF(double);
#if defined(__CUDA_BF16_TYPES_EXIST__)
DECL_ALL_REDUCE_ARG_SELF(__nv_bfloat16);
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__) && defined(NCCL_ENABLE_FP8)
DECL_ALL_REDUCE_ARG_SELF(__nv_fp8_e4m3);
DECL_ALL_REDUCE_ARG_SELF(__nv_fp8_e5m2);
#endif

#if defined(__CUDA_BF16_TYPES_EXIST__)
#define DECL_ALL_REDUCE_ARG_KERN_ALL_TYPES(RedT) \
  DECL_ALL_REDUCE_ARG_ALL(__nv_bfloat16, RedT);

// kernel where dequantization type is FP32
DECL_ALL_REDUCE_ARG_KERN_ALL_TYPES(float);
#endif
