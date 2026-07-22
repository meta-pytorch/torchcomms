// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/gpe/CtranGpeDev.h"

enum ElemTestType {
  kTestElemComplete,
  kTestElemFree,
  kTestElemRepost,
};

enum BcastTestAlgo {
  kTestDefaultBcast,
  kTestMultiPutBcast,
};

cudaError_t testKernMultiPutNotifyWrapper(
    ElemTestType testType,
    dim3 grid,
    dim3 blocks,
    cudaStream_t stream,
    KernelElem* elemList,
    CtranAlgoDeviceState* devState);

cudaError_t testKernMultiWaitNotifyWrapper(
    ElemTestType testType,
    dim3 grid,
    dim3 blocks,
    cudaStream_t stream,
    KernelElem* elemList,
    CtranAlgoDeviceState* devState);

cudaError_t testKernMultiReduceWrapper(
    ElemTestType testType,
    dim3 grid,
    dim3 blocks,
    cudaStream_t stream,
    KernelElem* elemList,
    int nSteps,
    CtranAlgoDeviceState* devState);

cudaError_t testKernBcastWrapper(
    BcastTestAlgo testAlgo,
    ElemTestType testType,
    dim3 grid,
    dim3 blocks,
    cudaStream_t stream,
    KernelElem* elemList,
    int nSteps,
    int nBcastBlocks,
    CtranAlgoDeviceState* devState);

// Launch the barrier correctness test kernel. Each of `nIters` iterations picks
// a rotating "laggard" rank (`iter % nLocalRanks`) that busy-waits before
// publishing `iter` into its NVL-visible `selfSlot`, so it reliably enters the
// barrier last. Every rank then barriers, reads every peer's slot via
// `peerSlots` and increments `errCount` on any token that is not exactly
// `iter`, then barriers again. A correct barrier makes all ranks wait for the
// laggard (fresh tokens, `errCount == 0`); a barrier that drops synchronization
// edges (e.g. for non-power-of-two ranks) lets a rank proceed early and read
// the laggard's stale token, producing a non-zero `errCount`.
cudaError_t testKernNvlBarrierLoopWrapper(
    dim3 grid,
    dim3 blocks,
    cudaStream_t stream,
    int rank,
    int nLocalRanks,
    int nIters,
    int* selfSlot,
    int** peerSlots,
    int* errCount,
    CtranAlgoDeviceState* devState);

cudaError_t testKernDequantizedAllToAllReduceWrapper(
    const void* sendBuff,
    void* recvBuff,
    dim3 grid,
    dim3 blocks,
    cudaStream_t stream);

template <typename T>
void* getReduceKernelFn(commRedOp_t op);

template <typename T>
void* getMultiReduceKernelFn(ElemTestType testType, commRedOp_t op);

template <typename T, typename RedT>
void* getDequantizedAllToAllReduceKernelFn(commRedOp_t op);

template <typename T>
void* getKernCopyFn();

template <typename T>
void* getNaiveKernCopyFn();
