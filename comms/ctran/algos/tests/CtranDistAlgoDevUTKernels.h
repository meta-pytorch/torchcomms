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
