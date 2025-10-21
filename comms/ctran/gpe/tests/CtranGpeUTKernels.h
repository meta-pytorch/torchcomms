// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/gpe/CtranGpeDev.h"

constexpr int numKElems = 10;

__global__ void CtranGpeTestKernel(
    int* flag,
    CtranAlgoDeviceState* devState_d,
    CtranKernelAllGatherArgs args);

struct CtranKernelCustomArgs {
  const int scaleFactor;
  int numElems;
  int* data;
};

__global__ void CtranGpeTestCustomArgsKernel(
    int* flag,
    CtranAlgoDeviceState* devState_d,
    CtranKernelCustomArgs args);

__global__ void CtranGpeTestStartAndExitKernel(int* flag);

__global__ void CtranGpeTestTerminateKernel(int* flag);

__global__ void CtranGpeTestKElemsKernel(
    int* flag,
    CtranAlgoDeviceState* devState_d,
    CtranKernelAllGatherArgs args);

__global__ void CtranGpeTestOneFlagKernel(
    int* flag,
    CtranAlgoDeviceState* devState_d,
    CtranKernelAllGatherArgs args);

__global__ void CtranGpeTestPerBlockFlagKernel(
    int* flag,
    CtranAlgoDeviceState* devState_d,
    CtranKernelAllGatherArgs args);

struct CtranKernelFtArgs {
  int* terminate;
};

__global__ void CtranGpeTestFtDisabledOobTerminateKernel(
    int* flag,
    CtranAlgoDeviceState* devState_d,
    CtranKernelFtArgs args);

__global__ void CtranGpeTestFtEnabledOobTerminateKernel(
    int* flag,
    CtranAlgoDeviceState* devState_d,
    CtranKernelFtArgs args);

__global__ void CtranGpeTestFtBaseKernel(
    int* flag,
    CtranAlgoDeviceState* devState_d,
    CtranKernelFtArgs args);

__global__ void CtranGpeTestFtShmAbortKernel(
    int* flag,
    CtranAlgoDeviceState* devState_d,
    CtranKernelFtArgs args);

__global__ void CtranGpeTestFtKernelSkipGpeStart(
    int* flag,
    CtranAlgoDeviceState* devState_d,
    CtranKernelFtArgs args);

__global__ void CtranGpeTestFtKernelSkipGpeTerminate(
    int* flag,
    CtranAlgoDeviceState* devState_d,
    CtranKernelFtArgs args);
