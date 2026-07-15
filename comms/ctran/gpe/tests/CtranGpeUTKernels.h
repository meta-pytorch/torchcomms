// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include "comms/ctran/algos/AllGather/Types.h"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/common/GpeRing.h"
#include "comms/ctran/gpe/CtranGpeDev.h"

constexpr int numKElems = 10;

__global__ void CtranGpeTestKernel(
    ctran::gpe::KernelFlagDev* flag,
    CtranAlgoDeviceState* devState_d,
    ctran::allgather::KernelArgs args);

struct CtranKernelCustomArgs {
  const int scaleFactor;
  int numElems;
  int* data;
};

__global__ void CtranGpeTestCustomArgsKernel(
    ctran::gpe::KernelFlagDev* flag,
    CtranAlgoDeviceState* devState_d,
    CtranKernelCustomArgs args);

__global__ void CtranGpeTestStartAndExitKernel(ctran::gpe::KernelFlagDev* flag);

__global__ void CtranGpeTestTerminateKernel(ctran::gpe::KernelFlagDev* flag);

__global__ void CtranGpeTestKElemsKernel(
    ctran::gpe::KernelFlagDev* flag,
    CtranAlgoDeviceState* devState_d,
    ctran::allgather::KernelArgs args);

__global__ void CtranGpeTestOneFlagKernel(
    ctran::gpe::KernelFlagDev* flag,
    CtranAlgoDeviceState* devState_d,
    ctran::allgather::KernelArgs args);

__global__ void CtranGpeTestPerBlockFlagKernel(
    ctran::gpe::KernelFlagDev* flag,
    CtranAlgoDeviceState* devState_d,
    ctran::allgather::KernelArgs args);

struct CtranKernelFtArgs {
  int* terminate;
};

__global__ void CtranGpeTestFtDisabledOobTerminateKernel(
    ctran::gpe::KernelFlagDev* flag,
    CtranAlgoDeviceState* devState_d,
    CtranKernelFtArgs args);

__global__ void CtranGpeTestFtEnabledOobTerminateKernel(
    ctran::gpe::KernelFlagDev* flag,
    CtranAlgoDeviceState* devState_d,
    CtranKernelFtArgs args);

__global__ void CtranGpeTestFtBaseKernel(
    ctran::gpe::KernelFlagDev* flag,
    CtranAlgoDeviceState* devState_d,
    CtranKernelFtArgs args);

__global__ void CtranGpeTestFtShmAbortKernel(
    ctran::gpe::KernelFlagDev* flag,
    CtranAlgoDeviceState* devState_d,
    CtranKernelFtArgs args);
