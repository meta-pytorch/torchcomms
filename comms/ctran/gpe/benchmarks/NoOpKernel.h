// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/ctran/algos/AllGather/Types.h"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/gpe/CtranGpeDev.h"

__global__ void NoOpKernel(
    int* flag,
    CtranAlgoDeviceState* devState,
    ctran::allgather::KernelArgs args);
