// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/ctran/gpe/benchmarks/NoOpKernel.h"

__global__ void NoOpKernel(
    ctran::gpe::KernelFlagDev* /*flag*/,
    CtranAlgoDeviceState* /*devState*/,
    ctran::allgather::KernelArgs /*args*/) {}
