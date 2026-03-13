// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/ctran/gpe/benchmarks/NoOpKernel.h"

__global__ void NoOpKernel(
    int* /*flag*/,
    CtranAlgoDeviceState* /*devState*/,
    ctran::allgather::KernelArgs /*args*/) {}
