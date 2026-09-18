// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/utils/commSpecs.h"

//------------------------------------------------------------------------------
// Benchmark Kernel Arguments
//------------------------------------------------------------------------------

struct ReduceKernelBenchArg {
  size_t count;
  int nsrcs;
  int ndsts;
  const void* srcs[CTRAN_MAX_NVL_PEERS];
  void* dsts[CTRAN_MAX_NVL_PEERS];
};

//------------------------------------------------------------------------------
// Kernel Accessor
//------------------------------------------------------------------------------

// nvcc gives the host-side launch stub of an explicitly instantiated
// __global__ template internal linkage, so the stub cannot be named from
// another translation unit. Hand the kernel address out of the .cu instead.
// Instantiated in ReduceKernelBench.cu for every <T, redOp> benchmarked.
template <typename T, commRedOp_t redOp>
void* getLocalReduceKernelFn();
