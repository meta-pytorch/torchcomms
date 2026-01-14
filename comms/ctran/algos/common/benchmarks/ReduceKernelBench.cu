// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/algos/DevAlgoImpl.cuh"
#include "comms/ctran/algos/common/benchmarks/ReduceKernelBench.h"
#include "comms/ctran/algos/localReduce.cuh"

//------------------------------------------------------------------------------
// Benchmark Kernel
//------------------------------------------------------------------------------

/**
 * Benchmark kernel for reduce operation.
 * This kernel measures the performance of the reduce operation
 * by iterating over multiple reduce calls.
 *
 * @param arg: reduce arguments (count, srcs, dsts)
 * @param iters: number of iterations to perform within a single kernel launch
 */
template <typename T, commRedOp_t redOp>
__global__ void __launch_bounds__(1024, 1)
    LocalReduceKernel(ReduceKernelBenchArg arg, int iters) {
  const T* srcs[CTRAN_MAX_NVL_PEERS];
  T* dsts[CTRAN_MAX_NVL_PEERS];

  for (int i = 0; i < arg.nsrcs; i++) {
    srcs[i] = reinterpret_cast<const T*>(arg.srcs[i]);
  }
  for (int i = 0; i < arg.ndsts; i++) {
    dsts[i] = reinterpret_cast<T*>(arg.dsts[i]);
  }

  for (int iter = 0; iter < iters; iter++) {
    localReduce<T, redOp>(arg.nsrcs, srcs, arg.ndsts, dsts, arg.count);
  }
}

//------------------------------------------------------------------------------
// Template Instantiations
//------------------------------------------------------------------------------

template __global__ void LocalReduceKernel<int, commSum>(
    ReduceKernelBenchArg arg,
    int iters);
