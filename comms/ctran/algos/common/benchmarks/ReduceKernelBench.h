// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/algos/CtranAlgoDev.h"

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
