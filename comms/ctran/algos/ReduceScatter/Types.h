// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/utils/commSpecs.h"

// Forward declaration
struct KernelElem;

namespace ctran::reducescatter {

struct KernelArgs {
  const void* sendbuff;
  void* recvbuff;
  commDataType_t datatype;
  size_t recvcount;
  bool stageCopy;
  KernelElem* intraReduce;
  // Reuse single interReduce for number of interNode reduce steps
  int nStepsInterReduce;
  KernelElem* interReduce;
};

} // namespace ctran::reducescatter
