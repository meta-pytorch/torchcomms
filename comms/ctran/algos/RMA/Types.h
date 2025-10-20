// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once
#include "comms/utils/commSpecs.h"

struct CtranKernelPutSignalArgs {
  uint64_t* signalAddr;
  uint64_t signalVal;
};

struct CtranKernelWaitSignalArgs {
  uint64_t* signalAddr;
  uint64_t cmpVal;
  commCmpOp_t cmpOp;
};

struct CtranKernelSignalArgs {
  uint64_t* signalAddr;
  uint64_t signalVal;
};
