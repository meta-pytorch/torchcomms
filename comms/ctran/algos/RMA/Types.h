// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once
#include "comms/utils/commSpecs.h"

namespace ctran::rma {

struct KernelPutNotifyArgs {
  bool isDirect;
  int peerLocalRank;
};

struct KernelWaitNotifyArgs {
  bool isDirect;
  int peerLocalRank;
};

struct KernelGetArgs {
  bool isDirect;
  int peerLocalRank;
};

} // namespace ctran::rma

struct CtranKernelPutSignalArgs {
  uint64_t* signalAddr;
  uint64_t signalVal;
};

struct CtranKernelWaitSignalArgs {
  uint64_t* signalAddr;
  uint64_t cmpVal;
};

struct CtranKernelSignalArgs {
  uint64_t* signalAddr;
  uint64_t signalVal;
};
