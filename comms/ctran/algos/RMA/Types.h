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

struct CtranKernelSignalArgs {
  uint64_t* signalAddr;
  // Per-peer signal counter in mapped pinned host memory. The kernel
  // atomically increments this counter and uses the result as the signal
  // value to store at signalAddr.
  uint64_t* signalCounter;
  // Whether the counter increment needs system-wide visibility (true for
  // IB path where the GPE host thread reads the counter, false for NVL
  // where only same-device kernels read it).
  bool signalCounterSystemScope;
};

using CtranKernelPutSignalArgs = CtranKernelSignalArgs;

struct CtranKernelWaitSignalArgs {
  uint64_t* signalAddr;
  // Per-peer signal counter in mapped pinned host memory. The kernel
  // atomically increments this counter and spins on signalAddr until
  // the value reaches the counter result.
  uint64_t* signalCounter;
};
