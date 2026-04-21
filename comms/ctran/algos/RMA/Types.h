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
  // Device-side replay counter for CUDA graph mode. When non-null, the
  // kernel atomically increments this counter and uses the result as the
  // signal value (ignoring signalVal). This gives each graph replay a
  // unique monotonic value without modifying frozen graph args.
  uint64_t* replayCounter;
  // Whether the counter increment needs system-wide visibility (true for
  // IB path where the GPE host thread reads the counter, false for NVL
  // where only same-device kernels read it).
  bool replayCounterSystemScope;
};

struct CtranKernelWaitSignalArgs {
  uint64_t* signalAddr;
  uint64_t cmpVal;
  // Device-side replay counter for CUDA graph mode. When non-null, the
  // kernel reads this counter (already incremented by PutSignal on the
  // same stream) and uses it as the compare value (ignoring cmpVal).
  uint64_t* replayCounter;
};

struct CtranKernelSignalArgs {
  uint64_t* signalAddr;
  uint64_t signalVal;
  uint64_t* replayCounter;
  bool replayCounterSystemScope;
};
