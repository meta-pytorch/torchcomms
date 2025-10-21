// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef CTRAN_PERF_H_
#define CTRAN_PERF_H_

// Performance knobs for low-latency Ctran collectives.
struct PerfConfig {
  // If true, we can skip valid VC check and lock on VcStateMap. This can only
  // be enabled when NCCL_CTRAN_ENABLE_PRECONNECT is true.
  bool skipVcConnectionCheck = false;
};

struct DefaultPerfCollConfig {
  static constexpr bool skipVcConnectionCheck = false;
};
struct LowLatencyCollConfig {
  static constexpr bool skipVcConnectionCheck = true;
};

#endif
