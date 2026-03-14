// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>
#include <string>

#include "comms/ctran/profiler/AlgoProfilerReport.h"

namespace ctran {

// Abstract interface for reporting algo profiling data to a scuba backend.
// Implementations can target different tables (e.g., nccl_profiler_algo for
// NCCLX, mccl_operation_trace for MCCL).
class IAlgoProfilerReporter {
 public:
  virtual ~IAlgoProfilerReporter() = default;
  virtual void report(const AlgoProfilerReport& report) = 0;
};

} // namespace ctran
