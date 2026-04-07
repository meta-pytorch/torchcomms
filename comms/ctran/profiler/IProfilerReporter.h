// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/profiler/AlgoProfilerReport.h"

namespace ctran {

// Abstract interface for reporting algo profiling data to a scuba backend.
// Implementations can target different scuba tables.
class IProfilerReporter {
 public:
  virtual ~IProfilerReporter() = default;
  virtual void report(const AlgoProfilerReport& report) = 0;
};

} // namespace ctran
