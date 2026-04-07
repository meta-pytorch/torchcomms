// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/profiler/IProfilerReporter.h"

namespace ctran {

// Default reporter that logs algo profiling data to a scuba table.
class DefaultAlgoProfilerReporter : public IProfilerReporter {
 public:
  ~DefaultAlgoProfilerReporter() override = default;
  void report(const AlgoProfilerReport& report) override;
};

} // namespace ctran
