// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/profiler/IAlgoProfilerReporter.h"

namespace ctran {

// Reporter that logs algo profiling data to the nccl_profiler_algo scuba table.
// This is the default reporter used by NCCLX.
class NcclxAlgoProfilerReporter : public IAlgoProfilerReporter {
 public:
  ~NcclxAlgoProfilerReporter() override = default;
  void report(const AlgoProfilerReport& report) override;
};

} // namespace ctran
