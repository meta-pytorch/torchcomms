// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/ctran/profiler/IGpeProfilerReporter.h"

namespace ctran {

// Default reporter that logs per-tracepoint GPE thread profiler rows to a
// scuba table.
class DefaultGpeProfilerReporter : public IGpeProfilerReporter {
 public:
  ~DefaultGpeProfilerReporter() override = default;
  void report(const GpeProfilerReport& report) override;
};

} // namespace ctran
