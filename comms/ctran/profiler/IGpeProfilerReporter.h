// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/ctran/profiler/GpeProfilerReport.h"

namespace ctran {

// Sink for one report row. Production impl writes to Scuba; tests use
// gmock-based MockGpeProfilerReporter to capture and assert.
class IGpeProfilerReporter {
 public:
  virtual ~IGpeProfilerReporter() = default;
  virtual void report(const GpeProfilerReport& report) = 0;
};

} // namespace ctran
