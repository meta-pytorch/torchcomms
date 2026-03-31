// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/profiler/AlgoProfilerReport.h"
#include "comms/ctran/profiler/IAlgoProfilerReporter.h"

namespace ctran {

// Mock reporter that captures the report for verification.
// Extracted from ProfilerTest.cc for reuse across test files.
class MockAlgoProfilerReporter : public IAlgoProfilerReporter {
 public:
  void report(const AlgoProfilerReport& report) override {
    reportCalled_ = true;
    lastReport_ = report;
    // Deep-copy the AlgoContext since the pointer may become invalid
    if (report.algoContext) {
      capturedAlgoContext_ = *report.algoContext;
    }
    reportCount_++;
  }

  bool reportCalled_{false};
  AlgoProfilerReport lastReport_;
  AlgoContext capturedAlgoContext_;
  int reportCount_{0};
};

} // namespace ctran
