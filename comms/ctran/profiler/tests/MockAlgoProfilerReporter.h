// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <gmock/gmock.h>
#include "comms/ctran/profiler/AlgoProfilerReport.h"
#include "comms/ctran/profiler/IProfilerReporter.h"

namespace ctran {

// Mock reporter using GMock for call verification.
// Extracted from ProfilerTest.cc for reuse across test files.
class MockAlgoProfilerReporter : public IProfilerReporter {
 public:
  MOCK_METHOD(void, report, (const AlgoProfilerReport& report), (override));
};

} // namespace ctran
