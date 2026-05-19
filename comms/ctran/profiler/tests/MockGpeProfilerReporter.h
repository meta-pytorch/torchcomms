// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <gmock/gmock.h>

#include "comms/ctran/profiler/GpeProfilerReport.h"
#include "comms/ctran/profiler/IGpeProfilerReporter.h"

namespace ctran {

// GMock reporter for verifying GpeProfiler report() calls in tests.
class MockGpeProfilerReporter : public IGpeProfilerReporter {
 public:
  MOCK_METHOD(void, report, (const GpeProfilerReport& report), (override));
};

} // namespace ctran
