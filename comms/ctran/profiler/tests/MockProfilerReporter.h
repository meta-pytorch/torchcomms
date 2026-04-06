// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <gmock/gmock.h>

#include "comms/ctran/profiler/AlgoProfilerReport.h"
#include "comms/ctran/profiler/IProfilerReporter.h"

namespace ctran {

// GMock reporter for verifying profiler report calls in tests.
class MockProfilerReporter : public IProfilerReporter {
 public:
  MOCK_METHOD(void, report, (const AlgoProfilerReport& report), (override));
};

} // namespace ctran
