// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>
#include <string>

#include "comms/ctran/utils/ReporterType.h"

struct CommLogData;

namespace ctran {

struct DataContext {
  uint64_t totalBytes{0};
  std::string messageSizes{};
};

struct AlgoContext {
  std::string deviceName{};
  std::string algorithmName{};
  DataContext sendContext{};
  DataContext recvContext{};
  uint64_t peerRank{0};
};

// Data struct capturing all profiled algo metrics, decoupled from Profiler
// internals. Passed to IAlgoProfilerReporter::report().
struct AlgoProfilerReport {
  AlgoContext const* algoContext{nullptr};
  const CommLogData* logMetaData{nullptr};
  uint64_t opCount{0};
  uint64_t bufferRegistrationTimeUs{0};
  uint64_t controlSyncTimeUs{0};
  uint64_t dataTransferTimeUs{0};
  uint64_t collectiveDurationUs{0};
  uint64_t readyTs{0};
  uint64_t controlTs{0};
  uint64_t timeFromDataToCollEndUs{0};
};

} // namespace ctran
