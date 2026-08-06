// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <string_view>

namespace meta::comms::logger {

struct CommsLogMetadata {
  std::chrono::system_clock::time_point timestamp;
  uint64_t threadId;
  std::string_view filename;
  unsigned int lineNumber;
  std::string_view hostname;
  int processId;
  int threadContext;
  std::string_view threadName;
  std::string_view prefix;
};

std::string formatCommsLogMessage(
    std::string_view levelName,
    std::string_view message,
    const CommsLogMetadata& metadata);

} // namespace meta::comms::logger
