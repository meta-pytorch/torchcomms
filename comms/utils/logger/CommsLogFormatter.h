// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace meta::comms::logger {

/* Maximum number of bytes stored for a thread name. */
inline constexpr size_t kMaxLogThreadNameLength = 63;

/*
 * Sets the current thread's name for comms log records. Longer names are
 * truncated to kMaxLogThreadNameLength.
 */
void setLogThreadName(std::string_view name);

/*
 * Returns the current thread's log name, which defaults to "main". The returned
 * view remains valid until the name changes or the thread exits.
 */
std::string_view getLogThreadName();

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

// The first byte is the legacy severity initial used by stderr routing.
std::string formatCommsLogMessage(
    std::string_view levelName,
    std::string_view message,
    const CommsLogMetadata& metadata);

} // namespace meta::comms::logger
