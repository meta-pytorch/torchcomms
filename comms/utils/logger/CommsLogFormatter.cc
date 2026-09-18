// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/logger/CommsLogFormatter.h"

#include <algorithm>
#include <type_traits>

#include <fmt/chrono.h>
#include <fmt/format.h>

namespace meta::comms::logger {
namespace {

/*
 * Logging may continue after non-trivial TLS destructors have run. Constant
 * initialization avoids dynamic initialization, while trivial destruction
 * avoids registering a TLS destructor. Store the length to avoid scanning the
 * buffer for every log line.
 */
struct LogThreadName {
  char data[kMaxLogThreadNameLength + 1];
  size_t size;
};
static_assert(std::is_trivially_destructible_v<LogThreadName>);

thread_local LogThreadName logThreadName{"main", sizeof("main") - 1};

} // namespace

void setLogThreadName(std::string_view name) {
  const auto size = name.copy(logThreadName.data, kMaxLogThreadNameLength);
  logThreadName.data[size] = '\0';
  logThreadName.size = size;
}

std::string_view getLogThreadName() {
  return std::string_view{logThreadName.data, logThreadName.size};
}

std::string formatCommsLogMessage(
    std::string_view levelName,
    std::string_view message,
    const CommsLogMetadata& metadata) {
  const auto timeSinceEpoch = metadata.timestamp.time_since_epoch();
  const auto epochSeconds =
      std::chrono::duration_cast<std::chrono::seconds>(timeSinceEpoch);
  const auto usecs =
      std::chrono::duration_cast<std::chrono::microseconds>(timeSinceEpoch) -
      epochSeconds;
  const auto levelInitial = levelName.empty() ? '?' : levelName.front();

  const auto header = fmt::format(
      "{}{:%m%d %H:%M:%S}.{:06d} {:5d} {}:{}] {}:{}:{} [{}][{}] {} {} ",
      levelInitial,
      metadata.timestamp,
      usecs.count(),
      metadata.threadId,
      metadata.filename,
      metadata.lineNumber,
      metadata.hostname,
      metadata.processId,
      metadata.threadId,
      metadata.threadContext,
      metadata.threadName,
      metadata.prefix,
      levelName);

  std::string output;
  if (message.find('\n') == std::string_view::npos) {
    output.reserve(header.size() + message.size() + 1);
    output.append(header);
    output.append(message);
    output.push_back('\n');
    return output;
  }

  output.reserve(
      ((header.size() + 1) * std::count(message.begin(), message.end(), '\n')) +
      message.size());
  std::size_t begin = 0;
  while (true) {
    auto end = message.find('\n', begin);
    if (end == std::string_view::npos) {
      end = message.size();
    }
    output.append(header);
    output.append(message.substr(begin, end - begin));
    output.push_back('\n');
    if (end == message.size()) {
      break;
    }
    begin = end + 1;
  }
  return output;
}

} // namespace meta::comms::logger
