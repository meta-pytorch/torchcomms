// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/logger/CommsLogFormatter.h"

#include <algorithm>

#include <fmt/chrono.h>
#include <fmt/format.h>

namespace meta::comms::logger {

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
