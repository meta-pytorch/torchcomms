// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/logger/LoggingFormat.h"

#include <string_view>
#include <utility>

#include <folly/logging/LogCategory.h>
#include <folly/logging/LogMessage.h>
#include <folly/logging/LogName.h>

#include "comms/utils/logger/CommsLogFormatter.h"

namespace meta::comms::logger {

folly::LogLevel loggerLevelToFollyLogLevel(LogLevel level) {
  switch (level) {
    case LogLevel::NONE:
    case LogLevel::VERSION:
      return folly::LogLevel::FATAL;
    case LogLevel::ERROR:
      return folly::LogLevel::ERR;
    case LogLevel::WARN:
      return folly::LogLevel::WARN;
    case LogLevel::INFO:
      return folly::LogLevel::INFO;
    case LogLevel::ABORT:
    case LogLevel::TRACE:
      return folly::LogLevel::DBG;
    default:
      return folly::LogLevel::UNINITIALIZED;
  }
}

std::string_view getGlogLevelName(folly::LogLevel level) {
  if (level < folly::LogLevel::INFO) {
    return "VERBOSE";
  } else if (level < folly::LogLevel::WARN) {
    return "INFO";
  } else if (level < folly::LogLevel::ERR) {
    return "WARN";
  } else if (level < folly::LogLevel::CRITICAL) {
    return "ERROR";
  } else if (level < folly::LogLevel::DFATAL) {
    return "CRITICAL";
  }
  return "FATAL";
}

folly::StringPiece getCategoryNthParent(folly::StringPiece category, int n) {
  for (auto i = 0; i < n; i++) {
    category = ::folly::LogName::getParent(category);
  }
  return category;
}

std::string NcclLogFormatter::formatMessage(
    const folly::LogMessage& message,
    const folly::LogCategory* /* handlerCategory */) {
  const auto processMetadata = detail::getLogProcessMetadata();
  if (message.getLevel() >= folly::LogLevel::ERR) {
    /*
     * Errors are recorded to Scuba at their call sites. Clear any stale native
     * stack here so a bare legacy XLOG(ERR) falls back to the per-frame chain.
     */
    detail::setLastErrorFromLegacyLog(message.getMessage());
  }

  const int cudaDev = threadContextFn_();
  const auto basename = message.getFileBaseName();
  return formatCommsLogMessage(
      getGlogLevelName(message.getLevel()),
      message.getMessage(),
      {.timestamp = message.getTimestamp(),
       .threadId = message.getThreadID(),
       .filename = std::string_view{basename.data(), basename.size()},
       .lineNumber = message.getLineNumber(),
       .hostname = processMetadata.hostname,
       .processId = processMetadata.processId,
       .threadContext = cudaDev,
       .threadName = getLogThreadName(),
       .prefix = prefix_});
}

NcclLogFormatter::NcclLogFormatter(
    std::string prefix,
    std::function<int(void)> threadContextFn)
    : prefix_(std::move(prefix)),
      threadContextFn_(std::move(threadContextFn)) {}

} // namespace meta::comms::logger
