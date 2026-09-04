// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <functional>
#include <string>

#include <fmt/format.h>
#include <folly/Range.h>
#include <folly/logging/LogFormatter.h>
#include <folly/logging/LogLevel.h>

#include "comms/utils/logger/CommsLogging.h"

namespace meta::comms::logger {

folly::LogLevel loggerLevelToFollyLogLevel(LogLevel level);

folly::StringPiece getCategoryNthParent(folly::StringPiece category, int n);

fmt::memory_buffer getLogPrefix(LogLevel level);

class NcclLogFormatter : public folly::LogFormatter {
 public:
  NcclLogFormatter(
      std::string prefix,
      std::function<int(void)> threadContextFn);

  std::string formatMessage(
      const folly::LogMessage& message,
      const folly::LogCategory* handlerCategory) override;

 private:
  std::string prefix_;
  std::function<int(void)> threadContextFn_;
};

} // namespace meta::comms::logger
