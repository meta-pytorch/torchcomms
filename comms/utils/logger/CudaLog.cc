// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/logger/CudaLog.h"

#include <array>
#include <cstdarg>
#include <cstdio>
#include <string>
#include <vector>

#include "comms/utils/logger/SpdlogLogger.h"

namespace meta::comms::logger {
namespace {

spdlog::level::level_enum toSpdlogLevel(CudaLogLevel level) {
  switch (level) {
    case CudaLogLevel::DBG:
      return spdlog::level::debug;
    case CudaLogLevel::INFO:
      return spdlog::level::info;
    case CudaLogLevel::WARN:
      return spdlog::level::warn;
    case CudaLogLevel::ERR:
      return spdlog::level::err;
  }
  return spdlog::level::off;
}

} // namespace

CommsSpdlogLogger* tryGetSpdlogLoggerForCuda() noexcept {
  try {
    return &getSpdlogLogger();
  } catch (...) {
    reportCommsLoggingFailureToStderr("ERROR");
    return nullptr;
  }
}

bool shouldLogFromCuda(
    const CommsSpdlogLogger& logger,
    CudaLogLevel level) noexcept {
  try {
    return logger.should_log(toSpdlogLevel(level));
  } catch (...) {
    reportCommsLoggingFailureToStderr("ERROR");
    return false;
  }
}

void logFromCuda(
    CommsSpdlogLogger& logger,
    CudaLogLevel level,
    const char* filename,
    int line,
    const char* function,
    const char* format,
    ...) noexcept {
  try {
    constexpr std::size_t kStackBufferSize = 512;
    std::array<char, kStackBufferSize> stackBuffer{};
    va_list args;
    va_start(args, format);
    const int required =
        std::vsnprintf(stackBuffer.data(), stackBuffer.size(), format, args);
    va_end(args);
    if (required < 0) {
      reportCommsLoggingFailureToStderr("ERROR");
      return;
    }

    std::string message;
    if (static_cast<std::size_t>(required) < stackBuffer.size()) {
      message.assign(stackBuffer.data(), static_cast<std::size_t>(required));
    } else {
      std::vector<char> buffer(static_cast<std::size_t>(required) + 1);
      va_start(args, format);
      const int written =
          std::vsnprintf(buffer.data(), buffer.size(), format, args);
      va_end(args);
      if (written != required) {
        reportCommsLoggingFailureToStderr("ERROR");
        return;
      }
      message.assign(buffer.data(), static_cast<std::size_t>(required));
    }

    logger.log(
        spdlog::source_loc{filename, line, function},
        toSpdlogLevel(level),
        message);
  } catch (...) {
    reportCommsLoggingFailureToStderr("ERROR");
  }
}

} // namespace meta::comms::logger
