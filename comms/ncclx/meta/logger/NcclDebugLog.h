// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <string_view>

#include "nccl_common.h"

#include "meta/NcclxLogger.h"

namespace ncclx::logging {

inline constexpr spdlog::level::level_enum ncclLogLevelToSpdlogLevel(
    ncclDebugLogLevel level) {
  switch (level) {
    case NCCL_LOG_ERROR:
      return spdlog::level::err;
    case NCCL_LOG_WARN:
      return spdlog::level::warn;
    case NCCL_LOG_TRACE:
      return spdlog::level::debug;
    case NCCL_LOG_NONE:
    case NCCL_LOG_VERSION:
    case NCCL_LOG_INFO:
    case NCCL_LOG_ABORT:
      return spdlog::level::info;
  }
  return spdlog::level::info;
}

inline void writeNcclLog(
    ncclDebugLogLevel level,
    const char* file,
    const char* func,
    int line,
    std::string_view message) {
  meta::comms::logger::getSpdlogLogger(kNcclxLoggerName)
      .log(
          spdlog::source_loc{file, line, func},
          ncclLogLevelToSpdlogLevel(level),
          message);
}

} // namespace ncclx::logging
