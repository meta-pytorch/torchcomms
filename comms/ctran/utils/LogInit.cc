// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/utils/LogInit.h"

#include <cuda_runtime.h>

#include <folly/synchronization/CallOnce.h>

#include "comms/ctran/utils/CtranLogger.h"
#include "comms/utils/cvars/nccl_cvars.h" // @manual=fbcode//comms/utils/cvars:ncclx-cvars
#include "comms/utils/logger/LogUtils.h"
#include "comms/utils/logger/LoggingFormat.h"

namespace ctran::logging {

namespace {
spdlog::level::level_enum loggerLevelToSpdlogLevel(
    meta::comms::logger::LogLevel level) {
  switch (level) {
    case meta::comms::logger::LogLevel::NONE:
    case meta::comms::logger::LogLevel::VERSION:
      // COMMS_LOG_FATAL bypasses this threshold; off suppresses only
      // non-fatal messages for these modes.
      return spdlog::level::off;
    case meta::comms::logger::LogLevel::ERROR:
      return spdlog::level::err;
    case meta::comms::logger::LogLevel::WARN:
      return spdlog::level::warn;
    case meta::comms::logger::LogLevel::INFO:
      return spdlog::level::info;
    case meta::comms::logger::LogLevel::ABORT:
    case meta::comms::logger::LogLevel::TRACE:
      return spdlog::level::debug;
  }
  return spdlog::level::off;
}

} // namespace

namespace {
folly::once_flag ctranLoggingInitOnceFlag;

void initCtranLoggingImpl() {
  meta::comms::logger::initCommLogging();
  meta::comms::logger::configureSpdlogLogger(
      kCtranLoggerName,
      "CTRAN",
      meta::comms::logger::parseDebugFile(NCCL_DEBUG_FILE.c_str()),
      []() {
        int cudaDev = -1;
        (void)cudaGetDevice(&cudaDev);
        return cudaDev;
      },
      [](std::string_view message) {
        meta::comms::logger::setLastError(std::string{message}, {});
      },
      NCCL_DEBUG_LOGGING_ASYNC);
  meta::comms::logger::getSpdlogLogger(kCtranLoggerName)
      .set_level(loggerLevelToSpdlogLevel(
          meta::comms::logger::getLoggerDebugLevel(NCCL_DEBUG)));
}
} // anonymous namespace

void initCtranLogging(bool alwaysInit) {
  if (alwaysInit) {
    initCtranLoggingImpl();
  } else {
    folly::call_once(ctranLoggingInitOnceFlag, initCtranLoggingImpl);
  }
}

}; // namespace ctran::logging
