// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/utils/LogInit.h"

#include <cuda_runtime.h>

#include <folly/synchronization/CallOnce.h>

#include "comms/ctran/ibverbx/utils/IbverbxLogger.h"
#include "comms/ctran/utils/CtranLogger.h"
#include "comms/utils/cvars/nccl_cvars.h" // @manual=fbcode//comms/utils/cvars:ncclx-cvars
#include "comms/utils/logger/CommsLogging.h"
#include "comms/utils/logger/LoggerRuntime.h"

namespace ctran::logging {

namespace {
folly::once_flag ctranLoggingInitOnceFlag;

spdlog::level::level_enum toSpdlogLevel(const ibverbx::logging::Level level) {
  switch (level) {
    case ibverbx::logging::Level::DBG:
      return spdlog::level::debug;
    case ibverbx::logging::Level::INFO:
      return spdlog::level::info;
    case ibverbx::logging::Level::WARN:
      return spdlog::level::warn;
    case ibverbx::logging::Level::ERR:
      return spdlog::level::err;
    case ibverbx::logging::Level::FATAL:
      return spdlog::level::critical;
  }
  return spdlog::level::err;
}

/*
 * ibverbx logs to stderr unless a host claims delivery. Route it through the
 * CTRAN logger so its lines keep landing wherever NCCL_DEBUG_FILE points,
 * gated by NCCL_DEBUG, rather than splitting onto a second destination.
 *
 * FATAL takes the same path CTRAN_LOG(FATAL) does: drain the async queue first,
 * then deliver synchronously. ibverbx aborts the moment this returns, so a
 * queued line would otherwise die with the process -- the one line that must
 * survive. It also bypasses the level gate, since a crash message is not
 * something NCCL_DEBUG should be able to suppress. A throw here must not escape
 * into ibverbx, where it would unwind past the abort and let the failed
 * assertion continue.
 */
void logIbverbxThroughCtran(
    const ibverbx::logging::Level level,
    const char* const file,
    const int line,
    const char* const function,
    const std::string& message) {
  const spdlog::source_loc location{file, line, function};

  if (level == ibverbx::logging::Level::FATAL) {
    try {
      auto& logger = ::ctran::logging::getCtranLogger();
      meta::comms::logger::shutdownSpdlogForFatal();
      logger.logFatal(location, message);
    } catch (...) {
      meta::comms::logger::abortAfterCommsLoggingFailure();
    }
    return;
  }

  ::ctran::logging::getCtranLogger().log(
      location, toSpdlogLevel(level), message);
}

void initCtranLoggingImpl() {
  meta::comms::logger::initCommLoggerRuntime();
  const auto logFilePath =
      meta::comms::logger::parseDebugFile(NCCL_DEBUG_FILE.c_str());
  const auto threadContextFn = []() {
    int cudaDev = -1;
    (void)cudaGetDevice(&cudaDev);
    return cudaDev;
  };
  const auto errorCallback = [](std::string_view message) {
    meta::comms::logger::setLastError(std::string{message}, {});
  };
  const auto logLevel = meta::comms::logger::loggerLevelToSpdlogLevel(
      meta::comms::logger::getLoggerDebugLevel(NCCL_DEBUG));

  meta::comms::logger::configureCommsAndNamedSpdlogLoggers(
      kCtranLoggerName,
      "CTRAN",
      logFilePath,
      threadContextFn,
      errorCallback,
      NCCL_DEBUG_LOGGING_ASYNC,
      logLevel);

  // After the logger is configured, so the first forwarded line already sees
  // the real level and destination.
  ibverbx::logging::setLogSink(logIbverbxThroughCtran);
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
