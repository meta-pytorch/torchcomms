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
folly::once_flag ctranLoggingInitOnceFlag;

void initCtranLoggingImpl() {
  meta::comms::logger::initCommLogging();
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
