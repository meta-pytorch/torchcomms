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
      .set_level(
          meta::comms::logger::loggerLevelToSpdlogLevel(
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
