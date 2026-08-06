// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/utils/LogInit.h"

#include <cuda_runtime.h>

#include <folly/synchronization/CallOnce.h>

#include "comms/utils/cvars/nccl_cvars.h" // @manual=fbcode//comms/utils/cvars:ncclx-cvars
#include "comms/utils/logger/LogUtils.h"
#include "comms/utils/logger/Logger.h"
#include "comms/utils/logger/LoggingFormat.h"
#include "comms/utils/logger/SpdlogLogger.h"

namespace ctran::logging {

namespace {

/*
 * On AMD, after hipification, the file path has format:
 * buck-out/v2/gen/fbcode/{hash}/comms/ctran/dir1/dir2/__{ori_file_name}_hipify_gen__/out/{ori_file_name},
 * The XLOG stripBuckV2Prefix https://fburl.com/code/utkyckpn doesn't handle the
 * hipified file path, so we do our own work to find the correct ctran component
 * prefix here.
 */
std::string getHipCtranCategory() {
  std::string fullFilePath = XLOG_FILENAME;
  static constexpr std::string_view kCtranComponent{"comms/ctran"};
  auto ctranComponentIdx = fullFilePath.find(kCtranComponent);
  return fullFilePath.substr(0, ctranComponentIdx + kCtranComponent.size());
};

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
  NcclLogger::init(
      NcclLoggerInitConfig{
          .contextName = std::string{kCtranCategory},
          .logPrefix = "CTRAN",
          .logFilePath =
              meta::comms::logger::parseDebugFile(NCCL_DEBUG_FILE.c_str()),
          .logLevel = meta::comms::logger::loggerLevelToFollyLogLevel(
              meta::comms::logger::getLoggerDebugLevel(NCCL_DEBUG)),
          .threadContextFn = []() {
            int cudaDev = -1;
            (void)cudaGetDevice(&cudaDev);
            return cudaDev;
          }});
  // Init logging for CTRAN header
  NcclLogger::init(
      NcclLoggerInitConfig{
          .contextName = std::string{kCtranHeaderCategory},
          .logPrefix = "CTRAN",
          .logFilePath =
              meta::comms::logger::parseDebugFile(NCCL_DEBUG_FILE.c_str()),
          .logLevel = meta::comms::logger::loggerLevelToFollyLogLevel(
              meta::comms::logger::getLoggerDebugLevel(NCCL_DEBUG)),
          .threadContextFn = []() {
            int cudaDev = -1;
            (void)cudaGetDevice(&cudaDev);
            return cudaDev;
          }});
#if defined(USE_ROCM)
  NcclLogger::init(
      NcclLoggerInitConfig{
          .contextName = getHipCtranCategory(),
          .logPrefix = "CTRAN",
          .logFilePath =
              meta::comms::logger::parseDebugFile(NCCL_DEBUG_FILE.c_str()),
          .logLevel = meta::comms::logger::loggerLevelToFollyLogLevel(
              meta::comms::logger::getLoggerDebugLevel(NCCL_DEBUG)),
          .threadContextFn = []() {
            int cudaDev = -1;
            (void)cudaGetDevice(&cudaDev);
            return cudaDev;
          }});
#endif
  meta::comms::logger::configureSpdlogLogger(
      kCtranSpdlogContext,
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
  meta::comms::logger::getSpdlogLogger(kCtranSpdlogContext)
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
