// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/utils/LogInit.h"

#include <cuda_runtime.h>

#include <folly/synchronization/CallOnce.h>

#include "comms/utils/cvars/nccl_cvars.h" // @manual=fbcode//comms/utils/cvars:ncclx-cvars
#include "comms/utils/logger/LogUtils.h"
#include "comms/utils/logger/Logger.h"
#include "comms/utils/logger/LoggingFormat.h"

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
            cudaGetDevice(&cudaDev);
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
            cudaGetDevice(&cudaDev);
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
            cudaGetDevice(&cudaDev);
            return cudaDev;
          }});
#endif
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
