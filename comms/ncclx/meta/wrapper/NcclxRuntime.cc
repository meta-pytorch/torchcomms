// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "meta/wrapper/NcclxRuntime.h"

#include <mutex>

#include "comms/utils/InitFolly.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/Logger.h"
#include "comms/utils/logger/LoggingFormat.h"

#include "cuda_runtime_api.h"

namespace meta::comms::ncclx {

void ncclxInitLogger() {
  NcclLogger::init(
      NcclLoggerInitConfig{
          .contextName = "comms.ncclx",
          .logPrefix = "NCCL",
          .logFilePath =
              meta::comms::logger::parseDebugFile(NCCL_DEBUG_FILE.c_str()),
          .logLevel = meta::comms::logger::loggerLevelToFollyLogLevel(
              meta::comms::logger::getLoggerDebugLevel(NCCL_DEBUG)),
          .threadContextFn = []() {
            int cudaDev = -1;
            cudaGetDevice(&cudaDev);
            return cudaDev;
          }});
  // Init logging for NCCL header inside meta directory.
  // This is due to the buck2 behavior of copying the header files to the
  // buck-out directory.
  // For logging in src/include headers, they are using NCCL logging
  // (INFO/WARN/ERROR) which will inherit the logging category from debug.cc
  NcclLogger::init(
      NcclLoggerInitConfig{
          .contextName = "meta",
          .logPrefix = "NCCL",
          .logFilePath =
              meta::comms::logger::parseDebugFile(NCCL_DEBUG_FILE.c_str()),
          .logLevel = meta::comms::logger::loggerLevelToFollyLogLevel(
              meta::comms::logger::getLoggerDebugLevel(NCCL_DEBUG)),
          .threadContextFn = []() {
            int cudaDev = -1;
            cudaGetDevice(&cudaDev);
            return cudaDev;
          }});
}

void ncclxInitRuntime(void (*loadEnvFiles)()) {
  static std::once_flag once;
  std::call_once(once, [loadEnvFiles] {
    meta::comms::initFolly();
    ncclCvarInit();
    loadEnvFiles();
    ncclxInitLogger();
  });
}

} // namespace meta::comms::ncclx
