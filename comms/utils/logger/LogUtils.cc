// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>

#include "comms/utils/logger/LogUtils.h"

#include <folly/synchronization/CallOnce.h>

#include "comms/utils/cvars/nccl_cvars.h" // @manual=fbcode//comms/utils/cvars:ncclx-cvars
#include "comms/utils/logger/Logger.h"
#include "comms/utils/logger/LoggingFormat.h"

namespace meta::comms::logger {

namespace {
static uint64_t subSystemMask = 0; // Bitmask of enabled subsystems

folly::once_flag commLoggingInitOnceFlag;

void initCommLoggingImpl() {
  setSubSystemMask(
      meta::comms::logger::parseDebugSubsysMask(NCCL_DEBUG_SUBSYS.c_str()));
  NcclLogger::init(NcclLoggerInitConfig{
      .contextName = std::string{kCommsUtilsCategory},
      .logPrefix = "COMM",
      .logFilePath =
          meta::comms::logger::parseDebugFile(NCCL_DEBUG_FILE.c_str()),
      .logLevel = meta::comms::logger::loggerLevelToFollyLogLevel(
          meta::comms::logger::getLoggerDebugLevel(NCCL_DEBUG)),
      .threadContextFn = []() {
        int cudaDev = -1;
        (void)cudaGetDevice(&cudaDev);
        return cudaDev;
      }});
}
} // anonymous namespace

void setSubSystemMask(uint64_t subSystemMask_) {
  subSystemMask = subSystemMask_;
}

bool isEnabledSubSystemBitwise(uint64_t subSystem) {
  return subSystemMask & subSystem;
}

void initCommLogging(bool alwaysInit) {
  if (alwaysInit) {
    initCommLoggingImpl();
  } else {
    folly::call_once(commLoggingInitOnceFlag, initCommLoggingImpl);
  }
}
} // namespace meta::comms::logger
