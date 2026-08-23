// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "meta/wrapper/NcclxRuntime.h"

#include <mutex>
#include <string>
#include <string_view>

#include "comms/utils/InitFolly.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"
#include "comms/utils/logger/LoggingFormat.h"
#include "comms/utils/logger/SpdlogLogger.h"
#include "meta/NcclxLogger.h"

#include "cuda_runtime_api.h"

namespace meta::comms::ncclx {

void ncclxInitLogger() {
  // Shared CollTrace code still emits raw XLOG under comms.utils.
  meta::comms::logger::initCommLogging();
  meta::comms::logger::configureSpdlogLogger(
      ::ncclx::logging::kNcclxLoggerName,
      "NCCL",
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
  meta::comms::logger::getSpdlogLogger(::ncclx::logging::kNcclxLoggerName)
      .set_level(
          meta::comms::logger::loggerLevelToSpdlogLevel(
              meta::comms::logger::getLoggerDebugLevel(NCCL_DEBUG)));
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
