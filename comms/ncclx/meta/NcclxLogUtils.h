// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <fmt/format.h>

#include "comms/utils/logger/CommsLogging.h"
#include "comms/utils/logger/LogTypes.h"
#include "meta/NcclxLogger.h"

#define NCCLX_LOG_SUBSYS(level, subsys, ...)            \
  NCCLX_LOG_IF(                                         \
      level,                                            \
      ::meta::comms::logger::isEnabledSubSystemBitwise( \
          static_cast<uint64_t>([]() {                  \
            using namespace ::meta::comms::logger;      \
            return subsys;                              \
          }())),                                        \
      __VA_ARGS__)

#define NCCLX_LOG_STREAM_EVERY_MS(level, ms) \
  COMMS_LOGGER_STREAM_EVERY_MS(::ncclx::logging::getNcclxLogger(), level, ms)

#define NCCLX_ERR(code, ...)                                                  \
  do {                                                                        \
    const auto _ncclx_error_message = fmt::format(__VA_ARGS__);               \
    NCCLX_LOG(ERR, "{}", _ncclx_error_message);                               \
    ::meta::comms::logger::logCommErrorToScuba((code), _ncclx_error_message); \
  } while (false)
