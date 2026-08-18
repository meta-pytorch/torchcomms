// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <fmt/format.h>

#include "comms/utils/logger/LogUtils.h"
#include "meta/NcclxLogger.h"

#define NCCLX_LOG_SUBSYS(level, subsys, ...) \
  NCCLX_LOG_IF(level, CLOGF_ENABLED(subsys), __VA_ARGS__)

#define NCCLX_ERR(code, ...)                                                  \
  do {                                                                        \
    const auto _ncclx_error_message = fmt::format(__VA_ARGS__);               \
    NCCLX_LOG(ERR, "{}", _ncclx_error_message);                               \
    ::meta::comms::logger::logCommErrorToScuba((code), _ncclx_error_message); \
  } while (false)
