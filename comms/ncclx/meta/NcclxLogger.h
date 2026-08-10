// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <string_view>

#include "comms/utils/logger/SpdlogLogger.h"

namespace ncclx::logging {

inline constexpr std::string_view kNcclxLoggerName = "comms.ncclx";

} // namespace ncclx::logging

#define NCCLX_LOG(level, ...) \
  COMMS_LOG_NAMED(::ncclx::logging::kNcclxLoggerName, level, __VA_ARGS__)
