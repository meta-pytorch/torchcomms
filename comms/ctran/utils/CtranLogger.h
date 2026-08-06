// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <string_view>

#include "comms/utils/logger/SpdlogLogger.h"

namespace ctran::logging {

inline constexpr std::string_view kCtranLoggerName = "comms.ctran";

} // namespace ctran::logging

#define CTRAN_LOG(level, ...) \
  COMMS_LOG_NAMED(::ctran::logging::kCtranLoggerName, level, __VA_ARGS__)
