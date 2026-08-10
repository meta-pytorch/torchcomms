// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <string_view>

#include "comms/utils/logger/RateLimit.h"
#include "comms/utils/logger/SpdlogLogger.h"

namespace ctran::logging {

inline constexpr std::string_view kCtranLoggerName = "comms.ctran";

} // namespace ctran::logging

#define CTRAN_LOG(level, ...) \
  COMMS_LOG_NAMED(::ctran::logging::kCtranLoggerName, level, __VA_ARGS__)

#define CTRAN_LOG_FIRST_N_IMPL(level, spdlog_level, n, ...)                    \
  do {                                                                         \
    auto& _ctran_logger = ::meta::comms::logger::getSpdlogLogger(              \
        ::ctran::logging::kCtranLoggerName);                                   \
    if (_ctran_logger.should_log(spdlog_level) && [&] {                        \
          struct ctran_log_first_n_tag {};                                     \
          return ::meta::comms::logger::firstNExact<ctran_log_first_n_tag>(n); \
        }()) {                                                                 \
      CTRAN_LOG(level, __VA_ARGS__);                                           \
    }                                                                          \
  } while (false)

#define CTRAN_LOG_FIRST_N_WARN(n, ...) \
  CTRAN_LOG_FIRST_N_IMPL(WARN, ::spdlog::level::warn, n, __VA_ARGS__)
#define CTRAN_LOG_FIRST_N_ERR(n, ...) \
  CTRAN_LOG_FIRST_N_IMPL(ERR, ::spdlog::level::err, n, __VA_ARGS__)
#define CTRAN_LOG_FIRST_N(level, n, ...) \
  CTRAN_LOG_FIRST_N_##level(n, __VA_ARGS__)
