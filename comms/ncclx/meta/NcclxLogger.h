// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <string_view>

#include "comms/utils/logger/RateLimit.h"
#include "comms/utils/logger/SpdlogLogger.h"

namespace ncclx::logging {

inline constexpr std::string_view kNcclxLoggerName = "comms.ncclx";

} // namespace ncclx::logging

#define NCCLX_LOG(level, ...) \
  COMMS_LOG_NAMED(::ncclx::logging::kNcclxLoggerName, level, __VA_ARGS__)

#define NCCLX_LOG_FIRST_N_IMPL(level, spdlog_level, n, ...)                    \
  do {                                                                         \
    auto& _ncclx_logger = ::meta::comms::logger::getSpdlogLogger(              \
        ::ncclx::logging::kNcclxLoggerName);                                   \
    if (_ncclx_logger.should_log(spdlog_level) && [&] {                        \
          struct ncclx_log_first_n_tag {};                                     \
          return ::meta::comms::logger::firstNExact<ncclx_log_first_n_tag>(n); \
        }()) {                                                                 \
      NCCLX_LOG(level, __VA_ARGS__);                                           \
    }                                                                          \
  } while (false)

#define NCCLX_LOG_FIRST_N_WARN(n, ...) \
  NCCLX_LOG_FIRST_N_IMPL(WARN, ::spdlog::level::warn, n, __VA_ARGS__)
#define NCCLX_LOG_FIRST_N_DBG(n, ...) \
  NCCLX_LOG_FIRST_N_IMPL(DBG, ::spdlog::level::debug, n, __VA_ARGS__)
#define NCCLX_LOG_FIRST_N_INFO(n, ...) \
  NCCLX_LOG_FIRST_N_IMPL(INFO, ::spdlog::level::info, n, __VA_ARGS__)
#define NCCLX_LOG_FIRST_N_ERR(n, ...) \
  NCCLX_LOG_FIRST_N_IMPL(ERR, ::spdlog::level::err, n, __VA_ARGS__)
#define NCCLX_LOG_FIRST_N_CRITICAL(n, ...) \
  NCCLX_LOG_FIRST_N_IMPL(CRITICAL, ::spdlog::level::critical, n, __VA_ARGS__)
#define NCCLX_LOG_FIRST_N(level, n, ...) \
  NCCLX_LOG_FIRST_N_##level(n, __VA_ARGS__)
