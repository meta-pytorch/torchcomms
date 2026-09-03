// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <string_view>

#include "comms/utils/logger/RateLimit.h"
#include "comms/utils/logger/SpdlogLogger.h"

namespace ncclx::logging {

inline constexpr std::string_view kNcclxLoggerName = "comms.ncclx";

inline meta::comms::logger::CommsSpdlogLogger& getNcclxLogger() {
  static auto& logger /* library-local */ =
      meta::comms::logger::getSpdlogLogger(kNcclxLoggerName);
  return logger;
}

} // namespace ncclx::logging

#define NCCLX_LOG_IMPL(spdlog_level, spdlog_macro, ...)                  \
  do {                                                                   \
    static auto& _ncclx_logger = ::meta::comms::logger::getSpdlogLogger( \
        ::ncclx::logging::kNcclxLoggerName);                             \
    if (_ncclx_logger.should_log(spdlog_level)) {                        \
      spdlog_macro(&_ncclx_logger, __VA_ARGS__);                         \
    }                                                                    \
  } while (false)

#define NCCLX_LOG_DBG(...) \
  NCCLX_LOG_IMPL(::spdlog::level::debug, COMMS_LOGGER_DEBUG, __VA_ARGS__)
#define NCCLX_LOG_INFO(...) \
  NCCLX_LOG_IMPL(::spdlog::level::info, SPDLOG_LOGGER_INFO, __VA_ARGS__)
#define NCCLX_LOG_WARN(...) \
  NCCLX_LOG_IMPL(::spdlog::level::warn, SPDLOG_LOGGER_WARN, __VA_ARGS__)
#define NCCLX_LOG_ERR(...) \
  NCCLX_LOG_IMPL(::spdlog::level::err, SPDLOG_LOGGER_ERROR, __VA_ARGS__)
#define NCCLX_LOG_CRITICAL(...) \
  NCCLX_LOG_IMPL(::spdlog::level::critical, SPDLOG_LOGGER_CRITICAL, __VA_ARGS__)
#define NCCLX_LOG_FATAL(...)                                             \
  do {                                                                   \
    static auto& _ncclx_logger = ::meta::comms::logger::getSpdlogLogger( \
        ::ncclx::logging::kNcclxLoggerName);                             \
    COMMS_LOG_FATAL_IMPL(_ncclx_logger, __VA_ARGS__);                    \
  } while (false)
#define NCCLX_LOG(level, ...) NCCLX_LOG_##level(__VA_ARGS__)
#define NCCLX_LOG_STREAM_IF(level, condition) \
  COMMS_LOGGER_STREAM_IF(::ncclx::logging::getNcclxLogger(), level, condition)
#define NCCLX_LOG_STREAM(level) \
  COMMS_LOGGER_STREAM(::ncclx::logging::getNcclxLogger(), level)

#define NCCLX_LOG_IF_IMPL(level, spdlog_level, condition, ...)           \
  do {                                                                   \
    static auto& _ncclx_logger = ::meta::comms::logger::getSpdlogLogger( \
        ::ncclx::logging::kNcclxLoggerName);                             \
    if (_ncclx_logger.should_log(spdlog_level) && (condition)) {         \
      NCCLX_LOG(level, __VA_ARGS__);                                     \
    }                                                                    \
  } while (false)

#define NCCLX_LOG_IF_WARN(condition, ...) \
  NCCLX_LOG_IF_IMPL(WARN, ::spdlog::level::warn, condition, __VA_ARGS__)
#define NCCLX_LOG_IF_DBG(condition, ...) \
  NCCLX_LOG_IF_IMPL(DBG, ::spdlog::level::debug, condition, __VA_ARGS__)
#define NCCLX_LOG_IF_INFO(condition, ...) \
  NCCLX_LOG_IF_IMPL(INFO, ::spdlog::level::info, condition, __VA_ARGS__)
#define NCCLX_LOG_IF_ERR(condition, ...) \
  NCCLX_LOG_IF_IMPL(ERR, ::spdlog::level::err, condition, __VA_ARGS__)
#define NCCLX_LOG_IF_CRITICAL(condition, ...) \
  NCCLX_LOG_IF_IMPL(CRITICAL, ::spdlog::level::critical, condition, __VA_ARGS__)
// Legacy XLOGF_IF never filters FATAL, so its condition is always evaluated.
#define NCCLX_LOG_IF_FATAL(condition, ...) \
  do {                                     \
    if ((condition)) {                     \
      NCCLX_LOG(FATAL, __VA_ARGS__);       \
    }                                      \
  } while (false)
#define NCCLX_LOG_IF(level, condition, ...) \
  NCCLX_LOG_IF_##level(condition, __VA_ARGS__)

#define NCCLX_LOG_FIRST_N_IMPL(level, spdlog_level, n, ...)                    \
  do {                                                                         \
    static auto& _ncclx_logger = ::meta::comms::logger::getSpdlogLogger(       \
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
