// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <string_view>

#include "comms/utils/logger/RateLimit.h"
#include "comms/utils/logger/SpdlogLogger.h"

namespace ctran::logging {

inline constexpr std::string_view kCtranLoggerName = "comms.ctran";

} // namespace ctran::logging

#define CTRAN_LOG_IMPL(spdlog_level, spdlog_macro, ...)                  \
  do {                                                                   \
    static auto& _ctran_logger = ::meta::comms::logger::getSpdlogLogger( \
        ::ctran::logging::kCtranLoggerName);                             \
    if (_ctran_logger.should_log(spdlog_level)) {                        \
      spdlog_macro(&_ctran_logger, __VA_ARGS__);                         \
    }                                                                    \
  } while (false)

#define CTRAN_LOG_DBG(...) \
  CTRAN_LOG_IMPL(::spdlog::level::debug, COMMS_LOGGER_DEBUG, __VA_ARGS__)
#define CTRAN_LOG_INFO(...) \
  CTRAN_LOG_IMPL(::spdlog::level::info, SPDLOG_LOGGER_INFO, __VA_ARGS__)
#define CTRAN_LOG_WARN(...) \
  CTRAN_LOG_IMPL(::spdlog::level::warn, SPDLOG_LOGGER_WARN, __VA_ARGS__)
#define CTRAN_LOG_ERR(...) \
  CTRAN_LOG_IMPL(::spdlog::level::err, SPDLOG_LOGGER_ERROR, __VA_ARGS__)
#define CTRAN_LOG_CRITICAL(...) \
  CTRAN_LOG_IMPL(::spdlog::level::critical, SPDLOG_LOGGER_CRITICAL, __VA_ARGS__)
#define CTRAN_LOG_FATAL(...)                                             \
  do {                                                                   \
    static auto& _ctran_logger = ::meta::comms::logger::getSpdlogLogger( \
        ::ctran::logging::kCtranLoggerName);                             \
    COMMS_LOG_FATAL_IMPL(_ctran_logger, __VA_ARGS__);                    \
  } while (false)
#define CTRAN_LOG(level, ...) CTRAN_LOG_##level(__VA_ARGS__)
#define CTRAN_LOG_STREAM_IF(level, condition) \
  COMMS_LOG_NAMED_STREAM_IF(                  \
      ::ctran::logging::kCtranLoggerName, level, condition)
#define CTRAN_LOG_STREAM(level) \
  COMMS_LOG_NAMED_STREAM(::ctran::logging::kCtranLoggerName, level)

#define CTRAN_LOG_SYNC_ERR(...)                                          \
  do {                                                                   \
    static auto& _ctran_logger = ::meta::comms::logger::getSpdlogLogger( \
        ::ctran::logging::kCtranLoggerName);                             \
    if (_ctran_logger.should_log(::spdlog::level::err)) {                \
      _ctran_logger.logSynchronous(                                      \
          ::spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION},     \
          ::spdlog::level::err,                                          \
          __VA_ARGS__);                                                  \
    }                                                                    \
  } while (false)

#define CTRAN_LOG_IF_IMPL(level, spdlog_level, condition, ...)           \
  do {                                                                   \
    static auto& _ctran_logger = ::meta::comms::logger::getSpdlogLogger( \
        ::ctran::logging::kCtranLoggerName);                             \
    if (_ctran_logger.should_log(spdlog_level) && (condition)) {         \
      CTRAN_LOG(level, __VA_ARGS__);                                     \
    }                                                                    \
  } while (false)

#define CTRAN_LOG_IF_WARN(condition, ...) \
  CTRAN_LOG_IF_IMPL(WARN, ::spdlog::level::warn, condition, __VA_ARGS__)
#define CTRAN_LOG_IF_DBG(condition, ...) \
  CTRAN_LOG_IF_IMPL(DBG, ::spdlog::level::debug, condition, __VA_ARGS__)
#define CTRAN_LOG_IF_INFO(condition, ...) \
  CTRAN_LOG_IF_IMPL(INFO, ::spdlog::level::info, condition, __VA_ARGS__)
#define CTRAN_LOG_IF_ERR(condition, ...) \
  CTRAN_LOG_IF_IMPL(ERR, ::spdlog::level::err, condition, __VA_ARGS__)
#define CTRAN_LOG_IF_CRITICAL(condition, ...) \
  CTRAN_LOG_IF_IMPL(CRITICAL, ::spdlog::level::critical, condition, __VA_ARGS__)
// Legacy XLOGF_IF never filters FATAL, so its condition is always evaluated.
#define CTRAN_LOG_IF_FATAL(condition, ...) \
  do {                                     \
    if ((condition)) {                     \
      CTRAN_LOG(FATAL, __VA_ARGS__);       \
    }                                      \
  } while (false)
#define CTRAN_LOG_IF(level, condition, ...) \
  CTRAN_LOG_IF_##level(condition, __VA_ARGS__)

#define CTRAN_LOG_FIRST_N_IMPL(level, spdlog_level, n, ...)                    \
  do {                                                                         \
    static auto& _ctran_logger = ::meta::comms::logger::getSpdlogLogger(       \
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
