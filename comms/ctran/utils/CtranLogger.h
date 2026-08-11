// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <string_view>

// The standalone RCCL/conda build is the only AMD/HIP build and it has no
// spdlog, which SpdlogLogger.h hard-requires. Fall back to folly XLOG there,
// which is what these call sites used before the spdlog migration: LogInit.cc
// already configures the folly "comms.ctran" categories on ROCm, so the level
// gating, CTRAN prefix, NCCL_DEBUG_FILE output and per-thread device context
// are unchanged.
#if defined(__HIP_PLATFORM_AMD__)
#include <folly/logging/xlog.h>
#else
#include "comms/utils/logger/RateLimit.h"
#include "comms/utils/logger/SpdlogLogger.h"
#endif

namespace ctran::logging {

inline constexpr std::string_view kCtranLoggerName = "comms.ctran";

} // namespace ctran::logging

#if defined(__HIP_PLATFORM_AMD__)

#define CTRAN_LOG(level, ...) XLOGF(level, __VA_ARGS__)
#define CTRAN_LOG_FIRST_N_WARN(n, ...) XLOGF_FIRST_N(WARN, n, __VA_ARGS__)
#define CTRAN_LOG_FIRST_N_ERR(n, ...) XLOGF_FIRST_N(ERR, n, __VA_ARGS__)

// XLOGF_IF is the legacy form the spdlog CTRAN_LOG_IF_* macros were modelled
// on, so FATAL needs no special case here: it already evaluates the condition
// unconditionally.
#define CTRAN_LOG_IF_WARN(condition, ...) XLOGF_IF(WARN, condition, __VA_ARGS__)
#define CTRAN_LOG_IF_DBG(condition, ...) XLOGF_IF(DBG, condition, __VA_ARGS__)
#define CTRAN_LOG_IF_INFO(condition, ...) XLOGF_IF(INFO, condition, __VA_ARGS__)
#define CTRAN_LOG_IF_ERR(condition, ...) XLOGF_IF(ERR, condition, __VA_ARGS__)
#define CTRAN_LOG_IF_CRITICAL(condition, ...) \
  XLOGF_IF(CRITICAL, condition, __VA_ARGS__)
#define CTRAN_LOG_IF_FATAL(condition, ...) \
  XLOGF_IF(FATAL, condition, __VA_ARGS__)

#else

#define CTRAN_LOG(level, ...) \
  COMMS_LOG_NAMED(::ctran::logging::kCtranLoggerName, level, __VA_ARGS__)

#define CTRAN_LOG_IF_IMPL(level, spdlog_level, condition, ...)    \
  do {                                                            \
    auto& _ctran_logger = ::meta::comms::logger::getSpdlogLogger( \
        ::ctran::logging::kCtranLoggerName);                      \
    if (_ctran_logger.should_log(spdlog_level) && (condition)) {  \
      CTRAN_LOG(level, __VA_ARGS__);                              \
    }                                                             \
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

#endif

#define CTRAN_LOG_IF(level, condition, ...) \
  CTRAN_LOG_IF_##level(condition, __VA_ARGS__)

#define CTRAN_LOG_FIRST_N(level, n, ...) \
  CTRAN_LOG_FIRST_N_##level(n, __VA_ARGS__)
