// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <chrono>

#include "comms/ctran/utils/CtranLogger.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

#define CTRAN_LOG_SUBSYS(level, subsys, ...) \
  CTRAN_LOG_IF(level, CLOGF_ENABLED(subsys), __VA_ARGS__)

// The interval is fixed on first use at each expansion site, matching
// CLOGF_EVERY_MS; callers must pass a call-site constant.
#define CTRAN_LOG_EVERY_MS(level, ms, ...)                          \
  CTRAN_LOG_IF(                                                     \
      level,                                                        \
      [_ctran_log_every_ms = (ms)] {                                \
        static ::meta::comms::logger::IntervalRateLimiter           \
            ctran_log_rate_limiter(                                 \
                1, std::chrono::milliseconds(_ctran_log_every_ms)); \
        return ctran_log_rate_limiter.check();                      \
      }(),                                                          \
      __VA_ARGS__)

#define CTRAN_LOG_TRACE(subsys, format, ...)                             \
  do {                                                                   \
    if (NCCL_CTRAN_ENABLE_TRACE_LOG) {                                   \
      CTRAN_LOG_SUBSYS(                                                  \
          INFO, subsys, "[TRACE] {}: " format, __func__, ##__VA_ARGS__); \
    }                                                                    \
  } while (false)

#define CTRAN_ERR(code, ...)                                                  \
  do {                                                                        \
    const auto _ctran_error_message = fmt::format(__VA_ARGS__);               \
    CTRAN_LOG(ERR, "{}", _ctran_error_message);                               \
    ::meta::comms::logger::logCommErrorToScuba((code), _ctran_error_message); \
  } while (false)
