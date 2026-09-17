// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <chrono>
#include <cstdint>

#include <folly/logging/xlog.h>

#include "comms/utils/logger/LogTypes.h"
#include "comms/utils/logger/RateLimit.h"

// This file defines logging APIs modelled atop `folly/log.h`.

/*
 *
 * Usage:
 *   CLOGF(INFO, "Processing data: {}", dataName);
 *
 * Output example:
 *   [GPU 0] Processing data: input_tensor
 *
 * @param level The log level (DBG, INFO, WARN, ERR, FATAL)
 * @param fmt Format string (printf-style)
 * @param ... Format arguments
 */
#define CLOGF(level, ...) XLOGF(level, ##__VA_ARGS__)

/**
 * Usage:
 *   CLOGF_IF(INFO, size > threshold, "Large data: {} bytes", size);
 *
 * Note: allows for manual CLOGF_SUBSYS with multiple subsystems when
 * used together with CLOGF_ENABLED
 *
 * Usage:
 *  CLOGF_IF(INFO, CLOGF_ENABLED(ALLOC) | CLOGF_ENABLED(P2P), "{} bytes", sz);
 *
 * @param level The log level (DBG, INFO, WARN, ERR, FATAL)
 * @param cond Condition that determines whether to log
 * @param fmt Format string (printf-style)
 * @param ... Format arguments
 */
#define CLOGF_IF(level, ...) XLOGF_IF(level, ##__VA_ARGS__)

/**
 * Helper to check if a subsystem is enabled.
 *
 * Usage:
 *   CLOGF_ENABLED(ALLOC)
 *   CLOGF_ENABLED(COLL | P2P)
 *
 * @param subsys meta::comms::logger::SubSystem enum name or bitwise OR
 * combination e.g. (COLL, NET, COLL | P2P)
 */
#define CLOGF_ENABLED(subsys)                       \
  ::meta::comms::logger::isEnabledSubSystemBitwise( \
      static_cast<uint64_t>([]() {                  \
        using namespace ::meta::comms::logger;      \
        return subsys;                              \
      }()))

/**
 * This will conditionally log the message if the sub-system logging is
 * enabled.
 *
 * Usage:
 *   CLOGF_SUBSYS(INFO, COLL, "Processing data: {}", dataName);
 *
 * @param level The log level (DBG, INFO, WARN, ERR, FATAL)
 * @param subsys meta::comms::logger::SubSystem enum name e.g. (COLL, NET)
 * @param fmt Format string (printf-style)
 * @param ... Format arguments
 */
#define CLOGF_SUBSYS(level, subsys, fmt, ...) \
  XLOGF_IF(level, CLOGF_ENABLED(subsys), fmt, ##__VA_ARGS__)

#define CLOGF_FIRST_N(level, n, fmt, ...)                                    \
  CLOGF_IF(                                                                  \
      level,                                                                 \
      [&] {                                                                  \
        struct comms_log_first_n_tag {};                                     \
        return ::meta::comms::logger::firstNExact<comms_log_first_n_tag>(n); \
      }(),                                                                   \
      fmt,                                                                   \
      ##__VA_ARGS__)

#define CLOGF_EVERY_MS(level, ms, fmt, ...)                         \
  CLOGF_IF(                                                         \
      level,                                                        \
      [_comms_log_every_ms = ms] {                                  \
        static ::meta::comms::logger::IntervalRateLimiter           \
            comms_log_rate_limiter(                                 \
                1, std::chrono::milliseconds(_comms_log_every_ms)); \
        return comms_log_rate_limiter.check();                      \
      }(),                                                          \
      fmt,                                                          \
      ##__VA_ARGS__)
