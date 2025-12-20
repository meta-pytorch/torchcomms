// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <folly/Format.h>
#include <folly/logging/xlog.h>

#include "comms/utils/logger/LoggingFormat.h"

//
// This file defines Logging APIs modelled atop `folly/log.h`. Atop it intends
// to add some additional features like GPU index prefixing and conditional
// logging per sub-system
//
// Use this API specifically for logging in the context of GPU code. Otherwise,
// prefer using `folly/log.h` directly for performance reasons.
//

namespace meta::comms::logger {

constexpr std::string_view kCommsUtilsCategory = "comms.utils";

/**
 * Bitwise OR of all sub-systems that needs to be enabled.
 */
void setSubSystemMask(uint64_t subSystemMask);

bool isEnabledSubSystemBitwise(uint64_t subSystem);

/**
 * Initialize logging for Comms. By default it only initializes once globlally
 * and no-op for future calls on the process.
 *
 * @param alwaysInit If true, always initialize logging, for testing purpose.
 */
void initCommLogging(bool alwaysInit = false);

}; // namespace meta::comms::logger

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
 * This will conditionally log the message if the sub-system
 * logging is enabled.
 * Usage:
 *   CLOGF_SUBSYS(INFO, COLL, "Processing data: {}", dataName);
 *
 * @param level The log level (DBG, INFO, WARN, ERR, FATAL)
 * @param subsys meta::comms::logger::SubSystem enum name e.g. (COLL, NET)
 * @param cond Condition that determines whether to log
 * @param fmt Format string (printf-style)
 * @param ... Format arguments
 */
#define CLOGF_SUBSYS(level, subsys, fmt, ...) \
  CLOGF_IF(level, CLOGF_ENABLED(subsys), fmt, ##__VA_ARGS__)

#define CLOGF_FIRST_N(level, n, fmt, ...)                                      \
  CLOGF_IF(                                                                    \
      level,                                                                   \
      [&] {                                                                    \
        struct folly_detail_xlog_tag {};                                       \
        return ::folly::detail::xlogFirstNExactImpl<folly_detail_xlog_tag>(n); \
      }(),                                                                     \
      fmt,                                                                     \
      ##__VA_ARGS__)

#define CLOGF_EVERY_MS(level, ms, fmt, ...)                           \
  CLOGF_IF(                                                           \
      level,                                                          \
      [_folly_detail_xlog_ms = ms] {                                  \
        static ::folly::logging::IntervalRateLimiter                  \
            folly_detail_xlog_limiter(                                \
                1, std::chrono::milliseconds(_folly_detail_xlog_ms)); \
        return folly_detail_xlog_limiter.check();                     \
      }(),                                                            \
      fmt,                                                            \
      ##__VA_ARGS__)

/* Trace level log API. Use cvar NCCL_CTRAN_ENABLE_TRACE_LOG to control for
 * backward compatibility  */
#define CLOGF_TRACE(subsys, fmt, ...)                                          \
  do {                                                                         \
    if (NCCL_CTRAN_ENABLE_TRACE_LOG) {                                         \
      CLOGF_SUBSYS(INFO, subsys, "[TRACE] {}: " fmt, __func__, ##__VA_ARGS__); \
    }                                                                          \
  } while (0);

/**
 * Helper to check if a subsystem is enabled.
 *
 * Usage:
 *   CLOGF_ENABLED(ALLOC)
 *   CLOGF_ENABLED(COLL | P2P)
 *
 * @param subsys meta::comms::logger::SubSystem enum name or bitwise OR
 * combination e.g. (COLL, NET, COLL | P2P)
 * we use lambda to allow user call CLOGF_ENABLED() with \
 * subsystem without specifying a full name e.g. \
 * CLOGF_ENABLED(COLL | P2P) but not exposing using namespace \
 * meta::comms::logger \
 */
#define CLOGF_ENABLED(subsys)                       \
  ::meta::comms::logger::isEnabledSubSystemBitwise( \
      static_cast<uint64_t>([]() {                  \
        using namespace ::meta::comms::logger;      \
        return subsys;                              \
      }()))
