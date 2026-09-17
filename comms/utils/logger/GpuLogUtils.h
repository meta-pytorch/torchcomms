// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <string_view>

#include <folly/Format.h>

#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"
#include "comms/utils/logger/LoggingFormat.h"

namespace meta::comms::logger {

constexpr std::string_view kCommsUtilsCategory = "comms.utils";

/**
 * Initialize logging for Comms. By default it only initializes once globally
 * and is a no-op for future calls in the process.
 *
 * @param alwaysInit If true, always initialize logging, for testing purposes.
 */
void initCommLogging(bool alwaysInit = false);

} // namespace meta::comms::logger

/**
 * CERR(code, fmt, ...) logs an ERR-level message (like CLOGF(ERR, ...)) AND
 * records one Scuba error record carrying the commResult_t `code`. Use it at
 * fatal/root-cause CTRAN error sites; keep plain CLOGF(ERR, ...) for
 * recoverable/propagated logs that should not create an error record.
 */
#define CERR(code, ...)                         \
  do {                                          \
    XLOGF(ERR, ##__VA_ARGS__);                  \
    ::meta::comms::logger::logCommErrorToScuba( \
        (code), fmt::format(__VA_ARGS__));      \
  } while (0)

/* Trace level log API. Use cvar NCCL_CTRAN_ENABLE_TRACE_LOG to control for
 * backward compatibility  */
#define CLOGF_TRACE(subsys, fmt, ...)                                          \
  do {                                                                         \
    if (NCCL_CTRAN_ENABLE_TRACE_LOG) {                                         \
      CLOGF_SUBSYS(INFO, subsys, "[TRACE] {}: " fmt, __func__, ##__VA_ARGS__); \
    }                                                                          \
  } while (0);
