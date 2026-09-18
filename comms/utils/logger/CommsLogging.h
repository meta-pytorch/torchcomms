// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include "comms/utils/commSpecs.h"
#include "comms/utils/logger/LogTypes.h"
#include "comms/utils/logger/SpdlogLogger.h"

namespace meta::comms::logger {

uint64_t parseDebugSubsysMask(const char* ncclDebugSubsysEnv);

std::string parseDebugFile(const char* ncclDebugFileEnv);

LogLevel getLoggerDebugLevel(std::string_view levelStr);
LogLevel getNcclLoggerDebugLevel(std::string_view levelStr);

/*
 * The fallback applies only when value is absent. Any present value other
 * than a recognized false token enables logging, matching NCCL cvar parsing.
 */
bool parseDebugLoggingAsync(const char* value, bool valueWhenUnset);

void initProcMetaData();

void initThreadMetaData(std::string_view threadName);

std::string getLastCommsError();

namespace detail {

struct ProcessMetadata {
  std::string_view hostname;
  int processId;
};

ProcessMetadata getLogProcessMetadata();

void setLastErrorFromLegacyLog(std::string_view message);

} // namespace detail

/*
 * TODO: remove once ncclx v2_29 is retired. The only remaining producer is
 * v2_29's ncclMetaDebugLogWithScuba (via ERR_WITH_SCUBA/WARN_WITH_SCUBA);
 * v2_30 and ctran record the last-error stack via captureNativeErrorStack() +
 * setLastError() instead (native-stack-only).
 */
void appendErrorToStack(std::string error);

/*
 * Record an ERROR-level log to the nccl_structured_logging Scuba table as a
 * single error record (top-level message in the exception_message column,
 * native stack in the stack_trace column). The caller is responsible for
 * gating this on NCCL_SCUBA_LOG_ERROR_ENABLED and for capturing the native
 * stack once so it can be shared across reporters.
 */
__attribute__((visibility("default"))) void logErrorToScuba(
    const std::string& message,
    int code,
    const std::string& errorName,
    const std::vector<std::string>& stack);

/*
 * Update the ncclGetLastError() state with the latest error message and its
 * pre-captured native stack. Unconditional; independent of Scuba logging.
 */
void setLastError(const std::string& message, std::vector<std::string> stack);

/*
 * Record a CTRAN ERR-level error to Scuba, carrying the commResult_t code.
 * Gated by NCCL_SCUBA_LOG_ERROR_ENABLED; a no-op when disabled.
 */
__attribute__((visibility("default"))) void logCommErrorToScuba(
    commResult_t code,
    const std::string& message);

} // namespace meta::comms::logger

#define COMMS_ERR(code, ...)                                                  \
  do {                                                                        \
    const auto _comms_error_message = fmt::format(__VA_ARGS__);               \
    COMMS_LOG(ERR, "{}", _comms_error_message);                               \
    ::meta::comms::logger::logCommErrorToScuba((code), _comms_error_message); \
  } while (false)

#define COMMS_NAMED_THREAD_START_EXT(threadName, rank, commHash, commDesc)                \
  do {                                                                                    \
    meta::comms::logger::initThreadMetaData(threadName);                                  \
    COMMS_LOG(                                                                            \
        INFO,                                                                             \
        "[COMMS THREAD] Starting {} thread for rank {} commHash {:#x} commDesc {} at {}", \
        threadName,                                                                       \
        rank,                                                                             \
        commHash,                                                                         \
        commDesc,                                                                         \
        __func__);                                                                        \
  } while (0);
