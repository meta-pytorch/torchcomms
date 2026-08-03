// Copyright (c) Meta Platforms, Inc. and affiliates.

// Meta logging entry points hoisted out of the forked upstream debug.cc to
// keep the NCCLX fork's delta against pristine NCCL small. These share the
// debug state and lazy initialization owned by debug.cc via
// DebugExtInternal.h; both compile into the same NCCLX library.

#include <cstdarg>
#include <cstdio>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <folly/logging/LogLevel.h>
#include <folly/logging/LogStreamProcessor.h>
#include <folly/logging/xlog.h>

#include "core.h"
#include "debug.h"
#include "meta/logger/DebugExtInternal.h"

#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/ErrorStackUtil.h"
#include "comms/utils/logger/LogUtils.h"
#include "comms/utils/logger/LoggingFormat.h"

// These are Meta's logging implementations, kept out of the baseline debug.cc.
// Guarded to v2_30+ since older versions keep their own copy in debug.cc.
// TODO T279903668: Cleanup version check after v2_29 removal
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 0)

void ncclMetaDebugLogError(
    ncclResult_t code,
    unsigned long flags,
    const char* file,
    const char* func,
    int line,
    const char* fmt,
    ...) {
  // Format the message once (same vsnprintf pattern as ncclMetaDebugLog).
  size_t logLen = 0;
  va_list vargs;
  va_start(vargs, fmt);
  logLen += std::vsnprintf(nullptr, 0, fmt, vargs);
  va_end(vargs);

  std::vector<char> buffer(logLen + 1); // +1 for null terminator
  va_start(vargs, fmt);
  // vsnprintf copy at most buf_size - 1 characters
  std::vsnprintf(buffer.data(), buffer.size(), fmt, vargs);
  va_end(vargs);

  // Emit the folly ERR log via the common path so glog still shows 'E' and
  // honors level/subsys filtering. Delegating also preserves the
  // ncclDebugNoWarn downgrade and the ncclLastError save consistently with
  // every other logging macro.
  ncclMetaDebugLog(
      NCCL_LOG_ERROR, flags, file, func, line, "%s", buffer.data());

  const std::string message{buffer.data()};
  // Capture the expensive native stack once and share it across all reporters.
  std::vector<std::string> stack;
  if (NCCL_SCUBA_STACK_TRACE_ON_ERROR_ENABLED) {
    stack = ::meta::comms::logger::captureNativeErrorStack();
  }
  // ncclGetLastError state is independent of Scuba error logging.
  ::meta::comms::logger::setLastError(message, stack);
  // Scuba error record: hidden, env-gated; skip when downgraded via
  // ncclDebugNoWarn.
  if (ncclDebugNoWarn == 0 && NCCL_SCUBA_LOG_ERROR_ENABLED) {
    ::meta::comms::logger::logErrorToScuba(
        message, static_cast<int>(code), ncclCodeToString(code), stack);
  }
}

void ncclSetMyThreadLoggingName(std::string_view name) {
  meta::comms::logger::initThreadMetaData(name);
}

/* Meta's logging function with separate file and func parameters.
 * Used by the VERSION, WARN, ERR, INFO, TRACE_CALL, and TRACE macros.
 * Unlike ncclDebugLog (which combines file/func into filefunc for OFI plugin
 * compatibility), this passes file and func separately to LogStreamProcessor
 * so that folly can correctly resolve log levels and categories.
 */

void ncclMetaDebugLog(
    ncclDebugLogLevel level,
    unsigned long flags,
    const char* file,
    const char* func,
    int line,
    const char* fmt,
    ...) {
  int gotLevel =
      COMPILER_ATOMIC_LOAD(&ncclDebugLevel, std::memory_order_acquire);

  if (ncclDebugNoWarn != 0 &&
      (level == NCCL_LOG_WARN || level == NCCL_LOG_ERROR)) {
    level = NCCL_LOG_INFO;
    flags = ncclDebugNoWarn;
  }

  // Save the last error (WARN) as a human readable string
  if (level == NCCL_LOG_WARN || level == NCCL_LOG_ERROR) {
    std::lock_guard<std::mutex> lock(ncclDebugMutex);
    va_list vargs;
    va_start(vargs, fmt);
    ncclDebugSaveLastError(fmt, vargs);
    va_end(vargs);
  }

  if (gotLevel >= 0 && (gotLevel < level || (flags & ncclDebugMask) == 0)) {
    return;
  }

  {
    std::lock_guard<std::mutex> lock(ncclDebugMutex);
    if (ncclDebugLevel < 0) {
      ncclDebugInit();
    }
    if (ncclDebugLevel < level || ((flags & ncclDebugMask) == 0)) {
      return;
    }
  }

  std::stringstream logStream;
  auto logLevel = folly::LogLevel::INFO;
  if (level == NCCL_LOG_WARN) {
    logLevel = folly::LogLevel::WARN;
  } else if (level == NCCL_LOG_INFO || level == NCCL_LOG_VERSION) {
    logLevel = folly::LogLevel::INFO;
  } else if (level == NCCL_LOG_TRACE) {
    logLevel = folly::LogLevel::DBG;
  } else if (level == NCCL_LOG_ERROR) {
    logLevel = folly::LogLevel::ERR;
  }

  size_t logLen = 0;
  va_list vargs;
  va_start(vargs, fmt);
  logLen += std::vsnprintf(nullptr, 0, fmt, vargs);
  va_end(vargs);

  std::vector<char> buffer(logLen + 1); // +1 for null terminator
  va_start(vargs, fmt);
  // vsnprintf copy at most buf_size - 1 characters
  std::vsnprintf(buffer.data(), buffer.size(), fmt, vargs);
  va_end(vargs);
  logStream << buffer.data();

  auto logStr = logStream.str();
  // logging to specified stdout/stderr/file
  folly::LogStreamProcessor(
      XLOG_GET_CATEGORY(),
      logLevel,
      file,
      line,
      func,
      folly::LogStreamProcessor::AppendType::APPEND)
          .stream()
      << logStr;
}

#endif
