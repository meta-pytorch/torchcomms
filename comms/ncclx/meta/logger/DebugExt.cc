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

#include "debug.h"
#include "meta/logger/DebugExtInternal.h"

#include "comms/ctran/utils/ErrorStackTraceUtil.h"
#include "comms/utils/logger/LoggingFormat.h"

void ncclSetMyThreadLoggingName(std::string_view name) {
  meta::comms::logger::initThreadMetaData(name);
}

void ncclMetaDebugLogWithScuba(
    ncclDebugLogLevel level,
    unsigned long flags,
    const char* file,
    const char* func,
    int line,
    const char* fmt,
    ...) {
  char buffer[256];
  va_list vargs;
  va_start(vargs, fmt);
  (void)vsnprintf(buffer, sizeof(buffer), fmt, vargs);
  va_end(vargs);
  ::meta::comms::logger::appendErrorToStack(std::string{buffer});
  ErrorStackTraceUtil::logErrorMessage(std::string{buffer});
  ncclMetaDebugLog(level, flags, file, func, line, "%s", buffer);
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
