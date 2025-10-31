/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <chrono>

#include <folly/logging/LogLevel.h>
#include <folly/logging/LogStreamProcessor.h>
#include <fmt/printf.h>

#include "core.h"

#include "comms/utils/logger/LogUtils.h"
#include "comms/utils/logger/LoggingFormat.h"
#include "comms/ctran/utils/ErrorStackTraceUtil.h"
#include "comms/utils/cvars/nccl_cvars.h"

int ncclDebugLevel = -1;
thread_local int ncclDebugNoWarn = 0;
char ncclLastError[1024] = ""; // Global string for the last error in human readable form
static uint64_t ncclDebugMask = NCCL_INIT | NCCL_BOOTSTRAP | NCCL_ENV; // Default debug sub-system mask is INIT and ENV
std::string ncclDebugLogFileStr;
std::mutex ncclDebugLock;
std::chrono::steady_clock::time_point ncclEpoch;

void ncclMetaDebugInit() {
  std::lock_guard<std::mutex> guard(ncclDebugLock);

  if (ncclDebugLevel != -1) {
    return;
  }
  int tempNcclDebugLevel = static_cast<int>(meta::comms::logger::getLoggerDebugLevel(NCCL_DEBUG));

  if (!NCCL_DEBUG_SUBSYS.empty()) {
    ncclDebugMask = meta::comms::logger::parseDebugSubsysMask(NCCL_DEBUG_SUBSYS.c_str());
    // NCCLX -> Enable CTRAN subsystems logging as per NCCL_DEBUG_SUBSYS
    meta::comms::logger::setSubSystemMask(ncclDebugMask);
  }

  /* Parse and expand the NCCL_DEBUG_FILE path and
   * then create the debug file. But don't bother unless the
   * NCCL_DEBUG level is > VERSION
   */
   if (tempNcclDebugLevel > NCCL_LOG_VERSION && !NCCL_DEBUG_FILE.empty()) {
    ncclDebugLogFileStr = meta::comms::logger::parseDebugFile(NCCL_DEBUG_FILE.c_str());
  }

  ncclEpoch = std::chrono::steady_clock::now();
  __atomic_store_n(&ncclDebugLevel, tempNcclDebugLevel, __ATOMIC_RELEASE);
}

void ncclMetaDebugLogWithScuba(ncclDebugLogLevel level, unsigned long flags, const char *file, const char *func, int line, const char *fmt, ...) {
  char buffer[256];
  va_list vargs;
  va_start(vargs, fmt);
  (void) vsnprintf(buffer, sizeof(buffer), fmt, vargs);
  va_end(vargs);
  ::meta::comms::logger::appendErrorToStack(std::string{buffer});
  ErrorStackTraceUtil::logErrorMessage(std::string{buffer});
  ncclMetaDebugLog(level, flags, file, func, line, "%s", buffer);
}

/* Common logging function used by the INFO, WARN and TRACE macros
 * Also exported to the dynamically loadable Net transport modules so
 * they can share the debugging mechanisms and output files
 */
void ncclMetaDebugLog(ncclDebugLogLevel level, unsigned long flags, const char *file, const char *func, int line, const char *fmt, ...) {
  if (__atomic_load_n(&ncclDebugLevel, __ATOMIC_ACQUIRE) == -1) ncclMetaDebugInit();
  if (ncclDebugNoWarn != 0 && (level == NCCL_LOG_WARN || level == NCCL_LOG_ERROR)) { level = NCCL_LOG_INFO; flags = ncclDebugNoWarn; }

  // Save the last error (WARN) as a human readable string
  if (level == NCCL_LOG_WARN || level == NCCL_LOG_ERROR) {
    std::lock_guard<std::mutex> guard(ncclDebugLock);
    va_list vargs;
    va_start(vargs, fmt);
    (void) vsnprintf(ncclLastError, sizeof(ncclLastError), fmt, vargs);
    va_end(vargs);
  }
  if (ncclDebugLevel < level || ((flags & ncclDebugMask) == 0)) return;

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

NCCL_API(void, ncclResetDebugInit);
void ncclResetDebugInit() {
  // Cleans up from a previous ncclMetaDebugInit() and reruns.
  // Use this after changing NCCL_DEBUG and related parameters in the environment.
  __atomic_load_n(&ncclDebugLevel, __ATOMIC_ACQUIRE);
  ncclDebugLevel = -1;
  ncclMetaDebugInit();
}

void ncclSetThreadName(pthread_t thread, const char *fmt, ...) {
  // pthread_setname_np is nonstandard GNU extension
  // needs the following feature test macro
#ifdef _GNU_SOURCE
  if (NCCL_SET_THREAD_NAME != 1) return;
  char threadName[NCCL_THREAD_NAMELEN];
  va_list vargs;
  va_start(vargs, fmt);
  vsnprintf(threadName, NCCL_THREAD_NAMELEN, fmt, vargs);
  va_end(vargs);
  pthread_setname_np(thread, threadName);
#endif
}

void ncclSetMyThreadLoggingName(std::string_view name) {
  meta::comms::logger::initThreadMetaData(name);
}
