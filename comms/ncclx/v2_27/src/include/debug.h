/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_INT_DEBUG_H_
#define NCCL_INT_DEBUG_H_

#include "nccl.h"
#include "nccl_common.h"
#include <stdio.h>
#include <chrono>
#include <string.h>
#include <sstream>
#include <iomanip>
#include <pthread.h>
#include <string_view>

// Conform to pthread and NVTX standard
#define NCCL_THREAD_NAMELEN 16

extern int ncclDebugLevel;
extern std::string ncclDebugLogFileStr;

void ncclMetaDebugLog(ncclDebugLogLevel level, unsigned long flags, const char *file, const char *func, int line, const char *fmt, ...) __attribute__ ((format (printf, 6, 7)));

void ncclMetaDebugLogWithScuba(ncclDebugLogLevel level, unsigned long flags, const char *file, const char *func, int line, const char *fmt, ...) __attribute__ ((format (printf, 6, 7)));

void ncclMetaDebugInit();

// Let code temporarily downgrade WARN into INFO
extern thread_local int ncclDebugNoWarn;
extern char ncclLastError[];

#define VERSION(...) ncclMetaDebugLog(NCCL_LOG_VERSION, NCCL_ALL, __FILE__, __func__, __LINE__, __VA_ARGS__)
#define WARN(...) ncclMetaDebugLog(NCCL_LOG_WARN, NCCL_ALL, __FILE__, __func__, __LINE__, __VA_ARGS__)
#define INFO(FLAGS, ...) ncclMetaDebugLog(NCCL_LOG_INFO, (FLAGS), __FILE__, __func__, __LINE__, __VA_ARGS__)
#define TRACE_CALL(...) ncclMetaDebugLog(NCCL_LOG_TRACE, NCCL_CALL, __FILE__, __func__, __LINE__, __VA_ARGS__)
#define ERR(...) ncclMetaDebugLog(NCCL_LOG_ERROR, NCCL_ALL, __FILE__, __func__, __LINE__, __VA_ARGS__)

#define WARN_WITH_SCUBA(...) ncclMetaDebugLogWithScuba(NCCL_LOG_WARN, NCCL_ALL, __FILE__, __func__, __LINE__, __VA_ARGS__)
#define ERR_WITH_SCUBA(...) ncclMetaDebugLogWithScuba(NCCL_LOG_ERROR, NCCL_ALL, __FILE__, __func__, __LINE__, __VA_ARGS__)

#ifdef ENABLE_TRACE
#define TRACE(FLAGS, ...) ncclMetaDebugLog(NCCL_LOG_TRACE, (FLAGS), __func__, __LINE__, __VA_ARGS__)
#else
#define TRACE(...)
#endif

void ncclSetThreadName(pthread_t thread, const char *fmt, ...);

void ncclResetDebugInit();

void ncclSetMyThreadLoggingName(std::string_view name);

#define NCCL_NAMED_THREAD_START(threadName)       \
  do {                                            \
    ncclSetMyThreadLoggingName(threadName);       \
    INFO(                                         \
        NCCL_INIT,                                \
        "[NCCL THREAD] Starting %s thread at %s", \
        threadName,                               \
        __func__);                                \
  } while (0);

#define NCCL_NAMED_THREAD_START_EXT(threadName, rank, commHash, commDesc)              \
  do {                                                                                 \
    ncclSetMyThreadLoggingName(threadName);                                            \
    INFO(                                                                              \
        NCCL_INIT,                                                                     \
        "[NCCL THREAD] Starting %s thread for rank %d commHash %lx commDesc %s at %s", \
        threadName,                                                                    \
        rank,                                                                          \
        commHash,                                                                      \
        commDesc.c_str(),                                                              \
        __func__);                                                                     \
  } while (0);

inline std::string getTime(void) {
  auto now = std::chrono::system_clock::now();
  std::time_t now_c = std::chrono::system_clock::to_time_t(now);
  auto now_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    now.time_since_epoch()) %
      1000000;

  std::stringstream timeSs;
  struct tm nowTm;
  localtime_r(&now_c, &nowTm);
  timeSs << std::put_time(&nowTm, "%FT%T.") << std::setfill('0')
         << std::setw(6) << now_us.count();
  return timeSs.str();
}

#endif
