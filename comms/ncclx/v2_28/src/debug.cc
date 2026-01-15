/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "nccl_net.h"
#include <ctime>
#include <stdlib.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <sys/syscall.h>
#include <chrono>
#include "param.h"
#include <mutex>
#include "env.h"

#include <cstdio>
#include <vector>
#include <sstream>
#include <folly/logging/LogLevel.h>
#include <folly/logging/LogStreamProcessor.h>
#include <folly/logging/xlog.h>

#include "comms/utils/logger/LogUtils.h"
#include "comms/utils/logger/LoggingFormat.h"
#include "comms/ctran/utils/ErrorStackTraceUtil.h"

#define NCCL_DEBUG_RESET_TRIGGERED (-2)

int ncclDebugLevel = -1;
static uint32_t ncclDebugTimestampLevels = 0;     // bitmaps of levels that have timestamps turned on
static char ncclDebugTimestampFormat[256];        // with space for subseconds
static int ncclDebugTimestampSubsecondsStart;     // index where the subseconds starts
static uint64_t ncclDebugTimestampMaxSubseconds;  // Max number of subseconds plus 1, used in duration ratio
static int ncclDebugTimestampSubsecondDigits;     // Number of digits to display
static int pid = -1;
static char hostname[1024];
thread_local int ncclDebugNoWarn = 0;
char ncclLastError[1024] = ""; // Global string for the last error in human readable form
uint64_t ncclDebugMask = 0;
FILE *ncclDebugFile = stdout;
static std::mutex ncclDebugMutex;
static std::chrono::steady_clock::time_point ncclEpoch;
static bool ncclWarnSetDebugInfo = false;

typedef const char* (*ncclGetEnvFunc_t)(const char*);

// This function must be called with ncclDebugLock locked!
static void ncclDebugInit() {
  bool envPluginInitialized = ncclEnvPluginInitialized();
  const char* nccl_debug;
  if (envPluginInitialized) {
    nccl_debug = ncclGetEnv("NCCL_DEBUG");
  } else {
    nccl_debug = getenv("NCCL_DEBUG");
  }
  int tempNcclDebugLevel = -1;
  uint64_t tempNcclDebugMask = NCCL_INIT | NCCL_BOOTSTRAP | NCCL_ENV; // Default debug sub-system mask
  if (ncclDebugLevel == NCCL_DEBUG_RESET_TRIGGERED && ncclDebugFile != stdout) {
    // Finish the reset initiated via ncclResetDebugInit().
    fclose(ncclDebugFile);
    ncclDebugFile = stdout;
  }

  if (nccl_debug == NULL) {
    tempNcclDebugLevel = NCCL_LOG_NONE;
  } else if (strcasecmp(nccl_debug, "VERSION") == 0) {
    tempNcclDebugLevel = NCCL_LOG_VERSION;
  } else if (strcasecmp(nccl_debug, "WARN") == 0) {
    tempNcclDebugLevel = NCCL_LOG_WARN;
  } else if (strcasecmp(nccl_debug, "INFO") == 0) {
    tempNcclDebugLevel = NCCL_LOG_INFO;
  } else if (strcasecmp(nccl_debug, "ABORT") == 0) {
    tempNcclDebugLevel = NCCL_LOG_ABORT;
  } else if (strcasecmp(nccl_debug, "TRACE") == 0) {
    tempNcclDebugLevel = NCCL_LOG_TRACE;
  }

  /* Parse the NCCL_DEBUG_SUBSYS env var
   * This can be a comma separated list such as INIT,COLL
   * or ^INIT,COLL etc
   */
  const char* ncclDebugSubsysEnv;
  if (envPluginInitialized) {
    ncclDebugSubsysEnv = ncclGetEnv("NCCL_DEBUG_SUBSYS");
  } else {
    ncclDebugSubsysEnv = getenv("NCCL_DEBUG_SUBSYS");
  }
  if (ncclDebugSubsysEnv != NULL) {
    int invert = 0;
    if (ncclDebugSubsysEnv[0] == '^') { invert = 1; ncclDebugSubsysEnv++; }
    tempNcclDebugMask = invert ? ~0ULL : 0ULL;
    char *ncclDebugSubsys = strdup(ncclDebugSubsysEnv);
    char *subsys = strtok(ncclDebugSubsys, ",");
    while (subsys != NULL) {
      uint64_t mask = 0;
      if (strcasecmp(subsys, "INIT") == 0) {
        mask = NCCL_INIT;
      } else if (strcasecmp(subsys, "COLL") == 0) {
        mask = NCCL_COLL;
      } else if (strcasecmp(subsys, "P2P") == 0) {
        mask = NCCL_P2P;
      } else if (strcasecmp(subsys, "SHM") == 0) {
        mask = NCCL_SHM;
      } else if (strcasecmp(subsys, "NET") == 0) {
        mask = NCCL_NET;
      } else if (strcasecmp(subsys, "GRAPH") == 0) {
        mask = NCCL_GRAPH;
      } else if (strcasecmp(subsys, "TUNING") == 0) {
        mask = NCCL_TUNING;
      } else if (strcasecmp(subsys, "ENV") == 0) {
        mask = NCCL_ENV;
      } else if (strcasecmp(subsys, "ALLOC") == 0) {
        mask = NCCL_ALLOC;
      } else if (strcasecmp(subsys, "CALL") == 0) {
        mask = NCCL_CALL;
      } else if (strcasecmp(subsys, "PROXY") == 0) {
        mask = NCCL_PROXY;
      } else if (strcasecmp(subsys, "NVLS") == 0) {
        mask = NCCL_NVLS;
      } else if (strcasecmp(subsys, "BOOTSTRAP") == 0) {
        mask = NCCL_BOOTSTRAP;
      } else if (strcasecmp(subsys, "REG") == 0) {
        mask = NCCL_REG;
      } else if (strcasecmp(subsys, "PROFILE") == 0) {
        mask = NCCL_PROFILE;
      } else if (strcasecmp(subsys, "RAS") == 0) {
        mask = NCCL_RAS;
      } else if (strcasecmp(subsys, "ALL") == 0) {
        mask = NCCL_ALL;
      }
      if (mask) {
        if (invert) tempNcclDebugMask &= ~mask; else tempNcclDebugMask |= mask;
      }
      subsys = strtok(NULL, ",");
    }
    free(ncclDebugSubsys);
  }

  const char* ncclWarnSetDebugInfoEnv;
  if (envPluginInitialized) {
    ncclWarnSetDebugInfoEnv = ncclGetEnv("NCCL_WARN_ENABLE_DEBUG_INFO");
  } else {
    ncclWarnSetDebugInfoEnv = getenv("NCCL_WARN_ENABLE_DEBUG_INFO");
  }
  if (ncclWarnSetDebugInfoEnv != NULL && strlen(ncclWarnSetDebugInfoEnv) > 0) {
    int64_t value;
    errno = 0;
    value = strtoll(ncclWarnSetDebugInfoEnv, NULL, 0);
    if (!errno)
      ncclWarnSetDebugInfo = value;
  }

  // Determine which debug levels will have timestamps.
  const char* timestamps;
  if (envPluginInitialized) {
    timestamps = ncclGetEnv("NCCL_DEBUG_TIMESTAMP_LEVELS");
  } else {
    timestamps = getenv("NCCL_DEBUG_TIMESTAMP_LEVELS");
  }
  if (timestamps == nullptr) {
    ncclDebugTimestampLevels = (1<<NCCL_LOG_WARN);
  } else {
    int invert = 0;
    if (timestamps[0] == '^') { invert = 1; ++timestamps; }
    ncclDebugTimestampLevels = invert ? ~0U : 0U;
    char *timestampsDup = strdup(timestamps);
    char *level = strtok(timestampsDup, ",");
    while (level != NULL) {
      uint32_t mask = 0;
      if (strcasecmp(level, "ALL") == 0) {
        mask = ~0U;
      } else if (strcasecmp(level, "VERSION") == 0) {
        mask = (1<<NCCL_LOG_VERSION);
      } else if (strcasecmp(level, "WARN") == 0) {
        mask = (1<<NCCL_LOG_WARN);
      } else if (strcasecmp(level, "INFO") == 0) {
        mask = (1<<NCCL_LOG_INFO);
      } else if (strcasecmp(level, "ABORT") == 0) {
        mask = (1<<NCCL_LOG_ABORT);
      } else if (strcasecmp(level, "TRACE") == 0) {
        mask = (1<<NCCL_LOG_TRACE);
      } else {
        // Silently fail.
      }
      if (mask) {
        if (invert) ncclDebugTimestampLevels &= ~mask;
        else ncclDebugTimestampLevels |= mask;
      }
      level = strtok(NULL, ",");
    }
    free(timestampsDup);
  }

  // Store a copy of the timestamp format with space for the subseconds, if used.
  const char* tsFormat;
  if (envPluginInitialized) {
    tsFormat = ncclGetEnv("NCCL_DEBUG_TIMESTAMP_FORMAT");
  } else {
    tsFormat = getenv("NCCL_DEBUG_TIMESTAMP_FORMAT");
  }
  if (tsFormat == nullptr) tsFormat = "[%F %T] ";
  ncclDebugTimestampSubsecondsStart = -1;
  // Find where the subseconds are in the format.
  for (int i=0; tsFormat[i] != '\0'; ++i) {
    if (tsFormat[i]=='%' && tsFormat[i+1]=='%') { // Next two chars are "%"
      // Skip the next character, too, and restart checking after that.
      ++i;
      continue;
    }
    if (tsFormat[i]=='%' &&                               // Found a percentage
        ('1' <= tsFormat[i+1] && tsFormat[i+1] <= '9') && // Next char is a digit between 1 and 9 inclusive
        tsFormat[i+2]=='f'                                // Two characters later is an "f"
        ) {
      constexpr int replaceLen = sizeof("%Xf") - 1;
      ncclDebugTimestampSubsecondDigits = tsFormat[i+1] - '0';
      if (ncclDebugTimestampSubsecondDigits + strlen(tsFormat) - replaceLen > sizeof(ncclDebugTimestampFormat) - 1) {
        // Won't fit; fall back on the default.
        break;
      }
      ncclDebugTimestampSubsecondsStart = i;
      ncclDebugTimestampMaxSubseconds = 1;

      memcpy(ncclDebugTimestampFormat, tsFormat, i);
      for (int j=0; j<ncclDebugTimestampSubsecondDigits; ++j) {
        ncclDebugTimestampFormat[i+j] = ' ';
        ncclDebugTimestampMaxSubseconds *= 10;
      }
      strcpy(ncclDebugTimestampFormat+i+ncclDebugTimestampSubsecondDigits, tsFormat+i+replaceLen);
      break;
    }
  }
  if (ncclDebugTimestampSubsecondsStart == -1) {
    if (strlen(tsFormat) < sizeof(ncclDebugTimestampFormat)) {
      strcpy(ncclDebugTimestampFormat, tsFormat);
    } else {
      strcpy(ncclDebugTimestampFormat, "[%F %T] ");
    }
  }

  // Replace underscore with spaces... it is hard to put spaces in command line parameters.
  for (int i=0; ncclDebugTimestampFormat[i] != '\0'; ++i) {
    if (ncclDebugTimestampFormat[i]=='_') ncclDebugTimestampFormat[i] = ' ';
  }

  // Cache pid and hostname
  getHostName(hostname, 1024, '.');
  pid = getpid();

  /* Parse and expand the NCCL_DEBUG_FILE path and
   * then create the debug file. But don't bother unless the
   * NCCL_DEBUG level is > VERSION
   */
  const char* ncclDebugFileEnv;
  if (envPluginInitialized) {
    ncclDebugFileEnv = ncclGetEnv("NCCL_DEBUG_FILE");
  } else {
    ncclDebugFileEnv = getenv("NCCL_DEBUG_FILE");
  }
  if (tempNcclDebugLevel > NCCL_LOG_VERSION && ncclDebugFileEnv != NULL) {
    int c = 0;
    char debugFn[PATH_MAX+1] = "";
    char *dfn = debugFn;
    while (ncclDebugFileEnv[c] != '\0' && (dfn - debugFn) < PATH_MAX) {
      if (ncclDebugFileEnv[c++] != '%') {
        *dfn++ = ncclDebugFileEnv[c-1];
        continue;
      }
      switch (ncclDebugFileEnv[c++]) {
        case '%': // Double %
          *dfn++ = '%';
          break;
        case 'h': // %h = hostname
          dfn += snprintf(dfn, PATH_MAX + 1 - (dfn - debugFn), "%s", hostname);
          break;
        case 'p': // %p = pid
          dfn += snprintf(dfn, PATH_MAX + 1 - (dfn - debugFn), "%d", pid);
          break;
        default: // Echo everything we don't understand
          *dfn++ = '%';
          if ((dfn - debugFn) < PATH_MAX) {
            *dfn++ = ncclDebugFileEnv[c-1];
          }
          break;
      }
      if ((dfn - debugFn) > PATH_MAX) {
        // snprintf wanted to overfill the buffer: set dfn to the end
        // of the buffer (for null char) and it will naturally exit
        // the loop.
        dfn = debugFn + PATH_MAX;
      }
    }
    *dfn = '\0';
    if (debugFn[0] != '\0') {
      FILE *file = fopen(debugFn, "w");
      if (file != nullptr) {
        setlinebuf(file); // disable block buffering
        ncclDebugFile = file;
      }
    }
  }

  ncclEpoch = std::chrono::steady_clock::now();
  ncclDebugMask = tempNcclDebugMask;

  // NCCLX -> Enable CTRAN subsystems logging as per NCCL_DEBUG_SUBSYS
  meta::comms::logger::setSubSystemMask(ncclDebugMask);

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
  ncclDebugLog(level, flags, file, func, line, "%s", buffer);
}

/* Common logging function used by the INFO, WARN and TRACE macros
 * Also exported to the dynamically loadable Net transport modules so
 * they can share the debugging mechanisms and output files
 */
void ncclDebugLog(ncclDebugLogLevel level, unsigned long flags, const char *file, const char *func, int line, const char *fmt, ...) {
  int gotLevel = __atomic_load_n(&ncclDebugLevel, __ATOMIC_ACQUIRE);

  if (ncclDebugNoWarn != 0 && (level == NCCL_LOG_WARN || level == NCCL_LOG_ERROR)) { level = NCCL_LOG_INFO; flags = ncclDebugNoWarn; }

  // Save the last error (WARN) as a human readable string
  if (level == NCCL_LOG_WARN || level == NCCL_LOG_ERROR) {
    std::lock_guard<std::mutex> lock(ncclDebugMutex);
    va_list vargs;
    va_start(vargs, fmt);
    (void) vsnprintf(ncclLastError, sizeof(ncclLastError), fmt, vargs);
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

// Non-deprecated version for internal use.
extern "C"
__attribute__ ((visibility("default")))
void ncclResetDebugInitInternal() {
  // Cleans up from a previous ncclDebugInit() and reruns.
  // Use this after changing NCCL_DEBUG and related parameters in the environment.
  std::lock_guard<std::mutex> lock(ncclDebugMutex);
  // Let ncclDebugInit() know to complete the reset.
  __atomic_store_n(&ncclDebugLevel, NCCL_DEBUG_RESET_TRIGGERED, __ATOMIC_RELEASE);
}

// In place of: NCCL_API(void, ncclResetDebugInit);
__attribute__ ((visibility("default")))
__attribute__ ((alias("ncclResetDebugInit")))
void pncclResetDebugInit();
extern "C"
__attribute__ ((visibility("default")))
__attribute__ ((weak))
__attribute__ ((deprecated("ncclResetDebugInit is not supported as part of the NCCL API and will be removed in the future")))
void ncclResetDebugInit();


void ncclResetDebugInit() {
  // This is now deprecated as part of the NCCL API. It will be removed
  // from the API in the future. It is still available as an
  // exported symbol.
  ncclResetDebugInitInternal();
}

NCCL_PARAM(SetThreadName, "SET_THREAD_NAME", 0);

void ncclSetThreadName(pthread_t thread, const char *fmt, ...) {
  // pthread_setname_np is nonstandard GNU extension
  // needs the following feature test macro
#ifdef _GNU_SOURCE
  if (ncclParamSetThreadName() != 1) return;
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
