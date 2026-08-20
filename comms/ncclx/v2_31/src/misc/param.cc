/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "param.h"
#include "param/param.h"
#include "debug.h"
#include "env.h"

#include <algorithm>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <mutex>
#include <unordered_set>
#include "os.h"

#include "cuda_runtime_api.h"

#include <folly/synchronization/CallOnce.h>

#include "comms/utils/InitFolly.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"
#include "comms/utils/logger/LoggingFormat.h"
#include "meta/NcclxLogger.h"

namespace {

spdlog::level::level_enum loggerLevelToSpdlogLevel(meta::comms::logger::LogLevel level) {
  switch (level) {
  case meta::comms::logger::LogLevel::NONE:
  case meta::comms::logger::LogLevel::VERSION:
    return spdlog::level::off;
  case meta::comms::logger::LogLevel::ERROR:
    return spdlog::level::err;
  case meta::comms::logger::LogLevel::WARN:
    return spdlog::level::warn;
  case meta::comms::logger::LogLevel::INFO:
    return spdlog::level::info;
  case meta::comms::logger::LogLevel::ABORT:
  case meta::comms::logger::LogLevel::TRACE:
    return spdlog::level::debug;
  }
  return spdlog::level::off;
}

} // namespace

const char* userHomeDir() {
  return getenv("HOME");
}

void setEnvFile(const char* fileName) {
  FILE* file = fopen(fileName, "r");
  if (file == NULL) return;

  char line[4096];
  char envVar[1024];
  char envValue[1024];
  while (fgets(line, (int)sizeof(line), file) != NULL) {
    size_t len = strlen(line);
    if (len > 0 && line[len - 1] == '\n') line[--len] = '\0';
    if (len > 0 && line[len - 1] == '\r') line[--len] = '\0';
    if (line[0] == '#') continue;
    int s = 0;
    while (line[s] != '\0' && line[s] != '=') s++;
    if (line[s] == '\0') continue;
    strncpy(envVar, line, std::min(1023, s));
    envVar[std::min(1023, s)] = '\0';
    s++;
    strncpy(envValue, line + s, 1023);
    envValue[1023] = '\0';
    ncclOsSetEnv(envVar, envValue);
  }
  fclose(file);
}

static void initEnvFunc() {
  char confFilePath[1024];
  const char* userFile = std::getenv("NCCL_CONF_FILE");
  if (userFile && strlen(userFile) > 0) {
    snprintf(confFilePath, sizeof(confFilePath), "%s", userFile);
    setEnvFile(confFilePath);
  } else {
    const char* userDir = userHomeDir();
    if (userDir) {
      snprintf(confFilePath, sizeof(confFilePath), "%s/.nccl.conf", userDir);
      setEnvFile(confFilePath);
    }
  }
  snprintf(confFilePath, sizeof(confFilePath), "/etc/nccl.conf");
  setEnvFile(confFilePath);
}

void initEnv() {
  // folly::call_once, not std::call_once: initFolly()/ncclCvarInit() can throw,
  // and std::call_once deadlocks rather than retrying on a throwing callable.
  static folly::once_flag once;
  folly::call_once(once, [] {
    meta::comms::initFolly();
    ncclCvarInit();
    // As early as the sink can be registered: it is configured from the
    // NCCL_DEBUG* cvars, so it cannot precede ncclCvarInit(). Anything logged
    // before this point goes to an unconfigured logger and is lost, so keep
    // initEnvFunc() -- which reads conf files and can warn -- after it.
    initNcclLogger();
    initEnvFunc();
  });
}

static void ncclGetCachePolicy(char const* env, int8_t* noCache) {
  *noCache = ncclParamIsCacheDisabled(env) ? /*noCache*/ 1 : /*cache*/ 0;
}

int64_t ncclLoadParam(char const* env, int64_t deftVal, int64_t uninitialized, int64_t* cache, int8_t* noCache) {
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);

  // noCache is only load/stored within the mutex, no need for atomic
  if (*noCache == /*uninitialized*/ -1) ncclGetCachePolicy(env, noCache);

  if (COMPILER_ATOMIC_LOAD(cache, std::memory_order_relaxed) != uninitialized) {
    return COMPILER_ATOMIC_LOAD(cache, std::memory_order_relaxed);
  }

  // Read the environment variable
  const char* str = ncclGetEnv(env);
  int64_t value = deftVal;

  if (str && strlen(str) > 0) {
    errno = 0;
    value = strtoll(str, nullptr, 0);
    if (errno) {
      value = deftVal;
      INFO(NCCL_ALL, "Invalid value %s for %s, using default %lld.", str, env, (long long)deftVal);
    } else {
      INFO(NCCL_ENV, "%s set by environment to %lld.", env, (long long)value);
    }
  }

  if (*noCache == /*cache*/ 0) COMPILER_ATOMIC_STORE(cache, value, std::memory_order_relaxed);
  return value;
}

const char* ncclGetEnv(const char* name) {
  ncclInitEnv();
  return ncclEnvPluginGetEnv(name);
}

void initNcclLogger() {
  // Shared CollTrace code still emits raw XLOG under comms.utils.
  meta::comms::logger::initCommLogging();
  meta::comms::logger::configureSpdlogLogger(
    ncclx::logging::kNcclxLoggerName, "NCCL", meta::comms::logger::parseDebugFile(NCCL_DEBUG_FILE.c_str()),
    []() {
      int cudaDev = -1;
      (void)cudaGetDevice(&cudaDev);
      return cudaDev;
    },
    [](std::string_view message) { meta::comms::logger::setLastError(std::string{message}, {}); },
    NCCL_DEBUG_LOGGING_ASYNC);
  meta::comms::logger::getSpdlogLogger(ncclx::logging::kNcclxLoggerName)
    .set_level(loggerLevelToSpdlogLevel(meta::comms::logger::getLoggerDebugLevel(NCCL_DEBUG)));
}
