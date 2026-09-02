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
#include <atomic>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <mutex>
#include <unordered_set>
#include "os.h"

#include "comms/utils/logger/LoggingFormat.h"
#include "comms/utils/logger/LoggerRuntime.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/InitFolly.h"
#include "meta/NcclxLogger.h"

#include "cuda_runtime_api.h"

const char* userHomeDir() {
  return getenv("HOME");
}

void setEnvFile(const char* fileName) {
  FILE * file = fopen(fileName, "r");
  if (file == NULL) return;

  char line[4096];
  char envVar[1024];
  char envValue[1024];
  while (fgets(line, (int)sizeof(line), file) != NULL) {
    size_t len = strlen(line);
    if (len > 0 && line[len-1] == '\n') line[--len] = '\0';
    if (len > 0 && line[len-1] == '\r') line[--len] = '\0';
    if (line[0] == '#') continue;
    int s = 0;
    while (line[s] != '\0' && line[s] != '=') s++;
    if (line[s] == '\0') continue;
    strncpy(envVar, line, std::min(1023,s));
    envVar[std::min(1023,s)] = '\0';
    s++;
    strncpy(envValue, line+s, 1023);
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
  static std::once_flag once;
  std::call_once(once, [] {
    meta::comms::initFolly();
    ncclCvarInit();
    initEnvFunc();
    initNcclLogger();
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

  if (COMPILER_ATOMIC_LOAD(cache, std::memory_order_relaxed) != uninitialized) return COMPILER_ATOMIC_LOAD(cache, std::memory_order_relaxed);

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
  /*
   * The plugin is published only after its initialization callback succeeds.
   * Query it directly once published so logger reset during ncclInitEnv() does
   * not recursively enter the active call_once.
   */
  if (!ncclEnvPluginInitialized()) {
    ncclInitEnv();
  }
  return ncclEnvPluginGetEnv(name);
}

static const char* getNcclLoggerEnv(const char* name) {
  return ncclEnvPluginInitialized() ? ncclEnvPluginGetEnv(name)
                                    : std::getenv(name);
}

static std::atomic<bool> ncclLoggerIsInitialized{false};

bool ncclLoggerInitialized() noexcept {
  return ncclLoggerIsInitialized.load(std::memory_order_acquire);
}

void initNcclLogger(bool configureCommsLogger) noexcept {
  try {
    meta::comms::logger::initCommLoggerRuntime();
    const auto subSystemMask = meta::comms::logger::parseDebugSubsysMask(
        getNcclLoggerEnv("NCCL_DEBUG_SUBSYS"));
    const auto logFilePath = meta::comms::logger::parseDebugFile(
        getNcclLoggerEnv("NCCL_DEBUG_FILE"));
    const auto threadContextFn = []() {
      int cudaDev = -1;
      (void)cudaGetDevice(&cudaDev);
      return cudaDev;
    };
    const auto errorCallback = [](std::string_view message) {
      meta::comms::logger::setLastError(std::string{message}, {});
    };
    const auto* debugLevelValue = getNcclLoggerEnv("NCCL_DEBUG");
    const auto logLevel = meta::comms::logger::loggerLevelToSpdlogLevel(
        meta::comms::logger::getNcclLoggerDebugLevel(
            debugLevelValue == nullptr ? std::string_view{}
                                       : std::string_view{debugLevelValue}));
    const auto asyncLogging = meta::comms::logger::parseDebugLoggingAsync(
        getNcclLoggerEnv("NCCL_DEBUG_LOGGING_ASYNC"),
        NCCL_DEBUG_LOGGING_ASYNC);

    const auto configureLoggers = [&](std::string_view outputPath) {
      meta::comms::logger::configureCommsAndNamedSpdlogLoggers(
          ncclx::logging::kNcclxLoggerName,
          "NCCL",
          outputPath,
          threadContextFn,
          errorCallback,
          asyncLogging,
          logLevel,
          configureCommsLogger);
    };
    try {
      configureLoggers(logFilePath);
    } catch (const spdlog::spdlog_ex&) {
      /*
       * Keep the shared and NCCLX loggers on one destination. If either file
       * backend cannot be created, retry both on stdout rather than leaving a
       * partially configured split route.
       */
      configureLoggers({});
    }
    meta::comms::logger::setSubSystemMask(subSystemMask);
    ncclLoggerIsInitialized.store(true, std::memory_order_release);
  } catch (...) {
    meta::comms::logger::reportCommsLoggingFailureToStderr("ERROR");
  }
}
