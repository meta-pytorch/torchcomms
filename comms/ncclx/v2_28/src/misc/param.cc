/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "param.h"
#include "debug.h"
#include "env.h"

#include <algorithm>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <pthread.h>
#include <mutex>
#include <pwd.h>

#include "comms/utils/logger/Logger.h"
#include "comms/utils/logger/LoggingFormat.h"
#include "meta/analyzer/NCCLXCommsTracingServiceUtil.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/colltrace/CollTraceFunc.h"
#include "meta/colltrace/CollTraceLegacyHandle.h"
#include "comms/ctran/colltrace/CollTraceWrapper.h"
#include "comms/utils/cvars/nccl_baseline_adapter.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/InitFolly.h"

#include "meta/algoconf/AlgoConfig.h"
#include "cuda_runtime_api.h"

using namespace meta::comms::colltrace;

void initLegacyColltraceForCtran() {
  setCollTraceLegacyHandleFunc(
      [](CtranComm* comm,
         const std::vector<std::unique_ptr<OpElem>>& opElems,
         const KernelConfig& kernelConfig,
         const bool isLegacy) -> std::unique_ptr<ICollTraceHandle> {
        return std::make_unique<CollTraceLegacyHandle>(
            comm,
            ncclx::colltrace::collTraceAquireEventCtran(
                comm, opElems, kernelConfig, isLegacy),
            CollTraceLegacyHandle::HandleType::ctran);
      });
}

const char* userHomeDir() {
  struct passwd *pwUser = getpwuid(getuid());
  return pwUser == NULL ? NULL : pwUser->pw_dir;
}

void setEnvFile(const char* fileName) {
  FILE * file = fopen(fileName, "r");
  if (file == NULL) return;

  char *line = NULL;
  char envVar[1024];
  char envValue[1024];
  size_t n = 0;
  ssize_t read;
  while ((read = getline(&line, &n, file)) != -1) {
    if (line[0] == '#') continue;
    if (line[read-1] == '\n') line[read-1] = '\0';
    int s=0; // Env Var Size
    while (line[s] != '\0' && line[s] != '=') s++;
    if (line[s] == '\0') continue;
    strncpy(envVar, line, std::min(1023,s));
    envVar[std::min(1023,s)] = '\0';
    s++;
    strncpy(envValue, line+s, 1023);
    envValue[1023]='\0';
    setenv(envVar, envValue, 0);
    //printf("%s : %s->%s\n", fileName, envVar, envValue);
  }
  if (line) free(line);
  fclose(file);
}

static void initEnvFunc() {
  char confFilePath[1024];
  const char* userFile = getenv("NCCL_CONF_FILE");
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
    initLegacyColltraceForCtran();
    ncclx::NCCLXCommsTracingServiceUtil::startService();
    ncclx::algoconf::setupGlobalHints();
  });
}

void ncclLoadParam(char const* env, int64_t deftVal, int64_t uninitialized, int64_t* cache) {
  nccl_baseline_adapter::ncclLoadParam(env, deftVal, uninitialized, cache);
}

const char* ncclGetEnvStr(std::string_view name) {
  ncclInitEnv();
  return nccl_baseline_adapter::ncclGetEnvImpl(name.data());
}

void initNcclLogger() {
  NcclLogger::init(NcclLoggerInitConfig{
    .contextName = "comms.ncclx",
    .logPrefix = "NCCL",
    .logFilePath = meta::comms::logger::parseDebugFile(NCCL_DEBUG_FILE.c_str()),
    .logLevel = meta::comms::logger::loggerLevelToFollyLogLevel(
        meta::comms::logger::getLoggerDebugLevel(NCCL_DEBUG)),
    .threadContextFn = []() {
      int cudaDev = -1;
      cudaGetDevice(&cudaDev);
      return cudaDev;
    }});
    // Init logging for NCCL header inside meta directory.
    // This is due to the buck2 behavior of copying the header files to the
    // buck-out directory.
    // For logging in src/include headers, they are using NCCL logging
    // (INFO/WARN/ERROR) which will inherit the loggging category from debug.cc
    NcclLogger::init(NcclLoggerInitConfig{
      .contextName = "meta",
      .logPrefix = "NCCL",
      .logFilePath = meta::comms::logger::parseDebugFile(NCCL_DEBUG_FILE.c_str()),
      .logLevel = meta::comms::logger::loggerLevelToFollyLogLevel(
          meta::comms::logger::getLoggerDebugLevel(NCCL_DEBUG)),
      .threadContextFn = []() {
        int cudaDev = -1;
        cudaGetDevice(&cudaDev);
        return cudaDev;
      }});
}
