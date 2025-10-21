/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "param.h"
#include "debug.h"

#include <algorithm>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <pthread.h>
#include <pwd.h>

#include "comms/utils/logger/Logger.h"
#include "comms/utils/logger/LoggingFormat.h"
#include "meta/analyzer/NCCLXCommsTracingServiceUtil.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/colltrace/CollTraceFunc.h"
#include "meta/colltrace/CollTraceLegacyHandle.h"
#include "comms/ctran/tracing/CollTraceWrapper.h"
#include "comms/utils/InitFolly.h"

#include "meta/algoconf/AlgoConfig.h"
#include "cuda_runtime_api.h"

std::once_flag initOnceFlag;

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

void initEnv() {
  std::call_once(initOnceFlag, [] {
    meta::comms::initFolly();
    ncclCvarInit();
    initNcclLogger();
    initLegacyColltraceForCtran();
    ncclx::NCCLXCommsTracingServiceUtil::startService();
    ncclx::algoconf::setupGlobalHints();
  });
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
