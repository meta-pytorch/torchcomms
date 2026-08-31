/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "paramUT.h"

#include <mutex>

#include <folly/Singleton.h>
#include <folly/init/Init.h>
#include <folly/synchronization/HazptrThreadPoolExecutor.h>

#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LoggerRuntime.h"

std::once_flag initOnceFlag;

void initFolly() {
  // Adapted from folly/init/Init.cpp
  // We can't use folly::init directly because:
  // - we don't have gflags
  // - Training stack already initialized the signal handler

  // Move from the registration phase to the "you can actually instantiate
  // things now" phase.
  folly::SingletonVault::singleton()->registrationComplete();

  // Set the default hazard pointer domain to use a thread pool executor
  // for asynchronous reclamation
  folly::enable_hazptr_thread_pool_executor();
}

void initEnv() {
  std::call_once(initOnceFlag, [] {
    initFolly();
    ncclCvarInit();
    meta::comms::logger::initCommLoggerRuntime();
  });
}
