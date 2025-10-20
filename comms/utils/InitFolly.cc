// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/InitFolly.h"

#include <cstdlib>

#include <folly/Singleton.h>
#include <folly/init/Init.h>
#include <folly/logging/Init.h>
#include <folly/synchronization/HazptrThreadPoolExecutor.h>

namespace meta::comms {

void initFolly() {
  // Adapted from folly/init/Init.cpp
  // We can't use folly::init directly because:
  // - we don't have gflags
  // - Launcher already initialized the signal handler

  // Move from the registration phase to the "you can actually instantiate
  // things now" phase.
  folly::SingletonVault::singleton()->registrationComplete();

  auto const follyLoggingEnv = std::getenv(folly::kLoggingEnvVarName);
  auto const follyLoggingEnvOr = follyLoggingEnv ? follyLoggingEnv : "";
  folly::initLoggingOrDie(follyLoggingEnvOr);

  // Set the default hazard pointer domain to use a thread pool executor
  // for asynchronous reclamation
  folly::enable_hazptr_thread_pool_executor();
}

} // namespace meta::comms
