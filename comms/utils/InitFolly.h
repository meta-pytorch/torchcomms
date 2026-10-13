// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdlib>

#include <folly/Singleton.h>
#include <folly/init/Init.h>
#include <folly/logging/Init.h>
#include <folly/synchronization/HazptrThreadPoolExecutor.h>

namespace meta::comms {

// Initializes the folly linked into the calling shared library. libnccl,
// libmccl and libcommsutils each carry their own statically linked, hidden
// folly, so each runs this once on its own copy; header-only and hidden so
// that no library can ever resolve another's.
//
// Adapted from folly/init/Init.cpp. folly::init itself is not usable here:
// there are no gflags to parse, and the launcher already installed the signal
// handlers.
__attribute__((visibility("hidden"))) inline void initFolly() {
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
