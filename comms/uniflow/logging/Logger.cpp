// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/logging/Logger.h"

#include <spdlog/async.h>
#include <spdlog/cfg/env.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace uniflow::logging {

namespace {
constexpr const char* kLoggerName = "uniflow";
constexpr const char* kLogPattern = "%L%m%d %H:%M:%S.%f %t %s:%#] %v";
constexpr size_t kAsyncQueueSize = 8192;
constexpr size_t kAsyncThreadCount = 1;

// Creates the async non-blocking logger. Called exactly once via
// function-local static in getLogger().
std::shared_ptr<spdlog::logger> createLogger() {
  // Async non-blocking: 1 bg thread, 8192-slot lock-free ring buffer.
  // overrun_oldest policy — drops oldest message when full, never blocks.
  if (!spdlog::thread_pool()) {
    spdlog::init_thread_pool(kAsyncQueueSize, kAsyncThreadCount);
  }
  auto logger =
      spdlog::create_async_nb<spdlog::sinks::stderr_color_sink_mt>(kLoggerName);

  // To switch to synchronous logging (no background thread), replace the
  // two lines above with:
  // auto logger = spdlog::stderr_color_mt(kLoggerName);

  logger->set_pattern(kLogPattern);

  // Apply SPDLOG_LEVEL env var (e.g., SPDLOG_LEVEL="uniflow=debug").
  spdlog::cfg::load_env_levels();
  return logger;
}
} // namespace

// Thread-safe lazy initialization via C++11 "magic static".
spdlog::logger* getLogger() {
  static auto logger = createLogger();
  return logger.get();
}

} // namespace uniflow::logging
