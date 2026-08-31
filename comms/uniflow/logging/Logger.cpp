// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/logging/Logger.h"

#include <memory>

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

/*
 * Thread-safe lazy initialization via C++11 "magic static", deliberately
 * leaked. A process that exits without joining its worker threads leaves them
 * logging while __cxa_atexit runs; a static shared_ptr would destroy the
 * logger and its sink out from under them and hand every later caller a
 * dangling pointer. Leaking behind a pointer registers no destructor, so the
 * logger outlives every thread that can reach it. Same reasoning as
 * comms/utils/logger/SpdlogLogger.cc.
 *
 * spdlog's registry still stops the shared thread pool at exit, so async
 * delivery stops working at that point. Messages are then dropped rather than
 * written through a destroyed sink: async_logger sees an expired weak_ptr to
 * the pool and raises, which spdlog catches and routes to the logger's error
 * handler -- by default a stderr line rate-limited to one per second, so the
 * drop is reported but the record itself is lost.
 */
spdlog::logger* getLogger() {
  static auto* logger = new std::shared_ptr<spdlog::logger>{createLogger()};
  return logger->get();
}

namespace testing {

/*
 * Reach spdlog's global registry from this library's own translation unit. A
 * test cannot call spdlog::shutdown() or spdlog::thread_pool() for itself:
 * spdlog is linked statically, so the test binary can hold its own copy of the
 * registry's function-local static and would tear down or inspect a registry
 * this logger never used -- passing without exercising anything.
 */
void shutdownRegistryForTesting() {
  ::spdlog::shutdown();
}

bool globalThreadPoolAliveForTesting() {
  return ::spdlog::thread_pool() != nullptr;
}

} // namespace testing

} // namespace uniflow::logging
