// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/logger/SpdlogLogger.h"

#include <cstddef>
#include <memory>

#include <spdlog/async.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace meta::comms::logger {
namespace {

constexpr auto kLoggerName = "comms";
constexpr auto kLogPattern = "%L%m%d %H:%M:%S.%f %t %s:%#] %v";
constexpr size_t kAsyncQueueSize = 8192;
constexpr size_t kAsyncThreadCount = 1;

std::shared_ptr<spdlog::logger> createLogger() {
  if (!spdlog::thread_pool()) {
    spdlog::init_thread_pool(kAsyncQueueSize, kAsyncThreadCount);
  }

  auto logger =
      spdlog::create_async_nb<spdlog::sinks::stderr_color_sink_mt>(kLoggerName);
  logger->set_pattern(kLogPattern);
  return logger;
}

} // namespace

spdlog::logger& getSpdlogLogger() {
  static auto logger = createLogger();
  return *logger;
}

} // namespace meta::comms::logger
