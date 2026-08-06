// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/logger/SpdlogLogger.h"

#include <sys/syscall.h>
#include <unistd.h>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <spdlog/async.h>
#include <spdlog/pattern_formatter.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include "comms/utils/logger/CommsLogFormatter.h"

namespace meta::comms::logger {
namespace {

constexpr std::string_view kLoggerName = "comms";
constexpr size_t kAsyncQueueSize = 8192;
constexpr size_t kAsyncThreadCount = 1;
thread_local std::string threadName = "main";
// Guard the full callback chain so it cannot recurse through another context
// logger and eventually re-enter the originating callback.
thread_local bool errorCallbackInProgress = false;

class ErrorCallbackGuard {
 public:
  ErrorCallbackGuard() {
    errorCallbackInProgress = true;
  }
  ~ErrorCallbackGuard() {
    errorCallbackInProgress = false;
  }

  ErrorCallbackGuard(const ErrorCallbackGuard&) = delete;
  ErrorCallbackGuard& operator=(const ErrorCallbackGuard&) = delete;
  ErrorCallbackGuard(ErrorCallbackGuard&&) = delete;
  ErrorCallbackGuard& operator=(ErrorCallbackGuard&&) = delete;
};

class StderrRoutingSink final : public spdlog::sinks::sink {
 public:
  StderrRoutingSink()
      : sink_(std::make_shared<spdlog::sinks::stderr_color_sink_mt>()) {}

  void log(const spdlog::details::log_msg& message) override {
    if (should_log(message.level) &&
        shouldWriteCommsLogToStderr(message.level)) {
      sink_->log(message);
    }
  }

  void flush() override {
    sink_->flush();
  }

  void set_pattern(const std::string& pattern) override {
    sink_->set_pattern(pattern);
  }

  void set_formatter(std::unique_ptr<spdlog::formatter> formatter) override {
    sink_->set_formatter(std::move(formatter));
  }

 private:
  std::shared_ptr<spdlog::sinks::sink> sink_;
};

std::unique_ptr<spdlog::formatter> makeFormatter() {
  return std::make_unique<spdlog::pattern_formatter>(
      "%v", spdlog::pattern_time_type::local, "");
}

std::string getHostName() {
  char hostname[HOST_NAME_MAX + 1];
  if (gethostname(hostname, sizeof(hostname)) != 0) {
    return "unknown";
  }
  hostname[HOST_NAME_MAX] = '\0';
  if (auto* domain = std::strchr(hostname, '.')) {
    *domain = '\0';
  }
  return hostname;
}

std::string_view getSpdlogLevelName(spdlog::level::level_enum level) {
  switch (level) {
    case spdlog::level::trace:
    case spdlog::level::debug:
      return "VERBOSE";
    case spdlog::level::info:
      return "INFO";
    case spdlog::level::warn:
      return "WARN";
    case spdlog::level::err:
      return "ERROR";
    case spdlog::level::critical:
      return "CRITICAL";
    case spdlog::level::off:
    case spdlog::level::n_levels:
      return "UNKNOWN";
  }
  return "UNKNOWN";
}

std::string_view getBaseName(const char* filename) {
  if (filename == nullptr) {
    return {};
  }
  const auto* slash = std::strrchr(filename, '/');
  return slash == nullptr ? filename : slash + 1;
}

uint64_t getCurrentThreadId() {
  static thread_local const auto threadId =
      static_cast<uint64_t>(::syscall(SYS_gettid));
  return threadId;
}

std::shared_ptr<spdlog::logger> createLogger(std::string name) {
  if (!spdlog::thread_pool()) {
    spdlog::init_thread_pool(kAsyncQueueSize, kAsyncThreadCount);
  }

  auto logger = spdlog::create_async_nb<spdlog::sinks::stderr_color_sink_mt>(
      std::move(name));
  logger->set_formatter(makeFormatter());
  return logger;
}

} // namespace

bool shouldWriteCommsLogToStderr(spdlog::level::level_enum level) {
  return level >= spdlog::level::warn && level < spdlog::level::off;
}

CommsSpdlogLogger::CommsSpdlogLogger()
    : CommsSpdlogLogger(std::string{kLoggerName}) {}

CommsSpdlogLogger::CommsSpdlogLogger(std::string name)
    : logger_(createLogger(std::move(name))),
      outputSink_(
          std::make_shared<spdlog::sinks::dist_sink_mt>(logger_->sinks())) {
  logger_->sinks() = {outputSink_};
  storeConfiguration(std::make_shared<const Configuration>());
}

std::string_view CommsSpdlogLogger::getLevelName(
    spdlog::level::level_enum level) {
  return getSpdlogLevelName(level);
}

bool CommsSpdlogLogger::should_log(spdlog::level::level_enum level) const {
  return logger_->should_log(level);
}

const std::string& CommsSpdlogLogger::name() const {
  return logger_->name();
}

void CommsSpdlogLogger::set_level(spdlog::level::level_enum level) {
  logger_->set_level(level);
}

void CommsSpdlogLogger::flush() {
  logger_->flush();
}

void CommsSpdlogLogger::log(
    spdlog::source_loc location,
    spdlog::level::level_enum level,
    std::string_view message) {
  if (should_log(level)) {
    logFormatted(location, level, getLevelName(level), message, false);
  }
}

void CommsSpdlogLogger::logFatal(
    spdlog::source_loc location,
    std::string_view message) {
  logFormatted(location, spdlog::level::critical, "FATAL", message, true);
}

void CommsSpdlogLogger::configure(
    std::string prefix,
    std::function<int(void)> threadContextFn,
    std::function<void(std::string_view)> errorCallback) {
  if (!threadContextFn) {
    threadContextFn = []() { return 0; };
  }
  std::shared_ptr<const Configuration> configuration =
      std::make_shared<const Configuration>(Configuration{
          std::move(prefix),
          std::move(threadContextFn),
          std::move(errorCallback)});
  storeConfiguration(std::move(configuration));
}

std::shared_ptr<const CommsSpdlogLogger::Configuration>
CommsSpdlogLogger::loadConfiguration() const {
#if defined(__cpp_lib_atomic_shared_ptr) && \
    __cpp_lib_atomic_shared_ptr >= 201711L
  return configuration_.load(std::memory_order_acquire);
#else
  return std::atomic_load_explicit(&configuration_, std::memory_order_acquire);
#endif
}

void CommsSpdlogLogger::storeConfiguration(
    std::shared_ptr<const Configuration> configuration) {
#if defined(__cpp_lib_atomic_shared_ptr) && \
    __cpp_lib_atomic_shared_ptr >= 201711L
  configuration_.store(std::move(configuration), std::memory_order_release);
#else
  std::atomic_store_explicit(
      &configuration_, std::move(configuration), std::memory_order_release);
#endif
}

void CommsSpdlogLogger::configureOutput(std::string_view logFilePath) {
  std::vector<std::shared_ptr<spdlog::sinks::sink>> sinks;
  if (logFilePath.empty()) {
    sinks.push_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
  } else {
    const auto path = std::string{logFilePath};
    try {
      sinks.push_back(
          std::make_shared<spdlog::sinks::basic_file_sink_mt>(path, false));
    } catch (const spdlog::spdlog_ex& error) {
      throw spdlog::spdlog_ex(
          "Failed to open comms log file '" + path + "': " + error.what());
    }
    sinks.push_back(std::make_shared<StderrRoutingSink>());
  }
  for (auto& sink : sinks) {
    sink->set_formatter(makeFormatter());
  }
  outputSink_->set_sinks(std::move(sinks));
}

void CommsSpdlogLogger::logFormatted(
    spdlog::source_loc location,
    spdlog::level::level_enum level,
    std::string_view levelName,
    std::string_view message,
    bool bypassLevelGate) {
  static const auto hostname = getHostName();
  static const auto processId = getpid();
  const auto configuration = loadConfiguration();
  if (level >= spdlog::level::err && configuration->errorCallback &&
      !errorCallbackInProgress) {
    ErrorCallbackGuard guard;
    try {
      configuration->errorCallback(message);
    } catch (...) {
    }
  }
  const auto formatted = formatCommsLogMessage(
      levelName,
      message,
      {std::chrono::system_clock::now(),
       getCurrentThreadId(),
       getBaseName(location.filename),
       static_cast<unsigned int>(location.line),
       hostname,
       processId,
       configuration->threadContextFn(),
       threadName,
       configuration->prefix});
  if (bypassLevelGate) {
    spdlog::logger fatalLogger{
        logger_->name(), logger_->sinks().begin(), logger_->sinks().end()};
    fatalLogger.log(location, level, formatted);
    fatalLogger.flush();
  } else {
    logger_->log(location, level, formatted);
  }
}

CommsSpdlogLogger& getSpdlogLogger() {
  static CommsSpdlogLogger logger;
  return logger;
}

CommsSpdlogLogger& getSpdlogLogger(std::string_view contextName) {
  if (contextName == kLoggerName) {
    return getSpdlogLogger();
  }
  static std::shared_mutex mutex;
  static std::map<std::string, std::unique_ptr<CommsSpdlogLogger>, std::less<>>
      loggers;
  {
    std::shared_lock lock{mutex};
    if (const auto it = loggers.find(contextName); it != loggers.end()) {
      return *it->second;
    }
  }
  std::unique_lock lock{mutex};
  if (const auto it = loggers.find(contextName); it != loggers.end()) {
    return *it->second;
  }
  auto name = std::string{contextName};
  auto logger = std::make_unique<CommsSpdlogLogger>(name);
  const auto it = loggers.emplace(std::move(name), std::move(logger)).first;
  return *it->second;
}

void configureSpdlogLogger(
    std::string prefix,
    std::function<int(void)> threadContextFn) {
  getSpdlogLogger().configure(std::move(prefix), std::move(threadContextFn));
}

void configureSpdlogLogger(
    std::string_view contextName,
    std::string prefix,
    std::string_view logFilePath,
    std::function<int(void)> threadContextFn,
    std::function<void(std::string_view)> errorCallback) {
  auto& logger = getSpdlogLogger(contextName);
  logger.configure(
      std::move(prefix), std::move(threadContextFn), std::move(errorCallback));
  logger.configureOutput(logFilePath);
}

void setSpdlogThreadName(std::string_view name) {
  threadName = name;
}

} // namespace meta::comms::logger
