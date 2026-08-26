// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#ifndef SPDLOG_FMT_EXTERNAL
#error "SpdlogLogger requires SPDLOG_FMT_EXTERNAL from the build target"
#endif

#ifndef SPDLOG_ACTIVE_LEVEL
#error "SpdlogLogger requires SPDLOG_ACTIVE_LEVEL from the build target"
#endif

#include <spdlog/sinks/dist_sink.h>
#include <spdlog/spdlog.h>

#include "comms/utils/logger/LogTypes.h"
#include "comms/utils/logger/RateLimit.h"

namespace meta::comms::logger {

inline constexpr std::string_view kCommsLoggerName{"comms"};

class CommsSpdlogLogger {
 public:
  CommsSpdlogLogger();
  explicit CommsSpdlogLogger(std::string name);

  template <typename... Args>
  void log(
      spdlog::source_loc location,
      spdlog::level::level_enum level,
      spdlog::format_string_t<Args...> format,
      Args&&... args) {
    if (!should_log(level)) {
      return;
    }
    logFormatted(
        location,
        level,
        getLevelName(level),
        fmt::format(format, std::forward<Args>(args)...),
        false);
  }

  void log(
      spdlog::source_loc location,
      spdlog::level::level_enum level,
      std::string_view message);

  template <typename... Args>
  void logSynchronous(
      spdlog::source_loc location,
      spdlog::level::level_enum level,
      spdlog::format_string_t<Args...> format,
      Args&&... args) {
    if (!should_log(level)) {
      return;
    }
    logFormatted(
        location,
        level,
        getLevelName(level),
        fmt::format(format, std::forward<Args>(args)...),
        true);
  }

  void logSynchronous(
      spdlog::source_loc location,
      spdlog::level::level_enum level,
      std::string_view message);

  template <typename... Args>
  void logFatal(
      spdlog::source_loc location,
      spdlog::format_string_t<Args...> format,
      Args&&... args) {
    logFormatted(
        location,
        spdlog::level::critical,
        "FATAL",
        fmt::format(format, std::forward<Args>(args)...),
        true);
  }

  void logFatal(spdlog::source_loc location, std::string_view message);

  bool should_log(spdlog::level::level_enum level) const;
  const std::string& name() const;
  void set_level(spdlog::level::level_enum level);
  bool usesAsyncLogging() const;
  void flush();

  void configure(
      std::string prefix,
      std::function<int(void)> threadContextFn,
      std::function<void(std::string_view)> errorCallback = {},
      bool asyncLogging = true);
  void configureOutput(std::string_view logFilePath);

 private:
  struct Configuration {
    std::string prefix{"COMMS"};
    std::function<int(void)> threadContextFn{[]() { return 0; }};
    std::function<void(std::string_view)> errorCallback;
    bool asyncLogging{true};
  };

#if defined(__cpp_lib_atomic_shared_ptr) && \
    __cpp_lib_atomic_shared_ptr >= 201711L
  using ConfigurationStorage =
      std::atomic<std::shared_ptr<const Configuration>>;
#else
  // NCCLX also builds this target as C++17, before atomic<shared_ptr> exists.
  using ConfigurationStorage = std::shared_ptr<const Configuration>;
#endif

  static std::string_view getLevelName(spdlog::level::level_enum level);

  std::shared_ptr<const Configuration> loadConfiguration() const;
  void storeConfiguration(std::shared_ptr<const Configuration> configuration);

  void logFormatted(
      spdlog::source_loc location,
      spdlog::level::level_enum level,
      std::string_view levelName,
      std::string_view message,
      bool bypassLevelGate);

  std::shared_ptr<spdlog::logger> logger_;
  std::shared_ptr<spdlog::sinks::dist_sink_mt> outputSink_;
  std::shared_ptr<spdlog::logger> synchronousLogger_;
  ConfigurationStorage configuration_;
};

class CommsLogStreamBase {
 public:
  std::ostream& stream();

 protected:
  CommsLogStreamBase(
      CommsSpdlogLogger& logger,
      spdlog::source_loc location,
      spdlog::level::level_enum level);
  ~CommsLogStreamBase() = default;

  void log();
  [[noreturn]] void logFatalAndAbort() noexcept;

 private:
  CommsSpdlogLogger& logger_;
  spdlog::source_loc location_;
  spdlog::level::level_enum level_;
  std::ostringstream stream_;
};

class CommsLogStream final : public CommsLogStreamBase {
 public:
  CommsLogStream(
      CommsSpdlogLogger& logger,
      spdlog::source_loc location,
      spdlog::level::level_enum level);
  ~CommsLogStream() noexcept;
  CommsLogStream(const CommsLogStream&) = delete;
  CommsLogStream& operator=(const CommsLogStream&) = delete;
  CommsLogStream(CommsLogStream&&) = delete;
  CommsLogStream& operator=(CommsLogStream&&) = delete;
};

class CommsFatalLogStream final : public CommsLogStreamBase {
 public:
  CommsFatalLogStream(CommsSpdlogLogger& logger, spdlog::source_loc location);
  [[noreturn]] ~CommsFatalLogStream() noexcept;
  CommsFatalLogStream(const CommsFatalLogStream&) = delete;
  CommsFatalLogStream& operator=(const CommsFatalLogStream&) = delete;
  CommsFatalLogStream(CommsFatalLogStream&&) = delete;
  CommsFatalLogStream& operator=(CommsFatalLogStream&&) = delete;
};

CommsSpdlogLogger& getSpdlogLogger();
CommsSpdlogLogger& getSpdlogLogger(std::string_view loggerName);

void reportCommsLoggingFailureToStderr(const char* level) noexcept;
[[noreturn]] void abortAfterCommsLoggingFailure() noexcept;
void shutdownSpdlogForFatal();
CommsSpdlogLogger& getSpdlogLoggerForFatal(
    std::string_view loggerName) noexcept;

spdlog::level::level_enum loggerLevelToSpdlogLevel(LogLevel level);

void configureSpdlogLogger(
    std::string prefix,
    std::function<int(void)> threadContextFn);

void configureSpdlogLogger(
    std::string_view loggerName,
    std::string prefix,
    std::string_view logFilePath,
    std::function<int(void)> threadContextFn,
    std::function<void(std::string_view)> errorCallback,
    bool asyncLogging = true);

void configureCommsAndNamedSpdlogLoggers(
    std::string_view loggerName,
    std::string logPrefix,
    std::string_view logFilePath,
    std::function<int(void)> threadContextFn,
    std::function<void(std::string_view)> errorCallback,
    bool asyncLogging,
    spdlog::level::level_enum logLevel,
    bool configureCommsLogger = true);

void setSpdlogThreadName(std::string_view threadName);

bool shouldWriteCommsLogToStderr(std::string_view formattedMessage);

} // namespace meta::comms::logger

#define COMMS_LOG_IMPL(logger_expression, spdlog_level, spdlog_macro, ...) \
  do {                                                                     \
    auto& _comms_logger = (logger_expression);                             \
    if (_comms_logger.should_log(spdlog_level)) {                          \
      spdlog_macro(&_comms_logger, __VA_ARGS__);                           \
    }                                                                      \
  } while (false)

#define COMMS_LOGGER_DEBUG(logger, ...) \
  SPDLOG_LOGGER_CALL(logger, ::spdlog::level::debug, __VA_ARGS__)

/*
 * shutdownSpdlogForFatal() releases the library-owned async pool. It drains
 * once any in-flight log calls release their pool references; the synchronous
 * logger and its output sinks remain valid throughout.
 */
#define COMMS_LOG_FATAL_IMPL(logger_expression, ...)                 \
  do {                                                               \
    try {                                                            \
      auto& _comms_logger = (logger_expression);                     \
      _comms_logger.flush();                                         \
      ::meta::comms::logger::shutdownSpdlogForFatal();               \
      _comms_logger.logFatal(                                        \
          ::spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION}, \
          __VA_ARGS__);                                              \
    } catch (...) {                                                  \
      ::meta::comms::logger::abortAfterCommsLoggingFailure();        \
    }                                                                \
    std::abort();                                                    \
  } while (false)

#define COMMS_LOG_DBG(...)                      \
  COMMS_LOG_IMPL(                               \
      ::meta::comms::logger::getSpdlogLogger(), \
      ::spdlog::level::debug,                   \
      COMMS_LOGGER_DEBUG,                       \
      __VA_ARGS__)
#define COMMS_LOG_NAMED_DBG(logger_name, ...)              \
  COMMS_LOG_IMPL(                                          \
      ::meta::comms::logger::getSpdlogLogger(logger_name), \
      ::spdlog::level::debug,                              \
      COMMS_LOGGER_DEBUG,                                  \
      __VA_ARGS__)

#define COMMS_LOG_INFO(...)                     \
  COMMS_LOG_IMPL(                               \
      ::meta::comms::logger::getSpdlogLogger(), \
      ::spdlog::level::info,                    \
      SPDLOG_LOGGER_INFO,                       \
      __VA_ARGS__)
#define COMMS_LOG_WARN(...)                     \
  COMMS_LOG_IMPL(                               \
      ::meta::comms::logger::getSpdlogLogger(), \
      ::spdlog::level::warn,                    \
      SPDLOG_LOGGER_WARN,                       \
      __VA_ARGS__)
#define COMMS_LOG_ERR(...)                      \
  COMMS_LOG_IMPL(                               \
      ::meta::comms::logger::getSpdlogLogger(), \
      ::spdlog::level::err,                     \
      SPDLOG_LOGGER_ERROR,                      \
      __VA_ARGS__)
#define COMMS_LOG_CRITICAL(...)                 \
  COMMS_LOG_IMPL(                               \
      ::meta::comms::logger::getSpdlogLogger(), \
      ::spdlog::level::critical,                \
      SPDLOG_LOGGER_CRITICAL,                   \
      __VA_ARGS__)
#define COMMS_LOG_FATAL(...) \
  COMMS_LOG_FATAL_IMPL(::meta::comms::logger::getSpdlogLogger(), __VA_ARGS__)

#define COMMS_LOG_NAMED_INFO(logger_name, ...)             \
  COMMS_LOG_IMPL(                                          \
      ::meta::comms::logger::getSpdlogLogger(logger_name), \
      ::spdlog::level::info,                               \
      SPDLOG_LOGGER_INFO,                                  \
      __VA_ARGS__)
#define COMMS_LOG_NAMED_WARN(logger_name, ...)             \
  COMMS_LOG_IMPL(                                          \
      ::meta::comms::logger::getSpdlogLogger(logger_name), \
      ::spdlog::level::warn,                               \
      SPDLOG_LOGGER_WARN,                                  \
      __VA_ARGS__)
#define COMMS_LOG_NAMED_ERR(logger_name, ...)              \
  COMMS_LOG_IMPL(                                          \
      ::meta::comms::logger::getSpdlogLogger(logger_name), \
      ::spdlog::level::err,                                \
      SPDLOG_LOGGER_ERROR,                                 \
      __VA_ARGS__)
#define COMMS_LOG_NAMED_CRITICAL(logger_name, ...)         \
  COMMS_LOG_IMPL(                                          \
      ::meta::comms::logger::getSpdlogLogger(logger_name), \
      ::spdlog::level::critical,                           \
      SPDLOG_LOGGER_CRITICAL,                              \
      __VA_ARGS__)
#define COMMS_LOG_NAMED_FATAL(logger_name, ...) \
  COMMS_LOG_FATAL_IMPL(                         \
      ::meta::comms::logger::getSpdlogLogger(logger_name), __VA_ARGS__)

#define COMMS_LOG(level, ...) COMMS_LOG_##level(__VA_ARGS__)
#define COMMS_LOG_NAMED(logger_name, level, ...) \
  COMMS_LOG_NAMED_##level(logger_name, __VA_ARGS__)

#define COMMS_LOG_NAMED_STREAM_IMPL(logger_name, spdlog_level, condition) \
  if (static auto& _comms_stream_logger =                                 \
          ::meta::comms::logger::getSpdlogLogger(logger_name);            \
      !_comms_stream_logger.should_log(spdlog_level) || !(condition)) {   \
  } else                                                                  \
    ::meta::comms::logger::CommsLogStream(                                \
        _comms_stream_logger,                                             \
        ::spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION},        \
        spdlog_level)                                                     \
        .stream()

#define COMMS_LOG_NAMED_FATAL_STREAM_IMPL(logger_name, condition)      \
  if (static auto& _comms_stream_logger =                              \
          ::meta::comms::logger::getSpdlogLoggerForFatal(logger_name); \
      !(condition)) {                                                  \
  } else                                                               \
    ::meta::comms::logger::CommsFatalLogStream(                        \
        _comms_stream_logger,                                          \
        ::spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION})     \
        .stream()

#define COMMS_LOG_NAMED_STREAM_DBG_IF(logger_name, condition) \
  COMMS_LOG_NAMED_STREAM_IMPL(logger_name, ::spdlog::level::debug, condition)
#define COMMS_LOG_NAMED_STREAM_DBG5_IF(logger_name, condition) \
  COMMS_LOG_NAMED_STREAM_IMPL(logger_name, ::spdlog::level::trace, condition)
#define COMMS_LOG_NAMED_STREAM_INFO_IF(logger_name, condition) \
  COMMS_LOG_NAMED_STREAM_IMPL(logger_name, ::spdlog::level::info, condition)
#define COMMS_LOG_NAMED_STREAM_WARN_IF(logger_name, condition) \
  COMMS_LOG_NAMED_STREAM_IMPL(logger_name, ::spdlog::level::warn, condition)
#define COMMS_LOG_NAMED_STREAM_WARNING_IF(logger_name, condition) \
  COMMS_LOG_NAMED_STREAM_WARN_IF(logger_name, condition)
#define COMMS_LOG_NAMED_STREAM_ERR_IF(logger_name, condition) \
  COMMS_LOG_NAMED_STREAM_IMPL(logger_name, ::spdlog::level::err, condition)
#define COMMS_LOG_NAMED_STREAM_CRITICAL_IF(logger_name, condition) \
  COMMS_LOG_NAMED_STREAM_IMPL(logger_name, ::spdlog::level::critical, condition)
#define COMMS_LOG_NAMED_STREAM_FATAL_IF(logger_name, condition) \
  COMMS_LOG_NAMED_FATAL_STREAM_IMPL(logger_name, condition)

#define COMMS_LOG_NAMED_STREAM_IF_IMPL(logger_name, level, condition) \
  COMMS_LOG_NAMED_STREAM_##level##_IF(logger_name, condition)
#define COMMS_LOG_NAMED_STREAM_IF(logger_name, level, condition) \
  COMMS_LOG_NAMED_STREAM_IF_IMPL(logger_name, level, condition)
#define COMMS_LOG_NAMED_STREAM(logger_name, level) \
  COMMS_LOG_NAMED_STREAM_IF(logger_name, level, true)
#define COMMS_LOG_STREAM(level) \
  COMMS_LOG_NAMED_STREAM(::meta::comms::logger::kCommsLoggerName, level)

#define COMMS_LOG_RATE_LIMITABLE_DBG true
#define COMMS_LOG_RATE_LIMITABLE_DBG5 true
#define COMMS_LOG_RATE_LIMITABLE_INFO true
#define COMMS_LOG_RATE_LIMITABLE_WARN true
#define COMMS_LOG_RATE_LIMITABLE_WARNING true
#define COMMS_LOG_RATE_LIMITABLE_ERR true
#define COMMS_LOG_RATE_LIMITABLE_CRITICAL true
#define COMMS_LOG_RATE_LIMITABLE_FATAL false

/*
 * The interval is fixed on first use at each expansion site. Disabled levels
 * do not consume the rate-limit budget.
 */
#define COMMS_LOG_NAMED_STREAM_EVERY_MS(logger_name, level, ms)            \
  COMMS_LOG_NAMED_STREAM_IF(                                               \
      logger_name, level, [_comms_log_stream_every_ms = (ms)] {            \
        static_assert(                                                     \
            COMMS_LOG_RATE_LIMITABLE_##level,                              \
            "FATAL logging cannot be rate limited");                       \
        static ::meta::comms::logger::IntervalRateLimiter                  \
            comms_log_stream_rate_limiter(                                 \
                1, std::chrono::milliseconds(_comms_log_stream_every_ms)); \
        return comms_log_stream_rate_limiter.check();                      \
      }())
#define COMMS_LOG_STREAM_EVERY_MS(level, ms) \
  COMMS_LOG_NAMED_STREAM_EVERY_MS(           \
      ::meta::comms::logger::kCommsLoggerName, level, ms)

#define COMMS_LOGGER_STREAM_IMPL(logger_expression, spdlog_level, condition) \
  if (auto& _comms_stream_logger = (logger_expression);                      \
      !_comms_stream_logger.should_log(spdlog_level) || !(condition)) {      \
  } else                                                                     \
    ::meta::comms::logger::CommsLogStream(                                   \
        _comms_stream_logger,                                                \
        ::spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION},           \
        spdlog_level)                                                        \
        .stream()

#define COMMS_LOGGER_FATAL_STREAM_IMPL(logger_expression, condition)    \
  if (auto& _comms_stream_logger = (logger_expression); !(condition)) { \
  } else                                                                \
    ::meta::comms::logger::CommsFatalLogStream(                         \
        _comms_stream_logger,                                           \
        ::spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION})      \
        .stream()

#define COMMS_LOGGER_STREAM_DBG_IF(logger_expression, condition) \
  COMMS_LOGGER_STREAM_IMPL(logger_expression, ::spdlog::level::debug, condition)
#define COMMS_LOGGER_STREAM_DBG5_IF(logger_expression, condition) \
  COMMS_LOGGER_STREAM_IMPL(logger_expression, ::spdlog::level::trace, condition)
#define COMMS_LOGGER_STREAM_INFO_IF(logger_expression, condition) \
  COMMS_LOGGER_STREAM_IMPL(logger_expression, ::spdlog::level::info, condition)
#define COMMS_LOGGER_STREAM_WARN_IF(logger_expression, condition) \
  COMMS_LOGGER_STREAM_IMPL(logger_expression, ::spdlog::level::warn, condition)
#define COMMS_LOGGER_STREAM_WARNING_IF(logger_expression, condition) \
  COMMS_LOGGER_STREAM_WARN_IF(logger_expression, condition)
#define COMMS_LOGGER_STREAM_ERR_IF(logger_expression, condition) \
  COMMS_LOGGER_STREAM_IMPL(logger_expression, ::spdlog::level::err, condition)
#define COMMS_LOGGER_STREAM_CRITICAL_IF(logger_expression, condition) \
  COMMS_LOGGER_STREAM_IMPL(                                           \
      logger_expression, ::spdlog::level::critical, condition)
#define COMMS_LOGGER_STREAM_FATAL_IF(logger_expression, condition) \
  COMMS_LOGGER_FATAL_STREAM_IMPL(logger_expression, condition)

#define COMMS_LOGGER_STREAM_IF_IMPL(logger_expression, level, condition) \
  COMMS_LOGGER_STREAM_##level##_IF(logger_expression, condition)
#define COMMS_LOGGER_STREAM_IF(logger_expression, level, condition) \
  COMMS_LOGGER_STREAM_IF_IMPL(logger_expression, level, condition)
#define COMMS_LOGGER_STREAM(logger_expression, level) \
  COMMS_LOGGER_STREAM_IF(logger_expression, level, true)

#define COMMS_LOGGER_STREAM_EVERY_MS(logger_expression, level, ms)            \
  COMMS_LOGGER_STREAM_IF(                                                     \
      logger_expression, level, [_comms_logger_stream_every_ms = (ms)] {      \
        static_assert(                                                        \
            COMMS_LOG_RATE_LIMITABLE_##level,                                 \
            "FATAL logging cannot be rate limited");                          \
        static ::meta::comms::logger::IntervalRateLimiter                     \
            comms_logger_stream_rate_limiter(                                 \
                1, std::chrono::milliseconds(_comms_logger_stream_every_ms)); \
        return comms_logger_stream_rate_limiter.check();                      \
      }())

#define COMMS_LOGGER_STREAM_FIRST_N(logger_expression, level, n)              \
  COMMS_LOGGER_STREAM_IF(logger_expression, level, [&] {                      \
    static_assert(                                                            \
        COMMS_LOG_RATE_LIMITABLE_##level, "FATAL logging cannot be sampled"); \
    struct comms_logger_stream_first_n_tag {};                                \
    return ::meta::comms::logger::firstNExact<                                \
        comms_logger_stream_first_n_tag>(n);                                  \
  }())

#define COMMS_LOG_STREAM_FIRST_N(level, n) \
  COMMS_LOGGER_STREAM_FIRST_N(             \
      ::meta::comms::logger::getSpdlogLogger(), level, n)
