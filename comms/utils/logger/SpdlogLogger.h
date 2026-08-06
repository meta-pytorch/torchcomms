// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <atomic>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#ifndef SPDLOG_FMT_EXTERNAL
#error "SpdlogLogger requires SPDLOG_FMT_EXTERNAL from the build target"
#endif

#ifndef SPDLOG_ACTIVE_LEVEL
#error "SpdlogLogger requires SPDLOG_ACTIVE_LEVEL from the build target"
#endif

#include <spdlog/spdlog.h>

namespace meta::comms::logger {

class CommsSpdlogLogger {
 public:
  CommsSpdlogLogger();

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
  void flush();

  void configure(std::string prefix, std::function<int(void)> threadContextFn);

 private:
  struct Configuration {
    std::string prefix{"COMMS"};
    std::function<int(void)> threadContextFn{[]() { return 0; }};
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
  ConfigurationStorage configuration_;
};

CommsSpdlogLogger& getSpdlogLogger();

void configureSpdlogLogger(
    std::string prefix,
    std::function<int(void)> threadContextFn);

void setSpdlogThreadName(std::string_view threadName);

} // namespace meta::comms::logger

#define COMMS_LOG_IMPL(spdlog_level, spdlog_macro, ...)             \
  do {                                                              \
    auto& _comms_logger = ::meta::comms::logger::getSpdlogLogger(); \
    if (_comms_logger.should_log(spdlog_level)) {                   \
      spdlog_macro(&_comms_logger, __VA_ARGS__);                    \
    }                                                               \
  } while (false)

#define COMMS_LOG_DBG(...) \
  COMMS_LOG_IMPL(::spdlog::level::debug, SPDLOG_LOGGER_DEBUG, __VA_ARGS__)
#define COMMS_LOG_INFO(...) \
  COMMS_LOG_IMPL(::spdlog::level::info, SPDLOG_LOGGER_INFO, __VA_ARGS__)
#define COMMS_LOG_WARN(...) \
  COMMS_LOG_IMPL(::spdlog::level::warn, SPDLOG_LOGGER_WARN, __VA_ARGS__)
#define COMMS_LOG_ERR(...) \
  COMMS_LOG_IMPL(::spdlog::level::err, SPDLOG_LOGGER_ERROR, __VA_ARGS__)
#define COMMS_LOG_CRITICAL(...) \
  COMMS_LOG_IMPL(::spdlog::level::critical, SPDLOG_LOGGER_CRITICAL, __VA_ARGS__)
#define COMMS_LOG_FATAL(...)                                        \
  do {                                                              \
    auto& _comms_logger = ::meta::comms::logger::getSpdlogLogger(); \
    _comms_logger.flush();                                          \
    ::spdlog::shutdown();                                           \
    _comms_logger.logFatal(                                         \
        ::spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION},  \
        __VA_ARGS__);                                               \
    std::abort();                                                   \
  } while (false)

#define COMMS_LOG(level, ...) COMMS_LOG_##level(__VA_ARGS__)
