// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdlib>

#ifndef SPDLOG_FMT_EXTERNAL
#error "SpdlogLogger requires SPDLOG_FMT_EXTERNAL from the build target"
#endif

#ifndef SPDLOG_ACTIVE_LEVEL
#error "SpdlogLogger requires SPDLOG_ACTIVE_LEVEL from the build target"
#endif

#include <spdlog/spdlog.h>

namespace meta::comms::logger {

// Returns the lazily initialized, non-blocking logger for the comms facade.
spdlog::logger& getSpdlogLogger();

} // namespace meta::comms::logger

#define COMMS_LOG_DBG(...) \
  SPDLOG_LOGGER_DEBUG(&::meta::comms::logger::getSpdlogLogger(), __VA_ARGS__)
#define COMMS_LOG_INFO(...) \
  SPDLOG_LOGGER_INFO(&::meta::comms::logger::getSpdlogLogger(), __VA_ARGS__)
#define COMMS_LOG_WARN(...) \
  SPDLOG_LOGGER_WARN(&::meta::comms::logger::getSpdlogLogger(), __VA_ARGS__)
#define COMMS_LOG_ERR(...) \
  SPDLOG_LOGGER_ERROR(&::meta::comms::logger::getSpdlogLogger(), __VA_ARGS__)
#define COMMS_LOG_FATAL(...)                                        \
  do {                                                              \
    auto& _comms_logger = ::meta::comms::logger::getSpdlogLogger(); \
    _comms_logger.log(                                              \
        ::spdlog::source_loc{__FILE__, __LINE__, SPDLOG_FUNCTION},  \
        ::spdlog::level::critical,                                  \
        __VA_ARGS__);                                               \
    _comms_logger.flush();                                          \
    ::spdlog::shutdown();                                           \
    std::abort();                                                   \
  } while (false)

#define COMMS_LOG(level, ...) COMMS_LOG_##level(__VA_ARGS__)
