// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

namespace meta::comms::logger {

class CommsSpdlogLogger;

enum class CudaLogLevel : unsigned char {
  DBG,
  INFO,
  WARN,
  ERR,
};

CommsSpdlogLogger* tryGetSpdlogLoggerForCuda() noexcept;

bool shouldLogFromCuda(
    const CommsSpdlogLogger& logger,
    CudaLogLevel level) noexcept;

void logFromCuda(
    CommsSpdlogLogger& logger,
    CudaLogLevel level,
    const char* filename,
    int line,
    const char* function,
    const char* format,
    ...) noexcept __attribute__((format(printf, 6, 7)));

} // namespace meta::comms::logger

/*
 * Keep this facade free of C++ standard-library and spdlog headers because
 * cudafe parses it when compiling host logging statements in .cu files.
 */
#define COMMS_CUDA_LOG(level, ...)                          \
  do {                                                      \
    static auto* const _comms_cuda_logger =                 \
        ::meta::comms::logger::tryGetSpdlogLoggerForCuda(); \
    if (_comms_cuda_logger != nullptr &&                    \
        ::meta::comms::logger::shouldLogFromCuda(           \
            *_comms_cuda_logger,                            \
            ::meta::comms::logger::CudaLogLevel::level)) {  \
      ::meta::comms::logger::logFromCuda(                   \
          *_comms_cuda_logger,                              \
          ::meta::comms::logger::CudaLogLevel::level,       \
          __FILE__,                                         \
          __LINE__,                                         \
          static_cast<const char*>(__FUNCTION__),           \
          __VA_ARGS__);                                     \
    }                                                       \
  } while (false)
