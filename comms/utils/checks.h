// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

#include <fmt/format.h>
#include <folly/String.h>
#include <folly/logging/xlog.h>

#include "comms/utils/Conversion.h"
#include "comms/utils/commSpecs.h"
#include "comms/utils/logger/LogUtils.h"

// Base case
template <typename... ErrorCodes>
bool inCudaErrorCodes(cudaError_t res, cudaError_t error) {
  return res == error;
}

// Recursive case
template <typename... ErrorCodes>
  requires(std::same_as<ErrorCodes, cudaError_t> && ...)
bool inCudaErrorCodes(
    cudaError_t res,
    cudaError_t firstError,
    ErrorCodes... errorCodes) {
  return res == firstError || inCudaErrorCodes(res, errorCodes...);
}

inline ::meta::comms::CommsError getCommsErrorFromCudaError(
    cudaError_t error,
    const char* file,
    int line,
    const char* cmd) {
  return ::meta::comms::CommsError(
      fmt::format(
          "CUDA error in {}:{} {}: {}",
          file,
          line,
          cmd,
          cudaGetErrorString(error)),
      commUnhandledCudaError);
}

#define CUDA_CHECK_WITH_IGNORE(cmd, ...)                    \
  {                                                         \
    const auto res = cmd;                                   \
    if (!inCudaErrorCodes(res, cudaSuccess, __VA_ARGS__)) { \
      XLOG(FATAL) << fmt::format(                           \
          "CUDA error: {}:{} {}",                           \
          __FILE__,                                         \
          __LINE__,                                         \
          cudaGetErrorString(res));                         \
    }                                                       \
  }

#define CUDA_CHECK(cmd)             \
  {                                 \
    const auto err = cmd;           \
    if (err != cudaSuccess) {       \
      XLOG(FATAL) << fmt::format(   \
          "CUDA error: {}:{} {}",   \
          __FILE__,                 \
          __LINE__,                 \
          cudaGetErrorString(err)); \
    }                               \
  }

#define CUDA_CHECK_EXPECTED(cmd)                                      \
  {                                                                   \
    const auto err = cmd;                                             \
    if (err != cudaSuccess) {                                         \
      XLOG(ERR) << fmt::format("Call for {} failed", #cmd);           \
      return folly::makeUnexpected(                                   \
          getCommsErrorFromCudaError(err, __FILE__, __LINE__, #cmd)); \
    }                                                                 \
  }

#define NCCL_CHECK(cmd)             \
  {                                 \
    const auto err = cmd;           \
    if (err != ncclSuccess) {       \
      XLOG(FATAL) << fmt::format(   \
          "NCCL error: {}:{} {}",   \
          __FILE__,                 \
          __LINE__,                 \
          ncclGetErrorString(err)); \
    }                               \
  }

/**
 * Checks a CUDA command and throws std::runtime_error on failure.
 *
 * Usage:
 *   FB_CUDACHECKTHROW(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
 */
#define FB_CUDACHECKTHROW(cmd)                                      \
  do {                                                              \
    cudaError_t err = cmd;                                          \
    if (err != cudaSuccess) {                                       \
      CLOGF(                                                        \
          ERR,                                                      \
          "{}:{} Cuda failure {}",                                  \
          __FILE__,                                                 \
          __LINE__,                                                 \
          cudaGetErrorString(err));                                 \
      (void)cudaGetLastError();                                     \
      throw std::runtime_error(                                     \
          std::string("Cuda failure: ") + cudaGetErrorString(err)); \
    }                                                               \
  } while (false)

/**
 * Checks a folly::Expected result and throws std::runtime_error on error.
 *
 * Usage:
 *   auto result = someFunctionReturningExpected();
 *   FOLLY_EXPECTED_CHECKTHROW(result);
 */
#define FOLLY_EXPECTED_CHECKTHROW(RES)                                  \
  do {                                                                  \
    if (RES.hasError()) {                                               \
      CLOGF(                                                            \
          ERR,                                                          \
          "{}:{} -> {} ({})",                                           \
          __FILE__,                                                     \
          __LINE__,                                                     \
          RES.error().errNum,                                           \
          RES.error().errStr);                                          \
      throw std::runtime_error(                                         \
          std::string("COMM internal failure: ") + RES.error().errStr); \
    }                                                                   \
  } while (0)

/**
 * Checks a commResult_t and throws std::runtime_error on failure.
 *
 * Usage:
 *   FB_COMMCHECKTHROW(someCommFunction());
 */
#define FB_COMMCHECKTHROW(cmd)                         \
  do {                                                 \
    commResult_t RES = cmd;                            \
    if (RES != commSuccess && RES != commInProgress) { \
      CLOGF(                                           \
          ERR,                                         \
          "{}:{} -> {} ({})",                          \
          __FILE__,                                    \
          __LINE__,                                    \
          RES,                                         \
          ::meta::comms::commCodeToString(RES));       \
      throw std::runtime_error(                        \
          std::string("COMM internal failure: ") +     \
          ::meta::comms::commCodeToString(RES));       \
    }                                                  \
  } while (0)

/**
 * Checks a boolean statement and throws std::runtime_error on failure.
 *
 * Usage:
 *   FB_CHECKTHROW(ptr != nullptr, "Pointer must not be null");
 *   FB_CHECKTHROW(size > 0, "Size {} must be positive", size);
 */
#define FB_CHECKTHROW(statement, ...)                                          \
  do {                                                                         \
    if (!(statement)) {                                                        \
      auto errorMsg =                                                          \
          fmt::format("Check failed: {} - {}", #statement, __VA_ARGS__);       \
      CLOGF(ERR, errorMsg);                                                    \
      throw std::runtime_error(                                                \
          fmt::format(                                                         \
              "Check failed: {} - {}", #statement, fmt::format(__VA_ARGS__))); \
    }                                                                          \
  } while (0)

/**
 * Throws std::runtime_error with a commResult_t error code.
 *
 * Usage:
 *   FB_ERRORTHROW(commInternalError, "Failed to initialize resource");
 */
#define FB_ERRORTHROW(error, ...)                \
  do {                                           \
    CLOGF(ERR, ##__VA_ARGS__);                   \
    throw std::runtime_error(                    \
        std::string("COMM internal failure: ") + \
        ::meta::comms::commCodeToString(error)); \
  } while (0)
