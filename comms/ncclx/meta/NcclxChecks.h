// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include <fmt/format.h>
#include <folly/Expected.h>

#include "comms/utils/Conversion.h"
#include "comms/utils/commSpecs.h"
#include "meta/NcclxLogUtils.h"

namespace ncclx::logging::detail {

inline ::meta::comms::CommsError getCommsErrorFromCudaError(
    cudaError_t error,
    const char* file,
    int line,
    const char* command) {
  return ::meta::comms::CommsError(
      fmt::format(
          "CUDA error in {}:{} {}: {}",
          file,
          line,
          command,
          cudaGetErrorString(error)),
      commUnhandledCudaError);
}

} // namespace ncclx::logging::detail

#define NCCLX_CUDA_CHECK_EXPECTED(command)                               \
  do {                                                                   \
    const auto _ncclx_cuda_error = (command);                            \
    if (_ncclx_cuda_error != cudaSuccess) {                              \
      NCCLX_ERR(commUnhandledCudaError, "Call for {} failed", #command); \
      return folly::makeUnexpected(                                      \
          ::ncclx::logging::detail::getCommsErrorFromCudaError(          \
              _ncclx_cuda_error, __FILE__, __LINE__, #command));         \
    }                                                                    \
  } while (false)

#define NCCLX_CUDACHECKTHROW(command)             \
  do {                                            \
    const auto _ncclx_cuda_error = (command);     \
    if (_ncclx_cuda_error != cudaSuccess) {       \
      NCCLX_ERR(                                  \
          commUnhandledCudaError,                 \
          "{}:{} Cuda failure {}",                \
          __FILE__,                               \
          __LINE__,                               \
          cudaGetErrorString(_ncclx_cuda_error)); \
      (void)cudaGetLastError();                   \
      throw std::runtime_error(                   \
          std::string("Cuda failure: ") +         \
          cudaGetErrorString(_ncclx_cuda_error)); \
    }                                             \
  } while (false)

#define NCCLX_COMMCHECKTHROW(command)                           \
  do {                                                          \
    const commResult_t _ncclx_comm_result = (command);          \
    if (_ncclx_comm_result != commSuccess &&                    \
        _ncclx_comm_result != commInProgress) {                 \
      NCCLX_LOG(                                                \
          ERR,                                                  \
          "{}:{} -> {} ({})",                                   \
          __FILE__,                                             \
          __LINE__,                                             \
          _ncclx_comm_result,                                   \
          ::meta::comms::commCodeToString(_ncclx_comm_result)); \
      throw std::runtime_error(                                 \
          std::string("COMM internal failure: ") +              \
          ::meta::comms::commCodeToString(_ncclx_comm_result)); \
    }                                                           \
  } while (false)

#define NCCLX_ERRORTHROW(error, ...)             \
  do {                                           \
    NCCLX_LOG(ERR, __VA_ARGS__);                 \
    throw std::runtime_error(                    \
        std::string("COMM internal failure: ") + \
        ::meta::comms::commCodeToString(error)); \
  } while (false)
