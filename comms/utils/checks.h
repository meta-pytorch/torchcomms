// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <fmt/format.h>
#include <folly/logging/xlog.h>

#include "comms/utils/commSpecs.h"

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
