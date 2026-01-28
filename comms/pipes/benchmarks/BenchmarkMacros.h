// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>
#include <folly/logging/xlog.h>
#include <nccl.h>

namespace comms::pipes::benchmark {

// Benchmark iteration constants
constexpr int kWarmupIters = 20;
constexpr int kBenchmarkIters = 30;

// Default data buffer size for large message benchmarks (8MB)
constexpr std::size_t kDefaultDataBufferSize = 8 * 1024 * 1024;

// =============================================================================
// Error Checking Macros
// =============================================================================

// CUDA error checking macro for void functions
#define CUDA_CHECK_VOID(call)        \
  do {                               \
    cudaError_t err = call;          \
    if (err != cudaSuccess) {        \
      XLOGF(                         \
          ERR,                       \
          "CUDA error at {}:{}: {}", \
          __FILE__,                  \
          __LINE__,                  \
          cudaGetErrorString(err));  \
      return;                        \
    }                                \
  } while (0)

// NCCL error checking macro for void functions
#define NCCL_CHECK_VOID(call)        \
  do {                               \
    ncclResult_t res = call;         \
    if (res != ncclSuccess) {        \
      XLOGF(                         \
          ERR,                       \
          "NCCL error at {}:{}: {}", \
          __FILE__,                  \
          __LINE__,                  \
          ncclGetErrorString(res));  \
      return;                        \
    }                                \
  } while (0)

// CUDA error checking macro for float-returning functions
#define CUDA_CHECK(call)             \
  do {                               \
    cudaError_t err = call;          \
    if (err != cudaSuccess) {        \
      XLOGF(                         \
          ERR,                       \
          "CUDA error at {}:{}: {}", \
          __FILE__,                  \
          __LINE__,                  \
          cudaGetErrorString(err));  \
      return 0.0f;                   \
    }                                \
  } while (0)

// NCCL error checking macro for float-returning functions
#define NCCL_CHECK(call)             \
  do {                               \
    ncclResult_t res = call;         \
    if (res != ncclSuccess) {        \
      XLOGF(                         \
          ERR,                       \
          "NCCL error at {}:{}: {}", \
          __FILE__,                  \
          __LINE__,                  \
          ncclGetErrorString(res));  \
      return 0.0f;                   \
    }                                \
  } while (0)

// CUDA error checking macro for bool-returning functions
#define CUDA_CHECK_BOOL(call)        \
  do {                               \
    cudaError_t err = call;          \
    if (err != cudaSuccess) {        \
      XLOGF(                         \
          ERR,                       \
          "CUDA error at {}:{}: {}", \
          __FILE__,                  \
          __LINE__,                  \
          cudaGetErrorString(err));  \
      return false;                  \
    }                                \
  } while (0)

} // namespace comms::pipes::benchmark
