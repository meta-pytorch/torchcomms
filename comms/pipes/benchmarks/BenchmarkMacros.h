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

// Default cluster size for clustered kernel launches
constexpr int kDefaultClusterSize = 4;

// =============================================================================
// Auto-tune Message Size Thresholds
// =============================================================================

constexpr std::size_t kSmallMessageThreshold = 16 * 1024; // 16KB
constexpr std::size_t kMediumMessageThreshold = 256 * 1024; // 256KB
constexpr std::size_t kLargeMessageThreshold = 4 * 1024 * 1024; // 4MB

// Auto-tune configuration values for small messages (< 16KB)
constexpr std::size_t kSmallMsgChunkSize = 4 * 1024; // 4KB
constexpr std::size_t kSmallMsgStagingBuffer = 64 * 1024; // 64KB
constexpr int kSmallMsgNumBlocks = 4;
constexpr int kSmallMsgNumThreads = 256;

// Auto-tune configuration values for medium messages (16KB - 256KB)
constexpr std::size_t kMediumMsgChunkSize = 16 * 1024; // 16KB
constexpr std::size_t kMediumMsgStagingBuffer = 256 * 1024; // 256KB
constexpr int kMediumMsgNumBlocks = 8;
constexpr int kMediumMsgNumThreads = 256;

// Auto-tune configuration values for large messages (256KB - 4MB)
constexpr std::size_t kLargeMsgChunkSize = 64 * 1024; // 64KB
constexpr std::size_t kLargeMsgStagingBuffer = 1 * 1024 * 1024; // 1MB
constexpr int kLargeMsgNumBlocks = 16;
constexpr int kLargeMsgNumThreads = 512;

// Auto-tune configuration values for very large messages (>= 4MB)
constexpr std::size_t kVeryLargeMsgChunkSize = 128 * 1024; // 128KB
constexpr std::size_t kMaxStagingBufferSize = 64 * 1024 * 1024; // 64MB
constexpr int kVeryLargeMsgNumBlocks = 32;
constexpr int kVeryLargeMsgNumThreads = 512;

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
