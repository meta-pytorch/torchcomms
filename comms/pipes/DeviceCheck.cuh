// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdio>

namespace comms::pipes {

/**
 * PIPES_DEVICE_CHECK: Device-side assertion that remains active in @mode/opt.
 *
 * Unlike standard assert(), this macro is NOT disabled by NDEBUG and will
 * trigger a kernel trap in both debug and optimized builds.
 *
 * Behavior:
 * - On device: Prints diagnostic message and calls __trap() to abort the kernel
 * - On host: Evaluates to no-op (host-side checks should use standard assert)
 *
 * Usage:
 *   PIPES_DEVICE_CHECK(transport.type == TransportType::P2P_NVL);
 *   PIPES_DEVICE_CHECK(idx < size);
 *
 * Note: __trap() puts the CUDA device into an unrecoverable error state. After
 * a trap, cudaDeviceReset() is required to recover the device context.
 */
#ifdef __CUDA_ARCH__
#define PIPES_DEVICE_CHECK(expr)                                    \
  do {                                                              \
    if (!(expr)) {                                                  \
      printf(                                                       \
          "PIPES_DEVICE_CHECK: %s in %s at %s:%d block=(%u,%u,%u) " \
          "thread=(%u,%u,%u)\n",                                    \
          #expr,                                                    \
          __func__,                                                 \
          __FILE__,                                                 \
          __LINE__,                                                 \
          blockIdx.x,                                               \
          blockIdx.y,                                               \
          blockIdx.z,                                               \
          threadIdx.x,                                              \
          threadIdx.y,                                              \
          threadIdx.z);                                             \
      __trap();                                                     \
    }                                                               \
  } while (0)
#else
#define PIPES_DEVICE_CHECK(expr) ((void)0)
#endif

/**
 * PIPES_DEVICE_CHECK_MSG: Device-side assertion with custom message.
 *
 * Same as PIPES_DEVICE_CHECK but allows a custom message for added context.
 *
 * Usage:
 *   PIPES_DEVICE_CHECK_MSG(idx < size, "Index out of bounds");
 */
#ifdef __CUDA_ARCH__
#define PIPES_DEVICE_CHECK_MSG(expr, msg)                                \
  do {                                                                   \
    if (!(expr)) {                                                       \
      printf(                                                            \
          "PIPES_DEVICE_CHECK: %s (%s) in %s at %s:%d block=(%u,%u,%u) " \
          "thread=(%u,%u,%u)\n",                                         \
          #expr,                                                         \
          msg,                                                           \
          __func__,                                                      \
          __FILE__,                                                      \
          __LINE__,                                                      \
          blockIdx.x,                                                    \
          blockIdx.y,                                                    \
          blockIdx.z,                                                    \
          threadIdx.x,                                                   \
          threadIdx.y,                                                   \
          threadIdx.z);                                                  \
      __trap();                                                          \
    }                                                                    \
  } while (0)
#else
#define PIPES_DEVICE_CHECK_MSG(expr, msg) ((void)0)
#endif

} // namespace comms::pipes
