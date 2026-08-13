// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>
#include <cstdio>

#include "comms/common/fault_tolerance/AbortDevice.cuh"
#include "comms/prims/core/DeviceMacros.cuh"
#include "comms/prims/transport/amd/HipHostCompat.h"

namespace comms::prims {

using AbortDevice = comms::fault_tolerance::AbortDevice;

__device__ __forceinline__ uint64_t gpu_clock64() {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__)
  return wall_clock64();
#elif defined(__CUDA_ARCH__)
  return clock64();
#else
  return 0;
#endif
}

} // namespace comms::prims

#if PIPES_IS_DEVICE_COMPILE
#define ABORT_TRAP_IF_ABORTED(abort, group, fmt, ...)              \
  do {                                                             \
    if ((group).is_leader()) {                                     \
      switch ((abort).check()) {                                   \
        case ::comms::fault_tolerance::AbortCheckResult::CONTINUE: \
        case ::comms::fault_tolerance::AbortCheckResult::SKIP:     \
          break;                                                   \
        case ::comms::fault_tolerance::AbortCheckResult::TRAP:     \
          printf("CUDA ABORT ERROR: " fmt "\n", ##__VA_ARGS__);    \
          PIPES_DEVICE_TRAP();                                     \
          break;                                                   \
      }                                                            \
    }                                                              \
  } while (0)
#else
#define ABORT_TRAP_IF_ABORTED(abort, group, fmt, ...) \
  do {                                                \
    (void)(abort);                                    \
    (void)(group);                                    \
  } while (0)
#endif

#if PIPES_IS_DEVICE_COMPILE
#define ABORT_TRAP_IF_ABORTED_SINGLE(abort, fmt, ...)            \
  do {                                                           \
    switch ((abort).check()) {                                   \
      case ::comms::fault_tolerance::AbortCheckResult::CONTINUE: \
      case ::comms::fault_tolerance::AbortCheckResult::SKIP:     \
        break;                                                   \
      case ::comms::fault_tolerance::AbortCheckResult::TRAP:     \
        printf("CUDA ABORT ERROR: " fmt "\n", ##__VA_ARGS__);    \
        PIPES_DEVICE_TRAP();                                     \
        break;                                                   \
    }                                                            \
  } while (0)
#else
#define ABORT_TRAP_IF_ABORTED_SINGLE(abort, fmt, ...) \
  do {                                                \
    (void)(abort);                                    \
  } while (0)
#endif

#if PIPES_IS_DEVICE_COMPILE
#define ABORT_RETURN_FALSE_IF_ABORTED_SINGLE(abort, fmt, ...)    \
  do {                                                           \
    switch ((abort).check()) {                                   \
      case ::comms::fault_tolerance::AbortCheckResult::CONTINUE: \
        break;                                                   \
      case ::comms::fault_tolerance::AbortCheckResult::SKIP:     \
        return false;                                            \
      case ::comms::fault_tolerance::AbortCheckResult::TRAP:     \
        printf("CUDA ABORT ERROR: " fmt "\n", ##__VA_ARGS__);    \
        PIPES_DEVICE_TRAP();                                     \
        return false;                                            \
    }                                                            \
  } while (0)
#else
#define ABORT_RETURN_FALSE_IF_ABORTED_SINGLE(abort, fmt, ...) \
  do {                                                        \
    (void)(abort);                                            \
  } while (0)
#endif

#if PIPES_IS_DEVICE_COMPILE
#define ABORT_BREAK_IF_ABORTED_SINGLE(abort, aborted, fmt, ...)  \
  do {                                                           \
    switch ((abort).check()) {                                   \
      case ::comms::fault_tolerance::AbortCheckResult::CONTINUE: \
        break;                                                   \
      case ::comms::fault_tolerance::AbortCheckResult::SKIP:     \
        aborted = true;                                          \
        break;                                                   \
      case ::comms::fault_tolerance::AbortCheckResult::TRAP:     \
        printf("CUDA ABORT ERROR: " fmt "\n", ##__VA_ARGS__);    \
        PIPES_DEVICE_TRAP();                                     \
        aborted = true;                                          \
        break;                                                   \
    }                                                            \
  } while (0)
#else
#define ABORT_BREAK_IF_ABORTED_SINGLE(abort, aborted, fmt, ...) \
  do {                                                          \
    (void)(abort);                                              \
    (void)(aborted);                                            \
  } while (0)
#endif

#define TIMEOUT_TRAP_IF_EXPIRED(abort, group, fmt, ...) \
  ABORT_TRAP_IF_ABORTED(abort, group, fmt, ##__VA_ARGS__)

#define TIMEOUT_TRAP_IF_EXPIRED_SINGLE(abort, fmt, ...) \
  ABORT_TRAP_IF_ABORTED_SINGLE(abort, fmt, ##__VA_ARGS__)

#define TIMEOUT_RETURN_FALSE_IF_ABORTED_SINGLE(abort, fmt, ...) \
  ABORT_RETURN_FALSE_IF_ABORTED_SINGLE(abort, fmt, ##__VA_ARGS__)

#define TIMEOUT_BREAK_IF_ABORTED_SINGLE(abort, aborted, fmt, ...) \
  ABORT_BREAK_IF_ABORTED_SINGLE(abort, aborted, fmt, ##__VA_ARGS__)
