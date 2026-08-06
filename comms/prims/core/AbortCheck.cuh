// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdio>

#include "comms/common/fault_tolerance/AbortDevice.cuh"
#include "comms/prims/core/DeviceMacros.cuh"
#include "comms/prims/transport/amd/HipHostCompat.h"

namespace comms::prims {

using AbortDevice = comms::fault_tolerance::AbortDevice;

} // namespace comms::prims

#if PIPES_IS_DEVICE_COMPILE
#define ABORT_TRAP_IF_ABORTED(abort, group, fmt, ...)       \
  do {                                                      \
    if ((abort).checkExpired(group)) {                      \
      printf("CUDA ABORT ERROR: " fmt "\n", ##__VA_ARGS__); \
      PIPES_DEVICE_TRAP();                                  \
    }                                                       \
  } while (0)
#else
#define ABORT_TRAP_IF_ABORTED(abort, group, fmt, ...) \
  do {                                                \
    (void)(abort);                                    \
    (void)(group);                                    \
  } while (0)
#endif

#if PIPES_IS_DEVICE_COMPILE
#define ABORT_TRAP_IF_ABORTED_SINGLE(abort, fmt, ...)       \
  do {                                                      \
    if ((abort).checkExpired()) {                           \
      printf("CUDA ABORT ERROR: " fmt "\n", ##__VA_ARGS__); \
      PIPES_DEVICE_TRAP();                                  \
    }                                                       \
  } while (0)
#else
#define ABORT_TRAP_IF_ABORTED_SINGLE(abort, fmt, ...) \
  do {                                                \
    (void)(abort);                                    \
  } while (0)
#endif
