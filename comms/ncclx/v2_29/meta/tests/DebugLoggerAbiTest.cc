// Copyright (c) Meta Platforms, Inc. and affiliates.

/**
 * This test verifies that the ncclDebugLogger_t typedef matches the baseline
 * NCCL signature. This is critical for ABI compatibility with network plugins
 * (like OFI/EFA) that are compiled against baseline NCCL headers.
 *
 * The baseline NCCL ncclDebugLog signature has 5 parameters:
 *   void (*)(ncclDebugLogLevel level, unsigned long flags,
 *            const char *filefunc, int line, const char *fmt, ...)
 *
 * If NCCLx changes this to a 6-parameter version (separate file and func),
 * plugins compiled against baseline NCCL will crash when loaded.
 *
 * See: https://github.com/NVIDIA/nccl/blob/master/src/include/nccl_common.h
 */

#include <gtest/gtest.h>
#include <type_traits>

#include "debug.h"
#include "nccl_common.h"

// Expected baseline NCCL signature (5 parameters: level, flags, filefunc, line,
// fmt, ...)
using BaselineNcclDebugLogger_t = void (*)(
    ncclDebugLogLevel level,
    unsigned long flags,
    const char* filefunc,
    int line,
    const char* fmt,
    ...);

// Incompatible NCCLx signature (6 parameters: level, flags, file, func, line,
// fmt, ...) This is what we want to PREVENT
using IncompatibleNcclDebugLogger_t = void (*)(
    ncclDebugLogLevel level,
    unsigned long flags,
    const char* file,
    const char* func,
    int line,
    const char* fmt,
    ...);

TEST(DebugLoggerAbiTest, NcclDebugLoggerMatchesBaselineSignature) {
  // Verify ncclDebugLogger_t matches the 5-parameter baseline NCCL signature
  // This is a compile-time check - if the types don't match, compilation fails
  static_assert(
      std::is_same_v<ncclDebugLogger_t, BaselineNcclDebugLogger_t>,
      "ncclDebugLogger_t must match baseline NCCL 5-parameter signature "
      "(level, flags, filefunc, line, fmt, ...) for plugin ABI compatibility. "
      "Do NOT add separate 'file' and 'func' parameters.");

  // Runtime check as well for test reporting
  EXPECT_TRUE((std::is_same_v<ncclDebugLogger_t, BaselineNcclDebugLogger_t>))
      << "ncclDebugLogger_t does not match baseline NCCL signature";
}

TEST(DebugLoggerAbiTest, NcclDebugLoggerIsNotIncompatibleSignature) {
  // Verify ncclDebugLogger_t is NOT the incompatible 6-parameter signature
  static_assert(
      !std::is_same_v<ncclDebugLogger_t, IncompatibleNcclDebugLogger_t>,
      "ncclDebugLogger_t must NOT use 6-parameter signature "
      "(level, flags, file, func, line, fmt, ...) - this breaks plugin ABI "
      "compatibility");

  EXPECT_FALSE(
      (std::is_same_v<ncclDebugLogger_t, IncompatibleNcclDebugLogger_t>))
      << "ncclDebugLogger_t has incompatible 6-parameter signature that breaks "
         "plugins";
}

TEST(DebugLoggerAbiTest, NcclDebugLogFunctionMatchesTypedef) {
  // This assignment would fail to compile if ncclDebugLog has wrong signature
  // ncclDebugLog is declared in debug.h
  ncclDebugLogger_t loggerPtr = &ncclDebugLog;
  EXPECT_NE(loggerPtr, nullptr);
}
