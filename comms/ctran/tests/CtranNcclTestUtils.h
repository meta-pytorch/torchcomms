// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <vector>

#include "comms/ctran/tests/CtranTestUtils.h"

#if !defined(USE_ROCM)
#include <nccl.h> // @manual
#endif

namespace ctran {

// ============================================================================
// NCCL-Specific Test Macros
// ============================================================================

#if !defined(USE_ROCM)
// Macro for checking NCCL results in tests
#define NCCLCHECK_TEST(cmd)                  \
  do {                                       \
    ncclResult_t r = cmd;                    \
    if (r != ncclSuccess) {                  \
      printf(                                \
          "Failed, NCCL error %s:%d '%s'\n", \
          __FILE__,                          \
          __LINE__,                          \
          ncclGetErrorString(r));            \
      exit(EXIT_FAILURE);                    \
    }                                        \
  } while (0)
#endif // !defined(USE_ROCM)

// ============================================================================
// NCCL-Specific Memory Helpers
// ============================================================================

#if !defined(USE_ROCM)
// Helper for freeing NCCL memory with reference count verification.
// This ensures all internal handles have been properly released.
ncclResult_t ncclMemFreeWithRefCheck(void* ptr);
#endif // !defined(USE_ROCM)

// ============================================================================
// NCCL-Specific Test Helpers
// ============================================================================

// CtranNcclTestHelpers extends CtranTestHelpers with NCCL memory support.
// Use this class when tests need to allocate memory with kMemNcclMemAlloc
// or kCuMemAllocDisjoint memory types.
//
// For tests that only use kMemCudaMalloc, use CtranTestHelpers directly.
class CtranNcclTestHelpers : public CtranTestHelpers {
 public:
  CtranNcclTestHelpers() = default;

  // Prepare buffer with specified memory type.
  // Supports NCCL memory allocation types:
  // - kMemCudaMalloc: Uses cudaMalloc
  // - kMemNcclMemAlloc: Uses ncclMemAlloc
  // - kCuMemAllocDisjoint: Uses commMemAllocDisjoint directly
  // For kCuMemAllocDisjoint, numSegments specifies how many disjoint physical
  // segments to allocate (default: 2).
  static void* prepareBuf(
      size_t bufSize,
      MemAllocType memType,
      std::vector<TestMemSegment>& segments,
      size_t numSegments = 2);

  // Release buffer allocated by prepareBuf.
  // For kCuMemAllocDisjoint, numSegments must match the value used in
  // prepareBuf.
  static void releaseBuf(
      void* buf,
      size_t bufSize,
      MemAllocType memType,
      size_t numSegments = 2);
};

} // namespace ctran
