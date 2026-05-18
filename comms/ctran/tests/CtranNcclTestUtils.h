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
      size_t numSegments = 2,
      bool refCheck = true);
};

// RAII wrapper around prepareBuf/releaseBuf for automatic lifetime management.
// Defaults to kMemNcclMemAlloc for NVL-registered buffers.
class TestDeviceBuffer {
 public:
  TestDeviceBuffer() = default;

  explicit TestDeviceBuffer(
      size_t size,
      MemAllocType memType = kMemNcclMemAlloc,
      size_t numSegments = 2,
      bool refCheck = false)
      : size_(size),
        memType_(memType),
        numSegments_(numSegments),
        refCheck_(refCheck) {
    ptr_ =
        CtranNcclTestHelpers::prepareBuf(size, memType, segments_, numSegments);
  }

  ~TestDeviceBuffer() {
    if (ptr_) {
      CtranNcclTestHelpers::releaseBuf(
          ptr_, size_, memType_, numSegments_, refCheck_);
    }
  }

  TestDeviceBuffer(const TestDeviceBuffer&) = delete;
  TestDeviceBuffer& operator=(const TestDeviceBuffer&) = delete;

  TestDeviceBuffer(TestDeviceBuffer&& other) noexcept
      : ptr_(other.ptr_),
        size_(other.size_),
        memType_(other.memType_),
        numSegments_(other.numSegments_),
        refCheck_(other.refCheck_),
        segments_(std::move(other.segments_)) {
    other.ptr_ = nullptr;
    other.size_ = 0;
  }

  TestDeviceBuffer& operator=(TestDeviceBuffer&& other) noexcept {
    if (this != &other) {
      if (ptr_) {
        CtranNcclTestHelpers::releaseBuf(
            ptr_, size_, memType_, numSegments_, refCheck_);
      }
      ptr_ = other.ptr_;
      size_ = other.size_;
      memType_ = other.memType_;
      numSegments_ = other.numSegments_;
      refCheck_ = other.refCheck_;
      segments_ = std::move(other.segments_);
      other.ptr_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  void* get() const {
    return ptr_;
  }

  size_t size() const {
    return size_;
  }

 private:
  void* ptr_{nullptr};
  size_t size_{0};
  MemAllocType memType_{};
  size_t numSegments_{2};
  bool refCheck_{true};
  std::vector<TestMemSegment> segments_{};
};

} // namespace ctran
