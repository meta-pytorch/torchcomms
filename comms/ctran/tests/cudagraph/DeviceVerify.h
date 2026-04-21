// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace ctran::testing {

// Launch a kernel that compares two buffers byte-by-byte on the GPU.
// Mismatches are counted via atomicAdd into mismatchCount_d.
// Fully async — enqueued on `stream`, no host sync.
void launchCompareBuffers(
    const void* actual,
    const void* expected,
    size_t numBytes,
    unsigned int* mismatchCount_d,
    cudaStream_t stream);

// RAII wrapper for a device-side mismatch counter.
// Accumulates mismatches across multiple launchCompareBuffers calls.
// Call assertZero() after final device sync to check results.
class DeviceMismatchCounter {
 public:
  DeviceMismatchCounter() {
    cudaMalloc(&ptr_, sizeof(unsigned int));
    cudaMemset(ptr_, 0, sizeof(unsigned int));
  }

  ~DeviceMismatchCounter() {
    if (ptr_) {
      cudaFree(ptr_);
    }
  }

  DeviceMismatchCounter(const DeviceMismatchCounter&) = delete;
  DeviceMismatchCounter& operator=(const DeviceMismatchCounter&) = delete;

  unsigned int* ptr() {
    return ptr_;
  }

  unsigned int get() {
    unsigned int val = 0;
    cudaMemcpy(&val, ptr_, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    return val;
  }

  void assertZero() {
    EXPECT_EQ(get(), 0u) << "Device-side buffer comparison found mismatches";
  }

 private:
  unsigned int* ptr_{nullptr};
};

} // namespace ctran::testing
