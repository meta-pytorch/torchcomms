// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// RAII wrapper for HIP device memory allocation.
// Consolidates the HipDeviceBuffer class previously duplicated across
// multiple AMD benchmark and test files.

#pragma once

#include <cstddef>

#include <hip/hip_runtime.h>

namespace pipes_gda {

class HipDeviceBuffer {
 public:
  explicit HipDeviceBuffer(size_t size) : size_(size) {
    hipError_t err = hipMalloc(&ptr_, size);
    if (err != hipSuccess) {
      ptr_ = nullptr;
    }
  }

  ~HipDeviceBuffer() {
    if (ptr_) {
      (void)hipFree(ptr_);
    }
  }

  HipDeviceBuffer(const HipDeviceBuffer&) = delete;
  HipDeviceBuffer& operator=(const HipDeviceBuffer&) = delete;

  void* get() const {
    return ptr_;
  }

  size_t size() const {
    return size_;
  }

  explicit operator bool() const {
    return ptr_ != nullptr;
  }

 private:
  void* ptr_{nullptr};
  size_t size_{0};
};

} // namespace pipes_gda
