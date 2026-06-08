// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// =============================================================================
// HipHostCompat - HIP/AMD host-side compatibility layer (pipes-local)
// =============================================================================
//
// Provides two small HIP-side shims used by cross-platform pipes code:
//
// 1. **`__trap()` shim** — HIP doesn't expose `__trap()` in host pass the
//    way nvcc does. Aliases `__trap()` to `abort()` in the device pass so
//    cross-platform headers like `IbgdaBuffer.h` and `Timeout.cuh` that
//    use `__trap()` for fatal-error paths compile cleanly under hipcc.
//    Kept here (not in `HipDeviceCompat.h`) because it's lightweight and
//    used by cross-platform consumers that don't otherwise need AMD GCN
//    intrinsics.
//
// 2. **`meta::comms::DeviceBuffer`** — HIP-backed substitute for the
//    NVIDIA `comms::utils::DeviceBuffer` (CudaRAII) used by pipes test
//    runners. Provides hipMalloc-based RAII with the minimal interface
//    those runners need. Active only on AMD (`__HIP_PLATFORM_AMD__`).
//
// Scope: pipes-only. Lives in `comms/prims/transport/amd/` so the
// cross-platform plumbing stays scoped to this directory.
//
// For AMD GCN device-side intrinsic shims (used by IBGDA WQE construction
// + `pipes_gda_*` device code), see `HipDeviceCompat.h` instead.
// =============================================================================

#pragma once

// ---------------------------------------------------------------------------
// (1) __trap() shim — pull in hip_runtime.h so device-side blockIdx /
// threadIdx are visible to any header using them (e.g. printf-before-trap
// diagnostics in IbgdaBuffer.h).
// ---------------------------------------------------------------------------

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__)
#include <hip/hip_runtime.h>
#define __trap() abort()
#endif

// ---------------------------------------------------------------------------
// (2) HIP-backed `meta::comms::DeviceBuffer` substitute for AMD test runners
// ---------------------------------------------------------------------------
//
// The Buck AMD build path runs HIPify on `.cu`/`.cc` source files, which
// auto-renames `cuda*` runtime calls (`cudaSetDevice`, `cudaMemcpy`, etc.)
// to their `hip*` counterparts before compilation. So no macro renames
// are needed here for the runtime API surface.
//
// What HIPify does NOT do is provide a HIP-backed equivalent of
// `comms::utils::DeviceBuffer` (a CudaRAII wrapper that #includes
// `<cuda_runtime.h>`). On AMD, `<cuda_runtime.h>` doesn't exist and
// `comms::utils::DeviceBuffer` can't be linked. The substitute below
// uses the same name + namespace so test runners need no changes.

#ifdef __HIP_PLATFORM_AMD__

#include <cstddef>

#include <hip/hip_runtime_api.h>

// When comms/utils/CudaRAII.h is on the include path (e.g. via ctran deps),
// HIPify translates its CUDA calls to HIP, so the real DeviceBuffer/CudaEvent
// work on AMD — no substitutes needed. Only define substitutes for pipes-only
// AMD targets that don't link comms/utils:cuda_raii.
#if !__has_include("comms/utils/CudaRAII.h")

namespace meta::comms {

class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t size) : size_(size) {
    (void)hipMalloc(&ptr_, size);
  }

  ~DeviceBuffer() {
    if (ptr_ != nullptr) {
      (void)hipFree(ptr_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      if (ptr_ != nullptr) {
        (void)hipFree(ptr_);
      }
      ptr_ = other.ptr_;
      size_ = other.size_;
      other.ptr_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  void* get() const {
    return ptr_;
  }

  std::size_t size() const {
    return size_;
  }

 private:
  void* ptr_{nullptr};
  std::size_t size_{0};
};

// HIP-backed substitute for `comms::utils::CudaEvent`. Same name + namespace
// so benchmark/test code that timestamps with `cudaEvent_t` via this RAII
// wrapper compiles unchanged; surrounding `cudaEventRecord(ev.get(), ...)`
// calls get HIPify-rewritten to `hipEventRecord(ev.get(), ...)`.
class CudaEvent {
 public:
  CudaEvent() {
    (void)hipEventCreate(&event_);
  }

  ~CudaEvent() {
    if (event_ != nullptr) {
      (void)hipEventDestroy(event_);
    }
  }

  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;

  CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) {
    other.event_ = nullptr;
  }

  CudaEvent& operator=(CudaEvent&& other) noexcept {
    if (this != &other) {
      if (event_ != nullptr) {
        (void)hipEventDestroy(event_);
      }
      event_ = other.event_;
      other.event_ = nullptr;
    }
    return *this;
  }

  hipEvent_t get() const {
    return event_;
  }

 private:
  hipEvent_t event_{nullptr};
};

} // namespace meta::comms

#endif // !__has_include("comms/utils/CudaRAII.h")

#endif // __HIP_PLATFORM_AMD__
