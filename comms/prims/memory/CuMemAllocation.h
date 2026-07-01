// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

// `<cuda.h>` (driver API) and `<cuda_runtime.h>` are NVIDIA-only. On AMD the
// concrete VMM driver-API calls are unavailable and the impl bodies throw;
// stub the driver-API typedefs below so this header (and downstream consumers)
// still compiles. Mirrors NvlMemExchange.h.
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#else
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <cstddef>
#include <memory>

namespace comms::prims {

#if defined(__HIP_PLATFORM_AMD__)
// Stub CUDA driver-API typedefs the always-declared API references; concrete
// VMM calls are NVIDIA-only and live behind
// `#if !defined(__HIP_PLATFORM_AMD__) && CUDART_VERSION >= 12030` in the .cc.
// Types match real CUDA driver-API typedefs (`unsigned long long` /
// `int` / empty struct placeholder for CUmemAllocationProp).
using CUdevice = int;
using CUmemGenericAllocationHandle = unsigned long long;
using CUdeviceptr = unsigned long long;
struct CUmemAllocationProp {
  unsigned char _padding[1];
};
struct CUmemAccessDesc {
  unsigned char _padding[1];
};
#endif

/**
 * Builds a CUmemAllocationProp for a PINNED device allocation on `cuDev` with
 * the given requested handle-types mask. Sets gpuDirectRDMACapable when the
 * device reports GPUDirect-RDMA support. `requestedHandleTypesMask` may combine
 * multiple CU_MEM_HANDLE_TYPE_* flags so cuMemCreate can request both fabric
 * and POSIX FD support at once.
 *
 * Lives with CuMemAllocation (the physical-allocation owner) so that both
 * CuMemAllocation and NvlMemExchange can build props without a dependency
 * cycle.
 */
CUmemAllocationProp makeVmmAllocationProp(
    CUdevice cuDev,
    unsigned int requestedHandleTypesMask);

/**
 * CuMemAllocation - RAII owner of ONE physical CUDA VMM allocation handle
 * (`CUmemGenericAllocationHandle`).
 *
 * It owns only the physical handle: it does not reserve or map any virtual
 * address (that is CuMemMapping's job). It is meant to be co-owned via
 * std::shared_ptr by the things that map it (CuMemMapping) and by the higher
 * level handlers (GpuMemHandler's unicast allocation, MultimemHandler's
 * backing), so one physical allocation can be shared by value-RAII rather than
 * by leaking the raw handle.
 *
 * Three ways to obtain one — all three factories return a `unique_ptr` that
 * the caller can either keep as unique ownership or promote to `shared_ptr`
 * via implicit conversion:
 *  - create(): cuMemCreate a fresh physical allocation (requests fabric + POSIX
 *    FD with a POSIX-FD-only fallback when fabric is unavailable).
 *  - retain(ptr): cuMemRetainAllocationHandle an existing externally-mapped VMM
 *    buffer (the returned object owns that retain reference).
 *  - adopt(handle, ...): take ownership of an already-owned handle (e.g. one
 *    returned by NvlMemExchange::importShareableHandle).
 *
 * Each factory takes ownership of any handle it touches atomically: on
 * internal failure (CUDA error, allocator `bad_alloc`) the handle is released
 * before the exception propagates, so the caller never sees a leaked raw
 * handle and never has to write release-on-throw bookkeeping.
 *
 * In every case the destructor cuMemRelease()s the owned reference once. Not
 * copyable; movable (the moved-from object releases nothing). Requires CUDA
 * 12.3+; the factories throw on older toolkits.
 */
class CuMemAllocation {
 public:
  /**
   * cuMemCreate a fresh physical allocation on `cuDev`. Queries the RECOMMENDED
   * granularity, rounds `size` up to max(driverGranularity, alignFloor), then
   * cuMemCreate requesting `requestedHandleTypesMask`. If fabric was requested
   * but cuMemCreate reports CUDA_ERROR_NOT_PERMITTED / NOT_SUPPORTED (e.g. a
   * fabric-capable GPU without IMEX) it drops fabric and retries with the
   * remaining types. `alignFloor` must be 0 or a power of two; it raises the
   * size/granularity floor so the allocation can satisfy a larger granularity
   * requirement (e.g. a multicast object's granularity).
   */
  static std::unique_ptr<CuMemAllocation> create(
      CUdevice cuDev,
      std::size_t size,
      unsigned int requestedHandleTypesMask,
      std::size_t alignFloor = 0);

  /**
   * Retain the physical handle backing an existing VMM device pointer via
   * cuMemRetainAllocationHandle. The size is taken from cuMemGetAddressRange
   * and the supported handle types / device from the allocation properties.
   * Throws if `ptr` is not a VMM allocation.
   */
  static std::unique_ptr<CuMemAllocation> retain(void* ptr);

  /**
   * Take ownership of an already-owned physical handle (the caller transfers
   * one reference; the destructor releases it). Queries the handle's
   * properties + granularity internally so the caller does not have to hold
   * the raw handle past this call. Used for handles produced by
   * NvlMemExchange::importShareableHandle.
   */
  static std::unique_ptr<CuMemAllocation>
  adopt(CUmemGenericAllocationHandle handle, CUdevice device, std::size_t size);

  ~CuMemAllocation();

  CuMemAllocation(CuMemAllocation&& other) noexcept;
  CuMemAllocation& operator=(CuMemAllocation&& other) noexcept;

  CuMemAllocation(const CuMemAllocation&) = delete;
  CuMemAllocation& operator=(const CuMemAllocation&) = delete;

  CUmemGenericAllocationHandle handle() const {
    return handle_;
  }

  CUdevice device() const {
    return device_;
  }

  std::size_t size() const {
    return size_;
  }

  std::size_t granularity() const {
    return granularity_;
  }

  unsigned int supportedHandleTypes() const {
    return supportedHandleTypes_;
  }

  bool supportsFabric() const;

 private:
  // Pure scalar field init — cannot throw. Marked noexcept so the factories'
  // release-on-throw try/catch only has to worry about the `new` storage
  // allocation (the only remaining throw window), not the construction itself.
  CuMemAllocation(
      CUmemGenericAllocationHandle handle,
      CUdevice device,
      std::size_t size,
      std::size_t granularity,
      unsigned int supportedHandleTypes) noexcept;

  void release() noexcept;

  CUmemGenericAllocationHandle handle_{0};
  CUdevice device_{0};
  std::size_t size_{0};
  std::size_t granularity_{0};
  unsigned int supportedHandleTypes_{0};
};

} // namespace comms::prims
