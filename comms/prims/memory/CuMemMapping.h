// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

// `<cuda.h>` (driver API) and `<cuda_runtime.h>` are NVIDIA-only. On AMD the
// concrete VMM driver-API calls are unavailable and the impl bodies throw;
// `CuMemAllocation.h` declares the matching AMD stub typedefs for `CUdevice`
// etc., which this header reuses transitively.
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#else
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <cstddef>
#include <memory>

#include "comms/prims/memory/CuMemAllocation.h"

namespace comms::prims {

class CuMemAllocation;
class CuMulticastAllocation;

// RAII virtual-address mapping over a physical / multicast allocation.
//
// Runs the standard reserve/map/setAccess sequence used by both per-rank
// backing allocations (CuMemAllocation) and multicast objects
// (CuMulticastAllocation):
//
//   cuMemAddressReserve -> virtual address
//   cuMemMap            -> bind handle to virtual address
//   cuMemSetAccess      -> RW from the allocation's device
//
// The mapping does NOT release the handle: destruction only tears down the VA
// (cuMemUnmap + cuMemAddressFree). It DOES co-own the allocation that produced
// the handle, via a type-erased std::shared_ptr<void> keepAlive_, so the
// physical / multicast handle stays alive at least as long as this VA mapping.
// Drop order is correct: ~CuMemMapping unmaps/frees the VA first, then the
// keepAlive_ shared_ptr drops (releasing the handle only after the VA is gone).
//
// Construct via overAllocation() / overMulticast(). Movable so peer mappings
// can live in a std::vector. Not copyable. Requires CUDA 12.3+; the factories
// throw on older toolkits.
class CuMemMapping {
 public:
  // Map `alloc`'s physical handle into a fresh VA of `size` bytes aligned to
  // `granularity`, granting RW access on the allocation's device. The returned
  // mapping co-owns `alloc`.
  static CuMemMapping overAllocation(
      std::shared_ptr<CuMemAllocation> alloc,
      std::size_t size,
      std::size_t granularity);

  // Map `overlay`'s multicast handle into a fresh VA of `size` bytes aligned to
  // `granularity`, granting RW access on the device registered with the
  // multicast object. The returned mapping co-owns `overlay`.
  static CuMemMapping overMulticast(
      std::shared_ptr<CuMulticastAllocation> overlay,
      std::size_t size,
      std::size_t granularity);

  ~CuMemMapping();

  CuMemMapping(CuMemMapping&& other) noexcept;
  CuMemMapping& operator=(CuMemMapping&& other) noexcept;

  CuMemMapping(const CuMemMapping&) = delete;
  CuMemMapping& operator=(const CuMemMapping&) = delete;

  CUdeviceptr devicePtr() const {
    return devicePtr_;
  }

  std::size_t size() const {
    return size_;
  }

 private:
  CuMemMapping(
      CUdevice cuDev,
      CUmemGenericAllocationHandle handle,
      std::size_t size,
      std::size_t granularity,
      std::shared_ptr<void> keepAlive);

  void cleanup() noexcept;

  CUdeviceptr devicePtr_{0};
  std::size_t size_{0};
  bool mapped_{false};
  std::shared_ptr<void> keepAlive_;
};

} // namespace comms::prims
