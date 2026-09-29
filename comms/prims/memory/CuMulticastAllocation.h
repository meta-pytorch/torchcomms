// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

// `<cuda.h>` (driver API) and `<cuda_runtime.h>` are NVIDIA-only. Multicast
// is NVIDIA-only; on AMD the impl bodies throw and `CuMemAllocation.h`
// provides the stub `CUdevice` / `CUmemGenericAllocationHandle` typedefs the
// always-declared API references.
#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#else
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <cstddef>

#include "comms/prims/memory/CuMemAllocation.h"

namespace comms::prims {

#if defined(__HIP_PLATFORM_AMD__)
// Stub the multicast-only CUDA driver-API type so the always-declared API
// surface (create(prop)) compiles on AMD. The create() body throws on AMD.
struct CUmulticastObjectProp {
  unsigned char _padding[1];
};
#endif

/**
 * CuMulticastAllocation - RAII owner of one CUDA multicast object handle (a
 * `CUmemGenericAllocationHandle` returned by cuMulticastCreate, or imported on
 * peers).
 *
 * It owns only the multicast object handle. Mapping the multicast VA is
 * CuMemMapping's job; binding a physical allocation into the object is done via
 * bindMem(). Like CuMemAllocation, it is meant to be co-owned via
 * std::shared_ptr so a CuMemMapping can keep the multicast object alive for as
 * long as its VA mapping exists.
 *
 * Two ways to obtain one:
 *  - create(prop): cuMulticastCreate (team rank 0).
 *  - adopt(handle, size): take ownership of an imported multicast handle
 * (peers, after NvlMemExchange::importShareableHandle).
 *
 * The destructor cuMulticastUnbind()s the local binding (if bindMem ran) then
 * cuMemRelease()s the handle. Drop the multicast VA mapping (CuMemMapping)
 * BEFORE destroying this object. Not copyable; movable. Requires CUDA 12.3+.
 */
class CuMulticastAllocation {
 public:
  /** cuMulticastCreate a multicast object from `prop` (team rank 0). */
  static CuMulticastAllocation create(const CUmulticastObjectProp& prop);

  /**
   * Take ownership of an imported multicast object handle (peers). `size` is
   * the multicast object size (used for unbind at teardown).
   */
  static CuMulticastAllocation adopt(
      CUmemGenericAllocationHandle handle,
      std::size_t size);

  ~CuMulticastAllocation();

  CuMulticastAllocation(CuMulticastAllocation&& other) noexcept;
  CuMulticastAllocation& operator=(CuMulticastAllocation&& other) noexcept;

  CuMulticastAllocation(const CuMulticastAllocation&) = delete;
  CuMulticastAllocation& operator=(const CuMulticastAllocation&) = delete;

  /**
   * cuMulticastAddDevice: register `cuDev` with the multicast object.
   *
   * Single-device only: the destructor unbinds exactly one device (`device_`),
   * so calling addDevice() more than once on the same object would leak the
   * earlier devices' bindings. Subsequent calls throw to surface the misuse
   * rather than silently overwriting `device_`.
   */
  void addDevice(CUdevice cuDev);

  /**
   * cuMulticastBindMem: bind `physHandle` into the multicast object at
   * `mcOffset` (physical offset `physOffset`) for `size` bytes. Records the
   * binding so the destructor unbinds it.
   *
   * Single-bind only: the destructor unbinds exactly one (mcOffset, size)
   * range. Calling bindMem() more than once on the same object would leak
   * every binding except the last; subsequent calls throw to surface the
   * misuse rather than silently overwriting the recorded range.
   */
  void bindMem(
      CUmemGenericAllocationHandle physHandle,
      std::size_t mcOffset,
      std::size_t physOffset,
      std::size_t size);

  CUmemGenericAllocationHandle handle() const {
    return handle_;
  }

  /** The local device registered via addDevice() (0 before addDevice). */
  CUdevice device() const {
    return device_;
  }

  std::size_t size() const {
    return size_;
  }

 private:
  CuMulticastAllocation(CUmemGenericAllocationHandle handle, std::size_t size);

  void release() noexcept;

  CUmemGenericAllocationHandle handle_{0};
  CUdevice device_{0};
  // `size_` is the multicast OBJECT size (from cuMulticastCreate / passed to
  // adopt) and is the value the public `size()` accessor returns. `boundSize_`
  // + `boundMcOffset_` are the (size, mcOffset) actually passed to bindMem();
  // they're only meaningful when `bound_` is true and are what the destructor
  // passes to cuMulticastUnbind. They are distinct from `size_` because
  // adopt() never binds (peers import the handle but typically don't rebind),
  // and they may differ if bindMem is called with a partial range.
  std::size_t size_{0};
  std::size_t boundSize_{0};
  std::size_t boundMcOffset_{0};
  bool deviceAdded_{false};
  bool bound_{false};
};

} // namespace comms::prims
