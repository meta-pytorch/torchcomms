// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "comms/uniflow/Result.h"

namespace uniflow {

class CudaApi;
class CudaDriverApi;

/// Description of a DMA-BUF region exported from device memory and
/// suitable for use with `ibv_reg_dmabuf_mr`.
///
struct DmaBuff {
  /// dma-buf fd to pass to `ibv_reg_dmabuf_mr`. -1 means unset.
  int fd{-1};
  /// Byte offset of the buffer inside the dma-buf described by `fd`.
  uint64_t offset{0};
  /// Length of the registered region in bytes.
  size_t len{0};
  /// IOVA for the NIC. Per-export backends typically set this to the
  /// original VA so the NIC can reference device memory directly;
  /// pool-backed backends use 0.
  uint64_t iova{0};
  /// Registration bbase for this DmaBuff to be used when creating WQEs.
  /// This is to be subtracted from a device pointer to form the WQE address.
  uint64_t registrationBase{0};
};

/// DeviceAdapter abstracts device-specific host pinned memory allocation
/// for use with DMA from a NIC or accelerator.
class DeviceAdapter {
 public:
  virtual ~DeviceAdapter() = default;

  /// Allocate page-aligned, host-pinned memory suitable for DMA from a NIC
  /// or accelerator.
  virtual Result<void*> pinnedHostAlloc(size_t size) = 0;

  /// Release memory previously returned by pinnedHostAlloc().
  virtual Status pinnedHostFree(void* ptr) = 0;

  /// Return a device-accessible pointer for `hostPtr` (which must have been
  /// obtained from pinnedHostAlloc()).
  virtual Result<void*> hostGetDevicePointer(void* hostPtr) = 0;

  /// Query whether DMA-BUF export is available for `deviceId`.
  virtual Result<bool> isDmaBuffSupported(int deviceId) = 0;

  /// Export a DMA-BUF descriptor for `[ptr, ptr + len)` on `deviceId`.
  /// The caller must call `closeDmaBuff()` on the returned value once the
  /// resulting MR has been registered.
  virtual Result<DmaBuff>
  exportDmaBuff(int deviceId, void* ptr, size_t len) = 0;

  /// Translate a client-supplied device pointer (which may be an opaque
  /// allocation handle / UID) to the bare device address the NIC needs to
  /// subtract `registrationBase` from. Default: identity cast — backends
  /// that have no separate UID representation (CPU, CUDA) do not need to
  /// override this. Backends whose pointer encodes a device ordinal (or
  /// other metadata) must strip it here before any wire-address arithmetic.
  virtual uint64_t resolveDevicePointer(const void* ptr) const noexcept {
    return reinterpret_cast<uint64_t>(ptr);
  }

  /// Whether a failed or unavailable DMA-BUF export may fall back to a plain
  /// `ibv_reg_mr` over the segment's host pointer.
  ///
  /// True (default) for backends whose accelerator pointers are real process
  /// virtual addresses (e.g. CUDA): there DMA-BUF is an optimization for GPU
  /// Direct RDMA and a plain MR is a valid, if degraded, path.
  ///
  /// False for backends whose pointers are not host-registerable.
  virtual bool allowsRegMrFallback() const noexcept {
    return true;
  }

  /// Release any resources held by `buff`. For per-export backends this
  /// closes the fd; for pool-backed backends it is a no-op because the
  /// fd is owned by the device runtime and shared across pool members.
  virtual Status closeDmaBuff(DmaBuff& buff) = 0;
};

/// Returns a DeviceAdapter for the current build configuration.
///
/// The CUDA implementation accepts an optional CudaApi and CudaDriverApi
/// for dependency injection in tests; if null, default instances are
/// constructed. These parameters are ignored by non-CUDA implementations.
std::shared_ptr<DeviceAdapter> createDeviceAdapter(
    std::shared_ptr<CudaApi> cudaApi = nullptr,
    std::shared_ptr<CudaDriverApi> cudaDriverApi = nullptr);

/// Whether the NVLink transport is usable on this build's accelerator.
/// Selected at link time by the platform's backend: true for CUDA (NVLink
/// is an NVIDIA technology), false for backends with no NVLink. Prefer this
/// over constructing a DeviceAdapter to read the constant, since adapter
/// construction may load backend runtimes (e.g. libcuda) unnecessarily.
bool isNvlinkAvailable();

} // namespace uniflow
