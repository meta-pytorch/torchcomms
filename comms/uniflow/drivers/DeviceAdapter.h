// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

#include "comms/uniflow/Result.h"

namespace uniflow {

class CudaApi;
class CudaDriverApi;

using DeviceStream = void*;

/// Pure pointer metadata only: no device initialization, allocation or loading.
/// Registered implementations must not throw.
using DevicePointerResolver = std::function<uint64_t(const void*)>;

enum class DeviceCopyKind {
  HostToDevice,
  DeviceToHost,
  DeviceToDevice,
};

struct DeviceCapabilities {
  bool asyncCopy{false};
  bool streamSignal64{false};
  bool nonBlockingStreams{false};
  bool completion{false};
};

/// Owns a completion marker, not the copied buffers or the stream. The caller
/// must retain those resources until ready() returns true, wait() succeeds, or
/// the exact stream has been drained. Destruction does not wait for work.
class DeviceCompletion {
 public:
  virtual ~DeviceCompletion() = default;
  virtual Result<bool> ready() = 0;
  virtual Status wait() = 0;
};

/// How a device buffer is mapped when exported as a DMA-BUF.
enum class DmaBufMapping {
  /// Standard mapping. On Grace-based systems (e.g. GB300) NIC<->GPU traffic
  /// traverses the C2C link.
  Default,
  /// mlx5 Data Direct: map the buffer over the NIC's PCIe BAR1 path so the NIC
  /// DMAs straight to/from GPU HBM, bypassing the Grace C2C link. The exported
  /// fd must back an `mlx5dv_reg_dmabuf_mr` with
  /// `MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT`. Only meaningful on a Data-Direct-
  /// capable NIC; requires CUDA >= 12.8.
  DataDirect,
};

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
  /// Full length of the exported dma-buf (page-aligned, covering `offset` +
  /// `len`). mlx5 Data Direct registers the whole dma-buf from its aligned
  /// base (offset 0), so the DD MR uses this length rather than `len`.
  size_t dmaBufLen{0};
};

/// DeviceAdapter abstracts device-specific host pinned memory allocation
/// for use with DMA from a NIC or accelerator.
class DeviceAdapter {
 public:
  virtual ~DeviceAdapter() = default;

  /// Passive support query; it must not initialize a GPU or select a backend.
  virtual DeviceCapabilities capabilities() const noexcept {
    return {};
  }

  /// Operations use the context for deviceId and restore the caller's device.
  /// nullptr denotes that device's default stream. Whether a stream must be
  /// explicitly supplied is a transport policy, not an adapter policy.
  /// An error may follow a partial enqueue: retain buffers and drain the exact
  /// stream before releasing them. A failed drain is not completion.
  virtual Status copyAsync(
      int /* deviceId */,
      void* /* dst */,
      const void* /* src */,
      size_t /* count */,
      DeviceCopyKind /* kind */,
      DeviceStream /* stream */) {
    return Err(ErrCode::NotImplemented, "device asynchronous copy unavailable");
  }

  /// Signal a device-accessible address after earlier work on this same stream,
  /// preserving memory ordering. It has the same failure/lifetime rules as
  /// copy.
  virtual Status signal64(
      int /* deviceId */,
      DeviceStream /* stream */,
      uint64_t* /* address */,
      uint64_t /* value */) {
    return Err(ErrCode::NotImplemented, "device stream signal unavailable");
  }

  virtual Result<DeviceStream> createNonBlockingStream(int /* deviceId */) {
    return Err(ErrCode::NotImplemented, "device private streams unavailable");
  }

  /// Drain only the supplied stream, including a device-scoped default stream.
  virtual Status synchronizeStream(
      int /* deviceId */,
      DeviceStream /* stream */) {
    return Err(
        ErrCode::NotImplemented, "device stream synchronization unavailable");
  }

  /// Only destroy a private stream returned by this adapter, after its work has
  /// completed or been drained. Never destroy the default stream.
  virtual Status destroyStream(int /* deviceId */, DeviceStream /* stream */) {
    return Err(
        ErrCode::NotImplemented, "device stream destruction unavailable");
  }

  /// Mark work already enqueued on stream. ready()/wait() cover only that work,
  /// not later submissions. On failure, retain buffers and drain this stream.
  virtual Result<std::unique_ptr<DeviceCompletion>> recordCompletion(
      int /* deviceId */,
      DeviceStream /* stream */) {
    return Err(ErrCode::NotImplemented, "device completion unavailable");
  }

  /// Allocate page-aligned, host-pinned memory suitable for DMA from a NIC
  /// or accelerator.
  virtual Result<void*> pinnedHostAlloc(size_t size) = 0;

  /// Release memory previously returned by pinnedHostAlloc().
  virtual Status pinnedHostFree(void* ptr) = 0;

  /// Return a device-accessible pointer for `hostPtr` (which must have been
  /// obtained from pinnedHostAlloc()).
  virtual Result<void*> hostGetDevicePointer(void* hostPtr) = 0;

  /// Device-scoped pinned storage for GPU staging. The adapter retains vendor
  /// identity; callers retain deviceId through matching allocation/map/free.
  /// Legacy adapters may use context-independent host storage.
  virtual Result<void*> pinnedHostAlloc(int /* deviceId */, size_t size) {
    return pinnedHostAlloc(size);
  }

  virtual Status pinnedHostFree(int /* deviceId */, void* ptr) {
    return pinnedHostFree(ptr);
  }

  virtual Result<void*> hostGetDevicePointer(
      int /* deviceId */,
      void* hostPtr) {
    return hostGetDevicePointer(hostPtr);
  }

  /// Query whether DMA-BUF export is available for `deviceId`.
  virtual Result<bool> isDmaBuffSupported(int deviceId) = 0;

  /// Export a DMA-BUF descriptor for `[ptr, ptr + len)` on `deviceId`.
  /// `mapping` selects the NIC<->GPU route (see DmaBufMapping); `DataDirect`
  /// requires a Data-Direct-capable NIC and, on GB300, a buffer whose base is
  /// VMM-granularity (2 MB) aligned so the returned `offset` is 0.
  /// The caller must call `closeDmaBuff()` on the returned value once the
  /// resulting MR has been registered.
  virtual Result<DmaBuff> exportDmaBuff(
      int deviceId,
      void* ptr,
      size_t len,
      DmaBufMapping mapping = DmaBufMapping::Default) = 0;

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
