// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda_runtime_api.h> // @manual=third-party//cuda:cuda-lazy

#include <array>
#include <string>

#include "comms/uniflow/Result.h"

namespace uniflow {

/// Thin wrapper around CUDA runtime APIs.
/// All methods are virtual for mockability in unit tests.
/// Thread safety: CUDA runtime calls are internally thread-safe.
class CudaApi {
 public:
  virtual ~CudaApi() = default;

  // --- Device management ---

  virtual Status setDevice(int device);

  virtual Result<int> getDevice();

  virtual Result<bool> deviceCanAccessPeer(int device, int peerDevice);

  virtual Status deviceEnablePeerAccess(int peerDevice);

  virtual Result<int> getDeviceCount();

  virtual Status getDevicePCIBusId(char* pciBusId, int len, int device);

  /// Return the device architecture string. On AMD this is the gfx arch (e.g.
  /// "gfx942", possibly with feature flags appended); on NVIDIA it is
  /// "sm_<major><minor>".
  virtual Result<std::string> getDeviceArch(int device);

  // --- Host memory ---

  virtual Result<void*> hostAlloc(size_t size, unsigned int flags);

  virtual Status hostFree(void* ptr);

  virtual Result<void*> hostGetDevicePointer(void* hostPtr);

  // --- Memory copy ---

  virtual Status memcpyAsync(
      void* dst,
      const void* src,
      size_t count,
      cudaMemcpyKind kind,
      cudaStream_t stream);

#if CUDART_VERSION >= 12080
  // Batch DMA submission. Available on CUDA 12.8+; the underlying CUDA
  // signature changed in CUDA 13.0 (failIdx removed), but the wrapper
  // surface stays stable.
  virtual Status memcpyBatchAsync(
      void* const* dsts,
      const void* const* srcs,
      const size_t* sizes,
      size_t count,
      cudaStream_t stream);
#endif

  virtual Status memcpyPeerAsync(
      void* dst,
      int dstDevice,
      const void* src,
      int srcDevice,
      size_t count,
      cudaStream_t stream);

  /// Neutral device-to-device async copy: no vendor types in the signature, so
  /// callers above the driver seam (e.g. P2pTransport) stay un-hipified.
  /// @p stream is an opaque stream handle (cudaStream_t/hipStream_t) or nullptr
  /// for the default stream.
  virtual Status memcpyDeviceToDeviceAsync(
      void* dst,
      const void* src,
      size_t count,
      void* stream);

  /// Neutral stream synchronize: waits for all work on the opaque @p stream
  /// (cudaStream_t/hipStream_t; nullptr = default). Distinct name from the
  /// vendor-typed streamSynchronize to avoid an overload ambiguity for callers
  /// that pass a literal 0/nullptr.
  virtual Status synchronizeStream(void* stream);

  // --- Stream ---

  virtual Status streamSynchronize(cudaStream_t stream);

  // --- Event ---

  virtual Status eventCreate(cudaEvent_t* event);

  virtual Status eventRecord(cudaEvent_t event, cudaStream_t stream);

  /// Query whether a recorded event has completed.
  /// Returns true if the event has completed, false if still in-flight.
  virtual Result<bool> eventQuery(cudaEvent_t event);

  virtual Status eventDestroy(cudaEvent_t event);

  // --- Event (neutral) ---

  /// Opaque event handle. `cudaEvent_t`/`hipEvent_t` is a pointer type, so it
  /// round-trips through `void*`. These wrappers let callers above the seam do
  /// event-based completion (create → record → poll → destroy) without naming a
  /// vendor type, mirroring the neutral memcpy/stream wrappers above. They
  /// delegate to the vendor-typed event methods, so a mock that stubs those is
  /// exercised transparently.
  virtual Result<void*> createEvent();
  virtual Status recordEvent(void* event, void* stream);
  virtual Result<bool> queryEvent(void* event);
  virtual Status destroyEvent(void* event);

  // --- IPC (cross-process device memory sharing) ---

  /// Opaque IPC handle bytes. `cudaIpcMemHandle_t` / `hipIpcMemHandle_t` is a
  /// fixed 64-byte POD (`HIP_IPC_HANDLE_SIZE`); exposing it as a neutral byte
  /// array lets callers above the driver seam serialize/exchange handles
  /// without naming a vendor type (so they need no hipification).
  static constexpr size_t kIpcMemHandleSize = 64;
  using IpcMemHandle = std::array<uint8_t, kIpcMemHandleSize>;

  /// Export an IPC handle for a device allocation (cudaIpcGetMemHandle).
  virtual Result<IpcMemHandle> ipcGetMemHandle(void* devPtr);

  /// Open a peer's IPC handle, returning a device pointer mapped into this
  /// process (cudaIpcOpenMemHandle, lazy peer-access enable).
  virtual Result<void*> ipcOpenMemHandle(const IpcMemHandle& handle);

  /// Close a device pointer previously opened via ipcOpenMemHandle.
  virtual Status ipcCloseMemHandle(void* devPtr);

  /// Allocation base + size for a device pointer. Lets a segment that is a
  /// sub-range of a larger allocation be IPC-exported at its allocation base
  /// with the correct offset recorded. AMD uses the HIP runtime address-range
  /// API; NVIDIA is a runtime-only no-op that reports the pointer as its own
  /// base (whole-allocation / offset 0), leaving the NVIDIA path unchanged.
  struct MemRange {
    void* base{nullptr};
    size_t size{0};
  };
  virtual Result<MemRange> getMemAddressRange(void* devPtr);
};

/// RAII guard that saves the current CUDA device on construction and
/// restores it on destruction. Use this to avoid leaking device state
/// when temporarily switching devices.
class CudaDeviceGuard {
 public:
  explicit CudaDeviceGuard(CudaApi& api, int device);

  ~CudaDeviceGuard();

  CudaDeviceGuard(const CudaDeviceGuard&) = delete;
  CudaDeviceGuard(CudaDeviceGuard&&) = delete;
  CudaDeviceGuard& operator=(const CudaDeviceGuard&) = delete;
  CudaDeviceGuard& operator=(CudaDeviceGuard&&) = delete;

 private:
  CudaApi& api_;
  int prevDevice_{-1};
};

} // namespace uniflow
