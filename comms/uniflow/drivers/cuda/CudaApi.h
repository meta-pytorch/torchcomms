// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime_api.h> // @manual=third-party//cuda:cuda-lazy
#endif

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

  // --- Host memory ---

  virtual Result<void*> hostAlloc(size_t size, unsigned int flags);

  virtual Status hostFree(void* ptr);

  virtual Result<void*> hostGetDevicePointer(void* hostPtr);

  // --- Memory copy ---

  virtual Status memcpyAsync(
      void* dst,
      const void* src,
      size_t count,
#ifdef __HIP_PLATFORM_AMD__
      hipMemcpyKind kind,
      hipStream_t stream);
#else
      cudaMemcpyKind kind,
      cudaStream_t stream);
#endif

#if !defined(__HIP_PLATFORM_AMD__) && CUDART_VERSION >= 12080
  // Batch DMA submission. Available on CUDA 12.8+; the underlying CUDA
  // signature changed in CUDA 13.0 (failIdx removed), but the wrapper
  // surface stays stable.
  // Note: Not available on AMD/HIP - use individual memcpyAsync calls instead.
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
#ifdef __HIP_PLATFORM_AMD__
      hipStream_t stream);
#else
      cudaStream_t stream);
#endif

  // --- Stream ---

#ifdef __HIP_PLATFORM_AMD__
  virtual Status streamSynchronize(hipStream_t stream);
#else
  virtual Status streamSynchronize(cudaStream_t stream);
#endif

  // --- Event ---

#ifdef __HIP_PLATFORM_AMD__
  virtual Status eventCreate(hipEvent_t* event);

  virtual Status eventRecord(hipEvent_t event, hipStream_t stream);

  /// Query whether a recorded event has completed.
  /// Returns true if the event has completed, false if still in-flight.
  virtual Result<bool> eventQuery(hipEvent_t event);

  virtual Status eventDestroy(hipEvent_t event);
#else
  virtual Status eventCreate(cudaEvent_t* event);

  virtual Status eventRecord(cudaEvent_t event, cudaStream_t stream);

  /// Query whether a recorded event has completed.
  /// Returns true if the event has completed, false if still in-flight.
  virtual Result<bool> eventQuery(cudaEvent_t event);

  virtual Status eventDestroy(cudaEvent_t event);
#endif
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
