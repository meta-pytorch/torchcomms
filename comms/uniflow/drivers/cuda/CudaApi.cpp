// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/drivers/cuda/CudaApi.h"

#include <string>

#ifdef __HIP_PLATFORM_AMD__
// Checks a HIP runtime call and returns Err on failure, recording the
// stringified call (API name + args), source location, and hipError_t.
// Falls through on success.
#define HIP_RETURN_ERR(hip_err, api_name, code)                              \
  do {                                                                       \
    hipError_t _hip_err_ = (hip_err);                                        \
    if (_hip_err_ != hipSuccess) {                                           \
      return ::uniflow::Err(                                                 \
          code,                                                              \
          std::string(api_name " failed: ") + hipGetErrorString(_hip_err_) + \
              " [" __FILE__ ":" STRINGIFY(__LINE__) "]");                    \
    }                                                                        \
  } while (0)

// Convenience wrapper: evaluates `call`, stringifies it, and checks the result.
#define HIP_CHECK(call, code) HIP_RETURN_ERR(call, #call, code)
#else
// Checks a CUDA runtime call and returns Err on failure, recording the
// stringified call (API name + args), source location, and cudaError_t.
// Falls through on success.
#define CUDA_RETURN_ERR(cuda_err, api_name, code)                              \
  do {                                                                         \
    cudaError_t _cuda_err_ = (cuda_err);                                       \
    if (_cuda_err_ != cudaSuccess) {                                           \
      return ::uniflow::Err(                                                   \
          code,                                                                \
          std::string(api_name " failed: ") + cudaGetErrorString(_cuda_err_) + \
              " [" __FILE__ ":" STRINGIFY(__LINE__) "]");                      \
    }                                                                          \
  } while (0)

// Convenience wrapper: evaluates `call`, stringifies it, and checks the result.
#define CUDA_CHECK(call, code) CUDA_RETURN_ERR(call, #call, code)
#endif

namespace uniflow {

// --- Device management ---

Status CudaApi::setDevice(int device) {
#ifdef __HIP_PLATFORM_AMD__
  HIP_CHECK(hipSetDevice(device), ErrCode::DriverError);
#else
  CUDA_CHECK(cudaSetDevice(device), ErrCode::DriverError);
#endif
  return Ok();
}

Result<int> CudaApi::getDevice() {
  int device = -1;
#ifdef __HIP_PLATFORM_AMD__
  HIP_CHECK(hipGetDevice(&device), ErrCode::DriverError);
#else
  CUDA_CHECK(cudaGetDevice(&device), ErrCode::DriverError);
#endif
  return device;
}

Result<bool> CudaApi::deviceCanAccessPeer(int device, int peerDevice) {
  int canAccess = 0;
#ifdef __HIP_PLATFORM_AMD__
  HIP_CHECK(
      hipDeviceCanAccessPeer(&canAccess, device, peerDevice),
      ErrCode::DriverError);
#else
  CUDA_CHECK(
      cudaDeviceCanAccessPeer(&canAccess, device, peerDevice),
      ErrCode::DriverError);
#endif
  return canAccess != 0;
}

Status CudaApi::deviceEnablePeerAccess(int peerDevice) {
#ifdef __HIP_PLATFORM_AMD__
  auto err = hipDeviceEnablePeerAccess(peerDevice, 0);
  // hipErrorPeerAccessAlreadyEnabled is not an error for us.
  if (err == hipErrorPeerAccessAlreadyEnabled) {
    (void)hipGetLastError(); // Clear the error.
    return Ok();
  }
  HIP_RETURN_ERR(err, "hipDeviceEnablePeerAccess", ErrCode::DriverError);
#else
  auto err = cudaDeviceEnablePeerAccess(peerDevice, 0);
  // cudaErrorPeerAccessAlreadyEnabled is not an error for us.
  if (err == cudaErrorPeerAccessAlreadyEnabled) {
    cudaGetLastError(); // Clear the error.
    return Ok();
  }
  CUDA_RETURN_ERR(err, "cudaDeviceEnablePeerAccess", ErrCode::DriverError);
#endif
  return Ok();
}

Result<int> CudaApi::getDeviceCount() {
  int count = 0;
#ifdef __HIP_PLATFORM_AMD__
  HIP_CHECK(hipGetDeviceCount(&count), ErrCode::DriverError);
#else
  CUDA_CHECK(cudaGetDeviceCount(&count), ErrCode::DriverError);
#endif
  return count;
}

Status CudaApi::getDevicePCIBusId(char* pciBusId, int len, int device) {
#ifdef __HIP_PLATFORM_AMD__
  HIP_CHECK(hipDeviceGetPCIBusId(pciBusId, len, device), ErrCode::DriverError);
#else
  CUDA_CHECK(
      cudaDeviceGetPCIBusId(pciBusId, len, device), ErrCode::DriverError);
#endif
  return Ok();
}

// --- Host memory ---

Result<void*> CudaApi::hostAlloc(size_t size, unsigned int flags) {
  void* ptr = nullptr;
#ifdef __HIP_PLATFORM_AMD__
  HIP_CHECK(hipHostMalloc(&ptr, size, flags), ErrCode::DriverError);
#else
  CUDA_CHECK(cudaHostAlloc(&ptr, size, flags), ErrCode::DriverError);
#endif
  return ptr;
}

Status CudaApi::hostFree(void* ptr) {
#ifdef __HIP_PLATFORM_AMD__
  HIP_CHECK(hipHostFree(ptr), ErrCode::DriverError);
#else
  CUDA_CHECK(cudaFreeHost(ptr), ErrCode::DriverError);
#endif
  return Ok();
}

Result<void*> CudaApi::hostGetDevicePointer(void* hostPtr) {
  void* devPtr = nullptr;
#ifdef __HIP_PLATFORM_AMD__
  HIP_CHECK(hipHostGetDevicePointer(&devPtr, hostPtr, 0), ErrCode::DriverError);
#else
  CUDA_CHECK(
      cudaHostGetDevicePointer(&devPtr, hostPtr, 0), ErrCode::DriverError);
#endif
  return devPtr;
}

// --- Memory copy ---

#ifdef __HIP_PLATFORM_AMD__
Status CudaApi::memcpyAsync(
    void* dst,
    const void* src,
    size_t count,
    hipMemcpyKind kind,
    hipStream_t stream) {
  HIP_CHECK(
      hipMemcpyAsync(dst, src, count, kind, stream), ErrCode::DriverError);
  return Ok();
}
#else
Status CudaApi::memcpyAsync(
    void* dst,
    const void* src,
    size_t count,
    cudaMemcpyKind kind,
    cudaStream_t stream) {
  CUDA_CHECK(
      cudaMemcpyAsync(dst, src, count, kind, stream), ErrCode::DriverError);
  return Ok();
}
#endif

#if !defined(__HIP_PLATFORM_AMD__) && CUDART_VERSION >= 12080
Status CudaApi::memcpyBatchAsync(
    void* const* dsts,
    const void* const* srcs,
    const size_t* sizes,
    size_t count,
    cudaStream_t stream) {
  cudaMemcpyAttributes attr{};
  attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
  size_t attrsIdx = 0;
#if CUDART_VERSION < 13000
  // CUDA 12.8 takes non-const dsts/srcs/sizes and a trailing failIdx.
  size_t failIdx = SIZE_MAX;
  CUDA_CHECK(
      cudaMemcpyBatchAsync(
          const_cast<void**>(dsts),
          const_cast<void**>(srcs),
          const_cast<size_t*>(sizes),
          count,
          &attr,
          &attrsIdx,
          1,
          &failIdx,
          stream),
      ErrCode::DriverError);
#else
  // CUDA 13.0+ took const qualifiers and removed failIdx.
  CUDA_CHECK(
      cudaMemcpyBatchAsync(
          dsts, srcs, sizes, count, &attr, &attrsIdx, 1, stream),
      ErrCode::DriverError);
#endif
  return Ok();
}
#endif

#ifdef __HIP_PLATFORM_AMD__
Status CudaApi::memcpyPeerAsync(
    void* dst,
    int dstDevice,
    const void* src,
    int srcDevice,
    size_t count,
    hipStream_t stream) {
  HIP_CHECK(
      hipMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream),
      ErrCode::DriverError);
  return Ok();
}
#else
Status CudaApi::memcpyPeerAsync(
    void* dst,
    int dstDevice,
    const void* src,
    int srcDevice,
    size_t count,
    cudaStream_t stream) {
  CUDA_CHECK(
      cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, count, stream),
      ErrCode::DriverError);
  return Ok();
}
#endif

// --- Stream ---

#ifdef __HIP_PLATFORM_AMD__
Status CudaApi::streamSynchronize(hipStream_t stream) {
  HIP_CHECK(hipStreamSynchronize(stream), ErrCode::DriverError);
  return Ok();
}
#else
Status CudaApi::streamSynchronize(cudaStream_t stream) {
  CUDA_CHECK(cudaStreamSynchronize(stream), ErrCode::DriverError);
  return Ok();
}
#endif

// --- Event ---

#ifdef __HIP_PLATFORM_AMD__
Status CudaApi::eventCreate(hipEvent_t* event) {
  HIP_CHECK(
      hipEventCreateWithFlags(event, hipEventDisableTiming),
      ErrCode::DriverError);
  return Ok();
}

Status CudaApi::eventRecord(hipEvent_t event, hipStream_t stream) {
  HIP_CHECK(hipEventRecord(event, stream), ErrCode::DriverError);
  return Ok();
}

Result<bool> CudaApi::eventQuery(hipEvent_t event) {
  auto err = hipEventQuery(event);
  if (err == hipSuccess) {
    return true;
  }
  if (err == hipErrorNotReady) {
    // Not an error — clear the sticky error and report not-ready.
    (void)hipGetLastError();
    return false;
  }
  HIP_RETURN_ERR(err, "hipEventQuery", ErrCode::DriverError);
  return false; // unreachable
}

Status CudaApi::eventDestroy(hipEvent_t event) {
  HIP_CHECK(hipEventDestroy(event), ErrCode::DriverError);
  return Ok();
}
#else
Status CudaApi::eventCreate(cudaEvent_t* event) {
  CUDA_CHECK(
      cudaEventCreateWithFlags(event, cudaEventDisableTiming),
      ErrCode::DriverError);
  return Ok();
}

Status CudaApi::eventRecord(cudaEvent_t event, cudaStream_t stream) {
  CUDA_CHECK(cudaEventRecord(event, stream), ErrCode::DriverError);
  return Ok();
}

Result<bool> CudaApi::eventQuery(cudaEvent_t event) {
  auto err = cudaEventQuery(event);
  if (err == cudaSuccess) {
    return true;
  }
  if (err == cudaErrorNotReady) {
    // Not an error — clear the sticky error and report not-ready.
    cudaGetLastError();
    return false;
  }
  CUDA_RETURN_ERR(err, "cudaEventQuery", ErrCode::DriverError);
  return false; // unreachable
}

Status CudaApi::eventDestroy(cudaEvent_t event) {
  CUDA_CHECK(cudaEventDestroy(event), ErrCode::DriverError);
  return Ok();
}
#endif

// --- CudaDeviceGuard ---

CudaDeviceGuard::CudaDeviceGuard(CudaApi& api, int device) : api_(api) {
  auto prev = api_.getDevice();
  if (prev.hasValue()) {
    prevDevice_ = prev.value();
  }
  CHECK_THROW_ERROR(api_.setDevice(device));
}

CudaDeviceGuard::~CudaDeviceGuard() {
  if (prevDevice_ >= 0) {
    CHECK_THROW_ERROR(api_.setDevice(prevDevice_));
  }
}

} // namespace uniflow
