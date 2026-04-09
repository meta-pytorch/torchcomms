// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/drivers/cuda/CudaApi.h"

#include <string>

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

namespace uniflow {

// --- Device management ---

Status CudaApi::setDevice(int device) {
  CUDA_CHECK(cudaSetDevice(device), ErrCode::DriverError);
  return Ok();
}

Result<int> CudaApi::getDevice() {
  int device = -1;
  CUDA_CHECK(cudaGetDevice(&device), ErrCode::DriverError);
  return device;
}

Result<bool> CudaApi::deviceCanAccessPeer(int device, int peerDevice) {
  int canAccess = 0;
  CUDA_CHECK(
      cudaDeviceCanAccessPeer(&canAccess, device, peerDevice),
      ErrCode::DriverError);
  return canAccess != 0;
}

Status CudaApi::deviceEnablePeerAccess(int peerDevice) {
  auto err = cudaDeviceEnablePeerAccess(peerDevice, 0);
  // cudaErrorPeerAccessAlreadyEnabled is not an error for us.
  if (err == cudaErrorPeerAccessAlreadyEnabled) {
    cudaGetLastError(); // Clear the error.
    return Ok();
  }
  CUDA_RETURN_ERR(err, "cudaDeviceEnablePeerAccess", ErrCode::DriverError);
  return Ok();
}

Result<int> CudaApi::getDeviceCount() {
  int count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&count), ErrCode::DriverError);
  return count;
}

Status CudaApi::getDevicePCIBusId(char* pciBusId, int len, int device) {
  CUDA_CHECK(
      cudaDeviceGetPCIBusId(pciBusId, len, device), ErrCode::DriverError);
  return Ok();
}

// --- Memory copy ---

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

// --- Stream ---

Status CudaApi::streamSynchronize(cudaStream_t stream) {
  CUDA_CHECK(cudaStreamSynchronize(stream), ErrCode::DriverError);
  return Ok();
}

// --- Event ---

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
