// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/drivers/cuda/CudaApi.h"

#include <algorithm>
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
    // Clear the error. Cast to void: cudaGetLastError() is not nodiscard, but
    // its hipified twin hipGetLastError() is, so discard explicitly to keep
    // the single source compiling under -Werror on both platforms.
    (void)cudaGetLastError();
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

Result<std::string> CudaApi::getDeviceArch(int device) {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device), ErrCode::DriverError);
#if defined(__HIP_PLATFORM_AMD__)
  // AMD: gfx arch, e.g. "gfx942" (may carry feature flags like
  // "gfx942:sramecc+:xnack-"); callers prefix-match the gfxNNNN base.
  return std::string(prop.gcnArchName);
#else
  // NVIDIA: cudaDeviceProp has no gcnArchName; report the SM version. NVIDIA
  // compute capabilities use a single-digit minor by convention, so the
  // canonical "sm_<major><minor>" form (e.g. sm_90, sm_120) is unambiguous; if
  // a multi-digit minor ever ships this concatenation would need a separator.
  return "sm_" + std::to_string(prop.major) + std::to_string(prop.minor);
#endif
}

// --- Host memory ---

Result<void*> CudaApi::hostAlloc(size_t size, unsigned int flags) {
  void* ptr = nullptr;
  CUDA_CHECK(cudaHostAlloc(&ptr, size, flags), ErrCode::DriverError);
  return ptr;
}

Status CudaApi::hostFree(void* ptr) {
  CUDA_CHECK(cudaFreeHost(ptr), ErrCode::DriverError);
  return Ok();
}

Result<void*> CudaApi::hostGetDevicePointer(void* hostPtr) {
  void* devPtr = nullptr;
  CUDA_CHECK(
      cudaHostGetDevicePointer(&devPtr, hostPtr, 0), ErrCode::DriverError);
  return devPtr;
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

#if CUDART_VERSION >= 12080
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
    // Cast to void: hipGetLastError() is nodiscard on AMD (see above).
    (void)cudaGetLastError();
    return false;
  }
  CUDA_RETURN_ERR(err, "cudaEventQuery", ErrCode::DriverError);
  return false; // unreachable
}

Status CudaApi::eventDestroy(cudaEvent_t event) {
  CUDA_CHECK(cudaEventDestroy(event), ErrCode::DriverError);
  return Ok();
}

// --- IPC (cross-process device memory sharing) ---

static_assert(
    sizeof(cudaIpcMemHandle_t) == CudaApi::kIpcMemHandleSize,
    "IpcMemHandle byte size must match cudaIpcMemHandle_t/hipIpcMemHandle_t");

Result<CudaApi::IpcMemHandle> CudaApi::ipcGetMemHandle(void* devPtr) {
  cudaIpcMemHandle_t handle{};
  CUDA_CHECK(cudaIpcGetMemHandle(&handle, devPtr), ErrCode::DriverError);
  IpcMemHandle out{};
  const auto* bytes = reinterpret_cast<const uint8_t*>(&handle);
  std::copy_n(bytes, kIpcMemHandleSize, out.begin());
  return out;
}

Result<void*> CudaApi::ipcOpenMemHandle(const IpcMemHandle& handle) {
  cudaIpcMemHandle_t raw{};
  std::copy_n(
      handle.begin(), kIpcMemHandleSize, reinterpret_cast<uint8_t*>(&raw));
  void* ptr = nullptr;
  CUDA_CHECK(
      cudaIpcOpenMemHandle(&ptr, raw, cudaIpcMemLazyEnablePeerAccess),
      ErrCode::DriverError);
  return ptr;
}

Status CudaApi::ipcCloseMemHandle(void* devPtr) {
  CUDA_CHECK(cudaIpcCloseMemHandle(devPtr), ErrCode::DriverError);
  return Ok();
}

Result<CudaApi::MemRange> CudaApi::getMemAddressRange(void* devPtr) {
#if defined(__HIP_PLATFORM_AMD__)
  // AMD: HIP runtime address-range API (libamdhip64; no driver lib). Lets a
  // sub-allocation segment be IPC-exported at its allocation base with the
  // right offset. cuMemGetAddressRange hipifies to hipMemGetAddressRange; this
  // branch compiles only on AMD.
  CUdeviceptr base = 0;
  size_t size = 0;
  auto err =
      cuMemGetAddressRange(&base, &size, reinterpret_cast<CUdeviceptr>(devPtr));
  // Driver-API success sentinel (CUDA_SUCCESS, not the runtime cudaSuccess) to
  // match cuMemGetAddressRange; both hipify to hipSuccess on AMD. The manual
  // check is intentional: on failure we gracefully fall back to
  // whole-allocation rather than propagating an error (do not switch to
  // CUDA_CHECK).
  if (err != CUDA_SUCCESS) {
    // Range unavailable (e.g. non-VMM / host memory): fall back to
    // whole-allocation (base == ptr, offset 0), preserving prior behavior.
    return MemRange{devPtr, 0};
  }
  return MemRange{reinterpret_cast<void*>(base), size};
#else
  // NVIDIA: runtime-only, no driver dependency. Report the pointer as its own
  // base -> whole-allocation / offset 0, identical to the pre-existing path.
  return MemRange{devPtr, 0};
#endif
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
