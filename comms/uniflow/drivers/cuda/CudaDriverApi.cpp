// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/drivers/cuda/CudaDriverApi.h"
#include "comms/uniflow/drivers/Constants.h"

#include <cudaTypedefs.h>
#include <cuda_runtime.h>

#include <mutex>
#include <string>

namespace uniflow {

constexpr int CUDA_DRIVER_MIN_VERSION = 12040;

static_assert(
    CUDA_VERSION >= CUDA_DRIVER_MIN_VERSION,
    "CudaDriverApi requires CUDA 12.4 or later");

// ---------------------------------------------------------------------------
// Function pointer declarations using PFN types from <cudaTypedefs.h>
// ---------------------------------------------------------------------------

#define DECLARE_CUDA_PFN(symbol, version) \
  PFN_##symbol##_v##version pfn_##symbol = nullptr

namespace {

// NOLINTBEGIN(facebook-avoid-non-const-global-variables)

// --- Error ---
DECLARE_CUDA_PFN(cuGetErrorString, 6000);
DECLARE_CUDA_PFN(cuGetErrorName, 6000);

// --- Device ---
DECLARE_CUDA_PFN(cuDeviceGet, 2000);
DECLARE_CUDA_PFN(cuDeviceGetAttribute, 2000);

// --- cuMem VMM ---
DECLARE_CUDA_PFN(cuMemCreate, 10020);
DECLARE_CUDA_PFN(cuMemRelease, 10020);
DECLARE_CUDA_PFN(cuMemAddressReserve, 10020);
DECLARE_CUDA_PFN(cuMemAddressFree, 10020);
DECLARE_CUDA_PFN(cuMemMap, 10020);
DECLARE_CUDA_PFN(cuMemUnmap, 10020);
DECLARE_CUDA_PFN(cuMemSetAccess, 10020);
DECLARE_CUDA_PFN(cuMemGetAllocationGranularity, 10020);
DECLARE_CUDA_PFN(cuMemExportToShareableHandle, 10020);
DECLARE_CUDA_PFN(cuMemImportFromShareableHandle, 10020);
DECLARE_CUDA_PFN(cuMemGetHandleForAddressRange, 11070);
DECLARE_CUDA_PFN(cuMemRetainAllocationHandle, 11000);
DECLARE_CUDA_PFN(cuMemGetAddressRange, 3020);

#undef DECLARE_CUDA_PFN

std::once_flag g_initFlag;
Status g_initStatus{Ok()};
bool g_isDmaBufSupported[kMaxDevices]{false};
bool g_isCuMemSupported{false};
CUmemAllocationHandleType g_cuMemHandleType{
    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR};
// NOLINTEND(facebook-avoid-non-const-global-variables)

/// Convert a CUresult to Status.
Status cuRetToStatus(CUresult ret, const char* funcName) {
  if (ret == CUDA_SUCCESS) {
    return Ok();
  }
  std::string msg = funcName;
  msg += "() failed, ret = ";
  msg += std::to_string(ret);
  if (pfn_cuGetErrorString != nullptr) {
    const char* errStr = nullptr;
    if (pfn_cuGetErrorString(ret, &errStr) == CUDA_SUCCESS &&
        errStr != nullptr) {
      msg += ": ";
      msg += errStr;
    }
  }
  return Err(ErrCode::DriverError, std::move(msg));
}

// ---------------------------------------------------------------------------
// Symbol loading via cudaGetDriverEntryPoint
// ---------------------------------------------------------------------------

#if CUDART_VERSION >= 13000
#define LOAD_SYM(symbol, version, ignore)                                    \
  do {                                                                       \
    cudaDriverEntryPointQueryResult driverStatus =                           \
        cudaDriverEntryPointSymbolNotFound;                                  \
    cudaError_t res = cudaGetDriverEntryPointByVersion(                      \
        #symbol,                                                             \
        (void**)(&pfn_##symbol),                                             \
        version,                                                             \
        cudaEnableDefault,                                                   \
        &driverStatus);                                                      \
    if (res != cudaSuccess || driverStatus != cudaDriverEntryPointSuccess) { \
      if (!ignore) {                                                         \
        g_initStatus = Err(                                                  \
            ErrCode::DriverError, std::string("Failed to load ") + #symbol); \
        return;                                                              \
      }                                                                      \
    }                                                                        \
  } while (0)
#elif CUDART_VERSION >= 12000
#define LOAD_SYM(symbol, version, ignore)                                    \
  do {                                                                       \
    cudaDriverEntryPointQueryResult driverStatus =                           \
        cudaDriverEntryPointSymbolNotFound;                                  \
    cudaError_t res = cudaGetDriverEntryPoint(                               \
        #symbol, (void**)(&pfn_##symbol), cudaEnableDefault, &driverStatus); \
    if (res != cudaSuccess || driverStatus != cudaDriverEntryPointSuccess) { \
      if (!ignore) {                                                         \
        g_initStatus = Err(                                                  \
            ErrCode::DriverError, std::string("Failed to load ") + #symbol); \
        return;                                                              \
      }                                                                      \
    }                                                                        \
  } while (0)
#else
#define LOAD_SYM(symbol, version, ignore)                                    \
  do {                                                                       \
    cudaError_t res = cudaGetDriverEntryPoint(                               \
        #symbol, (void**)(&pfn_##symbol), cudaEnableDefault);                \
    if (res != cudaSuccess) {                                                \
      if (!ignore) {                                                         \
        g_initStatus = Err(                                                  \
            ErrCode::DriverError, std::string("Failed to load ") + #symbol); \
        return;                                                              \
      }                                                                      \
    }                                                                        \
  } while (0)
#endif

#define _PFN(name, ...)                                       \
  (pfn_##name == nullptr                                      \
       ? Err(ErrCode::DriverError, #name " symbol not found") \
       : cuRetToStatus(pfn_##name(__VA_ARGS__), #name))

void checkCuMemSupported(int cudaDev) {
  if (pfn_cuMemCreate == nullptr) {
    g_isCuMemSupported = false;
  } else {
    CUdevice currentDev;
    if (pfn_cuDeviceGet(&currentDev, cudaDev) != CUDA_SUCCESS) {
      g_isCuMemSupported = false;
    } else {
      int flag = 0;
      auto ret = pfn_cuDeviceGetAttribute(
          &flag,
          CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
          currentDev);
      g_isCuMemSupported = (ret == CUDA_SUCCESS && flag != 0);
    }
  }
}

void probeCuMemHandleType() {
  CUdevice cuDevice;
  const int deviceId = 0;
  auto devStatus = _PFN(cuDeviceGet, &cuDevice, deviceId);
  if (devStatus.hasError()) {
    return;
  }

  int fabricSupported = 0;
  auto attrStatus = _PFN(
      cuDeviceGetAttribute,
      &fabricSupported,
      CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
      cuDevice);
  if (attrStatus.hasError() || fabricSupported == 0) {
    return; // Query failed; keep default fd mode.
  }

  // The device attribute reports fabric as supported, but this requires
  // the IMEX daemon to be running and the full cuMem VMM pipeline to work.
  // Check for the IMEX daemon socket first — cuMemExportToShareableHandle
  // with CU_MEM_HANDLE_TYPE_FABRIC blocks indefinitely if the daemon is
  // not running (no timeout in the CUDA driver API).
  if (std::system("systemctl is-active --quiet nvidia-imex") != 0) {
    return;
  }

  // 1. Query granularity for fabric-typed allocations.
  CUmemAllocationProp probeProp{};
  probeProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  probeProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  probeProp.location.id = cuDevice;
  probeProp.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

  size_t probeGranularity = 0;
  if (_PFN(
          cuMemGetAllocationGranularity,
          &probeGranularity,
          &probeProp,
          CU_MEM_ALLOC_GRANULARITY_MINIMUM)
          .hasError()) {
    return;
  }

  // 2. Allocate a small probe buffer.
  CUmemGenericAllocationHandle probeHandle;
  if (_PFN(cuMemCreate, &probeHandle, probeGranularity, &probeProp, 0)
          .hasError()) {
    return;
  }

  // 3. Export to a fabric handle.
  CUmemFabricHandle fabricHandle;
  if (_PFN(
          cuMemExportToShareableHandle,
          &fabricHandle,
          probeHandle,
          CU_MEM_HANDLE_TYPE_FABRIC,
          0)
          .hasError()) {
    pfn_cuMemRelease(probeHandle);
    return;
  }

  // 4. Import the fabric handle back (loopback).
  CUmemGenericAllocationHandle importedHandle;
  if (_PFN(
          cuMemImportFromShareableHandle,
          &importedHandle,
          &fabricHandle,
          CU_MEM_HANDLE_TYPE_FABRIC)
          .hasError()) {
    pfn_cuMemRelease(probeHandle);
    return;
  }

  // 5. Reserve virtual address space.
  CUdeviceptr mappedPtr = 0;
  if (_PFN(
          cuMemAddressReserve,
          &mappedPtr,
          probeGranularity,
          probeGranularity,
          0,
          0)
          .hasError()) {
    pfn_cuMemRelease(importedHandle);
    pfn_cuMemRelease(probeHandle);
    return;
  }

  // 6. Map the imported allocation into the reserved VA range.
  if (_PFN(cuMemMap, mappedPtr, probeGranularity, 0, importedHandle, 0)
          .hasError()) {
    pfn_cuMemAddressFree(mappedPtr, probeGranularity);
    pfn_cuMemRelease(importedHandle);
    pfn_cuMemRelease(probeHandle);
    return;
  }

  // 7. Set read/write access for the local device.
  CUmemAccessDesc accessDesc{};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = deviceId;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  if (_PFN(cuMemSetAccess, mappedPtr, probeGranularity, &accessDesc, 1)
          .hasValue()) {
    g_cuMemHandleType = CU_MEM_HANDLE_TYPE_FABRIC;
  }

  // Cleanup: tear down in reverse order.
  pfn_cuMemUnmap(mappedPtr, probeGranularity);
  pfn_cuMemAddressFree(mappedPtr, probeGranularity);
  pfn_cuMemRelease(importedHandle);
  pfn_cuMemRelease(probeHandle);
}

void checkDmaBufSupported() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    return;
  }
  for (int i = 0; i < count; i++) {
    g_isDmaBufSupported[i] = false;
    CUdevice dev;
    if (pfn_cuDeviceGet(&dev, i) != CUDA_SUCCESS) {
      continue;
    }
    int flag = 0;
    pfn_cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, dev);
    g_isDmaBufSupported[i] = flag != 0;
  }
}

void doInit() {
  int cudaDev;
  auto ret = cudaGetDevice(&cudaDev); // Initialize the driver
  if (ret != cudaSuccess) {
    g_initStatus = Err(ErrCode::DriverError, "cudaGetDevice failed");
    return;
  }

  int cudaDriverVersion;
  cudaError_t err = cudaDriverGetVersion(&cudaDriverVersion);
  if (err != cudaSuccess) {
    g_initStatus =
        Err(ErrCode::DriverError,
            "Failed to get CUDA driver version: " + std::to_string(err));
    return;
  }

  if (cudaDriverVersion < CUDA_DRIVER_MIN_VERSION) {
    g_initStatus =
        Err(ErrCode::DriverError,
            "CUDA driver version " + std::to_string(cudaDriverVersion) +
                " is too old, need at least 12.4");
    return;
  }

  // --- Error ---
  LOAD_SYM(cuGetErrorString, 6000, 0);
  LOAD_SYM(cuGetErrorName, 6000, 0);

  // --- Device ---
  LOAD_SYM(cuDeviceGet, 2000, 0);
  LOAD_SYM(cuDeviceGetAttribute, 2000, 0);

  // --- cuMem VMM ---
  LOAD_SYM(cuMemCreate, 10020, 1);
  LOAD_SYM(cuMemRelease, 10020, 1);
  LOAD_SYM(cuMemAddressReserve, 10020, 1);
  LOAD_SYM(cuMemAddressFree, 10020, 1);
  LOAD_SYM(cuMemMap, 10020, 1);
  LOAD_SYM(cuMemUnmap, 10020, 1);
  LOAD_SYM(cuMemSetAccess, 10020, 1);
  LOAD_SYM(cuMemGetAllocationGranularity, 10020, 1);
  LOAD_SYM(cuMemExportToShareableHandle, 10020, 1);
  LOAD_SYM(cuMemImportFromShareableHandle, 10020, 1);
  LOAD_SYM(cuMemGetHandleForAddressRange, 11070, 1);
  LOAD_SYM(cuMemRetainAllocationHandle, 11000, 1);
  LOAD_SYM(cuMemGetAddressRange, 3020, 1);

  // Check if cuMem is supported
  checkCuMemSupported(cudaDev);

  // Check if DMA_BUF is supported
  checkDmaBufSupported();

  // Check if fabric handle type is supported
  probeCuMemHandleType();

  g_initStatus = Ok();
}

#undef LOAD_SYM

} // namespace

// ---------------------------------------------------------------------------
// CudaDriverApi implementation
// ---------------------------------------------------------------------------

#define CU_ENSURE_INIT()            \
  do {                              \
    auto _s = init();               \
    if (_s.hasError()) {            \
      return std::move(_s).error(); \
    }                               \
  } while (0)

#define CU_CALL(name, ...)                                         \
  do {                                                             \
    CU_ENSURE_INIT();                                              \
    if (pfn_##name == nullptr) {                                   \
      return Err(ErrCode::DriverError, #name " symbol not found"); \
    }                                                              \
    return cuRetToStatus(pfn_##name(__VA_ARGS__), #name);          \
  } while (0)

Status CudaDriverApi::init() {
  std::call_once(g_initFlag, doInit);
  return g_initStatus;
}

// --- Device ---

Status CudaDriverApi::cuDeviceGet(CUdevice* device, int ordinal) {
  CU_CALL(cuDeviceGet, device, ordinal);
}

Status CudaDriverApi::cuDeviceGetAttribute(
    int* pi,
    CUdevice_attribute attrib,
    CUdevice dev) {
  CU_CALL(cuDeviceGetAttribute, pi, attrib, dev);
}

// --- Error ---

Status CudaDriverApi::cuGetErrorString(CUresult error, const char** pStr) {
  CU_CALL(cuGetErrorString, error, pStr);
}

Status CudaDriverApi::cuGetErrorName(CUresult error, const char** pStr) {
  CU_CALL(cuGetErrorName, error, pStr);
}

// --- cuMem VMM ---

Status CudaDriverApi::cuMemCreate(
    CUmemGenericAllocationHandle* handle,
    size_t size,
    const CUmemAllocationProp* prop,
    unsigned long long flags) {
  CU_CALL(cuMemCreate, handle, size, prop, flags);
}

Status CudaDriverApi::cuMemRelease(CUmemGenericAllocationHandle handle) {
  CU_CALL(cuMemRelease, handle);
}

Status CudaDriverApi::cuMemAddressReserve(
    CUdeviceptr* ptr,
    size_t size,
    size_t alignment,
    CUdeviceptr addr,
    unsigned long long flags) {
  CU_CALL(cuMemAddressReserve, ptr, size, alignment, addr, flags);
}

Status CudaDriverApi::cuMemAddressFree(CUdeviceptr ptr, size_t size) {
  CU_CALL(cuMemAddressFree, ptr, size);
}

Status CudaDriverApi::cuMemMap(
    CUdeviceptr ptr,
    size_t size,
    size_t offset,
    CUmemGenericAllocationHandle handle,
    unsigned long long flags) {
  CU_CALL(cuMemMap, ptr, size, offset, handle, flags);
}

Status CudaDriverApi::cuMemUnmap(CUdeviceptr ptr, size_t size) {
  CU_CALL(cuMemUnmap, ptr, size);
}

Status CudaDriverApi::cuMemSetAccess(
    CUdeviceptr ptr,
    size_t size,
    const CUmemAccessDesc* desc,
    size_t count) {
  CU_CALL(cuMemSetAccess, ptr, size, desc, count);
}

Status CudaDriverApi::cuMemGetAllocationGranularity(
    size_t* granularity,
    const CUmemAllocationProp* prop,
    CUmemAllocationGranularity_flags option) {
  CU_CALL(cuMemGetAllocationGranularity, granularity, prop, option);
}

Status CudaDriverApi::cuMemRetainAllocationHandle(
    CUmemGenericAllocationHandle* handle,
    void* addr) {
  CU_CALL(cuMemRetainAllocationHandle, handle, addr);
}

Status CudaDriverApi::cuMemGetAddressRange_v2(
    CUdeviceptr* pbase,
    size_t* psize,
    CUdeviceptr dptr) {
  CU_CALL(cuMemGetAddressRange, pbase, psize, dptr);
}

Status CudaDriverApi::cuMemExportToShareableHandle(
    void* shareableHandle,
    CUmemGenericAllocationHandle handle,
    CUmemAllocationHandleType handleType,
    unsigned long long flags) {
  CU_CALL(
      cuMemExportToShareableHandle, shareableHandle, handle, handleType, flags);
}

Status CudaDriverApi::cuMemImportFromShareableHandle(
    CUmemGenericAllocationHandle* handle,
    void* osHandle,
    CUmemAllocationHandleType shHandleType) {
  CU_CALL(cuMemImportFromShareableHandle, handle, osHandle, shHandleType);
}

Status CudaDriverApi::cuMemGetHandleForAddressRange(
    void* handle,
    CUdeviceptr dptr,
    size_t size,
    CUmemRangeHandleType handleType,
    unsigned long long flags) {
  CU_CALL(cuMemGetHandleForAddressRange, handle, dptr, size, handleType, flags);
}

// --- supported APIs ---

Result<bool> CudaDriverApi::isDmaBufSupported(int cudaDev) {
  CU_ENSURE_INIT();
  if (cudaDev < 0 || cudaDev >= kMaxDevices) {
    return Err(
        ErrCode::InvalidArgument,
        "Invalid cudaDev: " + std::to_string(cudaDev));
  }
  return g_isDmaBufSupported[cudaDev];
}

Result<bool> CudaDriverApi::isCuMemSupported() {
  CU_ENSURE_INIT();
  return g_isCuMemSupported;
}

Result<CUmemAllocationHandleType> CudaDriverApi::getCuMemHandleType() {
  CU_ENSURE_INIT();
  return g_cuMemHandleType;
}

} // namespace uniflow
