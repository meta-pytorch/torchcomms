// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/drivers/cuda/CudaDriverApi.h"
#include "comms/uniflow/drivers/Constants.h"

// NVIDIA loads the CUDA driver entry points dynamically via PFN typedefs from
// <cudaTypedefs.h> + cudaGetDriverEntryPoint. HIP has no versioned PFN
// typedefs and exposes the driver (hip*) functions directly, so the AMD build
// (hipified: cu* -> hip*) calls them directly. The divergence is confined to
// the symbol-loading machinery below; the per-method bodies are shared via the
// CU_PFN()/CU_CALL() indirection.
#ifndef __HIP_PLATFORM_AMD__
#include <cudaTypedefs.h>
#endif
#include <cuda_runtime.h>

#if defined(__HIP_PLATFORM_AMD__)
// For the AMD GPUDirect-RDMA (ib_peer_mem / amdkfd) sysfs + kallsyms probe and
// the POSIX-fd import convention probe.
#include <dlfcn.h>
#include <link.h>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#endif

#include <mutex>
#include <string>

namespace uniflow {

#ifndef __HIP_PLATFORM_AMD__
constexpr int CUDA_DRIVER_MIN_VERSION = 12040;

static_assert(
    CUDA_VERSION >= CUDA_DRIVER_MIN_VERSION,
    "CudaDriverApi requires CUDA 12.4 or later");
#endif

// On NVIDIA, CU_PFN(name) resolves to the dynamically loaded pfn_<name>; on AMD
// (hipified) it resolves to the driver function itself, called directly.
#if defined(__HIP_PLATFORM_AMD__)
#define CU_PFN(name) name
#else
#define CU_PFN(name) pfn_##name
#endif

namespace {

// NOLINTBEGIN(facebook-avoid-non-const-global-variables)

#ifndef __HIP_PLATFORM_AMD__
// ---------------------------------------------------------------------------
// Function pointer declarations using PFN types from <cudaTypedefs.h> (NVIDIA)
// ---------------------------------------------------------------------------

#define DECLARE_CUDA_PFN(symbol, version) \
  PFN_##symbol##_v##version pfn_##symbol = nullptr

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

// --- Stream memory ops ---
DECLARE_CUDA_PFN(cuStreamWriteValue64, 11070);

#undef DECLARE_CUDA_PFN
#endif // !__HIP_PLATFORM_AMD__

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
  msg += std::to_string(static_cast<int>(ret));
  const char* errStr = nullptr;
#if defined(__HIP_PLATFORM_AMD__)
  if (cuGetErrorString(ret, &errStr) == CUDA_SUCCESS && errStr != nullptr) {
    msg += ": ";
    msg += errStr;
  }
#else
  if (pfn_cuGetErrorString != nullptr &&
      pfn_cuGetErrorString(ret, &errStr) == CUDA_SUCCESS && errStr != nullptr) {
    msg += ": ";
    msg += errStr;
  }
#endif
  return Err(ErrCode::DriverError, std::move(msg));
}

#ifndef __HIP_PLATFORM_AMD__
// ---------------------------------------------------------------------------
// Symbol loading via cudaGetDriverEntryPoint (NVIDIA only)
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
#endif // !__HIP_PLATFORM_AMD__

void checkCuMemSupported(int cudaDev) {
#ifndef __HIP_PLATFORM_AMD__
  if (pfn_cuMemCreate == nullptr) {
    g_isCuMemSupported = false;
    return;
  }
#endif
  CUdevice currentDev;
  if (CU_PFN(cuDeviceGet)(&currentDev, cudaDev) != CUDA_SUCCESS) {
    g_isCuMemSupported = false;
    return;
  }
  int flag = 0;
  auto ret = CU_PFN(cuDeviceGetAttribute)(
      &flag,
      CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
      currentDev);
  g_isCuMemSupported = (ret == CUDA_SUCCESS && flag != 0);
}

#ifndef __HIP_PLATFORM_AMD__
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
  LOAD_SYM(cuStreamWriteValue64, 11070, 1);

  // Check if cuMem is supported
  checkCuMemSupported(cudaDev);

  // Check if DMA_BUF is supported
  checkDmaBufSupported();

  // Check if fabric handle type is supported
  probeCuMemHandleType();

  g_initStatus = Ok();
}

#undef LOAD_SYM

#else // __HIP_PLATFORM_AMD__

// Detect GPUDirect-RDMA (peer-mem) support on AMD. There is no HIP query for
// this, so we probe the same signals as rccl/ctran (see
// comms/ctran/utils/HipGdrCheck.h): the amdkfd peer-mem sysfs version node, and
// the `ib_register_peer_memory_client` symbol in /proc/kallsyms as a fallback
// for native-OS ib_peer modules. Used to decide whether VRAM can be registered
// via dma-buf; callers fall back to plain ibv_reg_mr when this returns false.
bool amdGpuDirectRdmaSupported() {
  static const char* kMemoryPeersPaths[] = {
      "/sys/kernel/mm/memory_peers/amdkfd/version",
      "/sys/kernel/memory_peers/amdkfd/version",
      "/sys/memory_peers/amdkfd/version",
  };
  for (const char* path : kMemoryPeersPaths) {
    if (access(path, F_OK) == 0) {
      return true;
    }
  }

  // Fallback: look for the native ib_peer_mem registration symbol.
  FILE* fp = std::fopen("/proc/kallsyms", "r");
  if (fp == nullptr) {
    return false;
  }
  char buf[256];
  bool found = false;
  while (std::fgets(buf, sizeof(buf), fp) != nullptr) {
    if (std::strstr(buf, "ib_register_peer_memory_client") != nullptr) {
      found = true;
      break;
    }
  }
  std::fclose(fp);
  return found;
}

// A failed HIP driver call is also recorded as the thread's last runtime
// error, which a later cudaGetLastError-based launch check (e.g. PyTorch's)
// would report as its own. Clear the last error on exit only if it was clean
// on entry; a pending error may still be replaced by this scope's own error.
class LastErrorScrub {
 public:
  LastErrorScrub() : wasClean_{cudaPeekAtLastError() == cudaSuccess} {}

  ~LastErrorScrub() {
    if (wasClean_) {
      (void)cudaGetLastError();
    }
  }

  LastErrorScrub(const LastErrorScrub&) = delete;
  LastErrorScrub& operator=(const LastErrorScrub&) = delete;

 private:
  const bool wasClean_;
};

// How the runtime reads the osHandle argument of a POSIX-fd import. ROCm up to
// 7.0.2.1 dereferences it as an int*; later runtimes take the fd by value.
// Passing the fd by value to a pointer runtime crashes inside the runtime.
enum class FdConvention { Unknown, Pointer, Value };

// HIP_VERSION encodes major * 10^7 + minor * 10^5 + patch.
constexpr int kFirstValueConventionVersion = 70100000;
constexpr uintptr_t kNegativeLow32 = 0x80000000u;

int collectHipLibraries(dl_phdr_info* info, size_t /*size*/, void* data) {
  const char* name = info->dlpi_name;
  if (name != nullptr && std::strstr(name, "libamdhip64") != nullptr) {
    static_cast<std::vector<std::string>*>(data)->emplace_back(name);
  }
  return 0;
}

bool exportsHip71Node(const std::string& path) {
  void* library = dlopen(path.c_str(), RTLD_NOW | RTLD_NOLOAD);
  if (library == nullptr) {
    return false;
  }
  const bool found =
      dlvsym(library, "hipMemcpyBatchAsync", "hip_7.1") != nullptr;
  dlclose(library);
  return found;
}

// 7.0.2.2 reports the same version as 7.0.2.1 but already takes the fd by
// value; it is the first build to export the hip_7.1 symbol version node.
// Every loaded copy must export it: the copy serving the import cannot be
// identified here (fbcode binds HIP through lazy stubs), and a by-value call
// into a pointer runtime crashes.
bool runtimeTakesFdByValue() {
  int version = 0;
  if (cudaRuntimeGetVersion(&version) == cudaSuccess &&
      version >= kFirstValueConventionVersion) {
    return true;
  }
  std::vector<std::string> libraries;
  dl_iterate_phdr(collectHipLibraries, &libraries);
  return !libraries.empty() &&
      std::all_of(libraries.begin(), libraries.end(), exportsHip71Node);
}

// Returns a process-lifetime int whose address has bit 31 set in its low 32
// bits. A by-value runtime truncates that address to a negative fd and fails
// cleanly, which makes the pointer form safe to try on every runtime.
int* negativeFdSlot() {
  static int* const slot = []() -> int* {
    const auto page = static_cast<uintptr_t>(sysconf(_SC_PAGESIZE));
    const uintptr_t span = (uintptr_t{1} << 31) + 2 * page;
    void* reserved = mmap(
        nullptr,
        span,
        PROT_NONE,
        MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
        -1,
        0);
    if (reserved == MAP_FAILED) {
      return nullptr;
    }
    const auto base = reinterpret_cast<uintptr_t>(reserved);
    const uintptr_t end = base + span;
    const uintptr_t chosen = (base & kNegativeLow32) != 0
        ? base
        : ((base & ~uintptr_t{0xffffffff}) | kNegativeLow32);
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    void* slotPage = reinterpret_cast<void*>(chosen);
    if (mprotect(slotPage, page, PROT_READ | PROT_WRITE) != 0) {
      munmap(reserved, span);
      return nullptr;
    }
    if (chosen > base) {
      munmap(reserved, chosen - base);
    }
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    munmap(reinterpret_cast<void*>(chosen + page), end - chosen - page);
    return static_cast<int*>(slotPage);
  }();
  return slot;
}

// A by-value runtime reads the fd as the value of the void* osHandle argument.
void* fdAsOsHandle(int fd) {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  return reinterpret_cast<void*>(static_cast<intptr_t>(fd));
}

CUresult importFd(CUmemGenericAllocationHandle* handle, void* osHandle) {
  return cuMemImportFromShareableHandle(
      handle, osHandle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
}

// Resolves the convention on the first successful import and caches it. The
// pointer form goes first because it fails harmlessly on a by-value runtime;
// the by-value form is only tried when the runtime generation says by-value.
Status importPosixFdAmd(CUmemGenericAllocationHandle* handle, int fd) {
  static std::mutex mutex;
  static FdConvention convention{FdConvention::Unknown};
  const std::lock_guard<std::mutex> lock(mutex);
  const LastErrorScrub scrub;
  if (convention == FdConvention::Value) {
    return cuRetToStatus(
        importFd(handle, fdAsOsHandle(fd)), "importPosixFd(value)");
  }
  Status viaPointer =
      Err(ErrCode::ResourceExhausted, "importPosixFd: cannot map the fd slot");
  if (int* slot = negativeFdSlot(); slot != nullptr) {
    *slot = fd;
    const CUresult ret = importFd(handle, slot);
    if (ret == CUDA_SUCCESS) {
      convention = FdConvention::Pointer;
      return Ok();
    }
    viaPointer = cuRetToStatus(ret, "importPosixFd(pointer)");
  }
  if (convention == FdConvention::Pointer || !runtimeTakesFdByValue()) {
    return viaPointer;
  }
  const CUresult viaValue = importFd(handle, fdAsOsHandle(fd));
  if (viaValue == CUDA_SUCCESS) {
    convention = FdConvention::Value;
  }
  return cuRetToStatus(viaValue, "importPosixFd(value)");
}

// AMD driver entry points (hip*) are linked directly, so there is no PFN
// loading. Init initializes the driver, probes cuMem (VMM) support, and detects
// GPUDirect-RDMA via the peer-mem probe above. The NVIDIA-only fabric handle
// path does not apply — AMD uses POSIX-FD / dma-buf sharing.
void doInit() {
  const LastErrorScrub scrub;
  int cudaDev;
  auto ret = cudaGetDevice(&cudaDev); // Initialize the driver
  if (ret != cudaSuccess) {
    g_initStatus = Err(ErrCode::DriverError, "cudaGetDevice failed");
    return;
  }

  checkCuMemSupported(cudaDev);

  const bool gdrSupported = amdGpuDirectRdmaSupported();
  for (int i = 0; i < kMaxDevices; i++) {
    g_isDmaBufSupported[i] = gdrSupported;
  }

  g_initStatus = Ok();
}

#endif // __HIP_PLATFORM_AMD__

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

#if defined(__HIP_PLATFORM_AMD__)
// The Status carries the failure, so it must not also linger as the thread's
// last runtime error (see LastErrorScrub).
#define CU_CALL(name, ...)                            \
  do {                                                \
    CU_ENSURE_INIT();                                 \
    const LastErrorScrub scrub;                       \
    return cuRetToStatus(::name(__VA_ARGS__), #name); \
  } while (0)
#else
#define CU_CALL(name, ...)                                         \
  do {                                                             \
    CU_ENSURE_INIT();                                              \
    if (pfn_##name == nullptr) {                                   \
      return Err(ErrCode::DriverError, #name " symbol not found"); \
    }                                                              \
    return cuRetToStatus(pfn_##name(__VA_ARGS__), #name);          \
  } while (0)
#endif

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
#if defined(__HIP_PLATFORM_AMD__)

Status CudaDriverApi::cuMemGetAccess(
    unsigned long long* flags,
    const CUmemLocation* location,
    CUdeviceptr ptr) {
  CU_CALL(cuMemGetAccess, flags, location, ptr);
}

Status CudaDriverApi::importPosixFd(
    CUmemGenericAllocationHandle* handle,
    int fd) {
  CU_ENSURE_INIT();
  return importPosixFdAmd(handle, fd);
}
#endif

Status CudaDriverApi::memGetHandleForAddressRange(
    void* handle,
    CUdeviceptr dptr,
    size_t size,
    CUmemRangeHandleType handleType,
    unsigned long long flags) {
#if defined(__HIP_PLATFORM_AMD__)
  // Not every hipify-perl release maps cuMemGetHandleForAddressRange (ROCm 7.0
  // gates it behind -experimental), so call the HIP dma-buf fd export directly
  // (GPUDirect RDMA path). The prologue mirrors CU_CALL.
  CU_ENSURE_INIT();
  const LastErrorScrub scrub;
  return cuRetToStatus(
      ::hipMemGetHandleForAddressRange(handle, dptr, size, handleType, flags),
      "hipMemGetHandleForAddressRange");
#else
  CU_CALL(cuMemGetHandleForAddressRange, handle, dptr, size, handleType, flags);
#endif
}

// --- Stream memory ops ---

Status CudaDriverApi::streamWriteValue64(
    CUstream stream,
    CUdeviceptr addr,
    uint64_t value,
    unsigned int flags) {
  CU_CALL(cuStreamWriteValue64, stream, addr, value, flags);
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
