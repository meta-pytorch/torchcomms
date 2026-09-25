// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/memory/NvlMemExchange.h"

#include "comms/common/BitOps.cuh"
#include "comms/prims/bootstrap/TeamStatusAgreement.h"
#include "comms/prims/core/Checks.h"
#include "comms/prims/memory/CuMemAllocation.h"
#include "comms/prims/platform/CudaDriverLazy.h"

#include <glog/logging.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>

#include <sys/syscall.h>
#include <unistd.h>

namespace comms::prims {

namespace {

// RAII guard for an owned POSIX file descriptor. ::close()s on scope exit
// (and only on scope exit -- no detach escape hatch; copy + move are deleted
// so the fd cannot escape this scope). Used for the local POSIX-FD shareable
// handle that nvlMemExchangeVmm exports before allGather: keeps the fd closed
// on every exit path (success, allGather throw, or import failure)
// without the try/catch boilerplate. C++ has no std::scope_exit — P0052 /
// <experimental/scope> never landed — so we roll a 5-line guard rather than
// reach for folly (comms/prims is zero-folly).
class FdGuard {
 public:
  FdGuard() = default;
  ~FdGuard() {
    reset();
  }
  FdGuard(const FdGuard&) = delete;
  FdGuard& operator=(const FdGuard&) = delete;
  FdGuard(FdGuard&&) = delete;
  FdGuard& operator=(FdGuard&&) = delete;

  void reset(int fd = -1) {
    if (fd_ >= 0) {
      ::close(fd_);
    }
    fd_ = fd;
  }

 private:
  int fd_{-1};
};

// pidfd_open / pidfd_getfd are available in glibc 2.36+ but the syscall numbers
// are defined in <sys/syscall.h>. Hardcoding fallbacks here is unsafe on archs
// where the numbers differ (e.g. mips, riscv); fail at compile time instead so
// we never silently issue the wrong syscall.
#if !defined(SYS_pidfd_open) || !defined(SYS_pidfd_getfd)
#error \
    "SYS_pidfd_open / SYS_pidfd_getfd not defined for this architecture; NvlMemExchange POSIX-FD import requires Linux 5.6+ and glibc with the syscall numbers exposed."
#endif

constexpr std::size_t kTrialAllocSize = 2 * 1024 * 1024;

std::string errnoMessage(int err) {
  return std::error_code(err, std::generic_category()).message();
}

void* posixFdShareableHandle(int fd) {
  // CUDA's driver API encodes POSIX FD shareable handles in a `void*`.
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  return reinterpret_cast<void*>(static_cast<uintptr_t>(fd));
}

int duplicateRemoteFd(pid_t pid, int fd) {
  const int pidfd = static_cast<int>(syscall(SYS_pidfd_open, pid, 0));
  if (pidfd < 0) {
    throw std::runtime_error(
        "pidfd_open failed for POSIX FD import: " + errnoMessage(errno));
  }

  const int importedFd =
      static_cast<int>(syscall(SYS_pidfd_getfd, pidfd, fd, 0));
  const int savedErrno = errno;
  close(pidfd);
  if (importedFd < 0) {
    throw std::runtime_error(
        "pidfd_getfd failed for POSIX FD import: " + errnoMessage(savedErrno));
  }
  return importedFd;
}

#if CUDART_VERSION >= 12030
// Trial map + setAccess on the just-created allocation. Mirrors the legacy
// checkFabricHandleSupportedImpl probe so a device where create / export /
// import succeed but VA reservation / mapping / access setup fails (e.g.
// constrained VA space, IMEX/access policy issues) is classified as
// unsupported up front, instead of failing later in nvlMemExchangeVmm /
// importAndMapPeerMemory. Returns true if the full reserve+map+setAccess
// chain succeeds; cleans up everything it touched on either outcome.
bool trialMapAndSetAccess(
    CUdevice cuDev,
    CUmemGenericAllocationHandle handle,
    std::size_t allocSize,
    std::size_t granularity) {
  CUdeviceptr ptr = 0;
  auto err = pfn_cuMemAddressReserve(&ptr, allocSize, granularity, 0, 0);
  if (err != CUDA_SUCCESS) {
    return false;
  }
  err = pfn_cuMemMap(ptr, allocSize, 0, handle, 0);
  if (err != CUDA_SUCCESS) {
    pfn_cuMemAddressFree(ptr, allocSize);
    return false;
  }
  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = cuDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  err = pfn_cuMemSetAccess(ptr, allocSize, &accessDesc, 1);
  pfn_cuMemUnmap(ptr, allocSize);
  pfn_cuMemAddressFree(ptr, allocSize);
  return err == CUDA_SUCCESS;
}
#endif

bool fabricHandleSupported(CUdevice cuDev) {
#if CUDART_VERSION < 12030
  (void)cuDev;
  return false;
#else
  int fabricSupported = 0;
  auto err = pfn_cuDeviceGetAttribute(
      &fabricSupported,
      CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
      cuDev);
  if (err != CUDA_SUCCESS || fabricSupported == 0) {
    return false;
  }

  auto prop = makeVmmAllocationProp(cuDev, CU_MEM_HANDLE_TYPE_FABRIC);
  std::size_t granularity = 0;
  err = pfn_cuMemGetAllocationGranularity(
      &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  if (err != CUDA_SUCCESS) {
    return false;
  }

  const std::size_t allocSize =
      comms::bitops::roundUp(kTrialAllocSize, granularity);
  CUmemGenericAllocationHandle handle = 0;
  err = pfn_cuMemCreate(&handle, allocSize, &prop, 0);
  if (err != CUDA_SUCCESS) {
    return false;
  }

  FabricHandle fabricHandle = {};
  err = pfn_cuMemExportToShareableHandle(
      &fabricHandle, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0);
  if (err != CUDA_SUCCESS) {
    pfn_cuMemRelease(handle);
    return false;
  }

  CUmemGenericAllocationHandle importedHandle = 0;
  err = pfn_cuMemImportFromShareableHandle(
      &importedHandle, &fabricHandle, CU_MEM_HANDLE_TYPE_FABRIC);
  if (err != CUDA_SUCCESS) {
    pfn_cuMemRelease(handle);
    return false;
  }

  const bool mapOk =
      trialMapAndSetAccess(cuDev, handle, allocSize, granularity);
  pfn_cuMemRelease(importedHandle);
  pfn_cuMemRelease(handle);
  return mapOk;
#endif
}

// Probe whether the runtime sandbox permits the pidfd_open + pidfd_getfd pair
// that duplicateRemoteFd uses to import a peer's POSIX FD. seccomp policies,
// older kernels, or container restrictions can block these syscalls even when
// CUDA's POSIX-FD export/import + map round-trip succeeds; without this check
// selectShareableHandleType would return kPosixFd and the failure would only
// surface at exchange time as a thrown runtime_error with no fallback. Probes
// against the current process (self pidfd_getfd on a real fd) so we exercise
// the exact syscalls duplicateRemoteFd issues at exchange time.
bool pidfdGetfdSupported(int fd) {
  const int pidfd = static_cast<int>(syscall(SYS_pidfd_open, getpid(), 0));
  if (pidfd < 0) {
    return false;
  }
  const int dupedFd = static_cast<int>(syscall(SYS_pidfd_getfd, pidfd, fd, 0));
  close(pidfd);
  if (dupedFd < 0) {
    return false;
  }
  close(dupedFd);
  return true;
}

bool posixFdHandleSupported(CUdevice cuDev) {
#if CUDART_VERSION < 12030
  (void)cuDev;
  return false;
#else
  auto prop =
      makeVmmAllocationProp(cuDev, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
  std::size_t granularity = 0;
  auto err = pfn_cuMemGetAllocationGranularity(
      &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  if (err != CUDA_SUCCESS) {
    return false;
  }

  const std::size_t allocSize =
      comms::bitops::roundUp(kTrialAllocSize, granularity);
  CUmemGenericAllocationHandle handle = 0;
  err = pfn_cuMemCreate(&handle, allocSize, &prop, 0);
  if (err != CUDA_SUCCESS) {
    return false;
  }

  int fd = -1;
  err = pfn_cuMemExportToShareableHandle(
      &fd, handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
  if (err != CUDA_SUCCESS) {
    pfn_cuMemRelease(handle);
    return false;
  }

  // Probe pidfd_open + pidfd_getfd against ourselves while the exported fd is
  // still open. Mirrors duplicateRemoteFd's two-syscall pair so the capability
  // check matches what the cross-process exchange will actually do.
  if (!pidfdGetfdSupported(fd)) {
    close(fd);
    pfn_cuMemRelease(handle);
    return false;
  }

  CUmemGenericAllocationHandle importedHandle = 0;
  err = pfn_cuMemImportFromShareableHandle(
      &importedHandle,
      posixFdShareableHandle(fd),
      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
  close(fd);
  if (err != CUDA_SUCCESS) {
    pfn_cuMemRelease(handle);
    return false;
  }

  const bool mapOk =
      trialMapAndSetAccess(cuDev, handle, allocSize, granularity);
  pfn_cuMemRelease(importedHandle);
  pfn_cuMemRelease(handle);
  return mapOk;
#endif
}

// Import one peer's shareable handle (duplicates the peer fd internally for
// POSIX FD), wrap it in a co-ownable CuMemAllocation, map it into a fresh peer
// VA, and record the mapping + pointer in `result`. The imported handle is
// owned by the CuMemAllocation that the peer VA mapping co-owns, so dropping
// the mapping releases the handle -- no separate handle bookkeeping. cuMem
// granularity for the mapping is queried from the imported handle's properties.
void importAndMapPeerMemory(
    int32_t peer,
    const ShareableHandle& handle,
    std::size_t peerAllocatedSize,
    CUdevice cuDev,
    NvlPeerMem& result) {
#if CUDART_VERSION < 12030
  (void)peer;
  (void)handle;
  (void)peerAllocatedSize;
  (void)cuDev;
  (void)result;
  throw std::runtime_error("nvlMemExchangeVmm requires CUDA 12.3+");
#else
  // CuMemAllocation::adopt() takes ownership of the imported handle on entry:
  // on internal failure (CUDA query, allocator bad_alloc) it releases the
  // handle and rethrows, so there is no raw-handle window for the caller to
  // guard. The returned unique_ptr promotes implicitly to shared_ptr for
  // CuMemMapping::overAllocation's keep-alive contract.
  std::shared_ptr<CuMemAllocation> peerAlloc = CuMemAllocation::adopt(
      importShareableHandle(handle), cuDev, peerAllocatedSize);
  const std::size_t granularity = peerAlloc->granularity();

  result.vmmMappings.push_back(
      CuMemMapping::overAllocation(
          std::move(peerAlloc), peerAllocatedSize, granularity));
  result.peerPtrs.at(static_cast<std::size_t>(peer)) =
      // NOLINTNEXTLINE(performance-no-int-to-ptr): CUdeviceptr is an integer
      reinterpret_cast<void*>(result.vmmMappings.back().devicePtr());
#endif
}

struct LegacyVmmExchangeData {
  ShareableHandle handle{};
  std::size_t allocatedSize{};
  detail::TeamStatus status{};
};

constexpr std::size_t kWireHandleSize = 64;

struct alignas(8) PreparedVmmExchangeData {
  std::array<std::uint8_t, kWireHandleSize> handle{};
  std::uint64_t allocatedSize{0};
  std::int32_t pid{-1};
  std::int32_t fd{-1};
  std::uint8_t handleType{0};
  std::array<std::uint8_t, 3> padding{};
  detail::TeamStatus status{};
};

struct alignas(8) PreparedCudaIpcExchangeData {
  std::array<std::uint8_t, kWireHandleSize> handle{};
  detail::TeamStatus status{};
  std::array<std::uint8_t, 4> padding{};
};

static_assert(sizeof(FabricHandle) == kWireHandleSize);
static_assert(sizeof(cudaIpcMemHandle_t) == kWireHandleSize);
static_assert(sizeof(detail::TeamStatus) == 4);
static_assert(std::is_standard_layout_v<PreparedVmmExchangeData>);
static_assert(std::is_trivially_copyable_v<PreparedVmmExchangeData>);
static_assert(sizeof(PreparedVmmExchangeData) == 88);
static_assert(offsetof(PreparedVmmExchangeData, handle) == 0);
static_assert(offsetof(PreparedVmmExchangeData, allocatedSize) == 64);
static_assert(offsetof(PreparedVmmExchangeData, pid) == 72);
static_assert(offsetof(PreparedVmmExchangeData, fd) == 76);
static_assert(offsetof(PreparedVmmExchangeData, handleType) == 80);
static_assert(offsetof(PreparedVmmExchangeData, padding) == 81);
static_assert(offsetof(PreparedVmmExchangeData, status) == 84);
static_assert(std::is_standard_layout_v<PreparedCudaIpcExchangeData>);
static_assert(std::is_trivially_copyable_v<PreparedCudaIpcExchangeData>);
static_assert(sizeof(PreparedCudaIpcExchangeData) == 72);
static_assert(offsetof(PreparedCudaIpcExchangeData, handle) == 0);
static_assert(offsetof(PreparedCudaIpcExchangeData, status) == 64);
static_assert(offsetof(PreparedCudaIpcExchangeData, padding) == 68);

std::size_t checkedRankCount(int32_t rank, int32_t nRanks) {
  if (nRanks <= 0 || rank < 0 || rank >= nRanks) {
    throw std::invalid_argument("invalid NVLink exchange rank or rank count");
  }
  return static_cast<std::size_t>(nRanks);
}

std::uint64_t checkedWireSize(std::size_t size) {
  if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
    if (size > std::numeric_limits<std::uint64_t>::max()) {
      throw std::overflow_error("NVLink allocation size exceeds wire range");
    }
  }
  return static_cast<std::uint64_t>(size);
}

std::size_t checkedHostSize(std::uint64_t size) {
  if constexpr (sizeof(std::size_t) < sizeof(std::uint64_t)) {
    if (size > std::numeric_limits<std::size_t>::max()) {
      throw std::overflow_error("NVLink wire size exceeds host range");
    }
  }
  return static_cast<std::size_t>(size);
}

ShareableHandle decodePreparedVmmHandle(const PreparedVmmExchangeData& wire) {
  if (wire.handleType >
          static_cast<std::uint8_t>(ShareableHandleType::kPosixFd) ||
      wire.padding != std::array<std::uint8_t, 3>{}) {
    throw std::runtime_error("invalid prepared VMM exchange record");
  }
  ShareableHandle handle{};
  handle.type = static_cast<ShareableHandleType>(wire.handleType);
  std::memcpy(&handle.fabric, wire.handle.data(), sizeof(handle.fabric));
  handle.pid = wire.pid;
  handle.fd = wire.fd;
  return handle;
}

cudaIpcMemHandle_t decodePreparedCudaIpcHandle(
    const PreparedCudaIpcExchangeData& wire) {
  if (wire.padding != std::array<std::uint8_t, 4>{}) {
    throw std::runtime_error("invalid prepared cudaIpc exchange record");
  }
  cudaIpcMemHandle_t handle{};
  std::memcpy(&handle, wire.handle.data(), sizeof(handle));
  return handle;
}

void closeCudaIpcPeerMappings(NvlPeerMem& result, int32_t rank) noexcept {
  for (std::size_t peer = 0; peer < result.peerPtrs.size(); ++peer) {
    if (peer == static_cast<std::size_t>(rank)) {
      continue;
    }
    auto& peerPtr = result.peerPtrs[peer];
    if (peerPtr == nullptr) {
      continue;
    }
    const auto error = cudaIpcCloseMemHandle(peerPtr);
    if (error != cudaSuccess) {
      LOG(ERROR) << "cudaIpcCloseMemHandle failed during exchange rollback for "
                 << "rank " << peer << ": " << cudaGetErrorString(error);
    }
    peerPtr = nullptr;
  }
}

} // namespace

struct NvlMemExchangeWorkspace::Impl {
  Impl(int32_t rank, int32_t nRanks, void* localPtr)
      : rank(rank),
        nRanks(nRanks),
        localPtr(localPtr),
        vmmData(checkedRankCount(rank, nRanks)),
        cudaIpcData(checkedRankCount(rank, nRanks)),
        importStatus(checkedRankCount(rank, nRanks)) {
    peerMem.peerPtrs.assign(checkedRankCount(rank, nRanks), nullptr);
    peerMem.peerPtrs.at(static_cast<std::size_t>(rank)) = localPtr;
    peerMem.vmmMappings.reserve(static_cast<std::size_t>(nRanks - 1));
  }

  const int32_t rank;
  const int32_t nRanks;
  void* const localPtr;
  NvlPeerMem peerMem;
  std::vector<PreparedVmmExchangeData> vmmData;
  std::vector<PreparedCudaIpcExchangeData> cudaIpcData;
  std::vector<detail::TeamStatus> importStatus;
  bool consumed{false};
};

NvlMemExchangeWorkspace::NvlMemExchangeWorkspace(
    int32_t rank,
    int32_t nRanks,
    void* localPtr)
    : impl_(std::make_unique<Impl>(rank, nRanks, localPtr)) {}

NvlMemExchangeWorkspace::~NvlMemExchangeWorkspace() = default;

void NvlMemExchangeWorkspace::consume(
    int32_t rank,
    int32_t nRanks,
    void* localPtr) {
  if (rank != impl_->rank || nRanks != impl_->nRanks ||
      localPtr != impl_->localPtr) {
    throw std::invalid_argument(
        "NVLink exchange workspace identity does not match its construction");
  }
  if (impl_->consumed) {
    throw std::logic_error("NVLink exchange workspace was already consumed");
  }
  impl_->consumed = true;
}

CUmemAllocationHandleType toCudaHandleType(ShareableHandleType type) {
#if CUDART_VERSION < 12030
  (void)type;
  throw std::runtime_error("NvlMemExchange requires CUDA 12.3+");
#else
  switch (type) {
    case ShareableHandleType::kFabric:
      return CU_MEM_HANDLE_TYPE_FABRIC;
    case ShareableHandleType::kPosixFd:
      return CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    case ShareableHandleType::kUnsupported:
      break;
  }
  throw std::runtime_error("NvlMemExchange: unsupported handle type");
#endif
}

namespace {

// Per-device cache for selectShareableHandleType. Each probe runs full
// cuMemCreate/export/import/map trials so re-running per handler construction
// is expensive; cache the result per cudaDevice (matches
// MultimemHandler::isMultimemSupported's pattern and the legacy
// isFabricHandleSupported caching in GpuMemHandler.cc). Devices >=
// kSelectCacheSize fall through to the uncached path; in practice we have <8.
constexpr int kSelectCacheSize = 8;

ShareableHandleType selectShareableHandleTypeUncached(int cudaDevice) {
#if CUDART_VERSION < 12030
  (void)cudaDevice;
  return ShareableHandleType::kUnsupported;
#else
  if (cudaDevice < 0) {
    return ShareableHandleType::kUnsupported;
  }
  if (cuda_driver_lazy_init() != 0) {
    return ShareableHandleType::kUnsupported;
  }

  CUdevice cuDev = 0;
  if (pfn_cuDeviceGet(&cuDev, cudaDevice) != CUDA_SUCCESS) {
    return ShareableHandleType::kUnsupported;
  }

  if (fabricHandleSupported(cuDev)) {
    return ShareableHandleType::kFabric;
  }
  if (posixFdHandleSupported(cuDev)) {
    return ShareableHandleType::kPosixFd;
  }
  return ShareableHandleType::kUnsupported;
#endif
}

} // namespace

ShareableHandleType selectShareableHandleType(int cudaDevice) {
  if (cudaDevice < 0 || cudaDevice >= kSelectCacheSize) {
    // < 0 is a caller bug; the uncached impl returns kUnsupported for it.
    // >= kSelectCacheSize is "valid but exceeds the cache" — every handler
    // construction will now pay the full probe cost instead of hitting the
    // cache. Warn once (per process) so the developer notices and bumps
    // kSelectCacheSize.
    if (cudaDevice >= kSelectCacheSize) {
      LOG_FIRST_N(WARNING, 1)
          << "selectShareableHandleType: cudaDevice=" << cudaDevice
          << " exceeds the per-device cache size (kSelectCacheSize="
          << kSelectCacheSize
          << "); falling through to the uncached probe on every call. "
             "Consider bumping kSelectCacheSize.";
    }
    return selectShareableHandleTypeUncached(cudaDevice);
  }
  // Encode the resolved type plus one so zero remains the value-initialized
  // uncomputed sentinel.
  static std::array<std::atomic<int>, kSelectCacheSize> cache{};
  auto& slot = cache[static_cast<std::size_t>(cudaDevice)];
  const int cached = slot.load(std::memory_order_acquire);
  if (cached != 0) {
    return static_cast<ShareableHandleType>(cached - 1);
  }
  const auto resolved = selectShareableHandleTypeUncached(cudaDevice);
  slot.store(static_cast<int>(resolved) + 1, std::memory_order_release);
  return resolved;
}

ShareableHandle exportShareableHandle(
    CUmemGenericAllocationHandle handle,
    ShareableHandleType type) {
  ShareableHandle result;
#if CUDART_VERSION < 12030
  (void)handle;
  (void)type;
  throw std::runtime_error("NvlMemExchange requires CUDA 12.3+");
#else
  result.type = type;
  if (type == ShareableHandleType::kFabric) {
    checkCuError(
        pfn_cuMemExportToShareableHandle(
            &result.fabric, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0),
        "cuMemExportToShareableHandle for fabric handle failed");
  } else if (type == ShareableHandleType::kPosixFd) {
    int fd = -1;
    checkCuError(
        pfn_cuMemExportToShareableHandle(
            &fd, handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0),
        "cuMemExportToShareableHandle for POSIX FD failed");
    result.pid = getpid();
    result.fd = fd;
  } else {
    throw std::runtime_error(
        "NvlMemExchange: cannot export unsupported handle type");
  }
#endif
  return result;
}

CUmemGenericAllocationHandle importShareableHandle(const ShareableHandle& h) {
  CUmemGenericAllocationHandle imported = 0;
#if CUDART_VERSION < 12030
  (void)h;
  throw std::runtime_error("NvlMemExchange requires CUDA 12.3+");
#else
  if (h.type == ShareableHandleType::kFabric) {
    auto fabric = h.fabric;
    checkCuError(
        pfn_cuMemImportFromShareableHandle(
            &imported, &fabric, CU_MEM_HANDLE_TYPE_FABRIC),
        "cuMemImportFromShareableHandle for fabric handle failed");
  } else if (h.type == ShareableHandleType::kPosixFd) {
    const int dupFd = duplicateRemoteFd(h.pid, h.fd);
    const auto importResult = pfn_cuMemImportFromShareableHandle(
        &imported,
        posixFdShareableHandle(dupFd),
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
    close(dupFd);
    checkCuError(
        importResult, "cuMemImportFromShareableHandle for POSIX FD failed");
  } else {
    throw std::runtime_error(
        "NvlMemExchange: cannot import unsupported handle type");
  }
#endif
  return imported;
}

NvlPeerMem nvlMemExchangeVmm(
    meta::comms::IBootstrap& bootstrap,
    int32_t rank,
    int32_t nRanks,
    CUdevice cuDev,
    CUmemGenericAllocationHandle localHandle,
    void* localPtr,
    std::size_t allocatedSize,
    bool preferFabric,
    bool* localHandlePossiblyExposed) {
  NvlPeerMem result;
  result.peerPtrs.assign(static_cast<std::size_t>(nRanks), nullptr);
  result.peerPtrs[static_cast<std::size_t>(rank)] = localPtr;

#if CUDART_VERSION < 12030
  (void)bootstrap;
  (void)cuDev;
  (void)localHandle;
  (void)allocatedSize;
  (void)preferFabric;
  (void)localHandlePossiblyExposed;
  throw std::runtime_error("nvlMemExchangeVmm requires CUDA 12.3+");
#else
  const ShareableHandleType exportType = preferFabric
      ? ShareableHandleType::kFabric
      : ShareableHandleType::kPosixFd;
  std::vector<LegacyVmmExchangeData> allData(static_cast<std::size_t>(nRanks));
  std::vector<detail::TeamStatus> importStatus(
      static_cast<std::size_t>(nRanks));
  FdGuard localFdGuard;
  std::exception_ptr exportError;
  try {
    auto& localData = allData[static_cast<std::size_t>(rank)];
    localData.handle = exportShareableHandle(localHandle, exportType);
    localData.allocatedSize = allocatedSize;
    if (localData.handle.type == ShareableHandleType::kPosixFd) {
      localFdGuard.reset(localData.handle.fd);
    }
  } catch (...) {
    exportError = std::current_exception();
  }
  if (!exportError && localHandlePossiblyExposed != nullptr) {
    *localHandlePossiblyExposed = true;
  }
  detail::allGatherAndAgree(
      bootstrap,
      rank,
      nRanks,
      allData,
      exportError,
      "nvlMemExchangeVmm handle export");

  std::exception_ptr importError;
  try {
    for (int32_t peer = 0; peer < nRanks; ++peer) {
      if (peer == rank) {
        continue;
      }
      const auto peerIdx = static_cast<std::size_t>(peer);
      importAndMapPeerMemory(
          peer,
          allData[peerIdx].handle,
          allData[peerIdx].allocatedSize,
          cuDev,
          result);
    }
  } catch (...) {
    importError = std::current_exception();
  }
  try {
    detail::allGatherAndAgree(
        bootstrap,
        rank,
        nRanks,
        importStatus,
        importError,
        "nvlMemExchangeVmm peer import");
  } catch (...) {
    result.vmmMappings.clear();
    throw;
  }
  return result;
#endif
}

NvlPeerMem nvlMemExchangeVmmPrepared(
    meta::comms::IBootstrap& bootstrap,
    int32_t rank,
    int32_t nRanks,
    CUdevice cuDev,
    CUmemGenericAllocationHandle localHandle,
    void* localPtr,
    std::size_t allocatedSize,
    bool preferFabric,
    NvlMemExchangeWorkspace& workspace) {
  workspace.consume(rank, nRanks, localPtr);
  auto& result = workspace.impl_->peerMem;
  result.vmmMappings.clear();
  std::fill(result.peerPtrs.begin(), result.peerPtrs.end(), nullptr);
  result.peerPtrs.at(static_cast<std::size_t>(rank)) = localPtr;

#if CUDART_VERSION < 12030
  (void)bootstrap;
  (void)cuDev;
  (void)localHandle;
  (void)allocatedSize;
  (void)preferFabric;
  throw std::runtime_error("nvlMemExchangeVmm requires CUDA 12.3+");
#else
  const ShareableHandleType exportType = preferFabric
      ? ShareableHandleType::kFabric
      : ShareableHandleType::kPosixFd;
  auto& allData = workspace.impl_->vmmData;
  auto& importStatus = workspace.impl_->importStatus;
  std::fill(allData.begin(), allData.end(), PreparedVmmExchangeData{});
  std::fill(importStatus.begin(), importStatus.end(), detail::TeamStatus{});
  FdGuard localFdGuard;
  std::exception_ptr exportError;
  try {
    auto& localData = allData[static_cast<std::size_t>(rank)];
    const auto handle = exportShareableHandle(localHandle, exportType);
    std::memcpy(localData.handle.data(), &handle.fabric, sizeof(handle.fabric));
    localData.allocatedSize = checkedWireSize(allocatedSize);
    localData.pid = handle.pid;
    localData.fd = handle.fd;
    localData.handleType = static_cast<std::uint8_t>(handle.type);
    if (handle.type == ShareableHandleType::kPosixFd) {
      localFdGuard.reset(handle.fd);
    }
  } catch (...) {
    exportError = std::current_exception();
  }
  detail::allGatherAndAgree(
      bootstrap,
      rank,
      nRanks,
      allData,
      exportError,
      "nvlMemExchangeVmm handle export");

  // Every rank must reach the status exchange even if a local import fails.
  // Besides propagating that failure, this collective keeps each exported
  // POSIX FD open until every peer has finished attempting its imports.
  std::exception_ptr importError;
  try {
    for (int32_t peer = 0; peer < nRanks; ++peer) {
      if (peer == rank) {
        continue;
      }
      const auto peerIdx = static_cast<std::size_t>(peer);
      const auto handle = decodePreparedVmmHandle(allData[peerIdx]);
      importAndMapPeerMemory(
          peer,
          handle,
          checkedHostSize(allData[peerIdx].allocatedSize),
          cuDev,
          result);
    }
  } catch (...) {
    importError = std::current_exception();
  }
  try {
    detail::allGatherAndAgree(
        bootstrap,
        rank,
        nRanks,
        importStatus,
        importError,
        "nvlMemExchangeVmm peer import");
  } catch (...) {
    result.vmmMappings.clear();
    std::fill(result.peerPtrs.begin(), result.peerPtrs.end(), nullptr);
    result.peerPtrs.at(static_cast<std::size_t>(rank)) = localPtr;
    throw;
  }

  return std::move(result);
#endif
}

NvlPeerMem nvlMemExchangeCudaIpc(
    meta::comms::IBootstrap& bootstrap,
    int32_t rank,
    int32_t nRanks,
    void* localPtr,
    bool* localHandlePossiblyExposed) {
  NvlPeerMem result;
  result.peerPtrs.assign(static_cast<std::size_t>(nRanks), nullptr);
  result.peerPtrs[static_cast<std::size_t>(rank)] = localPtr;
  std::vector<cudaIpcMemHandle_t> allHandles(static_cast<std::size_t>(nRanks));

  cudaIpcMemHandle_t localHandle{};
  checkCudaError(
      cudaIpcGetMemHandle(&localHandle, localPtr),
      "cudaIpcGetMemHandle failed");
  allHandles[static_cast<std::size_t>(rank)] = localHandle;
  if (localHandlePossiblyExposed != nullptr) {
    *localHandlePossiblyExposed = true;
  }

  auto gatherResult =
      bootstrap
          .allGather(
              allHandles.data(), sizeof(cudaIpcMemHandle_t), rank, nRanks)
          .get();
  if (gatherResult != 0) {
    throw std::runtime_error("nvlMemExchangeCudaIpc allGather failed");
  }

  for (int32_t peer = 0; peer < nRanks; ++peer) {
    if (peer == rank) {
      continue;
    }
    const auto peerIdx = static_cast<std::size_t>(peer);
    checkCudaError(
        cudaIpcOpenMemHandle(
            &result.peerPtrs[peerIdx],
            allHandles[peerIdx],
            cudaIpcMemLazyEnablePeerAccess),
        "cudaIpcOpenMemHandle failed");
  }

  return result;
}

NvlPeerMem nvlMemExchangeCudaIpcPrepared(
    meta::comms::IBootstrap& bootstrap,
    int32_t rank,
    int32_t nRanks,
    void* localPtr,
    const cudaIpcMemHandle_t& localHandle,
    NvlMemExchangeWorkspace& workspace) {
  workspace.consume(rank, nRanks, localPtr);
  auto& result = workspace.impl_->peerMem;
  closeCudaIpcPeerMappings(result, rank);
  std::fill(result.peerPtrs.begin(), result.peerPtrs.end(), nullptr);
  result.peerPtrs.at(static_cast<std::size_t>(rank)) = localPtr;

  auto& allData = workspace.impl_->cudaIpcData;
  auto& importStatus = workspace.impl_->importStatus;
  std::fill(allData.begin(), allData.end(), PreparedCudaIpcExchangeData{});
  std::fill(importStatus.begin(), importStatus.end(), detail::TeamStatus{});
  std::memcpy(
      allData.at(static_cast<std::size_t>(rank)).handle.data(),
      &localHandle,
      sizeof(localHandle));

  // A nonzero bootstrap return can itself be observed asymmetrically; callers
  // must treat that as terminal and abort the communicator. Local export and
  // import failures are different: they are carried by these two agreements so
  // every team member remains on the same collective sequence.
  detail::allGatherAndAgree(
      bootstrap,
      rank,
      nRanks,
      allData,
      std::exception_ptr{},
      "nvlMemExchangeCudaIpc handle export");

  std::exception_ptr importError;
  try {
    for (int32_t peer = 0; peer < nRanks; ++peer) {
      if (peer == rank) {
        continue;
      }
      const auto peerIdx = static_cast<std::size_t>(peer);
      const auto peerHandle = decodePreparedCudaIpcHandle(allData[peerIdx]);
      checkCudaError(
          cudaIpcOpenMemHandle(
              &result.peerPtrs[peerIdx],
              peerHandle,
              cudaIpcMemLazyEnablePeerAccess),
          "cudaIpcOpenMemHandle failed");
    }
  } catch (...) {
    importError = std::current_exception();
  }

  try {
    detail::allGatherAndAgree(
        bootstrap,
        rank,
        nRanks,
        importStatus,
        importError,
        "nvlMemExchangeCudaIpc peer import");
  } catch (...) {
    closeCudaIpcPeerMappings(result, rank);
    throw;
  }

  return std::move(result);
}

} // namespace comms::prims
