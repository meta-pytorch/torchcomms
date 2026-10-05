// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/transport/p2p/P2pVmm.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cerrno>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "comms/uniflow/drivers/cuda/CudaDevicePtr.h"
#include "comms/uniflow/drivers/cuda/CudaDriverApi.h"
#include "comms/uniflow/logging/Logger.h"

// Fallback definitions for older kernel headers.
#ifndef SYS_pidfd_open
#define SYS_pidfd_open 434
#endif
#ifndef SYS_pidfd_getfd
#define SYS_pidfd_getfd 438
#endif

namespace uniflow {
namespace {

using VmmChunk = P2pRegistrationHandle::VmmChunk;
using VmmPayload = P2pRegistrationHandle::VmmPayload;

// Descriptors left free for the rest of the process (sockets, other
// libraries) when budgeting the fds an export keeps open.
constexpr uint64_t kFdHeadroom = 256;

class UniqueFd {
 public:
  explicit UniqueFd(int fd) noexcept : fd_{fd} {}
  ~UniqueFd() {
    reset();
  }
  UniqueFd(UniqueFd&& other) noexcept : fd_{other.release()} {}
  UniqueFd& operator=(UniqueFd&&) = delete;
  UniqueFd(const UniqueFd&) = delete;
  UniqueFd& operator=(const UniqueFd&) = delete;

  int get() const noexcept {
    return fd_;
  }
  int release() noexcept {
    return std::exchange(fd_, -1);
  }
  void reset() noexcept {
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
  }

 private:
  int fd_{-1};
};

struct ChunkSpan {
  uint64_t base{0};
  uint64_t size{0};
};

struct FileId {
  uint64_t dev{0};
  uint64_t inode{0};
};

Err vmmError(ErrCode code, std::string_view what) {
  return Err(code, "P2P VMM: " + std::string(what));
}

Err errnoError(std::string_view what) {
  return vmmError(
      ErrCode::DriverError,
      std::string(what) + ": " + std::system_category().message(errno));
}

uint64_t toAddress(CUdeviceptr ptr) {
  return reinterpret_cast<uint64_t>(ptr);
}

void* toHostPtr(uint64_t addr) {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  return reinterpret_cast<void*>(addr);
}

CUmemLocation deviceLocation(int deviceId) {
  CUmemLocation location{};
  location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  location.id = deviceId;
  return location;
}

bool isVmmAt(CudaDriverApi& driver, int deviceId, uint64_t addr) {
  unsigned long long flags = 0;
  const CUmemLocation location = deviceLocation(deviceId);
  return !driver.cuMemGetAccess(&flags, &location, toDevicePtr(addr))
              .hasError();
}

// Finds the chunks covering [begin, end) without retaining any, so a segment
// that cannot be exported leaves no retained handle behind.
Result<std::vector<ChunkSpan>> findChunks(
    CudaDriverApi& driver,
    int deviceId,
    uint64_t begin,
    uint64_t end,
    size_t maxChunks) {
  std::vector<ChunkSpan> spans;
  for (uint64_t cursor = begin; cursor < end;) {
    if (spans.size() == maxChunks) {
      return vmmError(
          ErrCode::ResourceExhausted,
          "segment spans more than " + std::to_string(maxChunks) + " chunks");
    }
    // Checked per chunk because retaining non-VMM memory crashes.
    if (!isVmmAt(driver, deviceId, cursor)) {
      return vmmError(
          ErrCode::InvalidArgument,
          "segment byte " + std::to_string(cursor - begin) +
              " is not VMM-mapped");
    }
    CUdeviceptr base{};
    size_t size = 0;
    CHECK_EXPR(
        driver.cuMemGetAddressRange_v2(&base, &size, toDevicePtr(cursor)));
    const uint64_t chunkBase = toAddress(base);
    if (chunkBase > cursor || size <= cursor - chunkBase) {
      return vmmError(
          ErrCode::DriverError, "chunk range does not contain its address");
    }
    if (!spans.empty() && chunkBase != cursor) {
      return vmmError(
          ErrCode::InvalidArgument, "segment chunks are not VA-contiguous");
    }
    spans.push_back(ChunkSpan{.base = chunkBase, .size = size});
    cursor = chunkBase + size;
  }
  return spans;
}

// Retains each chunk at the first segment byte it holds. The handles are never
// released: HIP's retain adds a reference only on some ROCm 7.0 builds, so a
// release could drop the owner's own reference.
Result<std::vector<CUmemGenericAllocationHandle>> retainChunks(
    CudaDriverApi& driver,
    uint64_t begin,
    const std::vector<ChunkSpan>& spans) {
  std::vector<CUmemGenericAllocationHandle> retained;
  retained.reserve(spans.size());
  for (const auto& span : spans) {
    CUmemGenericAllocationHandle handle{};
    CHECK_EXPR(driver.cuMemRetainAllocationHandle(
        &handle, toHostPtr(std::max(span.base, begin))));
    retained.push_back(handle);
  }
  return retained;
}

Result<uint64_t> countOpenFds() {
  DIR* dir = ::opendir("/proc/self/fd");
  if (dir == nullptr) {
    return errnoError("opendir(/proc/self/fd)");
  }
  uint64_t count = 0;
  while (const dirent* entry = ::readdir(dir)) {
    if (entry->d_name[0] != '.') {
      ++count;
    }
  }
  ::closedir(dir);
  // The directory stream's own descriptor was counted.
  return count - 1;
}

// An export keeps one fd per chunk open until deregistration so importers can
// pull them; skip VMM rather than exhaust the process fd table.
Status checkFdBudget(size_t chunks) {
  rlimit limit{};
  if (::getrlimit(RLIMIT_NOFILE, &limit) != 0) {
    return errnoError("getrlimit(RLIMIT_NOFILE)");
  }
  if (limit.rlim_cur == RLIM_INFINITY) {
    return Ok();
  }
  auto open = countOpenFds();
  CHECK_RETURN(open);
  if (open.value() + chunks + kFdHeadroom > limit.rlim_cur) {
    return vmmError(
        ErrCode::ResourceExhausted,
        std::to_string(chunks) + " chunk fds with " +
            std::to_string(open.value()) + " open and " +
            std::to_string(kFdHeadroom) +
            " reserved exceed the RLIMIT_NOFILE soft limit " +
            std::to_string(limit.rlim_cur));
  }
  return Ok();
}

Result<FileId> fileIdOf(int fd) {
  struct stat st{};
  if (::fstat(fd, &st) != 0) {
    return errnoError("fstat");
  }
  return FileId{
      .dev = static_cast<uint64_t>(st.st_dev),
      .inode = static_cast<uint64_t>(st.st_ino)};
}

// Exports each retained chunk as a close-on-exec POSIX fd; on success the
// returned chunks own the fds.
Result<std::vector<VmmChunk>> exportChunks(
    CudaDriverApi& driver,
    const std::vector<ChunkSpan>& spans,
    const std::vector<CUmemGenericAllocationHandle>& retained) {
  std::vector<UniqueFd> fds;
  std::vector<VmmChunk> chunks;
  fds.reserve(retained.size());
  chunks.reserve(retained.size());
  for (size_t i = 0; i < retained.size(); ++i) {
    int fd = -1;
    CHECK_EXPR(driver.cuMemExportToShareableHandle(
        &fd, retained[i], CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
    fds.emplace_back(fd);
    // Runtimes do not document O_CLOEXEC on exported fds. Setting it here
    // still leaves a window in which a concurrent fork and exec inherits one.
    const int flags = ::fcntl(fd, F_GETFD);
    if (flags < 0 || ::fcntl(fd, F_SETFD, flags | FD_CLOEXEC) != 0) {
      return errnoError("fcntl(FD_CLOEXEC)");
    }
    auto id = fileIdOf(fd);
    CHECK_RETURN(id);
    chunks.push_back(
        VmmChunk{
            .fd = fd,
            .dev = id.value().dev,
            .inode = id.value().inode,
            .size = spans[i].size});
  }
  for (auto& fd : fds) {
    fd.release();
  }
  return chunks;
}

// Duplicates a peer's exported fd, checking that the fd number still names
// the exported object (the peer may have closed it and reused the number).
Result<UniqueFd> pullChunkFd(int pidfd, const VmmChunk& chunk) {
  const int peerFd = chunk.fd;
  UniqueFd fd{static_cast<int>(::syscall(SYS_pidfd_getfd, pidfd, peerFd, 0))};
  if (fd.get() < 0) {
    return errnoError("pidfd_getfd(" + std::to_string(peerFd) + ")");
  }
  auto id = fileIdOf(fd.get());
  CHECK_RETURN(id);
  if (id.value().dev != chunk.dev || id.value().inode != chunk.inode) {
    return vmmError(
        ErrCode::InvalidArgument,
        "peer fd " + std::to_string(peerFd) +
            " no longer names the exported chunk");
  }
  return fd;
}

CUmemAccessDesc readWriteAccess(int deviceId) {
  CUmemAccessDesc desc{};
  desc.location = deviceLocation(deviceId);
  desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  return desc;
}

// Peer chunks mapped back to back into one reservation. Destruction (also
// the unwind path of a failed import) unmaps, frees the reservation, then
// releases the imported handles, logging failures.
class P2pVmmMapping final : public P2pMapping {
 public:
  P2pVmmMapping(
      std::shared_ptr<CudaDriverApi> driver,
      uint64_t base,
      uint64_t reserved)
      : driver_{std::move(driver)}, base_{base}, reserved_{reserved} {}

  ~P2pVmmMapping() override {
    uint64_t offset = 0;
    for (const uint64_t size : mappedSizes_) {
      logIfError(
          driver_->cuMemUnmap(toDevicePtr(base_ + offset), size), "unmap");
      offset += size;
    }
    logIfError(
        driver_->cuMemAddressFree(toDevicePtr(base_), reserved_),
        "address free");
    for (const auto handle : handles_) {
      logIfError(driver_->cuMemRelease(handle), "release");
    }
  }

  P2pVmmMapping(const P2pVmmMapping&) = delete;
  P2pVmmMapping& operator=(const P2pVmmMapping&) = delete;
  P2pVmmMapping(P2pVmmMapping&&) = delete;
  P2pVmmMapping& operator=(P2pVmmMapping&&) = delete;

  void* base() const noexcept {
    return toHostPtr(base_);
  }

  // Imports one peer chunk and maps it after the chunks mapped so far. The
  // pulled fd is closed right after the import, so at most one is held.
  Status mapNext(int pidfd, const VmmChunk& chunk) {
    CUmemGenericAllocationHandle handle{};
    {
      auto fd = pullChunkFd(pidfd, chunk);
      CHECK_RETURN(fd);
      CHECK_EXPR(driver_->importPosixFd(&handle, fd.value().get()));
    }
    handles_.push_back(handle);
    const uint64_t size = chunk.size;
    CHECK_EXPR(driver_->cuMemMap(
        toDevicePtr(base_ + mappedBytes_), size, 0, handle, 0));
    mappedSizes_.push_back(size);
    mappedBytes_ += size;
    return Ok();
  }

  // Transfers may run on either device's copy engines (get() copies on the
  // source device), so both need access to the mapping.
  Status grantAccess(int deviceId, int exporterDevice) {
    const std::array<CUmemAccessDesc, 2> descs{
        readWriteAccess(deviceId), readWriteAccess(exporterDevice)};
    const size_t count = deviceId == exporterDevice ? 1 : 2;
    return driver_->cuMemSetAccess(
        toDevicePtr(base_), mappedBytes_, descs.data(), count);
  }

 private:
  static void logIfError(const Status& st, std::string_view step) {
    if (st.hasError()) {
      UNIFLOW_LOG_ERROR(
          "P2P VMM: mapping teardown {} failed: {}",
          step,
          st.error().message());
    }
  }

  std::shared_ptr<CudaDriverApi> driver_;
  uint64_t base_{0};
  uint64_t reserved_{0};
  uint64_t mappedBytes_{0};
  std::vector<uint64_t> mappedSizes_;
  std::vector<CUmemGenericAllocationHandle> handles_;
};

} // namespace

Result<P2pProcessScope> currentProcessScope() {
  P2pProcessScope scope;
  const UniqueFd bootIdFd{
      ::open("/proc/sys/kernel/random/boot_id", O_RDONLY | O_CLOEXEC)};
  if (bootIdFd.get() < 0) {
    return errnoError("open(boot_id)");
  }
  // A UUID: 32 hex digits in dash-separated groups, then a newline.
  std::array<char, 64> text{};
  const ssize_t length = ::read(bootIdFd.get(), text.data(), text.size());
  if (length < 0) {
    return errnoError("read(boot_id)");
  }
  size_t digits = 0;
  for (size_t i = 0; i < static_cast<size_t>(length) && text[i] != '\n'; ++i) {
    const char c = text[i];
    if (c == '-') {
      continue;
    }
    int value = -1;
    if (c >= '0' && c <= '9') {
      value = c - '0';
    } else if (c >= 'a' && c <= 'f') {
      value = c - 'a' + 10;
    }
    if (value < 0 || digits == 2 * scope.bootId.size()) {
      return vmmError(ErrCode::DriverError, "malformed boot_id");
    }
    scope.bootId[digits / 2] |=
        static_cast<uint8_t>(digits % 2 == 0 ? value << 4 : value);
    ++digits;
  }
  if (digits != 2 * scope.bootId.size()) {
    return vmmError(ErrCode::DriverError, "malformed boot_id");
  }
  struct stat pidNamespace{};
  if (::stat("/proc/self/ns/pid", &pidNamespace) != 0) {
    return errnoError("stat(/proc/self/ns/pid)");
  }
  scope.pidNamespace = static_cast<uint64_t>(pidNamespace.st_ino);
  return scope;
}

P2pVmm::P2pVmm(int deviceId, std::shared_ptr<CudaDriverApi> driver)
    : deviceId_{deviceId}, driver_{std::move(driver)} {}

bool P2pVmm::isVmm(const void* ptr) const {
  auto supported = driver_->isCuMemSupported();
  return !supported.hasError() && supported.value() &&
      isVmmAt(*driver_, deviceId_, reinterpret_cast<uint64_t>(ptr));
}

Result<bool> P2pVmm::grantPeerAccess(
    void* ptr,
    size_t len,
    std::span<const int> peerDevices) const {
  const auto begin = reinterpret_cast<uint64_t>(ptr);
  if (len == 0 || len > std::numeric_limits<uint64_t>::max() - begin) {
    return vmmError(ErrCode::InvalidArgument, "invalid segment range");
  }
  if (peerDevices.empty()) {
    return true;
  }
  auto spans = findChunks(
      *driver_,
      deviceId_,
      begin,
      begin + len,
      std::numeric_limits<size_t>::max());
  if (spans.hasError()) {
    return false;
  }
  // The owner's device is listed again in case a runtime replaces, rather
  // than amends, the access of the locations it is not given.
  std::vector<CUmemAccessDesc> descs{readWriteAccess(deviceId_)};
  for (const int device : peerDevices) {
    descs.push_back(readWriteAccess(device));
  }
  // One call per chunk: a single call across chunks is not documented to
  // work on every runtime.
  for (const auto& span : spans.value()) {
    CHECK_EXPR(driver_->cuMemSetAccess(
        toDevicePtr(span.base), span.size, descs.data(), descs.size()));
  }
  return true;
}

Result<std::unique_ptr<P2pRegistrationHandle>> P2pVmm::exportSegment(
    void* ptr,
    size_t len) const {
  const auto begin = reinterpret_cast<uint64_t>(ptr);
  if (len == 0 || len > std::numeric_limits<uint64_t>::max() - begin) {
    return vmmError(ErrCode::InvalidArgument, "invalid segment range");
  }
  // Before any retain: a failure past that point leaves the chunks retained.
  auto scope = currentProcessScope();
  CHECK_RETURN(scope);
  auto spans =
      findChunks(*driver_, deviceId_, begin, begin + len, kP2pMaxVmmChunks);
  CHECK_RETURN(spans);
  CHECK_EXPR(checkFdBudget(spans.value().size()));
  auto retained = retainChunks(*driver_, begin, spans.value());
  CHECK_RETURN(retained);
  auto chunks = exportChunks(*driver_, spans.value(), retained.value());
  CHECK_RETURN(chunks);
  // len > 0, so a successful walk found at least one chunk.
  assert(!spans.value().empty());
  return std::make_unique<P2pRegistrationHandle>(VmmPayload{
      .scope = scope.value(),
      .ownerPid = static_cast<int32_t>(::getpid()),
      .exporterDevice = deviceId_,
      .offset = begin - spans.value().front().base,
      .size = len,
      .chunks = std::move(chunks).value(),
  });
}

Result<std::unique_ptr<P2pRemoteRegistrationHandle>> P2pVmm::importSegment(
    const VmmPayload& payload) const {
  // A pid names the exporter only within its own boot and PID namespace.
  auto scope = currentProcessScope();
  CHECK_RETURN(scope);
  if (scope.value() != payload.scope) {
    return vmmError(
        ErrCode::InvalidArgument,
        "exporter pid " + std::to_string(payload.ownerPid) +
            " is on another host or in another PID namespace");
  }
  const UniqueFd pidfd{
      static_cast<int>(::syscall(SYS_pidfd_open, payload.ownerPid, 0))};
  if (pidfd.get() < 0) {
    return errnoError("pidfd_open(" + std::to_string(payload.ownerPid) + ")");
  }

  uint64_t total = 0;
  for (const auto& chunk : payload.chunks) {
    total += chunk.size;
  }
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location = deviceLocation(deviceId_);
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  size_t granularity = 0;
  CHECK_EXPR(driver_->cuMemGetAllocationGranularity(
      &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  if (granularity == 0) {
    return vmmError(ErrCode::DriverError, "allocation granularity is zero");
  }
  if (total > std::numeric_limits<uint64_t>::max() - granularity) {
    return vmmError(ErrCode::InvalidArgument, "segment is too large to map");
  }
  const uint64_t reserved =
      (total + granularity - 1) / granularity * granularity;
  CUdeviceptr base{};
  CHECK_EXPR(
      driver_->cuMemAddressReserve(&base, reserved, 0, toDevicePtr(0), 0));

  auto mapping =
      std::make_unique<P2pVmmMapping>(driver_, toAddress(base), reserved);
  for (const auto& chunk : payload.chunks) {
    CHECK_EXPR(mapping->mapNext(pidfd.get(), chunk));
  }
  CHECK_EXPR(mapping->grantAccess(deviceId_, payload.exporterDevice));
  void* mappedBase = mapping->base();
  return std::make_unique<P2pRemoteRegistrationHandle>(
      mappedBase, payload.offset, payload.size, std::move(mapping));
}

} // namespace uniflow
