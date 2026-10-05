// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/transport/p2p/P2pTransport.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <fcntl.h>
#include <unistd.h>

#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "comms/uniflow/drivers/cuda/mock/MockCudaApi.h"
#include "comms/uniflow/executor/ScopedEventBaseThread.h"
#include "comms/uniflow/transport/p2p/P2pRegistrationHandle.h"
#if defined(__HIP_PLATFORM_AMD__)
#include <dirent.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <tuple>
#include <utility>

#include "comms/uniflow/drivers/cuda/CudaDevicePtr.h"
#include "comms/uniflow/drivers/cuda/mock/MockCudaDriverApi.h"
#include "comms/uniflow/transport/p2p/P2pVmm.h"
#endif

namespace uniflow {

// Helper that leverages the `friend class SegmentTest` declarations in
// RegisteredSegment / RemoteRegisteredSegment to build them (with handles) from
// tests, since production construction APIs are not yet available.
class SegmentTest {
 public:
  static RegisteredSegment makeRegisteredSegment(
      void* buf,
      size_t len,
      MemoryType memType,
      int deviceId) {
    return RegisteredSegment(buf, len, memType, deviceId);
  }

  static RemoteRegisteredSegment makeRemote(
      void* buf,
      size_t len,
      std::unique_ptr<RemoteRegistrationHandle> handle) {
    RemoteRegisteredSegment remote(buf, len);
    remote.handles_.push_back(std::move(handle));
    return remote;
  }
};

namespace {

using ::testing::_;
using ::testing::DoAll;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::SetArgPointee;
#if defined(__HIP_PLATFORM_AMD__)
using ::testing::AnyNumber;
using ::testing::HasSubstr;
using ::testing::InSequence;

using IpcPayload = P2pRegistrationHandle::IpcPayload;
using VmmPayload = P2pRegistrationHandle::VmmPayload;
using ChunkShape =
    std::vector<std::tuple<uint64_t, uint64_t, uint64_t>>; // dev, inode, size

constexpr uint64_t kMiB = uint64_t{1} << 20;
constexpr uint64_t kExportVa = 0x7f0000000000;
constexpr uint64_t kImportVa = 0x7e0000000000;

uint64_t addressOf(CUdeviceptr ptr) {
  return reinterpret_cast<uint64_t>(ptr);
}

CUmemGenericAllocationHandle fakeHandle(uint64_t id) {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  return reinterpret_cast<CUmemGenericAllocationHandle>(id);
}

void* hostPtr(uint64_t addr) {
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  return reinterpret_cast<void*>(addr);
}

struct stat statOf(int fd) {
  struct stat st{};
  EXPECT_EQ(::fstat(fd, &st), 0);
  return st;
}

bool sameFile(const struct stat& a, const struct stat& b) {
  return a.st_dev == b.st_dev && a.st_ino == b.st_ino;
}

std::vector<int> openFds() {
  std::vector<int> fds;
  DIR* dir = ::opendir("/proc/self/fd");
  EXPECT_NE(dir, nullptr);
  if (dir == nullptr) {
    return fds;
  }
  while (const dirent* entry = ::readdir(dir)) {
    const int fd = std::atoi(entry->d_name);
    if (entry->d_name[0] != '.' && fd != ::dirfd(dir)) {
      fds.push_back(fd);
    }
  }
  ::closedir(dir);
  return fds;
}

// Counts the open fds, fd included, that name the same file as fd.
int countFdsSharingFile(int fd) {
  const struct stat target = statOf(fd);
  int count = 0;
  for (const int other : openFds()) {
    struct stat st{};
    if (::fstat(other, &st) == 0 && sameFile(st, target)) {
      ++count;
    }
  }
  return count;
}

ChunkShape shapeOf(const VmmPayload& payload) {
  ChunkShape shape;
  for (const auto& chunk : payload.chunks) {
    shape.emplace_back(
        uint64_t{chunk.dev}, uint64_t{chunk.inode}, uint64_t{chunk.size});
  }
  return shape;
}

// Whether a registered handle shares its segment over HIP IPC on the wire.
bool sharesOverIpc(const RegistrationHandle& handle) {
  auto parsed = P2pRegistrationHandle::deserialize(handle.serialize());
  return !parsed.hasError() &&
      std::holds_alternative<IpcPayload>(parsed.value());
}

// A cuMemSetAccess action that grants the access and appends each device it
// is granted to onto devices.
auto recordGrantedDevices(std::vector<int>& devices) {
  return [&devices](
             auto, auto, const CUmemAccessDesc* desc, size_t count) -> Status {
    for (size_t i = 0; i < count; ++i) {
      EXPECT_EQ(desc[i].location.type, CU_MEM_LOCATION_TYPE_DEVICE);
      EXPECT_EQ(desc[i].flags, CU_MEM_ACCESS_FLAGS_PROT_READWRITE);
      devices.push_back(desc[i].location.id);
    }
    return Ok();
  };
}

// Lowers the RLIMIT_NOFILE soft limit until destruction.
class ScopedFdLimit {
 public:
  explicit ScopedFdLimit(rlim_t soft) {
    EXPECT_EQ(::getrlimit(RLIMIT_NOFILE, &saved_), 0);
    rlimit lowered = saved_;
    lowered.rlim_cur = soft;
    EXPECT_EQ(::setrlimit(RLIMIT_NOFILE, &lowered), 0);
  }
  ~ScopedFdLimit() {
    EXPECT_EQ(::setrlimit(RLIMIT_NOFILE, &saved_), 0);
  }
  ScopedFdLimit(const ScopedFdLimit&) = delete;
  ScopedFdLimit& operator=(const ScopedFdLimit&) = delete;

 private:
  rlimit saved_{};
};

// VA-contiguous VMM chunks at a fake address. Each chunk is backed by a memfd
// that stands in for the dmabuf the driver exports for it.
class FakeVmmAllocation {
 public:
  FakeVmmAllocation(uint64_t base, const std::vector<uint64_t>& chunkSizes) {
    for (const uint64_t size : chunkSizes) {
      const int memfd = ::memfd_create("p2p-vmm-chunk", MFD_CLOEXEC);
      EXPECT_GE(memfd, 0);
      chunks_.push_back(Chunk{.base = base, .size = size, .memfd = memfd});
      base += size;
    }
  }
  ~FakeVmmAllocation() {
    for (const auto& chunk : chunks_) {
      ::close(chunk.memfd);
    }
  }
  FakeVmmAllocation(const FakeVmmAllocation&) = delete;
  FakeVmmAllocation& operator=(const FakeVmmAllocation&) = delete;

  int memfd(size_t index) const {
    return chunks_.at(index).memfd;
  }

  static CUmemGenericAllocationHandle exporterHandle(size_t index) {
    return fakeHandle(0x100 + index);
  }

  static CUmemGenericAllocationHandle importedHandle(size_t index) {
    return fakeHandle(0x200 + index);
  }

  // The chunks an export of the whole allocation must describe.
  ChunkShape shape() const {
    ChunkShape shape;
    for (const auto& chunk : chunks_) {
      const struct stat st = statOf(chunk.memfd);
      shape.emplace_back(uint64_t{st.st_dev}, uint64_t{st.st_ino}, chunk.size);
    }
    return shape;
  }

  // What the peer process sends after exporting every chunk.
  VmmPayload payload(uint64_t offset, uint64_t size, int exporterDevice) const {
    auto scope = currentProcessScope();
    EXPECT_FALSE(scope.hasError());
    VmmPayload payload{
        .scope = scope.hasError() ? P2pProcessScope{} : scope.value(),
        .ownerPid = static_cast<int32_t>(::getpid()),
        .exporterDevice = exporterDevice,
        .offset = offset,
        .size = size,
        .chunks = {}};
    for (const auto& chunk : chunks_) {
      const struct stat st = statOf(chunk.memfd);
      payload.chunks.push_back(
          {.fd = ::dup(chunk.memfd),
           .dev = static_cast<uint64_t>(st.st_dev),
           .inode = static_cast<uint64_t>(st.st_ino),
           .size = chunk.size});
    }
    return payload;
  }

  // Answers the classifier, range, retain and export calls of an exporter.
  void serveExport(MockCudaDriverApi& driver) const {
    ON_CALL(driver, cuMemGetAccess(_, _, _))
        .WillByDefault([this](auto*, auto*, CUdeviceptr ptr) -> Status {
          if (find(addressOf(ptr)) == nullptr) {
            return Err(ErrCode::InvalidArgument, "not VMM-mapped");
          }
          return Ok();
        });
    ON_CALL(driver, cuMemGetAddressRange_v2(_, _, _))
        .WillByDefault(
            [this](CUdeviceptr* base, size_t* size, CUdeviceptr ptr) -> Status {
              const Chunk* chunk = find(addressOf(ptr));
              if (chunk == nullptr) {
                return Err(ErrCode::InvalidArgument, "not VMM-mapped");
              }
              *base = toDevicePtr(chunk->base);
              *size = chunk->size;
              return Ok();
            });
    ON_CALL(driver, cuMemRetainAllocationHandle(_, _))
        .WillByDefault(
            [this](CUmemGenericAllocationHandle* handle, void* addr) -> Status {
              const Chunk* chunk = find(reinterpret_cast<uint64_t>(addr));
              if (chunk == nullptr) {
                return Err(ErrCode::InvalidArgument, "not VMM-mapped");
              }
              *handle =
                  exporterHandle(static_cast<size_t>(chunk - chunks_.data()));
              return Ok();
            });
    ON_CALL(driver, cuMemExportToShareableHandle(_, _, _, _))
        .WillByDefault(
            [this](void* out, CUmemGenericAllocationHandle handle, auto, auto)
                -> Status {
              for (size_t i = 0; i < chunks_.size(); ++i) {
                if (exporterHandle(i) == handle) {
                  *static_cast<int*>(out) = ::dup(chunks_[i].memfd);
                  return Ok();
                }
              }
              return Err(ErrCode::InvalidArgument, "unknown handle");
            });
  }

  // Answers imports of fds pulled from payload(), checking that each fd
  // names a chunk of this allocation.
  void serveImport(MockCudaDriverApi& driver) const {
    ON_CALL(driver, importPosixFd(_, _))
        .WillByDefault(
            [this](CUmemGenericAllocationHandle* handle, int fd) -> Status {
              const struct stat pulled = statOf(fd);
              for (size_t i = 0; i < chunks_.size(); ++i) {
                if (sameFile(statOf(chunks_[i].memfd), pulled)) {
                  *handle = importedHandle(i);
                  return Ok();
                }
              }
              return Err(ErrCode::InvalidArgument, "fd names no chunk");
            });
  }

 private:
  struct Chunk {
    uint64_t base{0};
    uint64_t size{0};
    int memfd{-1};
  };

  const Chunk* find(uint64_t addr) const {
    for (const auto& chunk : chunks_) {
      if (addr >= chunk.base && addr - chunk.base < chunk.size) {
        return &chunk;
      }
    }
    return nullptr;
  }

  std::vector<Chunk> chunks_;
};
#endif

CudaApi::IpcMemHandle makePatternHandle() {
  CudaApi::IpcMemHandle h{};
  std::iota(h.begin(), h.end(), uint8_t{7});
  return h;
}

// bind()/getTopology() serialize a device id as a raw host-order int32; mirror
// that here so the wire shape lives in one place in the tests too.
std::vector<uint8_t> makeDeviceIdBytes(int32_t deviceId) {
  std::vector<uint8_t> bytes(sizeof(deviceId));
  std::copy_n(
      reinterpret_cast<const uint8_t*>(&deviceId),
      sizeof(deviceId),
      bytes.data());
  return bytes;
}

class P2pTransportFactoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mock_ = std::make_shared<NiceMock<MockCudaApi>>();
    // CudaDeviceGuard (used by register/import) needs these to succeed.
    ON_CALL(*mock_, getDevice()).WillByDefault(Return(Result<int>(0)));
    ON_CALL(*mock_, setDevice(_)).WillByDefault(Return(Ok()));
#if defined(__HIP_PLATFORM_AMD__)
    driver_ = std::make_shared<NiceMock<MockCudaDriverApi>>();
    ON_CALL(*driver_, isCuMemSupported())
        .WillByDefault(Return(Result<bool>(true)));
    // Plain device memory by default: the VMM classifier misses, as it does
    // for hipMalloc memory.
    ON_CALL(*driver_, cuMemGetAccess(_, _, _))
        .WillByDefault(Return(Err(ErrCode::InvalidArgument, "not VMM")));
#endif
  }

#if defined(__HIP_PLATFORM_AMD__)
  P2pTransportFactory makeFactory(bool enableVmm = true) {
    return P2pTransportFactory(
        /*deviceId=*/0, ebt_.getEventBase(), mock_, driver_, enableVmm);
  }

  // Serves an exporter whose memory is all VMM, reporting chunkAt(addr) as
  // the {base, size} of the chunk holding addr.
  template <typename ChunkAt>
  void serveChunkRanges(ChunkAt chunkAt) {
    ON_CALL(*driver_, cuMemGetAccess(_, _, _)).WillByDefault(Return(Ok()));
    ON_CALL(*driver_, cuMemGetAddressRange_v2(_, _, _))
        .WillByDefault(
            [chunkAt](
                CUdeviceptr* base, size_t* size, CUdeviceptr ptr) -> Status {
              const auto [chunkBase, chunkSize] = chunkAt(addressOf(ptr));
              *base = toDevicePtr(chunkBase);
              *size = chunkSize;
              return Ok();
            });
  }

  // Serves the reservation calls of an import that maps at kImportVa.
  void serveReservation(uint64_t granularity) {
    ON_CALL(*driver_, cuMemGetAllocationGranularity(_, _, _))
        .WillByDefault(DoAll(SetArgPointee<0>(granularity), Return(Ok())));
    ON_CALL(*driver_, cuMemAddressReserve(_, _, _, _, _))
        .WillByDefault(
            DoAll(SetArgPointee<0>(toDevicePtr(kImportVa)), Return(Ok())));
  }
#else
  P2pTransportFactory makeFactory() {
    return P2pTransportFactory(/*deviceId=*/0, ebt_.getEventBase(), mock_);
  }
#endif

  ScopedEventBaseThread ebt_;
  std::shared_ptr<NiceMock<MockCudaApi>> mock_;
#if defined(__HIP_PLATFORM_AMD__)
  std::shared_ptr<NiceMock<MockCudaDriverApi>> driver_;
#endif
};

TEST_F(P2pTransportFactoryTest, SupportedReflectsDeviceCount) {
  // AMD builds also gate on the device arch.
  ON_CALL(*mock_, getDeviceArch(_))
      .WillByDefault(Return(Result<std::string>("gfx950")));
  EXPECT_CALL(*mock_, getDeviceCount()).WillOnce(Return(Result<int>(2)));
  EXPECT_FALSE(P2pTransportFactory::supported(mock_).hasError());

  EXPECT_CALL(*mock_, getDeviceCount()).WillOnce(Return(Result<int>(0)));
  EXPECT_TRUE(P2pTransportFactory::supported(mock_).hasError());
}

TEST_F(P2pTransportFactoryTest, RegisterSegmentRejectsNonVram) {
  auto factory = makeFactory();
  int dummy = 0;
  Segment segment(&dummy, sizeof(dummy), MemoryType::DRAM, /*deviceId=*/0);
  EXPECT_TRUE(factory.registerSegment(segment).hasError());
}

TEST_F(P2pTransportFactoryTest, RegisterSegmentExportsIpcHandle) {
  const auto ipc = makePatternHandle();
  EXPECT_CALL(*mock_, ipcGetMemHandle(_)).WillOnce(Return(ipc));

  auto factory = makeFactory();
  int buf = 0;
  Segment segment(&buf, sizeof(buf), MemoryType::VRAM, /*deviceId=*/0);

  auto handle = factory.registerSegment(segment);
  ASSERT_FALSE(handle.hasError());

  // Inspect the serialized payload to confirm the exported fields.
  auto parsed = P2pRegistrationHandle::deserialize(handle.value()->serialize());
  ASSERT_FALSE(parsed.hasError());
  const auto& p = std::get<P2pRegistrationHandle::IpcPayload>(parsed.value());
  EXPECT_EQ(p.ipcHandle, ipc);
  EXPECT_EQ(p.ownerPid, static_cast<int32_t>(::getpid()));
  EXPECT_EQ(p.base, reinterpret_cast<uint64_t>(&buf));
  EXPECT_EQ(p.offset, 0u);
  EXPECT_EQ(p.size, sizeof(buf));
}

TEST_F(P2pTransportFactoryTest, RegisterSegmentSubAllocationRecordsOffset) {
  const auto ipc = makePatternHandle();
  // A segment that starts 64 bytes into a larger allocation. getMemAddressRange
  // reports the true allocation base (AMD behavior); the IPC handle must be
  // exported at that base and the segment's offset recorded.
  uint8_t alloc[256] = {};
  void* allocBase = alloc;
  void* segPtr = alloc + 64;

  EXPECT_CALL(*mock_, getMemAddressRange(segPtr))
      .WillOnce(Return(
          Result<CudaApi::MemRange>(
              CudaApi::MemRange{allocBase, sizeof(alloc)})));
  EXPECT_CALL(*mock_, ipcGetMemHandle(allocBase)).WillOnce(Return(ipc));

  auto factory = makeFactory();
  Segment segment(segPtr, 128, MemoryType::VRAM, /*deviceId=*/0);

  auto handle = factory.registerSegment(segment);
  ASSERT_FALSE(handle.hasError());

  auto parsed = P2pRegistrationHandle::deserialize(handle.value()->serialize());
  ASSERT_FALSE(parsed.hasError());
  const auto& p = std::get<P2pRegistrationHandle::IpcPayload>(parsed.value());
  EXPECT_EQ(p.ipcHandle, ipc);
  EXPECT_EQ(p.base, reinterpret_cast<uint64_t>(allocBase));
  EXPECT_EQ(p.offset, 64u);
  EXPECT_EQ(p.size, 128u);
}

TEST_F(P2pTransportFactoryTest, ImportSamePidReusesBaseWithoutIpcOpen) {
  int peerBuf = 0;
  P2pRegistrationHandle local(
      makePatternHandle(),
      /*ownerPid=*/static_cast<int32_t>(::getpid()),
      /*base=*/reinterpret_cast<uint64_t>(&peerBuf),
      /*offset=*/64,
      /*size=*/256);
  const auto payload = local.serialize();

  EXPECT_CALL(*mock_, ipcOpenMemHandle(_)).Times(0);

  auto factory = makeFactory();
  auto imported = factory.importSegment(256, payload);
  ASSERT_FALSE(imported.hasError());

  auto* remote =
      static_cast<P2pRemoteRegistrationHandle*>(imported.value().get());
  EXPECT_EQ(remote->mappedPtr(), reinterpret_cast<uint8_t*>(&peerBuf) + 64);
}

TEST_F(P2pTransportFactoryTest, ImportCrossPidOpensIpcHandle) {
  int sentinel = 0;
  void* opened = &sentinel;
  P2pRegistrationHandle local(
      makePatternHandle(),
      /*ownerPid=*/static_cast<int32_t>(::getpid()) + 1, // different process
      /*base=*/0xdead,
      /*offset=*/32,
      /*size=*/128);
  const auto payload = local.serialize();

  EXPECT_CALL(*mock_, ipcOpenMemHandle(_))
      .WillOnce(Return(Result<void*>(opened)));

  auto factory = makeFactory();
  auto imported = factory.importSegment(128, payload);
  ASSERT_FALSE(imported.hasError());

  auto* remote =
      static_cast<P2pRemoteRegistrationHandle*>(imported.value().get());
  EXPECT_EQ(remote->mappedPtr(), static_cast<uint8_t*>(opened) + 32);
}

TEST_F(P2pTransportFactoryTest, CreateTransportRejectsWhenNoPeerAccess) {
  EXPECT_CALL(*mock_, deviceCanAccessPeer(0, 1))
      .WillOnce(Return(Result<bool>(false)));

  auto factory = makeFactory();
  const auto topo = makeDeviceIdBytes(1);

  EXPECT_TRUE(factory.createTransport(topo).hasError());
}

TEST_F(P2pTransportFactoryTest, CreateTransportSucceedsWithPeerAccess) {
  EXPECT_CALL(*mock_, deviceCanAccessPeer(0, 1))
      .WillOnce(Return(Result<bool>(true)));

  auto factory = makeFactory();
  const auto topo = makeDeviceIdBytes(1);

  EXPECT_FALSE(factory.createTransport(topo).hasError());
}

TEST_F(P2pTransportFactoryTest, CreateTransportRejectsNegativePeerDevice) {
  // canConnect must reject a negative peer id up front rather than passing it
  // to deviceCanAccessPeer and surfacing an opaque driver error.
  EXPECT_CALL(*mock_, deviceCanAccessPeer(_, _)).Times(0);

  auto factory = makeFactory();
  const auto topo = makeDeviceIdBytes(-1);

  auto result = factory.createTransport(topo);
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::InvalidArgument);
}

TEST_F(P2pTransportFactoryTest, CreateTransportRejectsWrongSizedTopology) {
  EXPECT_CALL(*mock_, deviceCanAccessPeer(_, _)).Times(0);

  auto factory = makeFactory();
  const std::vector<uint8_t> tooShort(sizeof(int32_t) - 1, 0);

  auto result = factory.createTransport(tooShort);
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::InvalidArgument);
}

TEST(P2pTransportTest, ConnectEnablesPeerAccessForDifferentDevice) {
  auto mock = std::make_shared<NiceMock<MockCudaApi>>();
  ON_CALL(*mock, getDevice()).WillByDefault(Return(Result<int>(0)));
  ON_CALL(*mock, setDevice(_)).WillByDefault(Return(Ok()));
  // BOTH directions: local->peer for put, peer->local for get.
  EXPECT_CALL(*mock, deviceEnablePeerAccess(1)).WillOnce(Return(Ok()));
  EXPECT_CALL(*mock, deviceEnablePeerAccess(0)).WillOnce(Return(Ok()));

  ScopedEventBaseThread ebt;
  P2pTransport transport(/*deviceId=*/0, ebt.getEventBase(), mock);

  const auto info = makeDeviceIdBytes(1);

  transport.bind();
  EXPECT_FALSE(transport.connect(info).hasError());
  EXPECT_EQ(transport.state(), TransportState::Connected);
}

TEST(P2pTransportTest, ConnectRejectsNegativePeerDevice) {
  auto mock = std::make_shared<NiceMock<MockCudaApi>>();
  ON_CALL(*mock, getDevice()).WillByDefault(Return(Result<int>(0)));
  ON_CALL(*mock, setDevice(_)).WillByDefault(Return(Ok()));
  EXPECT_CALL(*mock, deviceEnablePeerAccess(_)).Times(0);

  ScopedEventBaseThread ebt;
  P2pTransport transport(/*deviceId=*/0, ebt.getEventBase(), mock);

  const auto info = makeDeviceIdBytes(-1);

  transport.bind();
  EXPECT_TRUE(transport.connect(info).hasError());
  EXPECT_NE(transport.state(), TransportState::Connected);
}

TEST_F(P2pTransportFactoryTest, ImportRejectsSegmentLengthMismatch) {
  P2pRegistrationHandle local(
      makePatternHandle(),
      /*ownerPid=*/static_cast<int32_t>(::getpid()),
      /*base=*/0x1000,
      /*offset=*/0,
      /*size=*/256);
  const auto payload = local.serialize();

  auto factory = makeFactory();
  // Requested segment length (64) disagrees with the payload's size (256).
  EXPECT_TRUE(factory.importSegment(64, payload).hasError());
}

TEST_F(P2pTransportFactoryTest, ImportRejectsOffsetPlusSizeOverflow) {
  // offset + size must not wrap; both come straight off the wire and drive raw
  // pointer math.
  constexpr uint64_t kSize = 256;
  P2pRegistrationHandle local(
      makePatternHandle(),
      /*ownerPid=*/static_cast<int32_t>(::getpid()),
      /*base=*/0x1000,
      /*offset=*/std::numeric_limits<uint64_t>::max() - kSize + 1,
      /*size=*/kSize);
  const auto payload = local.serialize();

  EXPECT_CALL(*mock_, ipcOpenMemHandle(_)).Times(0);

  auto factory = makeFactory();
  auto result = factory.importSegment(kSize, payload);
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::InvalidArgument);
}

#if defined(__HIP_PLATFORM_AMD__)
TEST_F(P2pTransportFactoryTest, RegisterNonVmmUsesIpcWithoutRetaining) {
  EXPECT_CALL(*driver_, cuMemGetAccess(_, _, _))
      .WillOnce([](auto*, const CUmemLocation* location, auto) -> Status {
        EXPECT_EQ(location->type, CU_MEM_LOCATION_TYPE_DEVICE);
        EXPECT_EQ(location->id, 0);
        return Err(ErrCode::InvalidArgument, "not VMM");
      });
  EXPECT_CALL(*driver_, cuMemRetainAllocationHandle(_, _)).Times(0);
  EXPECT_CALL(*mock_, ipcGetMemHandle(_)).WillOnce(Return(makePatternHandle()));

  auto factory = makeFactory();
  int buf = 0;
  Segment segment(&buf, sizeof(buf), MemoryType::VRAM, /*deviceId=*/0);
  auto handle = factory.registerSegment(segment);
  ASSERT_FALSE(handle.hasError());
  EXPECT_TRUE(sharesOverIpc(*handle.value()));
}

TEST_F(P2pTransportFactoryTest, RegisterWithoutCuMemSupportUsesIpc) {
  ON_CALL(*driver_, isCuMemSupported())
      .WillByDefault(Return(Result<bool>(false)));
  EXPECT_CALL(*driver_, cuMemGetAccess(_, _, _)).Times(0);
  EXPECT_CALL(*mock_, ipcGetMemHandle(_)).WillOnce(Return(makePatternHandle()));

  auto factory = makeFactory();
  int buf = 0;
  Segment segment(&buf, sizeof(buf), MemoryType::VRAM, /*deviceId=*/0);
  EXPECT_FALSE(factory.registerSegment(segment).hasError());
}

TEST_F(P2pTransportFactoryTest, RegisterWithVmmDisabledExportsVmmThroughIpc) {
  const FakeVmmAllocation alloc(kExportVa, {2 * kMiB});
  alloc.serveExport(*driver_);
  EXPECT_CALL(*driver_, isCuMemSupported()).Times(0);
  EXPECT_CALL(*driver_, cuMemGetAccess(_, _, _)).Times(0);
  EXPECT_CALL(*mock_, ipcGetMemHandle(_)).WillOnce(Return(makePatternHandle()));

  auto factory = makeFactory(/*enableVmm=*/false);
  Segment segment(hostPtr(kExportVa), 2 * kMiB, MemoryType::VRAM, 0);
  auto handle = factory.registerSegment(segment);
  ASSERT_FALSE(handle.hasError());
  EXPECT_TRUE(sharesOverIpc(*handle.value()));
}

TEST_F(P2pTransportFactoryTest, RegisterVmmExportsEachChunkAtOffset) {
  // The segment starts inside the first chunk and ends inside the last.
  const FakeVmmAllocation alloc(kExportVa, {2 * kMiB, 4 * kMiB, 2 * kMiB});
  alloc.serveExport(*driver_);
  constexpr uint64_t kOffset = 0x1000;
  constexpr uint64_t kLen = 7 * kMiB;
  // Exporters never release the chunk handles they retain.
  EXPECT_CALL(*driver_, cuMemRelease(_)).Times(0);
  EXPECT_CALL(*mock_, ipcGetMemHandle(_)).Times(0);

  auto factory = makeFactory();
  Segment segment(hostPtr(kExportVa + kOffset), kLen, MemoryType::VRAM, 0);
  auto handle = factory.registerSegment(segment);
  ASSERT_FALSE(handle.hasError());

  auto parsed = P2pRegistrationHandle::deserialize(handle.value()->serialize());
  ASSERT_FALSE(parsed.hasError());
  const auto& p = std::get<VmmPayload>(parsed.value());
  EXPECT_EQ(p.ownerPid, static_cast<int32_t>(::getpid()));
  EXPECT_EQ(p.exporterDevice, 0);
  EXPECT_EQ(p.offset, kOffset);
  EXPECT_EQ(p.size, kLen);
  EXPECT_EQ(shapeOf(p), alloc.shape());
  for (const auto& chunk : p.chunks) {
    const int fd = chunk.fd;
    const int flags = ::fcntl(fd, F_GETFD);
    ASSERT_NE(flags, -1) << "exported fd " << fd << " is closed";
    EXPECT_NE(flags & FD_CLOEXEC, 0) << "fd " << fd;
  }
}

TEST_F(P2pTransportFactoryTest, RegisterVmmExportFailureFallsBackToIpc) {
  const FakeVmmAllocation alloc(kExportVa, {2 * kMiB, 2 * kMiB});
  alloc.serveExport(*driver_);
  // The second export fails after the first chunk's fd was created.
  EXPECT_CALL(*driver_, cuMemExportToShareableHandle(_, _, _, _))
      .Times(AnyNumber());
  EXPECT_CALL(
      *driver_,
      cuMemExportToShareableHandle(
          _, FakeVmmAllocation::exporterHandle(1), _, _))
      .WillOnce(Return(Err(ErrCode::DriverError, "export failed")));
  // Unwinding closes the exported fds but keeps the retained handles.
  EXPECT_CALL(*driver_, cuMemRelease(_)).Times(0);
  EXPECT_CALL(*mock_, ipcGetMemHandle(_)).WillOnce(Return(makePatternHandle()));

  auto factory = makeFactory();
  Segment segment(hostPtr(kExportVa), 4 * kMiB, MemoryType::VRAM, 0);
  auto handle = factory.registerSegment(segment);
  ASSERT_FALSE(handle.hasError());
  EXPECT_TRUE(sharesOverIpc(*handle.value()));
  // Only the allocation's own memfd still names the first chunk.
  EXPECT_EQ(countFdsSharingFile(alloc.memfd(0)), 1);
}

TEST_F(P2pTransportFactoryTest, RegisterReportsBothErrorsWhenVmmAndIpcFail) {
  const FakeVmmAllocation alloc(kExportVa, {2 * kMiB});
  alloc.serveExport(*driver_);
  EXPECT_CALL(*driver_, cuMemExportToShareableHandle(_, _, _, _))
      .WillOnce(Return(Err(ErrCode::DriverError, "vmm export failed")));
  EXPECT_CALL(*mock_, ipcGetMemHandle(_))
      .WillOnce(Return(Err(ErrCode::InvalidArgument, "ipc export failed")));

  auto factory = makeFactory();
  Segment segment(hostPtr(kExportVa), 2 * kMiB, MemoryType::VRAM, 0);
  auto handle = factory.registerSegment(segment);
  ASSERT_TRUE(handle.hasError());
  EXPECT_EQ(handle.error().code(), ErrCode::InvalidArgument);
  EXPECT_THAT(handle.error().message(), HasSubstr("vmm export failed"));
  EXPECT_THAT(handle.error().message(), HasSubstr("ipc export failed"));
}

TEST_F(P2pTransportFactoryTest, RegisterVmmOverFdBudgetFallsBackToIpc) {
  const FakeVmmAllocation alloc(kExportVa, {2 * kMiB, 2 * kMiB});
  alloc.serveExport(*driver_);
  // The walk finds both chunks, then the budget rejects them before any
  // chunk is retained.
  EXPECT_CALL(*driver_, cuMemGetAddressRange_v2(_, _, _)).Times(2);
  EXPECT_CALL(*driver_, cuMemRetainAllocationHandle(_, _)).Times(0);
  EXPECT_CALL(*driver_, cuMemExportToShareableHandle(_, _, _, _)).Times(0);
  EXPECT_CALL(*mock_, ipcGetMemHandle(_)).WillOnce(Return(makePatternHandle()));

  auto factory = makeFactory();
  Segment segment(hostPtr(kExportVa), 4 * kMiB, MemoryType::VRAM, 0);
  // Room for the two chunk fds, not for the headroom kept beside them.
  const ScopedFdLimit limit(openFds().size() + 8);
  auto handle = factory.registerSegment(segment);
  ASSERT_FALSE(handle.hasError());
  EXPECT_TRUE(sharesOverIpc(*handle.value()));
}

TEST_F(
    P2pTransportFactoryTest,
    RegisterPartlyVmmSegmentUsesIpcWithoutRetaining) {
  // Only the first 2 MiB of the 3 MiB segment are VMM-mapped. The tail is
  // hipMalloc-like: its address range resolves, so only the classifier keeps
  // the walk from retaining it.
  const FakeVmmAllocation alloc(kExportVa, {2 * kMiB});
  alloc.serveExport(*driver_);
  const CUdeviceptr tail = toDevicePtr(kExportVa + 2 * kMiB);
  ON_CALL(*driver_, cuMemGetAddressRange_v2(_, _, tail))
      .WillByDefault(DoAll(
          SetArgPointee<0>(tail),
          SetArgPointee<1>(size_t{kMiB}),
          Return(Ok())));
  EXPECT_CALL(*driver_, cuMemGetAccess(_, _, _)).Times(AnyNumber());
  EXPECT_CALL(*driver_, cuMemGetAccess(_, _, tail));
  EXPECT_CALL(*driver_, cuMemRetainAllocationHandle(_, _)).Times(0);
  EXPECT_CALL(*driver_, cuMemExportToShareableHandle(_, _, _, _)).Times(0);
  EXPECT_CALL(*mock_, ipcGetMemHandle(_)).WillOnce(Return(makePatternHandle()));

  auto factory = makeFactory();
  Segment segment(hostPtr(kExportVa), 3 * kMiB, MemoryType::VRAM, 0);
  auto handle = factory.registerSegment(segment);
  ASSERT_FALSE(handle.hasError());
  EXPECT_TRUE(sharesOverIpc(*handle.value()));
}

TEST_F(P2pTransportFactoryTest, RegisterVmmOverChunkCapFallsBackToIpc) {
  // One page-sized chunk more than a payload may carry: the walk stops at
  // the cap without retaining any chunk.
  static constexpr uint64_t kChunk = 4096;
  serveChunkRanges([](uint64_t addr) {
    return std::pair<uint64_t, size_t>{addr / kChunk * kChunk, kChunk};
  });
  EXPECT_CALL(*driver_, cuMemGetAddressRange_v2(_, _, _))
      .Times(kP2pMaxVmmChunks);
  EXPECT_CALL(*driver_, cuMemRetainAllocationHandle(_, _)).Times(0);
  EXPECT_CALL(*mock_, ipcGetMemHandle(_)).WillOnce(Return(makePatternHandle()));

  auto factory = makeFactory();
  Segment segment(
      hostPtr(kExportVa),
      (uint64_t{kP2pMaxVmmChunks} + 1) * kChunk,
      MemoryType::VRAM,
      0);
  auto handle = factory.registerSegment(segment);
  ASSERT_FALSE(handle.hasError());
  EXPECT_TRUE(sharesOverIpc(*handle.value()));
}

TEST_F(P2pTransportFactoryTest, RegisterVmmRejectsRangeMissingItsAddress) {
  // The driver reports a chunk that starts past the address it was asked
  // about.
  serveChunkRanges([](uint64_t addr) {
    return std::pair<uint64_t, size_t>{addr + 4096, 2 * kMiB};
  });
  EXPECT_CALL(*driver_, cuMemGetAddressRange_v2(_, _, _)).Times(1);
  EXPECT_CALL(*driver_, cuMemRetainAllocationHandle(_, _)).Times(0);
  EXPECT_CALL(*mock_, ipcGetMemHandle(_)).WillOnce(Return(makePatternHandle()));

  auto factory = makeFactory();
  Segment segment(hostPtr(kExportVa), 2 * kMiB, MemoryType::VRAM, 0);
  auto handle = factory.registerSegment(segment);
  ASSERT_FALSE(handle.hasError());
  EXPECT_TRUE(sharesOverIpc(*handle.value()));
}

TEST_F(P2pTransportFactoryTest, RegisterVmmRejectsOverlappingChunks) {
  // The chunk holding the second 2 MiB starts 1 MiB before the first chunk
  // ends.
  serveChunkRanges([](uint64_t addr) {
    const uint64_t base =
        addr < kExportVa + 2 * kMiB ? kExportVa : kExportVa + kMiB;
    return std::pair<uint64_t, size_t>{base, 2 * kMiB};
  });
  EXPECT_CALL(*driver_, cuMemGetAddressRange_v2(_, _, _)).Times(2);
  EXPECT_CALL(*driver_, cuMemRetainAllocationHandle(_, _)).Times(0);
  EXPECT_CALL(*mock_, ipcGetMemHandle(_)).WillOnce(Return(makePatternHandle()));

  auto factory = makeFactory();
  Segment segment(hostPtr(kExportVa), 4 * kMiB, MemoryType::VRAM, 0);
  auto handle = factory.registerSegment(segment);
  ASSERT_FALSE(handle.hasError());
  EXPECT_TRUE(sharesOverIpc(*handle.value()));
}

TEST_F(P2pTransportFactoryTest, ImportVmmMapsChunksBackToBack) {
  const FakeVmmAllocation peer(kExportVa, {2 * kMiB, 4 * kMiB});
  peer.serveImport(*driver_);
  serveReservation(/*granularity=*/4 * kMiB);
  constexpr int kPeerDevice = 1;
  constexpr uint64_t kOffset = 0x3000;
  constexpr uint64_t kSize = 5 * kMiB;
  constexpr uint64_t kMapped = 6 * kMiB;
  constexpr uint64_t kReserved = 8 * kMiB; // kMapped at 4 MiB granularity
  P2pRegistrationHandle local(peer.payload(kOffset, kSize, kPeerDevice));

  std::vector<int> grantedDevices;
  {
    InSequence seq;
    EXPECT_CALL(*driver_, cuMemAddressReserve(_, kReserved, _, _, _));
    EXPECT_CALL(
        *driver_,
        cuMemMap(
            toDevicePtr(kImportVa),
            2 * kMiB,
            0,
            FakeVmmAllocation::importedHandle(0),
            0));
    EXPECT_CALL(
        *driver_,
        cuMemMap(
            toDevicePtr(kImportVa + 2 * kMiB),
            4 * kMiB,
            0,
            FakeVmmAllocation::importedHandle(1),
            0));
    EXPECT_CALL(*driver_, cuMemSetAccess(toDevicePtr(kImportVa), kMapped, _, _))
        .WillOnce(recordGrantedDevices(grantedDevices));
    EXPECT_CALL(*driver_, cuMemUnmap(toDevicePtr(kImportVa), 2 * kMiB));
    EXPECT_CALL(
        *driver_, cuMemUnmap(toDevicePtr(kImportVa + 2 * kMiB), 4 * kMiB));
    EXPECT_CALL(*driver_, cuMemAddressFree(toDevicePtr(kImportVa), kReserved));
    EXPECT_CALL(*driver_, cuMemRelease(FakeVmmAllocation::importedHandle(0)));
    EXPECT_CALL(*driver_, cuMemRelease(FakeVmmAllocation::importedHandle(1)));
  }

  auto factory = makeFactory();
  auto imported = factory.importSegment(kSize, local.serialize());
  ASSERT_FALSE(imported.hasError());

  auto* remote =
      static_cast<P2pRemoteRegistrationHandle*>(imported.value().get());
  EXPECT_EQ(
      remote->mappedPtr(), static_cast<uint8_t*>(hostPtr(kImportVa)) + kOffset);
  const std::vector<int> expectedDevices{0, kPeerDevice};
  EXPECT_EQ(grantedDevices, expectedDevices);
  // Pulled fds are closed once imported: only the peer's memfd and the fd
  // its handle owns remain.
  EXPECT_EQ(countFdsSharingFile(peer.memfd(0)), 2);
  EXPECT_EQ(countFdsSharingFile(peer.memfd(1)), 2);
  // Dropping the import tears the mapping down (checked in sequence above).
  imported.value().reset();
}

TEST_F(P2pTransportFactoryTest, ImportVmmRejectsReusedPeerFd) {
  const FakeVmmAllocation peer(kExportVa, {2 * kMiB});
  peer.serveImport(*driver_);
  serveReservation(/*granularity=*/2 * kMiB);
  EXPECT_CALL(*driver_, importPosixFd(_, _)).Times(0);
  EXPECT_CALL(*driver_, cuMemMap(_, _, _, _, _)).Times(0);
  EXPECT_CALL(*driver_, cuMemAddressFree(toDevicePtr(kImportVa), 2 * kMiB))
      .Times(2);
  auto factory = makeFactory();

  // The peer's fd number now names a different file than the one exported:
  // another inode, or the same inode number on another device.
  for (const bool otherDevice : {false, true}) {
    SCOPED_TRACE(otherDevice ? "device differs" : "inode differs");
    auto payload = peer.payload(0, 2 * kMiB, /*exporterDevice=*/0);
    auto& chunk = payload.chunks[0];
    if (otherDevice) {
      chunk.dev += 1;
    } else {
      chunk.inode += 1;
    }
    P2pRegistrationHandle local(std::move(payload));

    auto result = factory.importSegment(2 * kMiB, local.serialize());
    ASSERT_TRUE(result.hasError());
    EXPECT_EQ(result.error().code(), ErrCode::InvalidArgument);
    EXPECT_EQ(countFdsSharingFile(peer.memfd(0)), 2);
  }
}

TEST_F(P2pTransportFactoryTest, ImportVmmUnwindsWhenAMapFails) {
  const FakeVmmAllocation peer(kExportVa, {2 * kMiB, 2 * kMiB});
  peer.serveImport(*driver_);
  serveReservation(/*granularity=*/2 * kMiB);
  P2pRegistrationHandle local(peer.payload(0, 4 * kMiB, /*exporterDevice=*/1));
  const auto handle0 = FakeVmmAllocation::importedHandle(0);
  const auto handle1 = FakeVmmAllocation::importedHandle(1);
  EXPECT_CALL(*driver_, cuMemSetAccess(_, _, _, _)).Times(0);
  {
    InSequence seq;
    EXPECT_CALL(*driver_, cuMemMap(toDevicePtr(kImportVa), _, _, handle0, _));
    EXPECT_CALL(*driver_, cuMemMap(_, _, _, handle1, _))
        .WillOnce(Return(Err(ErrCode::DriverError, "map failed")));
    EXPECT_CALL(*driver_, cuMemUnmap(toDevicePtr(kImportVa), 2 * kMiB));
    EXPECT_CALL(*driver_, cuMemAddressFree(toDevicePtr(kImportVa), 4 * kMiB));
    EXPECT_CALL(*driver_, cuMemRelease(handle0));
    EXPECT_CALL(*driver_, cuMemRelease(handle1));
  }

  auto factory = makeFactory();
  auto result = factory.importSegment(4 * kMiB, local.serialize());
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::DriverError);
}

TEST_F(P2pTransportFactoryTest, ImportVmmUnwindsWhenAnImportFails) {
  const FakeVmmAllocation peer(kExportVa, {2 * kMiB, 2 * kMiB, 2 * kMiB});
  peer.serveImport(*driver_);
  serveReservation(/*granularity=*/2 * kMiB);
  P2pRegistrationHandle local(peer.payload(0, 6 * kMiB, /*exporterDevice=*/1));
  const auto handle0 = FakeVmmAllocation::importedHandle(0);
  EXPECT_CALL(*driver_, cuMemSetAccess(_, _, _, _)).Times(0);
  {
    InSequence seq;
    EXPECT_CALL(*driver_, importPosixFd(_, _));
    EXPECT_CALL(*driver_, cuMemMap(toDevicePtr(kImportVa), _, _, handle0, _));
    EXPECT_CALL(*driver_, importPosixFd(_, _))
        .WillOnce(Return(Err(ErrCode::DriverError, "import failed")));
    EXPECT_CALL(*driver_, cuMemUnmap(toDevicePtr(kImportVa), 2 * kMiB));
    EXPECT_CALL(*driver_, cuMemAddressFree(toDevicePtr(kImportVa), 6 * kMiB));
    EXPECT_CALL(*driver_, cuMemRelease(handle0));
  }

  auto factory = makeFactory();
  auto result = factory.importSegment(6 * kMiB, local.serialize());
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::DriverError);
  // The fd pulled for the failed import is closed and the third chunk's fd is
  // never pulled: only the peer's memfd and the fd its handle owns remain.
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(countFdsSharingFile(peer.memfd(i)), 2) << "chunk " << i;
  }
}

TEST_F(P2pTransportFactoryTest, ImportVmmUnwindsWhenGrantingAccessFails) {
  const FakeVmmAllocation peer(kExportVa, {2 * kMiB, 2 * kMiB});
  peer.serveImport(*driver_);
  serveReservation(/*granularity=*/2 * kMiB);
  P2pRegistrationHandle local(peer.payload(0, 4 * kMiB, /*exporterDevice=*/1));
  const auto handle0 = FakeVmmAllocation::importedHandle(0);
  const auto handle1 = FakeVmmAllocation::importedHandle(1);
  {
    InSequence seq;
    EXPECT_CALL(*driver_, cuMemMap(toDevicePtr(kImportVa), _, _, handle0, _));
    EXPECT_CALL(
        *driver_,
        cuMemMap(toDevicePtr(kImportVa + 2 * kMiB), _, _, handle1, _));
    EXPECT_CALL(
        *driver_, cuMemSetAccess(toDevicePtr(kImportVa), 4 * kMiB, _, _))
        .WillOnce(Return(Err(ErrCode::DriverError, "set access failed")));
    EXPECT_CALL(*driver_, cuMemUnmap(toDevicePtr(kImportVa), 2 * kMiB));
    EXPECT_CALL(
        *driver_, cuMemUnmap(toDevicePtr(kImportVa + 2 * kMiB), 2 * kMiB));
    EXPECT_CALL(*driver_, cuMemAddressFree(toDevicePtr(kImportVa), 4 * kMiB));
    EXPECT_CALL(*driver_, cuMemRelease(handle0));
    EXPECT_CALL(*driver_, cuMemRelease(handle1));
  }

  auto factory = makeFactory();
  auto result = factory.importSegment(4 * kMiB, local.serialize());
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::DriverError);
}

TEST_F(P2pTransportFactoryTest, ImportVmmFromSameDeviceGrantsAccessOnce) {
  const FakeVmmAllocation peer(kExportVa, {2 * kMiB});
  peer.serveImport(*driver_);
  serveReservation(/*granularity=*/2 * kMiB);
  P2pRegistrationHandle local(peer.payload(0, 2 * kMiB, /*exporterDevice=*/0));
  std::vector<int> grantedDevices;
  EXPECT_CALL(*driver_, cuMemSetAccess(_, _, _, _))
      .WillOnce(recordGrantedDevices(grantedDevices));

  auto factory = makeFactory();
  auto imported = factory.importSegment(2 * kMiB, local.serialize());
  ASSERT_FALSE(imported.hasError());
  const std::vector<int> expectedDevices{0};
  EXPECT_EQ(grantedDevices, expectedDevices);
}

TEST_F(P2pTransportFactoryTest, ImportVmmFailsCleanlyWhenOwnerExited) {
  const FakeVmmAllocation peer(kExportVa, {2 * kMiB});
  auto payload = peer.payload(0, 2 * kMiB, /*exporterDevice=*/0);
  // The owner is a child that exited and was reaped, so its pid names no
  // process.
  const pid_t child = ::fork();
  if (child == 0) {
    ::_exit(0);
  }
  ASSERT_GT(child, 0);
  ASSERT_EQ(::waitpid(child, nullptr, 0), child);
  payload.ownerPid = static_cast<int32_t>(child);
  P2pRegistrationHandle local(std::move(payload));
  EXPECT_CALL(*driver_, cuMemGetAllocationGranularity(_, _, _)).Times(0);
  EXPECT_CALL(*driver_, cuMemAddressReserve(_, _, _, _, _)).Times(0);
  EXPECT_CALL(*driver_, importPosixFd(_, _)).Times(0);
  auto factory = makeFactory();
  const size_t fdsBefore = openFds().size();

  auto result = factory.importSegment(2 * kMiB, local.serialize());
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::DriverError);
  EXPECT_EQ(openFds().size(), fdsBefore);
}

TEST_F(P2pTransportFactoryTest, ImportVmmRejectsPayloadFromAnotherScope) {
  const FakeVmmAllocation peer(kExportVa, {2 * kMiB});
  // A matching pid on another host or in another PID namespace names a
  // different process, so the import must not resolve it.
  const std::vector<std::pair<std::string, void (*)(P2pProcessScope&)>> cases{
      {"another boot", [](P2pProcessScope& scope) { scope.bootId[0] ^= 1; }},
      {"another PID namespace",
       [](P2pProcessScope& scope) { scope.pidNamespace ^= 1; }},
  };
  EXPECT_CALL(*driver_, cuMemGetAllocationGranularity(_, _, _)).Times(0);
  EXPECT_CALL(*driver_, cuMemAddressReserve(_, _, _, _, _)).Times(0);
  EXPECT_CALL(*driver_, importPosixFd(_, _)).Times(0);
  auto factory = makeFactory();
  for (const auto& [name, mutate] : cases) {
    SCOPED_TRACE(name);
    auto payload = peer.payload(0, 2 * kMiB, /*exporterDevice=*/0);
    mutate(payload.scope);
    P2pRegistrationHandle local(std::move(payload));
    const size_t fdsBefore = openFds().size();

    auto result = factory.importSegment(2 * kMiB, local.serialize());
    ASSERT_TRUE(result.hasError());
    EXPECT_EQ(result.error().code(), ErrCode::InvalidArgument);
    EXPECT_THAT(result.error().message(), HasSubstr("another host"));
    EXPECT_EQ(openFds().size(), fdsBefore);
  }
}

TEST(P2pVmmTest, CurrentProcessScopeMatchesProc) {
  std::ifstream bootIdFile("/proc/sys/kernel/random/boot_id");
  std::string bootIdText;
  ASSERT_TRUE(std::getline(bootIdFile, bootIdText));
  std::erase(bootIdText, '-');
  struct stat pidNamespace{};
  ASSERT_EQ(::stat("/proc/self/ns/pid", &pidNamespace), 0);

  auto scope = currentProcessScope();
  ASSERT_FALSE(scope.hasError()) << scope.error().message();
  std::string hex;
  for (const uint8_t byte : scope.value().bootId) {
    char digits[3];
    std::snprintf(digits, sizeof(digits), "%02x", byte);
    hex += digits;
  }
  EXPECT_EQ(hex, bootIdText);
  EXPECT_EQ(scope.value().pidNamespace, uint64_t{pidNamespace.st_ino});
}

TEST_F(P2pTransportFactoryTest, ImportVmmRejectsSegmentLengthMismatch) {
  const FakeVmmAllocation peer(kExportVa, {2 * kMiB});
  P2pRegistrationHandle local(peer.payload(0, 2 * kMiB, /*exporterDevice=*/0));
  EXPECT_CALL(*driver_, cuMemAddressReserve(_, _, _, _, _)).Times(0);

  auto factory = makeFactory();
  auto result = factory.importSegment(kMiB, local.serialize());
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::InvalidArgument);
}

TEST_F(P2pTransportFactoryTest, ImportRejectsVmmPayloadWhenVmmDisabled) {
  const FakeVmmAllocation peer(kExportVa, {2 * kMiB});
  P2pRegistrationHandle local(peer.payload(0, 2 * kMiB, /*exporterDevice=*/0));
  EXPECT_CALL(*driver_, cuMemAddressReserve(_, _, _, _, _)).Times(0);
  EXPECT_CALL(*driver_, importPosixFd(_, _)).Times(0);
  EXPECT_CALL(*mock_, ipcOpenMemHandle(_)).Times(0);

  auto factory = makeFactory(/*enableVmm=*/false);
  auto result = factory.importSegment(2 * kMiB, local.serialize());
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::NotImplemented);
}
#else
TEST_F(P2pTransportFactoryTest, ImportRejectsVmmPayload) {
  const int fd = ::open("/dev/null", O_RDONLY | O_CLOEXEC);
  ASSERT_GE(fd, 0);
  P2pRegistrationHandle local(
      P2pRegistrationHandle::VmmPayload{
          .ownerPid = static_cast<int32_t>(::getpid()),
          .exporterDevice = 0,
          .offset = 0,
          .size = 256,
          .chunks = {{.fd = fd, .inode = 1, .size = 4096}}});

  EXPECT_CALL(*mock_, ipcOpenMemHandle(_)).Times(0);

  auto factory = makeFactory();
  auto result = factory.importSegment(256, local.serialize());
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::NotImplemented);
}
#endif

TEST(P2pTransportTest, ConnectRequiresBindFirst) {
  auto mock = std::make_shared<NiceMock<MockCudaApi>>();
  ScopedEventBaseThread ebt;
  P2pTransport transport(/*deviceId=*/0, ebt.getEventBase(), mock);

  const auto info = makeDeviceIdBytes(1);

  // No bind() -> still Disconnected -> connect must be rejected.
  EXPECT_TRUE(transport.connect(info).hasError());
  EXPECT_NE(transport.state(), TransportState::Connected);
}

// Regression test for a cross-process get() fault on AMD: with only
// local->peer enabled, put() works and get() faults once the ranks are
// separate processes.
TEST(P2pTransportTest, ConnectEnablesPeerAccessInBothDirections) {
  auto mock = std::make_shared<NiceMock<MockCudaApi>>();

  int current = -1;
  ON_CALL(*mock, setDevice(_)).WillByDefault([&current](int d) {
    current = d;
    return Ok();
  });
  ON_CALL(*mock, getDevice()).WillByDefault([&current]() {
    return Result<int>(current);
  });

  EXPECT_CALL(*mock, deviceEnablePeerAccess(1)).WillOnce([&current](int) {
    EXPECT_EQ(current, 0) << "local->peer must be enabled with LOCAL current";
    return Ok();
  });
  EXPECT_CALL(*mock, deviceEnablePeerAccess(0)).WillOnce([&current](int) {
    EXPECT_EQ(current, 1) << "peer->local must be enabled with PEER current";
    return Ok();
  });

  ScopedEventBaseThread ebt;
  P2pTransport transport(/*deviceId=*/0, ebt.getEventBase(), mock);
  transport.bind();
  EXPECT_FALSE(transport.connect(makeDeviceIdBytes(1)).hasError());
  EXPECT_EQ(transport.state(), TransportState::Connected);
}

// A failure enabling the REVERSE direction must not leave the transport
// half-connected: state stays Initialized so the caller can retry or fall back.
TEST(P2pTransportTest, ConnectFailsClosedWhenReversePeerAccessFails) {
  auto mock = std::make_shared<NiceMock<MockCudaApi>>();
  ON_CALL(*mock, getDevice()).WillByDefault(Return(Result<int>(0)));
  ON_CALL(*mock, setDevice(_)).WillByDefault(Return(Ok()));
  EXPECT_CALL(*mock, deviceEnablePeerAccess(1)).WillOnce(Return(Ok()));
  EXPECT_CALL(*mock, deviceEnablePeerAccess(0))
      .WillOnce(Return(Err(ErrCode::DriverError, "no reverse peer access")));

  ScopedEventBaseThread ebt;
  P2pTransport transport(/*deviceId=*/0, ebt.getEventBase(), mock);
  transport.bind();
  EXPECT_TRUE(transport.connect(makeDeviceIdBytes(1)).hasError());
  EXPECT_EQ(transport.state(), TransportState::Initialized);
}

TEST(P2pTransportTest, BindDoesNotRegressConnectedState) {
  auto mock = std::make_shared<NiceMock<MockCudaApi>>();
  ON_CALL(*mock, getDevice()).WillByDefault(Return(Result<int>(0)));
  ON_CALL(*mock, setDevice(_)).WillByDefault(Return(Ok()));
  EXPECT_CALL(*mock, deviceEnablePeerAccess(1)).WillOnce(Return(Ok()));
  EXPECT_CALL(*mock, deviceEnablePeerAccess(0)).WillOnce(Return(Ok()));

  ScopedEventBaseThread ebt;
  P2pTransport transport(/*deviceId=*/0, ebt.getEventBase(), mock);
  transport.bind();

  const auto info = makeDeviceIdBytes(1);
  ASSERT_FALSE(transport.connect(info).hasError());
  ASSERT_EQ(transport.state(), TransportState::Connected);

  // bind() again must not regress a Connected transport to Initialized.
  transport.bind();
  EXPECT_EQ(transport.state(), TransportState::Connected);
}

// Fixture exercising the put/get transfer path: memcpy submission plus the
// vendor-typed event-completion seam (eventCreate/eventRecord/eventQuery/
// eventDestroy), mocked directly on MockCudaApi.
class P2pTransportPutGetTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mock_ = std::make_shared<NiceMock<MockCudaApi>>();
    ON_CALL(*mock_, getDevice()).WillByDefault(Return(Result<int>(0)));
    ON_CALL(*mock_, setDevice(_)).WillByDefault(Return(Ok()));
    ON_CALL(*mock_, deviceEnablePeerAccess(_)).WillByDefault(Return(Ok()));
    transport_ = std::make_unique<P2pTransport>(
        /*deviceId=*/0, ebt_.getEventBase(), mock_);

    // The remote handle maps to remoteBuf_; use a distinct fake VA as the
    // segment pointer so a bug that copies to the segment ptr instead of
    // mappedPtr() would fail. A null mapping keeps the destructor trivial.
    remoteSeg_ = SegmentTest::makeRemote(
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        reinterpret_cast<void*>(0xDEAD0000),
        sizeof(remoteBuf_),
        std::make_unique<P2pRemoteRegistrationHandle>(
            remoteBuf_,
            /*offset=*/0,
            sizeof(remoteBuf_),
            /*mapping=*/nullptr));
  }

  void connectTransport() {
    const auto info = makeDeviceIdBytes(1);
    transport_->bind();
    ASSERT_FALSE(transport_->connect(info).hasError());
  }

  ScopedEventBaseThread ebt_;
  std::shared_ptr<NiceMock<MockCudaApi>> mock_;
  std::unique_ptr<P2pTransport> transport_;
  uint8_t localBuf_[128]{};
  uint8_t remoteBuf_[128]{};
  RegisteredSegment localSeg_{SegmentTest::makeRegisteredSegment(
      localBuf_,
      sizeof(localBuf_),
      MemoryType::VRAM,
      /*deviceId=*/0)};
  std::optional<RemoteRegisteredSegment> remoteSeg_;
};

TEST_F(P2pTransportPutGetTest, PutCompletesViaEventPoll) {
  connectTransport();

  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  cudaEvent_t fakeEvent = reinterpret_cast<cudaEvent_t>(0x42);

  // put: dst = remote (mappedPtr()), src = local, default stream (nullptr).
  EXPECT_CALL(
      *mock_,
      memcpyAsync(
          remoteBuf_,
          localBuf_,
          sizeof(localBuf_),
          cudaMemcpyDeviceToDevice,
          nullptr))
      .WillOnce(Return(Ok()));
  EXPECT_CALL(*mock_, eventCreate(_))
      .WillOnce(DoAll(SetArgPointee<0>(fakeEvent), Return(Ok())));
  EXPECT_CALL(*mock_, eventRecord(fakeEvent, _)).WillOnce(Return(Ok()));
  // Not-ready first, then complete: exercises the EventBase re-dispatch poll.
  EXPECT_CALL(*mock_, eventQuery(fakeEvent))
      .WillOnce(Return(Result<bool>(false)))
      .WillOnce(Return(Result<bool>(true)));
  EXPECT_CALL(*mock_, eventDestroy(fakeEvent)).WillOnce(Return(Ok()));

  TransferRequest req{localSeg_.span(), remoteSeg_->span()};
  auto future = transport_->put(std::span(&req, 1));
  EXPECT_TRUE(future.get().hasValue());
}

TEST_F(P2pTransportPutGetTest, GetCompletesViaEvent) {
  connectTransport();

  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  cudaEvent_t fakeEvent = reinterpret_cast<cudaEvent_t>(0x43);

  // get: dst = local, src = remote (mappedPtr()).
  EXPECT_CALL(
      *mock_,
      memcpyAsync(
          localBuf_,
          remoteBuf_,
          sizeof(localBuf_),
          cudaMemcpyDeviceToDevice,
          nullptr))
      .WillOnce(Return(Ok()));
  EXPECT_CALL(*mock_, eventCreate(_))
      .WillOnce(DoAll(SetArgPointee<0>(fakeEvent), Return(Ok())));
  EXPECT_CALL(*mock_, eventRecord(fakeEvent, _)).WillOnce(Return(Ok()));
  EXPECT_CALL(*mock_, eventQuery(fakeEvent))
      .WillOnce(Return(Result<bool>(true)));
  EXPECT_CALL(*mock_, eventDestroy(fakeEvent)).WillOnce(Return(Ok()));

  TransferRequest req{localSeg_.span(), remoteSeg_->span()};
  auto future = transport_->get(std::span(&req, 1));
  EXPECT_TRUE(future.get().hasValue());
}

TEST_F(P2pTransportPutGetTest, PutMemcpyErrorDrainsAndFails) {
  connectTransport();

  EXPECT_CALL(
      *mock_,
      memcpyAsync(
          remoteBuf_,
          localBuf_,
          sizeof(localBuf_),
          cudaMemcpyDeviceToDevice,
          nullptr))
      .WillOnce(Return(Err(ErrCode::DriverError, "memcpy failed")));
  // On failure the stream is drained so buffers are safe to release, and no
  // completion event is created.
  EXPECT_CALL(*mock_, streamSynchronize(nullptr)).WillOnce(Return(Ok()));
  EXPECT_CALL(*mock_, eventCreate(_)).Times(0);

  TransferRequest req{localSeg_.span(), remoteSeg_->span()};
  auto future = transport_->put(std::span(&req, 1));
  auto status = future.get();
  ASSERT_TRUE(status.hasError());
  EXPECT_EQ(status.error().code(), ErrCode::DriverError);
}

TEST_F(P2pTransportPutGetTest, PutQueryEventErrorDrainsAndFails) {
  connectTransport();

  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  cudaEvent_t fakeEvent = reinterpret_cast<cudaEvent_t>(0x44);

  EXPECT_CALL(
      *mock_,
      memcpyAsync(
          remoteBuf_,
          localBuf_,
          sizeof(localBuf_),
          cudaMemcpyDeviceToDevice,
          nullptr))
      .WillOnce(Return(Ok()));
  EXPECT_CALL(*mock_, eventCreate(_))
      .WillOnce(DoAll(SetArgPointee<0>(fakeEvent), Return(Ok())));
  EXPECT_CALL(*mock_, eventRecord(fakeEvent, _)).WillOnce(Return(Ok()));
  // A hard query failure leaves the in-flight copies in an unknown state, so
  // the stream is drained (buffers safe to release) and the event destroyed
  // before the error surfaces on the promise.
  EXPECT_CALL(*mock_, eventQuery(fakeEvent))
      .WillOnce(Return(Err(ErrCode::DriverError, "queryEvent failed")));
  EXPECT_CALL(*mock_, streamSynchronize(nullptr)).WillOnce(Return(Ok()));
  EXPECT_CALL(*mock_, eventDestroy(fakeEvent)).WillOnce(Return(Ok()));

  TransferRequest req{localSeg_.span(), remoteSeg_->span()};
  auto future = transport_->put(std::span(&req, 1));
  auto status = future.get();
  ASSERT_TRUE(status.hasError());
  EXPECT_EQ(status.error().code(), ErrCode::DriverError);
}

// A remote handle with a null base yields a null mappedPtr(). Both put() and
// get() must reject it instead of doing nullptr + offset pointer math and
// handing the result to a device-to-device memcpy.
TEST_F(P2pTransportPutGetTest, RejectsNullMappedPointer) {
  connectTransport();

  auto handle = std::make_unique<P2pRemoteRegistrationHandle>(
      /*mappedBase=*/nullptr,
      /*offset=*/64,
      sizeof(remoteBuf_),
      /*mapping=*/nullptr);
  ASSERT_EQ(handle->mappedPtr(), nullptr);

  auto nullBaseSeg = SegmentTest::makeRemote(
      // NOLINTNEXTLINE(performance-no-int-to-ptr)
      reinterpret_cast<void*>(0xDEAD0000),
      sizeof(remoteBuf_),
      std::move(handle));

  EXPECT_CALL(*mock_, memcpyAsync(_, _, _, _, _)).Times(0);
  EXPECT_CALL(*mock_, eventCreate(_)).Times(0);

  TransferRequest req{localSeg_.span(), nullBaseSeg.span()};

  auto putStatus = transport_->put(std::span(&req, 1)).get();
  ASSERT_TRUE(putStatus.hasError());
  EXPECT_EQ(putStatus.error().code(), ErrCode::InvalidArgument);

  auto getStatus = transport_->get(std::span(&req, 1)).get();
  ASSERT_TRUE(getStatus.hasError());
  EXPECT_EQ(getStatus.error().code(), ErrCode::InvalidArgument);
}

} // namespace
} // namespace uniflow
