// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/transport/p2p/P2pRegistrationHandle.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include "comms/uniflow/drivers/cuda/mock/MockCudaApi.h"

namespace uniflow {
namespace {

using ::testing::_;
using ::testing::HasSubstr;
using ::testing::NiceMock;
using ::testing::Return;

using VmmChunk = P2pRegistrationHandle::VmmChunk;
using VmmPayload = P2pRegistrationHandle::VmmPayload;

CudaApi::IpcMemHandle makePatternHandle() {
  // Distinct nonzero bytes so a round-trip can't pass by accident (e.g. zeros).
  CudaApi::IpcMemHandle h{};
  std::iota(h.begin(), h.end(), uint8_t{1});
  return h;
}

int openNullFd() {
  return ::open("/dev/null", O_RDONLY | O_CLOEXEC);
}

bool isFdOpen(int fd) {
  return ::fcntl(fd, F_GETFD) != -1 || errno != EBADF;
}

template <typename T>
void appendRaw(std::vector<uint8_t>& out, T value) {
  const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
  out.insert(out.end(), bytes, bytes + sizeof(T));
}

struct WireChunk {
  int32_t fd;
  uint64_t dev;
  uint64_t inode;
  uint64_t size;
};

// Distinct nonzero bytes so a round-trip can't pass by accident.
const P2pProcessScope kScope{
    .bootId = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    .pidNamespace = 0xeffffffcULL,
};

// Builds a PosixFd payload field by field from the documented wire layout,
// independently of the implementation's serializer.
std::vector<uint8_t> makeVmmBytes(
    int32_t ownerPid,
    int32_t device,
    uint64_t offset,
    uint64_t size,
    const std::vector<WireChunk>& chunks,
    std::optional<uint32_t> declaredChunks = std::nullopt) {
  std::vector<uint8_t> out;
  appendRaw<uint8_t>(out, 1);
  out.insert(out.end(), kScope.bootId.begin(), kScope.bootId.end());
  appendRaw(out, kScope.pidNamespace);
  appendRaw(out, ownerPid);
  appendRaw(out, device);
  appendRaw(out, offset);
  appendRaw(out, size);
  appendRaw<uint32_t>(
      out, declaredChunks.value_or(static_cast<uint32_t>(chunks.size())));
  for (const auto& chunk : chunks) {
    appendRaw(out, chunk.fd);
    appendRaw(out, chunk.dev);
    appendRaw(out, chunk.inode);
    appendRaw(out, chunk.size);
  }
  return out;
}

using ChunkFields = std::tuple<int32_t, uint64_t, uint64_t, uint64_t>;

std::vector<ChunkFields> chunkFields(const std::vector<VmmChunk>& chunks) {
  std::vector<ChunkFields> fields;
  fields.reserve(chunks.size());
  for (const auto& chunk : chunks) {
    fields.emplace_back(chunk.fd, chunk.dev, chunk.inode, chunk.size);
  }
  return fields;
}

// Older peers exchange exactly these bytes with no mode byte, so a
// mixed-version pair must keep sharing IPC segments in both directions.
TEST(P2pRegistrationHandleTest, IpcUsesLegacyWireLayout) {
  const auto ipc = makePatternHandle();
  std::vector<uint8_t> legacy;
  appendRaw<int32_t>(legacy, 4321);
  appendRaw<uint64_t>(legacy, 0x1000);
  appendRaw<uint64_t>(legacy, 256);
  appendRaw<uint64_t>(legacy, 4096);
  legacy.insert(legacy.end(), ipc.begin(), ipc.end());

  P2pRegistrationHandle handle(
      ipc, /*ownerPid=*/4321, /*base=*/0x1000, /*offset=*/256, /*size=*/4096);
  EXPECT_EQ(handle.sharingMode(), P2pSharingMode::Ipc);
  EXPECT_EQ(handle.serialize(), legacy);

  auto parsed = P2pRegistrationHandle::deserialize(legacy);
  ASSERT_FALSE(parsed.hasError());
  const auto& p = std::get<P2pRegistrationHandle::IpcPayload>(parsed.value());
  EXPECT_EQ(p.ownerPid, 4321);
  EXPECT_EQ(p.base, 0x1000u);
  EXPECT_EQ(p.offset, 256u);
  EXPECT_EQ(p.size, 4096u);
  EXPECT_EQ(p.ipcHandle, ipc);
}

// An IPC payload has no mode byte, so its length decides even when the owner
// pid's low byte equals the PosixFd mode.
TEST(P2pRegistrationHandleTest, IpcLengthWinsOverModeLikeLeadingByte) {
  P2pRegistrationHandle handle(
      makePatternHandle(), /*ownerPid=*/0x101, /*base=*/0, /*offset=*/0, 64);
  const auto bytes = handle.serialize();
  ASSERT_EQ(bytes[0], static_cast<uint8_t>(P2pSharingMode::PosixFd));

  auto parsed = P2pRegistrationHandle::deserialize(bytes);
  ASSERT_FALSE(parsed.hasError());
  EXPECT_EQ(
      std::get<P2pRegistrationHandle::IpcPayload>(parsed.value()).ownerPid,
      0x101);
}

TEST(P2pRegistrationHandleTest, VmmSerializesDocumentedWireLayout) {
  const int fd0 = openNullFd();
  const int fd1 = openNullFd();
  ASSERT_GE(fd0, 0);
  ASSERT_GE(fd1, 0);
  // The segment ends exactly at the end of the last chunk.
  P2pRegistrationHandle handle(
      VmmPayload{
          .scope = kScope,
          .ownerPid = 4321,
          .exporterDevice = 3,
          .offset = 4096,
          .size = 8192,
          .chunks = {
              {.fd = fd0, .dev = 7, .inode = 11, .size = 4096},
              {.fd = fd1, .dev = 8, .inode = 22, .size = 8192}}});
  EXPECT_EQ(handle.sharingMode(), P2pSharingMode::PosixFd);

  const auto expected = makeVmmBytes(
      4321, 3, 4096, 8192, {{fd0, 7, 11, 4096}, {fd1, 8, 22, 8192}});
  EXPECT_EQ(handle.serialize(), expected);

  auto parsed = P2pRegistrationHandle::deserialize(expected);
  ASSERT_FALSE(parsed.hasError());
  const auto& p = std::get<VmmPayload>(parsed.value());
  EXPECT_EQ(p.scope, kScope);
  EXPECT_EQ(p.ownerPid, 4321);
  EXPECT_EQ(p.exporterDevice, 3);
  EXPECT_EQ(p.offset, 4096u);
  EXPECT_EQ(p.size, 8192u);
  const std::vector<ChunkFields> expectedChunks{
      {fd0, 7, 11, 4096}, {fd1, 8, 22, 8192}};
  EXPECT_EQ(chunkFields(p.chunks), expectedChunks);
}

TEST(P2pRegistrationHandleTest, VmmHandleClosesChunkFdsOnDestroy) {
  const int fd0 = openNullFd();
  const int fd1 = openNullFd();
  ASSERT_GE(fd0, 0);
  ASSERT_GE(fd1, 0);
  {
    P2pRegistrationHandle handle(
        VmmPayload{
            .ownerPid = 1,
            .exporterDevice = 0,
            .offset = 0,
            .size = 64,
            .chunks = {
                {.fd = fd0, .dev = 1, .inode = 1, .size = 4096},
                {.fd = fd1, .dev = 1, .inode = 2, .size = 4096}}});
    EXPECT_TRUE(isFdOpen(fd0));
    EXPECT_TRUE(isFdOpen(fd1));
  }
  EXPECT_FALSE(isFdOpen(fd0));
  EXPECT_FALSE(isFdOpen(fd1));
}

TEST(P2pRegistrationHandleTest, DeserializeAcceptsMaxChunkCount) {
  const std::vector<WireChunk> chunks(
      kP2pMaxVmmChunks, WireChunk{.fd = 3, .dev = 1, .inode = 1, .size = 4096});
  EXPECT_FALSE(
      P2pRegistrationHandle::deserialize(makeVmmBytes(1, 0, 0, 4096, chunks))
          .hasError());
}

TEST(P2pRegistrationHandleTest, DeserializeRejectsMalformedPayloads) {
  constexpr uint64_t kMax = std::numeric_limits<uint64_t>::max();
  const WireChunk ok{.fd = 3, .dev = 1, .inode = 1, .size = 4096};
  // Owner pid 1 makes the leading byte look like the PosixFd mode.
  auto ipcShort =
      P2pRegistrationHandle(makePatternHandle(), 1, 0, 0, 0).serialize();
  auto ipcLong = ipcShort;
  ipcShort.pop_back();
  ipcLong.push_back(0);
  auto vmmTruncatedHeader = makeVmmBytes(1, 0, 0, 64, {ok});
  vmmTruncatedHeader.resize(30);
  auto vmmTrailingByte = makeVmmBytes(1, 0, 0, 64, {ok});
  vmmTrailingByte.push_back(0);

  // Each case names the check that must reject it, so none passes by
  // tripping an earlier one. IPC-length mismatches with a PosixFd-like leading
  // byte fail somewhere in VMM parsing.
  struct Case {
    std::string name;
    std::vector<uint8_t> bytes;
    std::string reason;
  };
  const std::vector<Case> cases{
      {"empty", {}, "unrecognized payload"},
      {"unknown mode", {2, 0, 0, 0, 0}, "unrecognized payload"},
      {"IPC one byte short", ipcShort, ""},
      {"IPC one byte long", ipcLong, ""},
      {"VMM truncated header", vmmTruncatedHeader, "truncated"},
      {"VMM zero chunks", makeVmmBytes(1, 0, 0, 64, {}), "chunk count"},
      {"VMM over chunk cap",
       makeVmmBytes(1, 0, 0, 64, {}, kP2pMaxVmmChunks + 1),
       "chunk count"},
      {"VMM declared count exceeds bytes",
       makeVmmBytes(1, 0, 0, 64, {ok}, 2),
       "wrong size"},
      {"VMM trailing byte", vmmTrailingByte, "wrong size"},
      {"VMM invalid owner pid",
       makeVmmBytes(0, 0, 0, 64, {ok}),
       "owner pid or device"},
      {"VMM invalid device",
       makeVmmBytes(1, -1, 0, 64, {ok}),
       "owner pid or device"},
      {"VMM empty segment", makeVmmBytes(1, 0, 0, 0, {ok}), "empty"},
      {"VMM negative fd",
       makeVmmBytes(1, 0, 0, 64, {{-1, 1, 1, 4096}}),
       "invalid fd or size"},
      {"VMM zero-size chunk",
       makeVmmBytes(1, 0, 0, 64, {{3, 1, 1, 0}}),
       "invalid fd or size"},
      {"VMM chunk sizes overflow",
       makeVmmBytes(1, 0, 0, 64, {{3, 1, 1, kMax}, {4, 1, 2, 1}}),
       "overflow"},
      {"VMM offset past chunks",
       makeVmmBytes(1, 0, 4097, 1, {ok}),
       "past its chunks"},
      {"VMM segment past chunks",
       makeVmmBytes(1, 0, 4000, 97, {ok}),
       "past its chunks"},
  };
  for (const auto& c : cases) {
    SCOPED_TRACE(c.name);
    auto parsed = P2pRegistrationHandle::deserialize(c.bytes);
    ASSERT_TRUE(parsed.hasError());
    EXPECT_EQ(parsed.error().code(), ErrCode::InvalidArgument);
    EXPECT_THAT(parsed.error().message(), HasSubstr(c.reason));
  }
}

TEST(P2pRegistrationHandleTest, UsesNvlinkInterconnectTier) {
  P2pRegistrationHandle handle(makePatternHandle(), 1, 0, 0, 0);
  EXPECT_EQ(handle.transportType(), TransportType::NVLink);
}

TEST(P2pRemoteRegistrationHandleTest, MappedPtrAddsOffsetToBase) {
  alignas(64) uint8_t backing[512];
  P2pRemoteRegistrationHandle remote(
      backing, /*offset=*/128, /*size=*/64, /*mapping=*/nullptr);

  EXPECT_EQ(remote.mappedPtr(), backing + 128);
  EXPECT_EQ(remote.mappedSize(), 64u);
}

TEST(P2pRemoteRegistrationHandleTest, NullBaseYieldsNullMappedPtr) {
  P2pRemoteRegistrationHandle remote(
      /*mappedBase=*/nullptr, /*offset=*/128, /*size=*/64, /*mapping=*/nullptr);
  EXPECT_EQ(remote.mappedPtr(), nullptr);
}

TEST(P2pRemoteRegistrationHandleTest, IpcMappingClosesOnDestroy) {
  auto mock = std::make_shared<NiceMock<MockCudaApi>>();
  int sentinel = 0;
  void* base = &sentinel;
  EXPECT_CALL(*mock, ipcCloseMemHandle(base)).WillOnce(Return(Ok()));
  {
    P2pRemoteRegistrationHandle remote(
        base,
        /*offset=*/0,
        /*size=*/64,
        std::make_unique<P2pIpcMapping>(base, mock));
  }
}

TEST(P2pRemoteRegistrationHandleTest, BorrowedPointerClosesNothing) {
  auto mock = std::make_shared<NiceMock<MockCudaApi>>();
  int sentinel = 0;
  EXPECT_CALL(*mock, ipcCloseMemHandle(_)).Times(0);
  {
    P2pRemoteRegistrationHandle remote(
        &sentinel, /*offset=*/0, /*size=*/64, /*mapping=*/nullptr);
  }
}

TEST(P2pRemoteRegistrationHandleTest, IpcCloseFailureIsNotFatal) {
  auto mock = std::make_shared<NiceMock<MockCudaApi>>();
  int sentinel = 0;
  EXPECT_CALL(*mock, ipcCloseMemHandle(_))
      .WillOnce(Return(Err(ErrCode::DriverError, "close failed")));
  EXPECT_NO_THROW({ P2pIpcMapping mapping(&sentinel, mock); });
}

} // namespace
} // namespace uniflow
