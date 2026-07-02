// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/transport/p2p/P2pRegistrationHandle.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <numeric>
#include <vector>

#include "comms/uniflow/drivers/cuda/mock/MockCudaApi.h"

namespace uniflow {
namespace {

using ::testing::_;
using ::testing::NiceMock;
using ::testing::Return;

CudaApi::IpcMemHandle makePatternHandle() {
  // Distinct nonzero bytes so a round-trip can't pass by accident (e.g. zeros).
  CudaApi::IpcMemHandle h{};
  std::iota(h.begin(), h.end(), uint8_t{1});
  return h;
}

TEST(P2pRegistrationHandleTest, SerializeRoundTripPreservesFields) {
  const auto ipc = makePatternHandle();
  P2pRegistrationHandle handle(
      ipc, /*ownerPid=*/4321, /*base=*/0x1000, /*offset=*/256, /*size=*/4096);

  const auto bytes = handle.serialize();
  ASSERT_EQ(bytes.size(), P2pRegistrationHandle::kSerializedSize);

  auto parsed = P2pRegistrationHandle::deserialize(bytes);
  ASSERT_FALSE(parsed.hasError());
  const auto& p = parsed.value();
  EXPECT_EQ(p.ownerPid, 4321);
  EXPECT_EQ(p.base, 0x1000u);
  EXPECT_EQ(p.offset, 256u);
  EXPECT_EQ(p.size, 4096u);
  EXPECT_EQ(p.ipcHandle, ipc);
}

TEST(P2pRegistrationHandleTest, DeserializeRejectsWrongSize) {
  const std::vector<uint8_t> tooShort(
      P2pRegistrationHandle::kSerializedSize - 1, 0);
  EXPECT_TRUE(P2pRegistrationHandle::deserialize(tooShort).hasError());
}

TEST(P2pRegistrationHandleTest, UsesNvlinkInterconnectTier) {
  P2pRegistrationHandle handle(makePatternHandle(), 1, 0, 0, 0);
  EXPECT_EQ(handle.transportType(), TransportType::NVLink);
}

TEST(P2pRemoteRegistrationHandleTest, MappedPtrAddsOffsetToBase) {
  auto mock = std::make_shared<NiceMock<MockCudaApi>>();
  alignas(64) uint8_t backing[512];
  P2pRemoteRegistrationHandle remote(
      backing, /*offset=*/128, /*size=*/64, /*ownedByIpc=*/false, mock);

  EXPECT_EQ(remote.mappedPtr(), backing + 128);
  EXPECT_EQ(remote.mappedSize(), 64u);
}

TEST(P2pRemoteRegistrationHandleTest, CrossPidImportClosesIpcHandleOnDestroy) {
  auto mock = std::make_shared<NiceMock<MockCudaApi>>();
  int sentinel = 0;
  void* base = &sentinel;
  EXPECT_CALL(*mock, ipcCloseMemHandle(base)).WillOnce(Return(Ok()));
  {
    P2pRemoteRegistrationHandle remote(
        base, /*offset=*/0, /*size=*/64, /*ownedByIpc=*/true, mock);
  }
}

TEST(P2pRemoteRegistrationHandleTest, SamePidImportDoesNotCloseIpcHandle) {
  auto mock = std::make_shared<NiceMock<MockCudaApi>>();
  int sentinel = 0;
  EXPECT_CALL(*mock, ipcCloseMemHandle(_)).Times(0);
  {
    P2pRemoteRegistrationHandle remote(
        &sentinel, /*offset=*/0, /*size=*/64, /*ownedByIpc=*/false, mock);
  }
}

TEST(P2pRemoteRegistrationHandleTest, MoveConstructClosesExactlyOnce) {
  auto mock = std::make_shared<NiceMock<MockCudaApi>>();
  int sentinel = 0;
  void* base = &sentinel;
  // Across both src (moved-from) and dst destruction, close must happen once.
  EXPECT_CALL(*mock, ipcCloseMemHandle(base)).Times(1).WillOnce(Return(Ok()));
  {
    P2pRemoteRegistrationHandle src(
        base, /*offset=*/16, /*size=*/64, /*ownedByIpc=*/true, mock);
    P2pRemoteRegistrationHandle dst(std::move(src));
    EXPECT_EQ(dst.mappedPtr(), static_cast<uint8_t*>(base) + 16);
    // src is moved-from; its destructor must not close the mapping.
  }
}

TEST(P2pRemoteRegistrationHandleTest, MoveAssignClosesPriorAndSourceOnce) {
  auto mock = std::make_shared<NiceMock<MockCudaApi>>();
  int a = 0;
  int b = 0;
  void* baseA = &a; // destination's prior mapping
  void* baseB = &b; // source's mapping
  EXPECT_CALL(*mock, ipcCloseMemHandle(baseA)).Times(1).WillOnce(Return(Ok()));
  EXPECT_CALL(*mock, ipcCloseMemHandle(baseB)).Times(1).WillOnce(Return(Ok()));
  {
    P2pRemoteRegistrationHandle dst(
        baseA, /*offset=*/0, /*size=*/64, /*ownedByIpc=*/true, mock);
    P2pRemoteRegistrationHandle src(
        baseB, /*offset=*/0, /*size=*/64, /*ownedByIpc=*/true, mock);

    // Closes baseA (dst's prior mapping); dst takes over baseB; src is nulled.
    dst = std::move(src);
    EXPECT_EQ(dst.mappedPtr(), baseB);
    // On scope exit: dst closes baseB once; src (moved-from) closes nothing.
  }
}

} // namespace
} // namespace uniflow
