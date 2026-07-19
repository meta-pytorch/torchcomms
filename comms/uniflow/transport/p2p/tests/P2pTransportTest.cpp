// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/uniflow/transport/p2p/P2pTransport.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <unistd.h>

#include <memory>
#include <numeric>
#include <optional>

#include "comms/uniflow/drivers/cuda/mock/MockCudaApi.h"
#include "comms/uniflow/executor/ScopedEventBaseThread.h"
#include "comms/uniflow/transport/p2p/P2pRegistrationHandle.h"

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

CudaApi::IpcMemHandle makePatternHandle() {
  CudaApi::IpcMemHandle h{};
  std::iota(h.begin(), h.end(), uint8_t{7});
  return h;
}

class P2pTransportFactoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mock_ = std::make_shared<NiceMock<MockCudaApi>>();
    // CudaDeviceGuard (used by register/import) needs these to succeed.
    ON_CALL(*mock_, getDevice()).WillByDefault(Return(Result<int>(0)));
    ON_CALL(*mock_, setDevice(_)).WillByDefault(Return(Ok()));
  }

  P2pTransportFactory makeFactory() {
    return P2pTransportFactory(/*deviceId=*/0, ebt_.getEventBase(), mock_);
  }

  ScopedEventBaseThread ebt_;
  std::shared_ptr<NiceMock<MockCudaApi>> mock_;
};

TEST_F(P2pTransportFactoryTest, SupportedReflectsDeviceCount) {
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
  EXPECT_EQ(parsed.value().ipcHandle, ipc);
  EXPECT_EQ(parsed.value().ownerPid, static_cast<int32_t>(::getpid()));
  EXPECT_EQ(parsed.value().base, reinterpret_cast<uint64_t>(&buf));
  EXPECT_EQ(parsed.value().offset, 0u);
  EXPECT_EQ(parsed.value().size, sizeof(buf));
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
  EXPECT_EQ(parsed.value().ipcHandle, ipc);
  EXPECT_EQ(parsed.value().base, reinterpret_cast<uint64_t>(allocBase));
  EXPECT_EQ(parsed.value().offset, 64u);
  EXPECT_EQ(parsed.value().size, 128u);
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
  const int32_t peer = 1;
  std::vector<uint8_t> topo(sizeof(peer));
  std::copy_n(
      reinterpret_cast<const uint8_t*>(&peer), sizeof(peer), topo.data());

  EXPECT_TRUE(factory.createTransport(topo).hasError());
}

TEST_F(P2pTransportFactoryTest, CreateTransportSucceedsWithPeerAccess) {
  EXPECT_CALL(*mock_, deviceCanAccessPeer(0, 1))
      .WillOnce(Return(Result<bool>(true)));

  auto factory = makeFactory();
  const int32_t peer = 1;
  std::vector<uint8_t> topo(sizeof(peer));
  std::copy_n(
      reinterpret_cast<const uint8_t*>(&peer), sizeof(peer), topo.data());

  EXPECT_FALSE(factory.createTransport(topo).hasError());
}

TEST(P2pTransportTest, ConnectEnablesPeerAccessForDifferentDevice) {
  auto mock = std::make_shared<NiceMock<MockCudaApi>>();
  ON_CALL(*mock, getDevice()).WillByDefault(Return(Result<int>(0)));
  ON_CALL(*mock, setDevice(_)).WillByDefault(Return(Ok()));
  EXPECT_CALL(*mock, deviceEnablePeerAccess(1)).WillOnce(Return(Ok()));

  ScopedEventBaseThread ebt;
  P2pTransport transport(/*deviceId=*/0, ebt.getEventBase(), mock);

  const int32_t peer = 1;
  std::vector<uint8_t> info(sizeof(peer));
  std::copy_n(
      reinterpret_cast<const uint8_t*>(&peer), sizeof(peer), info.data());

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

  const int32_t peer = -1;
  std::vector<uint8_t> info(sizeof(peer));
  std::copy_n(
      reinterpret_cast<const uint8_t*>(&peer), sizeof(peer), info.data());

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

TEST(P2pTransportTest, ConnectRequiresBindFirst) {
  auto mock = std::make_shared<NiceMock<MockCudaApi>>();
  ScopedEventBaseThread ebt;
  P2pTransport transport(/*deviceId=*/0, ebt.getEventBase(), mock);

  const int32_t peer = 1;
  std::vector<uint8_t> info(sizeof(peer));
  std::copy_n(
      reinterpret_cast<const uint8_t*>(&peer), sizeof(peer), info.data());

  // No bind() -> still Disconnected -> connect must be rejected.
  EXPECT_TRUE(transport.connect(info).hasError());
  EXPECT_NE(transport.state(), TransportState::Connected);
}

TEST(P2pTransportTest, BindDoesNotRegressConnectedState) {
  auto mock = std::make_shared<NiceMock<MockCudaApi>>();
  ON_CALL(*mock, getDevice()).WillByDefault(Return(Result<int>(0)));
  ON_CALL(*mock, setDevice(_)).WillByDefault(Return(Ok()));
  EXPECT_CALL(*mock, deviceEnablePeerAccess(1)).WillOnce(Return(Ok()));

  ScopedEventBaseThread ebt;
  P2pTransport transport(/*deviceId=*/0, ebt.getEventBase(), mock);
  transport.bind();

  const int32_t peer = 1;
  std::vector<uint8_t> info(sizeof(peer));
  std::copy_n(
      reinterpret_cast<const uint8_t*>(&peer), sizeof(peer), info.data());
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
    // mappedPtr() would fail. ownedByIpc=false keeps the destructor trivial.
    remoteSeg_ = SegmentTest::makeRemote(
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        reinterpret_cast<void*>(0xDEAD0000),
        sizeof(remoteBuf_),
        std::make_unique<P2pRemoteRegistrationHandle>(
            remoteBuf_,
            /*offset=*/0,
            sizeof(remoteBuf_),
            /*ownedByIpc=*/false,
            mock_));
  }

  void connectTransport() {
    const int32_t peer = 1;
    std::vector<uint8_t> info(sizeof(peer));
    std::copy_n(
        reinterpret_cast<const uint8_t*>(&peer), sizeof(peer), info.data());
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

} // namespace
} // namespace uniflow
