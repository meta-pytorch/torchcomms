// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/MultiTransport.h"
#include "comms/uniflow/Segment.h"

#include <gtest/gtest.h>

namespace uniflow {

// --- Mock transport factory and handles ---

class MockRegistrationHandle : public RegistrationHandle {
 public:
  MockRegistrationHandle(TransportType type, std::vector<uint8_t> payload)
      : type_(type), payload_(std::move(payload)) {}

  TransportType transportType() const noexcept override {
    return type_;
  }
  std::vector<uint8_t> serialize() const override {
    return payload_;
  }

 private:
  TransportType type_;
  std::vector<uint8_t> payload_;
};

class MockRemoteRegistrationHandle : public RemoteRegistrationHandle {
 public:
  MockRemoteRegistrationHandle(TransportType type, std::vector<uint8_t> payload)
      : type_(type), payload_(std::move(payload)) {}

  TransportType transportType() const noexcept override {
    return type_;
  }
  const std::vector<uint8_t>& payload() const {
    return payload_;
  }

 private:
  TransportType type_;
  std::vector<uint8_t> payload_;
};

class MockTransport : public Transport {
 public:
  explicit MockTransport(TransportType type, TransportInfo bindData = {0x42})
      : name_(std::to_string(type)),
        transportType_(type),
        bindData_(std::move(bindData)) {}

  const std::string& name() const noexcept override {
    return name_;
  }
  TransportType transportType() const noexcept override {
    return transportType_;
  }
  TransportState state() const noexcept override {
    return TransportState::Disconnected;
  }
  TransportInfo bind() override {
    return bindData_;
  }
  Status connect(std::span<const uint8_t> remoteInfo) override {
    lastConnectData_.assign(remoteInfo.begin(), remoteInfo.end());
    if (failConnect_) {
      return Err(ErrCode::ConnectionFailed, "mock connect failure");
    }
    return Ok();
  }

  void setFailConnect(bool fail) {
    failConnect_ = fail;
  }
  const std::vector<uint8_t>& lastConnectData() const {
    return lastConnectData_;
  }
  std::future<Status> put(std::span<TransferRequest> reqs, RequestOptions)
      override {
    putCount += static_cast<int>(reqs.size());
    return make_ready_future<Status>(Ok());
  }
  std::future<Status> get(std::span<TransferRequest> reqs, RequestOptions)
      override {
    getCount += static_cast<int>(reqs.size());
    return make_ready_future<Status>(Ok());
  }

  int putCount{0};
  int getCount{0};
  std::future<Status> send(RegisteredSegment::Span, RequestOptions) override {
    std::promise<Status> p;
    p.set_value(ErrCode::NotImplemented);
    return p.get_future();
  }
  std::future<Result<size_t>> recv(RegisteredSegment::Span, RequestOptions)
      override {
    std::promise<Result<size_t>> p;
    p.set_value(ErrCode::NotImplemented);
    return p.get_future();
  }
  std::future<Status> send(Segment::Span, RequestOptions) override {
    std::promise<Status> p;
    p.set_value(ErrCode::NotImplemented);
    return p.get_future();
  }
  std::future<Result<size_t>> recv(Segment::Span, RequestOptions) override {
    std::promise<Result<size_t>> p;
    p.set_value(ErrCode::NotImplemented);
    return p.get_future();
  }
  void shutdown() override {}

 private:
  std::string name_;
  TransportType transportType_;
  TransportInfo bindData_;
  bool failConnect_{false};
  std::vector<uint8_t> lastConnectData_;
};

class MockTransportFactory : public TransportFactory {
 public:
  explicit MockTransportFactory(
      TransportType type,
      std::vector<uint8_t> handlePayload = {0x01},
      std::vector<uint8_t> topoData = {})
      : TransportFactory(type),
        handlePayload_(std::move(handlePayload)),
        topoData_(std::move(topoData)) {}

  Result<std::unique_ptr<RegistrationHandle>> registerSegment(
      Segment& /*segment*/) override {
    if (failRegister_) {
      return Err(ErrCode::DriverError, "mock register failure");
    }
    return std::make_unique<MockRegistrationHandle>(
        transportType_, handlePayload_);
  }

  Result<std::unique_ptr<RemoteRegistrationHandle>> importSegment(
      size_t /*segmentLength*/,
      std::span<const uint8_t> payload) override {
    if (failImport_) {
      return Err(ErrCode::DriverError, "mock import failure");
    }
    return std::make_unique<MockRemoteRegistrationHandle>(
        transportType_, std::vector<uint8_t>(payload.begin(), payload.end()));
  }

  Result<std::unique_ptr<Transport>> createTransport(
      std::span<const uint8_t> /*peerTopology*/) override {
    if (failCreateTransport_) {
      return Err(ErrCode::DriverError, "mock createTransport failure");
    }
    return std::make_unique<MockTransport>(transportType_);
  }

  std::vector<uint8_t> getTopology() override {
    return topoData_;
  }

  Status canConnect(std::span<const uint8_t> /*peerTopology*/) override {
    return Ok();
  }

  void setFailRegister(bool fail) {
    failRegister_ = fail;
  }
  void setFailImport(bool fail) {
    failImport_ = fail;
  }
  void setFailCreateTransport(bool fail) {
    failCreateTransport_ = fail;
  }

 private:
  std::vector<uint8_t> handlePayload_;
  std::vector<uint8_t> topoData_;
  bool failRegister_{false};
  bool failImport_{false};
  bool failCreateTransport_{false};
};

// SegmentTest is a friend of RegisteredSegment/RemoteRegisteredSegment,
// giving access to private constructors for test setup.
class SegmentTest {
 public:
  static RegisteredSegment makeRegisteredSegment(
      void* buf,
      size_t len,
      MemoryType memType = MemoryType::DRAM,
      int deviceId = -1) {
    return RegisteredSegment(buf, len, memType, deviceId);
  }

  static RemoteRegisteredSegment makeRemoteRegisteredSegment(
      void* buf,
      size_t len,
      MemoryType memType = MemoryType::DRAM,
      int deviceId = -1) {
    return RemoteRegisteredSegment(buf, len, memType, deviceId);
  }

  static const auto& getHandles(const RegisteredSegment& seg) {
    return seg.handles_;
  }

  static auto& getHandles(RegisteredSegment& seg) {
    return seg.handles_;
  }

  static const auto& getRemoteHandles(const RemoteRegisteredSegment& seg) {
    return seg.handles_;
  }

  static auto& getRemoteHandles(RemoteRegisteredSegment& seg) {
    return seg.handles_;
  }
};

// --- MultiTransport bind/connect tests ---

class MultiTransportTest : public ::testing::Test {
 protected:
  // Helper to add a MockTransport and return a raw pointer for inspection.
  MockTransport* addMock(
      MultiTransport& mt,
      TransportType type,
      TransportInfo bindData = {0x42}) {
    auto mock = std::make_unique<MockTransport>(type, std::move(bindData));
    auto* ptr = mock.get();
    mt.addTransport(std::move(mock));
    return ptr;
  }
};

TEST_F(MultiTransportTest, BindConnectRoundTrip) {
  // Sender side: bind
  MultiTransport sender(0);
  addMock(sender, TransportType::RDMA, {0x10, 0x20});
  addMock(sender, TransportType::NVLink, {0x30});

  auto bindResult = sender.bind();
  ASSERT_TRUE(bindResult.hasValue());

  // Receiver side: connect with sender's bind info
  MultiTransport receiver(1);
  auto* mock1 = addMock(receiver, TransportType::RDMA, {0xFF});
  auto* mock2 = addMock(receiver, TransportType::NVLink, {0xFF});

  auto connectResult = receiver.connect(bindResult.value());
  ASSERT_TRUE(connectResult.hasValue());

  // Verify each child transport received the correct subspan
  const std::vector<uint8_t> expected1{0x10, 0x20};
  const std::vector<uint8_t> expected2{0x30};
  EXPECT_EQ(mock1->lastConnectData(), expected1);
  EXPECT_EQ(mock2->lastConnectData(), expected2);
}

TEST_F(MultiTransportTest, ConnectRejectsEmptyInfo) {
  MultiTransport mt(-1);
  addMock(mt, TransportType::RDMA);

  TransportInfo empty;
  auto result = mt.connect(empty);
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::ConnectionFailed);
}

TEST_F(MultiTransportTest, ConnectRejectsTransportCountMismatch) {
  // Bind with 1 transport, connect with 2
  MultiTransport sender(-1);
  addMock(sender, TransportType::RDMA, {0xAA});
  auto bindResult = sender.bind();
  ASSERT_TRUE(bindResult.hasValue());

  MultiTransport receiver(0);
  addMock(receiver, TransportType::RDMA);
  addMock(receiver, TransportType::NVLink);

  auto result = receiver.connect(bindResult.value());
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::ConnectionFailed);
}

TEST_F(MultiTransportTest, ConnectRejectsTruncatedHeader) {
  MultiTransport sender(-1);
  addMock(sender, TransportType::RDMA, {0xAA});
  auto bindResult = sender.bind();
  ASSERT_TRUE(bindResult.hasValue());

  // Truncate after numTransports byte, removing the size field
  auto info = bindResult.value();
  info.resize(1);

  MultiTransport receiver(-1);
  addMock(receiver, TransportType::RDMA);

  auto result = receiver.connect(info);
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::ConnectionFailed);
}

TEST_F(MultiTransportTest, ConnectRejectsTruncatedData) {
  MultiTransport sender(-1);
  addMock(sender, TransportType::RDMA, {0xAA, 0xBB, 0xCC});
  auto bindResult = sender.bind();
  ASSERT_TRUE(bindResult.hasValue());

  // Truncate last byte of data
  auto info = bindResult.value();
  info.resize(info.size() - 1);

  MultiTransport receiver(-1);
  addMock(receiver, TransportType::RDMA);

  auto result = receiver.connect(info);
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::ConnectionFailed);
}

TEST_F(MultiTransportTest, ConnectPropagatesChildTransportError) {
  MultiTransport sender(-1);
  addMock(sender, TransportType::RDMA, {0xAA});
  auto bindResult = sender.bind();
  ASSERT_TRUE(bindResult.hasValue());

  MultiTransport receiver(-1);
  auto* mock = addMock(receiver, TransportType::RDMA);
  mock->setFailConnect(true);

  auto result = receiver.connect(bindResult.value());
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::ConnectionFailed);
}

TEST_F(MultiTransportTest, ConnectSingleTransport) {
  MultiTransport sender(-1);
  addMock(sender, TransportType::RDMA, {0xDE, 0xAD});
  auto bindResult = sender.bind();
  ASSERT_TRUE(bindResult.hasValue());

  MultiTransport receiver(-1);
  auto* mock = addMock(receiver, TransportType::RDMA);
  auto result = receiver.connect(bindResult.value());
  ASSERT_TRUE(result.hasValue());

  const std::vector<uint8_t> expected{0xDE, 0xAD};
  EXPECT_EQ(mock->lastConnectData(), expected);
}

// --- MultiTransportFactory tests ---

class MultiTransportFactoryTest : public ::testing::Test {
 protected:
  std::shared_ptr<MockTransportFactory> makeFactory(
      TransportType type,
      std::vector<uint8_t> payload = {0x01},
      std::vector<uint8_t> topo = {}) {
    return std::make_shared<MockTransportFactory>(
        type, std::move(payload), std::move(topo));
  }

  MultiTransportFactory makeMultiTransportFactory(
      std::vector<std::shared_ptr<TransportFactory>> factories) {
    return MultiTransportFactory(std::move(factories));
  }

  static const std::vector<std::unique_ptr<RemoteRegistrationHandle>>&
  getHandles(const RemoteRegisteredSegment& seg) {
    return seg.handles_;
  }
};

TEST_F(MultiTransportFactoryTest, RegisterSegmentSingleFactory) {
  auto rdma = makeFactory(TransportType::RDMA, {0xAA, 0xBB});
  MultiTransportFactory mtf = makeMultiTransportFactory({rdma});

  uint8_t buf[256];
  Segment seg(buf, sizeof(buf));
  auto result = mtf.registerSegment(seg);

  ASSERT_TRUE(result.hasValue());
  auto& regSeg = result.value();
  EXPECT_EQ(regSeg.data(), buf);
  EXPECT_EQ(regSeg.len(), sizeof(buf));
};

TEST_F(MultiTransportFactoryTest, RegisterSegmentMultipleFactories) {
  auto rdma = makeFactory(TransportType::RDMA, {0xAA});
  auto nvlink = makeFactory(TransportType::NVLink, {0xBB, 0xCC});
  MultiTransportFactory mtf = makeMultiTransportFactory({rdma, nvlink});

  uint8_t buf[128];
  Segment seg(buf, sizeof(buf));
  auto result = mtf.registerSegment(seg);

  ASSERT_TRUE(result.hasValue());
  auto& regSeg = result.value();
  EXPECT_EQ(regSeg.data(), buf);
  EXPECT_EQ(regSeg.len(), sizeof(buf));
}

TEST_F(MultiTransportFactoryTest, RegisterSegmentFailsIfFactoryFails) {
  auto rdma = makeFactory(TransportType::RDMA);
  rdma->setFailRegister(true);
  MultiTransportFactory mtf = makeMultiTransportFactory({rdma});

  uint8_t buf[64];
  Segment seg(buf, sizeof(buf));
  auto result = mtf.registerSegment(seg);

  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::DriverError);
}

TEST_F(MultiTransportFactoryTest, RegisterSegmentFailsOnSecondFactory) {
  auto rdma = makeFactory(TransportType::RDMA);
  auto nvlink = makeFactory(TransportType::NVLink);
  nvlink->setFailRegister(true);
  MultiTransportFactory mtf = makeMultiTransportFactory({rdma, nvlink});

  uint8_t buf[64];
  Segment seg(buf, sizeof(buf));
  auto result = mtf.registerSegment(seg);

  // Partial registration: RDMA succeeds, NVLink fails → one handle survives.
  ASSERT_TRUE(result.hasValue());
  auto& regSeg = result.value();
  EXPECT_EQ(regSeg.data(), buf);
  EXPECT_EQ(regSeg.len(), sizeof(buf));
  auto& handles = SegmentTest::getHandles(regSeg);
  ASSERT_EQ(handles.size(), 1);
  EXPECT_EQ(handles[0]->transportType(), TransportType::RDMA);
}

TEST_F(MultiTransportFactoryTest, RegisterAndImportRoundTrip) {
  auto rdma = makeFactory(TransportType::RDMA, {0x10, 0x20});
  MultiTransportFactory mtf = makeMultiTransportFactory({rdma});

  uint8_t buf[256];
  Segment seg(buf, sizeof(buf));
  auto regResult = mtf.registerSegment(seg);
  ASSERT_TRUE(regResult.hasValue());

  auto exported = regResult.value().exportId();
  auto importResult = mtf.importSegment(exported.value());

  ASSERT_TRUE(importResult.hasValue());
  auto& remote = importResult.value();
  EXPECT_EQ(remote.data(), buf);
  EXPECT_EQ(remote.len(), sizeof(buf));
}

TEST_F(MultiTransportFactoryTest, RegisterAndImportMultipleFactories) {
  auto rdma = makeFactory(TransportType::RDMA, {0xAA});
  auto nvlink = makeFactory(TransportType::NVLink, {0xBB, 0xCC});
  MultiTransportFactory mtf = makeMultiTransportFactory({rdma, nvlink});

  uint8_t buf[128];
  Segment seg(buf, sizeof(buf), MemoryType::VRAM, 2);
  auto regResult = mtf.registerSegment(seg);
  ASSERT_TRUE(regResult.hasValue());

  auto exported = regResult.value().exportId();
  ASSERT_TRUE(exported.hasValue());
  auto importResult = mtf.importSegment(exported.value());

  ASSERT_TRUE(importResult.hasValue());
  auto& remote = importResult.value();
  EXPECT_EQ(remote.data(), buf);
  EXPECT_EQ(remote.len(), sizeof(buf));
  EXPECT_EQ(remote.memType(), MemoryType::VRAM);
  EXPECT_EQ(remote.deviceId(), 2);
}

TEST_F(MultiTransportFactoryTest, ImportSegmentRejectsTruncated) {
  auto rdma = makeFactory(TransportType::RDMA);
  MultiTransportFactory mtf = makeMultiTransportFactory({rdma});

  std::vector<uint8_t> tooShort(10, 0);
  auto result = mtf.importSegment(tooShort);

  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::InvalidArgument);
}

TEST_F(MultiTransportFactoryTest, ImportSegmentRejectsUnknownTransportType) {
  // Register with RDMA factory, but import with a factory that only has NVLink.
  auto rdma = makeFactory(TransportType::RDMA, {0x01});
  MultiTransportFactory registerMtf = makeMultiTransportFactory({rdma});

  uint8_t buf[64];
  Segment seg(buf, sizeof(buf));
  auto regResult = registerMtf.registerSegment(seg);
  ASSERT_TRUE(regResult.hasValue());
  auto exported = regResult.value().exportId();
  ASSERT_TRUE(exported.hasValue());

  // Import with a factory that doesn't have RDMA.
  auto nvlink = makeFactory(TransportType::NVLink);
  MultiTransportFactory importMtf = makeMultiTransportFactory({nvlink});

  auto importResult = importMtf.importSegment(exported.value());
  ASSERT_TRUE(importResult.hasError());
  EXPECT_EQ(importResult.error().code(), ErrCode::InvalidArgument);
}

TEST_F(MultiTransportFactoryTest, ImportSegmentPropagatesFactoryError) {
  auto rdma = makeFactory(TransportType::RDMA, {0x01});
  MultiTransportFactory registerMtf = makeMultiTransportFactory({rdma});

  uint8_t buf[64];
  Segment seg(buf, sizeof(buf));
  auto regResult = registerMtf.registerSegment(seg);
  ASSERT_TRUE(regResult.hasValue());
  auto exported = regResult.value().exportId();
  ASSERT_TRUE(exported.hasValue());

  // Import with same transport type but factory fails.
  auto rdmaImport = makeFactory(TransportType::RDMA);
  rdmaImport->setFailImport(true);
  MultiTransportFactory importMtf = makeMultiTransportFactory({rdmaImport});

  auto importResult = importMtf.importSegment(exported.value());
  ASSERT_TRUE(importResult.hasError());
  EXPECT_EQ(importResult.error().code(), ErrCode::DriverError);
}

TEST_F(MultiTransportFactoryTest, RegisterSegmentZeroFactories) {
  MultiTransportFactory mtf = makeMultiTransportFactory({});

  uint8_t buf[64];
  Segment seg(buf, sizeof(buf));
  auto result = mtf.registerSegment(seg);

  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::DriverError);
}

TEST_F(MultiTransportFactoryTest, ImportSegmentHandlePayloadIntegrity) {
  std::vector<uint8_t> rdmaPayload = {0x10, 0x20, 0x30};
  auto rdma = makeFactory(TransportType::RDMA, rdmaPayload);
  MultiTransportFactory mtf = makeMultiTransportFactory({rdma});

  uint8_t buf[256];
  Segment seg(buf, sizeof(buf));
  auto regResult = mtf.registerSegment(seg);
  ASSERT_TRUE(regResult.hasValue());

  auto exported = regResult.value().exportId();
  ASSERT_TRUE(exported.hasValue());
  auto importResult = mtf.importSegment(exported.value());

  ASSERT_TRUE(importResult.hasValue());
  auto& remote = importResult.value();
  auto& handles = getHandles(remote);
  ASSERT_EQ(handles.size(), 1u);
  auto* rh = dynamic_cast<MockRemoteRegistrationHandle*>(handles[0].get());
  ASSERT_NE(rh, nullptr);
  EXPECT_EQ(rh->payload(), rdmaPayload);
}

// --- getTopology / createTransport tests ---

TEST_F(MultiTransportFactoryTest, GetTopologyAndCreateTransportRoundTrip) {
  auto rdma = makeFactory(TransportType::RDMA, {0x01}, {0xAA, 0xBB});
  auto nvlink = makeFactory(TransportType::NVLink, {0x01}, {0xCC});
  MultiTransportFactory mtf = makeMultiTransportFactory({rdma, nvlink});

  auto topo = mtf.getTopology();
  ASSERT_FALSE(topo.empty());

  // Peer with same factory types should create transport successfully.
  auto rdmaPeer = makeFactory(TransportType::RDMA, {0x01}, {0xAA, 0xBB});
  auto nvlinkPeer = makeFactory(TransportType::NVLink, {0x01}, {0xCC});
  MultiTransportFactory peerMtf =
      makeMultiTransportFactory({rdmaPeer, nvlinkPeer});

  auto result = peerMtf.createTransport(topo);
  ASSERT_TRUE(result.hasValue()) << result.error().message();
  EXPECT_NE(result.value(), nullptr);
}

TEST_F(MultiTransportFactoryTest, GetTopologyAndCreateTransportSingleFactory) {
  auto rdma = makeFactory(TransportType::RDMA, {0x01}, {0x10, 0x20, 0x30});
  MultiTransportFactory mtf = makeMultiTransportFactory({rdma});

  auto topo = mtf.getTopology();
  ASSERT_FALSE(topo.empty());

  auto rdmaPeer = makeFactory(TransportType::RDMA, {0x01}, {0x10, 0x20, 0x30});
  MultiTransportFactory peerMtf = makeMultiTransportFactory({rdmaPeer});

  auto result = peerMtf.createTransport(topo);
  ASSERT_TRUE(result.hasValue()) << result.error().message();
}

TEST_F(MultiTransportFactoryTest, CreateTransportRejectsEmptyTopology) {
  auto rdma = makeFactory(TransportType::RDMA, {0x01}, {0xAA});
  MultiTransportFactory mtf = makeMultiTransportFactory({rdma});

  std::span<const uint8_t> empty;
  auto result = mtf.createTransport(empty);
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::TopologyDisconnect);
}

TEST_F(MultiTransportFactoryTest, CreateTransportRejectsFactoryCountMismatch) {
  auto rdma = makeFactory(TransportType::RDMA, {0x01}, {0xAA});
  MultiTransportFactory singleMtf = makeMultiTransportFactory({rdma});
  auto topo = singleMtf.getTopology();

  // Peer has two factories — count mismatch.
  auto rdmaPeer = makeFactory(TransportType::RDMA, {0x01}, {0xAA});
  auto nvlinkPeer = makeFactory(TransportType::NVLink, {0x01}, {0xBB});
  MultiTransportFactory peerMtf =
      makeMultiTransportFactory({rdmaPeer, nvlinkPeer});

  auto result = peerMtf.createTransport(topo);
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::InvalidArgument);
}

TEST_F(MultiTransportFactoryTest, CreateTransportRejectsTransportTypeMismatch) {
  auto rdma = makeFactory(TransportType::RDMA, {0x01}, {0xAA});
  MultiTransportFactory senderMtf = makeMultiTransportFactory({rdma});
  auto topo = senderMtf.getTopology();

  // Peer has same count but different transport type.
  auto nvlink = makeFactory(TransportType::NVLink, {0x01}, {0xBB});
  MultiTransportFactory peerMtf = makeMultiTransportFactory({nvlink});

  auto result = peerMtf.createTransport(topo);
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::TopologyDisconnect);
}

TEST_F(MultiTransportFactoryTest, CreateTransportFailEmptyTransportError) {
  auto rdma = makeFactory(TransportType::RDMA, {0x01}, {0xAA});
  MultiTransportFactory senderMtf = makeMultiTransportFactory({rdma});
  auto topo = senderMtf.getTopology();

  // Peer has matching type but factory fails createTransport.
  auto rdmaPeer = makeFactory(TransportType::RDMA, {0x01}, {0xAA});
  rdmaPeer->setFailCreateTransport(true);
  MultiTransportFactory peerMtf = makeMultiTransportFactory({rdmaPeer});

  auto result = peerMtf.createTransport(topo);
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::TopologyDisconnect);
}

TEST_F(MultiTransportFactoryTest, GetTopologyAssertsOnEmptyTopoData) {
  auto rdma = makeFactory(TransportType::RDMA, {0x01}, {});
  MultiTransportFactory mtf = makeMultiTransportFactory({rdma});

  EXPECT_DEATH(mtf.getTopology(), "");
}

TEST_F(
    MultiTransportFactoryTest,
    CreateTransportRejectsTruncatedTopologyHeader) {
  auto rdma = makeFactory(TransportType::RDMA, {0x01}, {0xAA, 0xBB});
  MultiTransportFactory senderMtf = makeMultiTransportFactory({rdma});
  auto topo = senderMtf.getTopology();

  // Truncate so the per-transport header is incomplete.
  topo.resize(2);

  auto rdmaPeer = makeFactory(TransportType::RDMA, {0x01}, {0xAA, 0xBB});
  MultiTransportFactory peerMtf = makeMultiTransportFactory({rdmaPeer});

  auto result = peerMtf.createTransport(topo);
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::TopologyDisconnect);
}

TEST_F(MultiTransportFactoryTest, CreateTransportRejectsTruncatedTopologyData) {
  auto rdma = makeFactory(TransportType::RDMA, {0x01}, {0xAA, 0xBB, 0xCC});
  MultiTransportFactory senderMtf = makeMultiTransportFactory({rdma});
  auto topo = senderMtf.getTopology();

  // Truncate last byte of topology data.
  topo.resize(topo.size() - 1);

  auto rdmaPeer = makeFactory(TransportType::RDMA, {0x01}, {0xAA, 0xBB, 0xCC});
  MultiTransportFactory peerMtf = makeMultiTransportFactory({rdmaPeer});

  auto result = peerMtf.createTransport(topo);
  ASSERT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::TopologyDisconnect);
}

// --- Transport routing tests ---

struct TestSegments {
  uint8_t buf[64]{};
  RegisteredSegment regSeg;
  RemoteRegisteredSegment remoteSeg;

  explicit TestSegments(
      MemoryType memType,
      int deviceId = -1,
      const std::vector<TransportType>& handleTypes = {})
      : regSeg(
            SegmentTest::makeRegisteredSegment(
                buf,
                sizeof(buf),
                memType,
                deviceId)),
        remoteSeg(
            SegmentTest::makeRemoteRegisteredSegment(
                buf,
                sizeof(buf),
                memType,
                deviceId)) {
    for (auto type : handleTypes) {
      SegmentTest::getHandles(regSeg).emplace_back(
          std::make_unique<MockRegistrationHandle>(
              type, std::vector<uint8_t>{0x01}));
      SegmentTest::getRemoteHandles(remoteSeg).emplace_back(
          std::make_unique<MockRemoteRegistrationHandle>(
              type, std::vector<uint8_t>{0x01}));
    }
  }
};

TEST_F(MultiTransportTest, VramPutRoutesToNvLink) {
  int deviceId = 0;
  MultiTransport mt(deviceId);
  auto* nvlink = addMock(mt, TransportType::NVLink);
  auto* rdma = addMock(mt, TransportType::RDMA);

  TestSegments local(
      MemoryType::VRAM, deviceId, {TransportType::NVLink, TransportType::RDMA});
  TestSegments remote(
      MemoryType::VRAM, deviceId, {TransportType::NVLink, TransportType::RDMA});
  std::vector<TransferRequest> reqs = {
      {local.regSeg.span(size_t{0}, size_t{32}),
       remote.remoteSeg.span(size_t{0}, size_t{32})}};

  auto future = mt.put(std::move(reqs));
  EXPECT_TRUE(future.get().hasValue());
  EXPECT_EQ(nvlink->putCount, 1);
  EXPECT_EQ(rdma->putCount, 0);
}

TEST_F(MultiTransportTest, DramPutRoutesToRdma) {
  int deviceId = 0;
  MultiTransport mt(deviceId);
  auto* nvlink = addMock(mt, TransportType::NVLink);
  auto* rdma = addMock(mt, TransportType::RDMA);

  TestSegments local(MemoryType::DRAM, deviceId, {TransportType::RDMA});
  TestSegments remote(MemoryType::DRAM, deviceId, {TransportType::RDMA});
  std::vector<TransferRequest> reqs = {
      {local.regSeg.span(size_t{0}, size_t{32}),
       remote.remoteSeg.span(size_t{0}, size_t{32})}};

  auto future = mt.put(std::move(reqs));
  EXPECT_TRUE(future.get().hasValue());
  EXPECT_EQ(rdma->putCount, 1);
  EXPECT_EQ(nvlink->putCount, 0);
}

TEST_F(MultiTransportTest, VramGetRoutesToNvLink) {
  MultiTransport mt(0);
  auto* nvlink = addMock(mt, TransportType::NVLink);
  auto* rdma = addMock(mt, TransportType::RDMA);

  TestSegments local(
      MemoryType::VRAM, 0, {TransportType::NVLink, TransportType::RDMA});
  TestSegments remote(
      MemoryType::VRAM, 0, {TransportType::NVLink, TransportType::RDMA});
  std::vector<TransferRequest> reqs = {
      {local.regSeg.span(size_t{0}, size_t{32}),
       remote.remoteSeg.span(size_t{0}, size_t{32})}};

  auto future = mt.get(std::move(reqs));
  EXPECT_TRUE(future.get().hasValue());
  EXPECT_EQ(nvlink->getCount, 1);
  EXPECT_EQ(rdma->getCount, 0);
}

TEST_F(MultiTransportTest, DramGetRoutesToRdma) {
  MultiTransport mt(0);
  auto* rdma = addMock(mt, TransportType::RDMA);
  auto* nvlink = addMock(mt, TransportType::NVLink);

  TestSegments local(MemoryType::DRAM, -1, {TransportType::RDMA});
  TestSegments remote(MemoryType::DRAM, -1, {TransportType::RDMA});
  std::vector<TransferRequest> reqs = {
      {local.regSeg.span(size_t{0}, size_t{32}),
       remote.remoteSeg.span(size_t{0}, size_t{32})}};

  auto future = mt.get(std::move(reqs));
  EXPECT_TRUE(future.get().hasValue());
  EXPECT_EQ(rdma->getCount, 1);
  EXPECT_EQ(nvlink->getCount, 0);
}

TEST_F(MultiTransportTest, VramFallsBackToRdmaWhenNoNvLink) {
  MultiTransport mt(0);
  auto* rdma = addMock(mt, TransportType::RDMA);
  // No NVLink transport added — segments only have RDMA handles.

  TestSegments local(MemoryType::VRAM, 0, {TransportType::RDMA});
  TestSegments remote(MemoryType::VRAM, 0, {TransportType::RDMA});
  std::vector<TransferRequest> reqs = {
      {local.regSeg.span(size_t{0}, size_t{32}),
       remote.remoteSeg.span(size_t{0}, size_t{32})}};

  auto future = mt.put(std::move(reqs));
  EXPECT_TRUE(future.get().hasValue());
  EXPECT_EQ(rdma->putCount, 1);
}

TEST_F(MultiTransportTest, MultiplePutsBatchToSameTransport) {
  int deviceId = 0;
  MultiTransport mt(deviceId);
  auto* nvlink = addMock(mt, TransportType::NVLink);
  auto* rdma = addMock(mt, TransportType::RDMA);

  TestSegments v1(
      MemoryType::VRAM, deviceId, {TransportType::NVLink, TransportType::RDMA});
  TestSegments v2(
      MemoryType::VRAM, deviceId, {TransportType::NVLink, TransportType::RDMA});
  TestSegments vr1(
      MemoryType::VRAM, deviceId, {TransportType::NVLink, TransportType::RDMA});
  TestSegments vr2(
      MemoryType::VRAM, deviceId, {TransportType::NVLink, TransportType::RDMA});

  std::vector<TransferRequest> reqs = {
      {v1.regSeg.span(size_t{0}, size_t{32}),
       vr1.remoteSeg.span(size_t{0}, size_t{32})},
      {v2.regSeg.span(size_t{0}, size_t{32}),
       vr2.remoteSeg.span(size_t{0}, size_t{32})}};

  auto future = mt.put(std::move(reqs));
  EXPECT_TRUE(future.get().hasValue());
  EXPECT_EQ(nvlink->putCount, 2);
  EXPECT_EQ(rdma->putCount, 0);
}

TEST_F(MultiTransportTest, EmptyPutReturnsError) {
  MultiTransport mt(-1);
  addMock(mt, TransportType::RDMA);

  std::vector<TransferRequest> empty;
  auto future = mt.put(std::move(empty));
  auto status = future.get();
  EXPECT_TRUE(status.hasError());
  EXPECT_EQ(status.error().code(), ErrCode::InvalidArgument);
}

TEST_F(MultiTransportTest, PutWithNoTransportsReturnsError) {
  MultiTransport mt(-1);

  TestSegments local(MemoryType::DRAM);
  TestSegments remote(MemoryType::DRAM);
  std::vector<TransferRequest> reqs = {
      {local.regSeg.span(size_t{0}, size_t{32}),
       remote.remoteSeg.span(size_t{0}, size_t{32})}};

  auto future = mt.put(std::move(reqs));
  auto status = future.get();
  EXPECT_TRUE(status.hasError());
  EXPECT_EQ(status.error().code(), ErrCode::NotConnected);
}

TEST_F(MultiTransportTest, MixedMemoryTypesRejected) {
  MultiTransport mt(0);
  addMock(mt, TransportType::NVLink);
  addMock(mt, TransportType::RDMA);

  TestSegments vram1(
      MemoryType::VRAM, 0, {TransportType::NVLink, TransportType::RDMA});
  TestSegments dram1(MemoryType::DRAM, -1, {TransportType::RDMA});
  TestSegments vramR(
      MemoryType::VRAM, 0, {TransportType::NVLink, TransportType::RDMA});
  TestSegments dramR(MemoryType::DRAM, -1, {TransportType::RDMA});

  // One VRAM request + one DRAM request in the same batch.
  std::vector<TransferRequest> reqs = {
      {vram1.regSeg.span(size_t{0}, size_t{32}),
       vramR.remoteSeg.span(size_t{0}, size_t{32})},
      {dram1.regSeg.span(size_t{0}, size_t{32}),
       dramR.remoteSeg.span(size_t{0}, size_t{32})}};

  auto future = mt.put(std::move(reqs));
  auto status = future.get();
  EXPECT_TRUE(status.hasError());
  EXPECT_EQ(status.error().code(), ErrCode::InvalidArgument);
}

TEST_F(MultiTransportTest, VramWrongDeviceFallsBackToRdma) {
  // MultiTransport is for device 0, but request has deviceId=1.
  MultiTransport mt(0);
  auto* nvlink = addMock(mt, TransportType::NVLink);
  auto* rdma = addMock(mt, TransportType::RDMA);

  TestSegments local(
      MemoryType::VRAM, 1, {TransportType::NVLink, TransportType::RDMA});
  TestSegments remote(
      MemoryType::VRAM, 1, {TransportType::NVLink, TransportType::RDMA});
  std::vector<TransferRequest> reqs = {
      {local.regSeg.span(size_t{0}, size_t{32}),
       remote.remoteSeg.span(size_t{0}, size_t{32})}};

  auto future = mt.put(std::move(reqs));
  EXPECT_TRUE(future.get().hasValue());
  EXPECT_EQ(rdma->putCount, 1);
  EXPECT_EQ(nvlink->putCount, 0);
}

TEST_F(MultiTransportTest, CrossMemoryTypeFallsBackToRdma) {
  // Local VRAM, remote DRAM → should fall back to RDMA.
  MultiTransport mt(0);
  auto* nvlink = addMock(mt, TransportType::NVLink);
  auto* rdma = addMock(mt, TransportType::RDMA);

  TestSegments local(
      MemoryType::VRAM, 0, {TransportType::NVLink, TransportType::RDMA});
  TestSegments remote(MemoryType::DRAM, -1, {TransportType::RDMA});
  std::vector<TransferRequest> reqs = {
      {local.regSeg.span(size_t{0}, size_t{32}),
       remote.remoteSeg.span(size_t{0}, size_t{32})}};

  auto future = mt.put(std::move(reqs));
  EXPECT_TRUE(future.get().hasValue());
  EXPECT_EQ(rdma->putCount, 1);
  EXPECT_EQ(nvlink->putCount, 0);
}

TEST_F(MultiTransportTest, AsymmetricHandlesFallBackToRdma) {
  // Local has NVLink+RDMA, remote only has RDMA → NVLink path not possible.
  MultiTransport mt(0);
  auto* nvlink = addMock(mt, TransportType::NVLink);
  auto* rdma = addMock(mt, TransportType::RDMA);

  TestSegments local(
      MemoryType::VRAM, 0, {TransportType::NVLink, TransportType::RDMA});
  TestSegments remote(MemoryType::VRAM, 0, {TransportType::RDMA});
  std::vector<TransferRequest> reqs = {
      {local.regSeg.span(size_t{0}, size_t{32}),
       remote.remoteSeg.span(size_t{0}, size_t{32})}};

  auto future = mt.put(std::move(reqs));
  EXPECT_TRUE(future.get().hasValue());
  EXPECT_EQ(rdma->putCount, 1);
  EXPECT_EQ(nvlink->putCount, 0);
}

} // namespace uniflow
