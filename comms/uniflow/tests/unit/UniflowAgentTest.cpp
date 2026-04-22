// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/Uniflow.h"

#include <memory>
#include <thread>

#include <gtest/gtest.h>

#include "comms/uniflow/controller/TcpController.h"

namespace uniflow {

// ---------------------------------------------------------------------------
// Mock transport factory
// ---------------------------------------------------------------------------

class MockRegistrationHandle : public RegistrationHandle {
 public:
  explicit MockRegistrationHandle(TransportType type) : type_(type) {}
  TransportType transportType() const noexcept override {
    return type_;
  }
  std::vector<uint8_t> serialize() const override {
    return {static_cast<uint8_t>(type_)};
  }

 private:
  TransportType type_;
};

class MockTransport : public Transport {
 public:
  const std::string& name() const noexcept override {
    return name_;
  }
  TransportType transportType() const noexcept override {
    return TransportType::RDMA;
  }
  TransportState state() const noexcept override {
    return TransportState::Connected;
  }
  TransportInfo bind() override {
    return {0x42};
  }
  Status connect(std::span<const uint8_t>) override {
    return Ok();
  }
  std::future<Status> put(
      std::span<const TransferRequest>,
      const RequestOptions&) override {
    return make_ready_future<Status>(Ok());
  }
  std::future<Status> get(
      std::span<const TransferRequest>,
      const RequestOptions&) override {
    return make_ready_future<Status>(Ok());
  }
  std::future<Status> send(RegisteredSegment::Span, const RequestOptions&)
      override {
    return make_ready_future<Status>(ErrCode::NotImplemented);
  }
  std::future<Status> send(Segment::Span, const RequestOptions&) override {
    return make_ready_future<Status>(ErrCode::NotImplemented);
  }
  std::future<Result<size_t>> recv(
      RegisteredSegment::Span,
      const RequestOptions&) override {
    return make_ready_future<Result<size_t>>(size_t{0});
  }
  std::future<Result<size_t>> recv(Segment::Span, const RequestOptions&)
      override {
    return make_ready_future<Result<size_t>>(size_t{0});
  }
  void shutdown() override {}

 private:
  std::string name_{"mock"};
};

class MultiTransportFactoryTest {
 public:
  static std::shared_ptr<MultiTransportFactory> make(
      std::vector<std::shared_ptr<TransportFactory>> factories) {
    return std::shared_ptr<MultiTransportFactory>(
        new MultiTransportFactory(std::move(factories)));
  }
};

class MockTransportFactory : public TransportFactory {
 public:
  explicit MockTransportFactory(TransportType type) : TransportFactory(type) {}

  Result<std::unique_ptr<RegistrationHandle>> registerSegment(
      Segment&) override {
    return std::make_unique<MockRegistrationHandle>(transportType());
  }
  Result<std::unique_ptr<RemoteRegistrationHandle>> importSegment(
      size_t,
      std::span<const uint8_t>) override {
    return Err(ErrCode::NotImplemented, "mock");
  }
  Result<std::unique_ptr<Transport>> createTransport(
      std::span<const uint8_t>) override {
    return std::make_unique<MockTransport>();
  }
  std::vector<uint8_t> getTopology() override {
    return {static_cast<uint8_t>(transportType())};
  }

 private:
  Status canConnect(std::span<const uint8_t>) override {
    return Ok();
  }
};

// Helper to create UniflowAgent with mock factory via private constructor.
class UniflowAgentTest {
 public:
  static UniflowAgent create(
      const UniflowAgentConfig& config,
      std::shared_ptr<MultiTransportFactory> factory,
      std::unique_ptr<controller::Client> client = nullptr,
      std::unique_ptr<controller::Server> server = nullptr) {
    return UniflowAgent(
        config, std::move(factory), std::move(client), std::move(server));
  }
};

// ---------------------------------------------------------------------------
// getUniqueId tests
// ---------------------------------------------------------------------------

TEST(UniflowAgentTest, GetUniqueIdWithoutServer) {
  auto factory = MultiTransportFactoryTest::make(
      {std::make_shared<MockTransportFactory>(TransportType::RDMA)});

  auto agent =
      UniflowAgentTest::create({.deviceId = 0, .name = "test"}, factory);

  auto id = agent.getUniqueId();
  EXPECT_TRUE(id.hasError());
  EXPECT_EQ(id.error().code(), ErrCode::InvalidArgument);
}

TEST(UniflowAgentTest, GetUniqueIdWithTcpServer) {
  auto factory = MultiTransportFactoryTest::make(
      {std::make_shared<MockTransportFactory>(TransportType::RDMA)});

  auto tcpServer = std::make_unique<controller::TcpServer>("127.0.0.1:0");
  ASSERT_TRUE(tcpServer->init().hasValue());
  int port = tcpServer->getPort();
  ASSERT_GT(port, 0);

  auto agent = UniflowAgentTest::create(
      {.deviceId = 0, .name = "test"}, factory, nullptr, std::move(tcpServer));

  auto id = agent.getUniqueId();
  ASSERT_TRUE(id.hasValue());
  EXPECT_EQ(id.value(), "127.0.0.1:" + std::to_string(port));
}

TEST(UniflowAgentTest, GetUniqueIdWildcardResolvesToLoopback) {
  auto factory = MultiTransportFactoryTest::make(
      {std::make_shared<MockTransportFactory>(TransportType::RDMA)});

  for (const auto& addr : {"0.0.0.0:0", ":::0", "*:0", ":0"}) {
    SCOPED_TRACE(addr);
    auto tcpServer = std::make_unique<controller::TcpServer>(addr);
    auto status = tcpServer->init();
    if (status.hasError()) {
      continue; // Skip if address family not available (e.g. IPv6)
    }
    int port = tcpServer->getPort();
    ASSERT_GT(port, 0);

    auto agent = UniflowAgentTest::create(
        {.deviceId = 0, .name = "test"},
        factory,
        nullptr,
        std::move(tcpServer));

    auto id = agent.getUniqueId();
    ASSERT_TRUE(id.hasValue());
    EXPECT_EQ(id.value(), "127.0.0.1:" + std::to_string(port));
  }
}

// ---------------------------------------------------------------------------
// Full lifecycle tests
// ---------------------------------------------------------------------------

TEST(UniflowAgentTest, FullLifecycleWithTcpController) {
  auto factory = MultiTransportFactoryTest::make(
      {std::make_shared<MockTransportFactory>(TransportType::RDMA)});

  auto tcpServer = std::make_unique<controller::TcpServer>("127.0.0.1:0");
  ASSERT_TRUE(tcpServer->init().hasValue());
  int port = tcpServer->getPort();

  auto serverAgent = UniflowAgentTest::create(
      {.deviceId = 0, .name = "server"},
      factory,
      nullptr,
      std::move(tcpServer));

  auto serverId = serverAgent.getUniqueId();
  ASSERT_TRUE(serverId.hasValue());
  EXPECT_EQ(serverId.value(), "127.0.0.1:" + std::to_string(port));

  uint8_t buf[64];
  Segment seg(buf, sizeof(buf), MemoryType::VRAM, 0);
  auto regResult = serverAgent.registerSegment(seg);
  ASSERT_TRUE(regResult.hasValue());
  EXPECT_EQ(regResult.value().len(), sizeof(buf));

  auto clientAgent = UniflowAgentTest::create(
      {.deviceId = 1, .name = "client"},
      factory,
      std::make_unique<controller::TcpClient>());
  Result<std::unique_ptr<Connection>> serverConnResult =
      ErrCode::NotImplemented;
  Result<std::unique_ptr<Connection>> clientConnResult =
      ErrCode::NotImplemented;

  std::thread serverThread([&]() { serverConnResult = serverAgent.accept(); });
  std::thread clientThread(
      [&]() { clientConnResult = clientAgent.connect(serverId.value()); });

  serverThread.join();
  clientThread.join();

  ASSERT_TRUE(serverConnResult.hasValue())
      << serverConnResult.error().toString();
  ASSERT_TRUE(clientConnResult.hasValue())
      << clientConnResult.error().toString();
  EXPECT_NE(serverConnResult.value(), nullptr);
  EXPECT_NE(clientConnResult.value(), nullptr);
}

TEST(UniflowAgentTest, FullLifecycleWithWildcardAddress) {
  auto factory = MultiTransportFactoryTest::make(
      {std::make_shared<MockTransportFactory>(TransportType::RDMA)});

  auto tcpServer = std::make_unique<controller::TcpServer>("0.0.0.0:0");
  ASSERT_TRUE(tcpServer->init().hasValue());
  int port = tcpServer->getPort();

  auto serverAgent = UniflowAgentTest::create(
      {.deviceId = 0, .name = "server"},
      factory,
      nullptr,
      std::move(tcpServer));

  auto serverId = serverAgent.getUniqueId();
  ASSERT_TRUE(serverId.hasValue());
  EXPECT_EQ(serverId.value(), "127.0.0.1:" + std::to_string(port));

  uint8_t buf[64];
  Segment seg(buf, sizeof(buf), MemoryType::VRAM, 0);
  auto regResult = serverAgent.registerSegment(seg);
  ASSERT_TRUE(regResult.hasValue());
  EXPECT_EQ(regResult.value().len(), sizeof(buf));

  auto clientAgent = UniflowAgentTest::create(
      {.deviceId = 1, .name = "client"},
      factory,
      std::make_unique<controller::TcpClient>());
  Result<std::unique_ptr<Connection>> serverConnResult =
      ErrCode::NotImplemented;
  Result<std::unique_ptr<Connection>> clientConnResult =
      ErrCode::NotImplemented;

  std::thread serverThread([&]() { serverConnResult = serverAgent.accept(); });
  std::thread clientThread(
      [&]() { clientConnResult = clientAgent.connect(serverId.value()); });

  serverThread.join();
  clientThread.join();

  ASSERT_TRUE(serverConnResult.hasValue())
      << serverConnResult.error().toString();
  ASSERT_TRUE(clientConnResult.hasValue())
      << clientConnResult.error().toString();
  EXPECT_NE(serverConnResult.value(), nullptr);
  EXPECT_NE(clientConnResult.value(), nullptr);
}

// ---------------------------------------------------------------------------
// Error propagation
// ---------------------------------------------------------------------------

TEST(UniflowAgentTest, AcceptWithoutServer) {
  auto factory = MultiTransportFactoryTest::make(
      {std::make_shared<MockTransportFactory>(TransportType::RDMA)});

  auto agent =
      UniflowAgentTest::create({.deviceId = 0, .name = "test"}, factory);

  auto result = agent.accept();
  EXPECT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::InvalidArgument);
}

TEST(UniflowAgentTest, ConnectWithoutClient) {
  auto factory = MultiTransportFactoryTest::make(
      {std::make_shared<MockTransportFactory>(TransportType::RDMA)});

  auto agent =
      UniflowAgentTest::create({.deviceId = 0, .name = "test"}, factory);

  auto result = agent.connect("peer");
  EXPECT_TRUE(result.hasError());
  EXPECT_EQ(result.error().code(), ErrCode::InvalidArgument);
}

} // namespace uniflow
