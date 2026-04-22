// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/controller/TcpController.h"

#include <thread>

#include <gtest/gtest.h>

using namespace uniflow;
using namespace uniflow::controller;

struct AddrFamily {
  std::string serverAddr;
  std::string clientHost;
};

class TcpServerClientTest : public ::testing::TestWithParam<AddrFamily> {
 protected:
  std::string clientAddr(int port) const {
    return GetParam().clientHost + ":" + std::to_string(port);
  }
};

TEST_P(TcpServerClientTest, SuccessfulConnection) {
  TcpServer server(GetParam().serverAddr);
  auto status = server.init();
  if (status.hasError()) {
    GTEST_SKIP() << "Not available: " << status.error().toString();
  }
  int port = server.getPort();

  std::unique_ptr<Conn> serverConn;
  std::thread acceptThread([&]() { serverConn = server.accept().get(); });

  TcpClient client;
  auto clientConn = client.connect(clientAddr(port));
  EXPECT_NE(clientConn, nullptr) << "Client failed to connect";

  acceptThread.join();
  EXPECT_NE(serverConn, nullptr) << "Server failed to accept connection";
}

TEST_P(TcpServerClientTest, ServerShutdownWhileClientConnected) {
  auto server = std::make_unique<TcpServer>(GetParam().serverAddr);
  auto status = server->init();
  if (status.hasError()) {
    GTEST_SKIP() << "Not available: " << status.error().toString();
  }
  int port = server->getPort();

  std::unique_ptr<Conn> serverConn;
  std::thread acceptThread([&]() { serverConn = server->accept().get(); });

  TcpClient client;
  auto clientConn = client.connect(clientAddr(port));
  ASSERT_NE(clientConn, nullptr) << "Client failed to connect";

  acceptThread.join();
  ASSERT_NE(serverConn, nullptr) << "Server failed to accept connection";

  server.reset();

  EXPECT_NE(clientConn, nullptr);
}

TEST_P(TcpServerClientTest, ExplicitShutdownUnblocksAccept) {
  TcpServer server(GetParam().serverAddr);
  auto status = server.init();
  if (status.hasError()) {
    GTEST_SKIP() << "Not available: " << status.error().toString();
  }

  std::unique_ptr<Conn> serverConn;
  std::thread acceptThread([&]() { serverConn = server.accept().get(); });

  // Brief pause to let accept() enter its blocking state before shutdown
  // NOLINTNEXTLINE(facebook-hte-BadCall-sleep_for)
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  server.shutdown();

  acceptThread.join();
  EXPECT_EQ(serverConn, nullptr)
      << "accept() should return null after shutdown";
}

INSTANTIATE_TEST_SUITE_P(
    AddrFamilies,
    TcpServerClientTest,
    ::testing::Values(
        AddrFamily{"127.0.0.1:0", "127.0.0.1"},
        AddrFamily{":::0", "::1"}),
    [](const ::testing::TestParamInfo<AddrFamily>& info) {
      return info.param.clientHost == "127.0.0.1" ? "IPv4" : "IPv6";
    });

class TcpServerClientMiscTest : public ::testing::Test {};

TEST_F(TcpServerClientMiscTest, DoubleInitFails) {
  // IPv4
  {
    TcpServer server("127.0.0.1:0");
    ASSERT_TRUE(server.init().hasValue());
    Status status2 = server.init();
    EXPECT_TRUE(status2.hasError()) << "Second init() should fail";
    EXPECT_EQ(status2.error().code(), ErrCode::InvalidArgument);
  }

  // IPv6
  {
    TcpServer server(":::0");
    auto status1 = server.init();
    if (status1.hasError()) {
      GTEST_SKIP() << "IPv6 not available: " << status1.error().toString();
    }
    Status status2 = server.init();
    EXPECT_TRUE(status2.hasError()) << "Second init() should fail (IPv6)";
    EXPECT_EQ(status2.error().code(), ErrCode::InvalidArgument);
  }
}

TEST_F(TcpServerClientMiscTest, AddressParsingErrors) {
  // IPv4 server parsing
  EXPECT_THROW(TcpServer("127.0.0.1:invalid"), std::invalid_argument);
  EXPECT_THROW(
      TcpServer("127.0.0.1:99999999999999999999"), std::invalid_argument);
  EXPECT_THROW(TcpServer("127.0.0.1:70000"), std::invalid_argument);
  EXPECT_THROW(TcpServer("127.0.0.1:"), std::invalid_argument);
  EXPECT_THROW(TcpServer("127.0.0.1"), std::invalid_argument);

  // IPv6 server parsing
  EXPECT_THROW(TcpServer("::1:invalid"), std::invalid_argument);
  EXPECT_THROW(TcpServer(":::"), std::invalid_argument);

  TcpSocketConfig noRetryCfg = TcpSocketConfig{};
  noRetryCfg.connectRetries = 0;
  noRetryCfg.retryTimeout = std::chrono::milliseconds(0);
  TcpClient client(noRetryCfg);
  EXPECT_EQ(client.connect("127.0.0.1:invalid"), nullptr);
  EXPECT_EQ(client.connect("127.0.0.1"), nullptr);
  EXPECT_EQ(client.connect("localhost:8080"), nullptr);
  EXPECT_EQ(client.connect("::1:invalid"), nullptr);
}

TEST_F(TcpServerClientMiscTest, InvalidHostAddress) {
  // Invalid IPv4
  {
    TcpServer server("999.999.999.999:0");
    Status status = server.init();
    EXPECT_TRUE(status.hasError());
    EXPECT_EQ(status.error().code(), ErrCode::InvalidArgument);
  }

  // Invalid IPv6
  {
    TcpServer server("zzzz::1:0");
    Status status = server.init();
    EXPECT_TRUE(status.hasError());
    EXPECT_EQ(status.error().code(), ErrCode::InvalidArgument);
  }
}

TEST_F(TcpServerClientMiscTest, ConnectFailsWhenNoServerListening) {
  TcpSocketConfig noRetryCfg = TcpSocketConfig{};
  noRetryCfg.connectRetries = 0;
  noRetryCfg.retryTimeout = std::chrono::milliseconds(0);
  TcpClient client(noRetryCfg);

  // IPv4
  EXPECT_EQ(client.connect("127.0.0.1:59999"), nullptr);

  // IPv6
  EXPECT_EQ(client.connect("::1:59999"), nullptr);
}

TEST_F(TcpServerClientMiscTest, AcceptReturnsNullBeforeInit) {
  // IPv4
  {
    TcpServer server("127.0.0.1:0");
    EXPECT_EQ(server.accept().get(), nullptr);
  }

  // IPv6
  {
    TcpServer server(":::0");
    EXPECT_EQ(server.accept().get(), nullptr);
  }
}

TEST_F(TcpServerClientMiscTest, SyncAcceptReturnsReadyFuture) {
  TcpServer server("127.0.0.1:0");
  ASSERT_TRUE(server.init().hasValue());

  // accept() on a sync server with no client should eventually return
  // (after timeout/shutdown), and the future should be immediately ready.
  std::future<std::unique_ptr<Conn>> future;
  std::thread acceptThread([&]() { future = server.accept(); });

  // NOLINTNEXTLINE(facebook-hte-BadCall-sleep_for)
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  server.shutdown();
  acceptThread.join();

  // SyncAccept always returns a ready future
  ASSERT_TRUE(future.valid());
  EXPECT_EQ(
      future.wait_for(std::chrono::seconds(0)), std::future_status::ready);
  EXPECT_EQ(future.get(), nullptr);
}

TEST_F(TcpServerClientMiscTest, WildcardAddresses) {
  // IPv4 wildcard
  {
    TcpServer server("0.0.0.0:0");
    Status status = server.init();
    EXPECT_TRUE(status.hasValue())
        << "init() failed with 0.0.0.0: " << status.error().toString();
  }

  {
    TcpServer server("*:0");
    Status status = server.init();
    EXPECT_TRUE(status.hasValue())
        << "init() failed with *: " << status.error().toString();
  }

  // IPv6 wildcard
  {
    TcpServer server(":::0");
    Status status = server.init();
    if (status.hasError()) {
      // IPv6 not available, skip this sub-check
    } else {
      EXPECT_TRUE(status.hasValue())
          << "init() failed with ::: " << status.error().toString();
    }
  }
}

TEST_F(TcpServerClientMiscTest, EmptyHostBindsWildcard) {
  TcpServer server(":0");
  Status status = server.init();
  EXPECT_TRUE(status.hasValue())
      << "init() failed with empty host: " << status.error().toString();
}

class TcpSocketConfigTest : public ::testing::Test {};

TEST_F(TcpSocketConfigTest, DefaultConstructedMatchesProduction) {
  TcpSocketConfig cfg;
  EXPECT_EQ(cfg.connTimeout, std::chrono::seconds{30});
  EXPECT_EQ(cfg.socketBufSize, 1 << 20);
  EXPECT_EQ(cfg.tcpNoDelay, true);
  EXPECT_EQ(cfg.enableKeepalive, true);
  EXPECT_EQ(cfg.keepaliveIdle, std::chrono::seconds{60});
  EXPECT_EQ(cfg.keepaliveInterval, std::chrono::seconds{5});
  EXPECT_EQ(cfg.keepaliveCount, 3);
  EXPECT_EQ(cfg.userTimeout, std::chrono::milliseconds{60000});
  EXPECT_EQ(cfg.acceptRetryCnt, 5);
  EXPECT_EQ(cfg.connectRetries, 10u);
  EXPECT_EQ(cfg.retryTimeout, std::chrono::milliseconds{1000});
}

TEST_F(TcpSocketConfigTest, OsDefaultsIsAllNullopt) {
  auto cfg = TcpSocketConfig::osDefaults();
  EXPECT_FALSE(cfg.connTimeout.has_value());
  EXPECT_FALSE(cfg.socketBufSize.has_value());
  EXPECT_FALSE(cfg.tcpNoDelay.has_value());
  EXPECT_FALSE(cfg.enableKeepalive.has_value());
  EXPECT_FALSE(cfg.keepaliveIdle.has_value());
  EXPECT_FALSE(cfg.keepaliveInterval.has_value());
  EXPECT_FALSE(cfg.keepaliveCount.has_value());
  EXPECT_FALSE(cfg.userTimeout.has_value());
}

TEST_F(TcpSocketConfigTest, ValidateAcceptsDefaults) {
  TcpSocketConfig cfg;
  EXPECT_TRUE(cfg.validate().hasValue());
}

TEST_F(TcpSocketConfigTest, ValidateAcceptsOsDefaults) {
  auto cfg = TcpSocketConfig::osDefaults();
  EXPECT_TRUE(cfg.validate().hasValue());
}

TEST_F(TcpSocketConfigTest, OsDefaultConfigConnectionSucceeds) {
  auto cfg = TcpSocketConfig::osDefaults();

  TcpServer server("127.0.0.1:0", cfg);
  auto status = server.init();
  ASSERT_TRUE(status.hasValue()) << status.error().toString();

  std::unique_ptr<Conn> serverConn;
  std::thread acceptThread([&]() { serverConn = server.accept().get(); });

  TcpClient client(cfg);
  auto clientConn =
      client.connect("127.0.0.1:" + std::to_string(server.getPort()));
  EXPECT_NE(clientConn, nullptr);

  acceptThread.join();
  EXPECT_NE(serverConn, nullptr);
}

TEST_F(TcpSocketConfigTest, ValidateRejectsNegativeTimeout) {
  auto cfg = TcpSocketConfig{};
  cfg.connTimeout = std::chrono::seconds{-1};
  auto status = cfg.validate();
  EXPECT_TRUE(status.hasError());
  EXPECT_EQ(status.error().code(), ErrCode::InvalidArgument);
}

TEST_F(TcpSocketConfigTest, ValidateRejectsZeroBufferSize) {
  auto cfg = TcpSocketConfig{};
  cfg.socketBufSize = 0;
  auto status = cfg.validate();
  EXPECT_TRUE(status.hasError());
  EXPECT_EQ(status.error().code(), ErrCode::InvalidArgument);
}

TEST_F(TcpSocketConfigTest, ValidateRejectsZeroAcceptRetry) {
  auto cfg = TcpSocketConfig{};
  cfg.acceptRetryCnt = 0;
  auto status = cfg.validate();
  EXPECT_TRUE(status.hasError());
  EXPECT_EQ(status.error().code(), ErrCode::InvalidArgument);
}

TEST_F(TcpSocketConfigTest, InvalidConfigThrowsInServerConstructor) {
  auto cfg = TcpSocketConfig{};
  cfg.connTimeout = std::chrono::seconds{-1};
  EXPECT_THROW((TcpServer("127.0.0.1:0", cfg)), std::invalid_argument);
}

TEST_F(TcpSocketConfigTest, InvalidConfigThrowsInClientConstructor) {
  auto cfg = TcpSocketConfig{};
  cfg.socketBufSize = -1;
  EXPECT_THROW(TcpClient{cfg}, std::invalid_argument);
}

TEST_F(TcpSocketConfigTest, CustomConfigConnectionSucceeds) {
  auto cfg = TcpSocketConfig{};
  cfg.connTimeout = std::chrono::seconds{10};
  cfg.socketBufSize = 256 * 1024;

  TcpServer server("127.0.0.1:0", cfg);
  auto status = server.init();
  ASSERT_TRUE(status.hasValue()) << status.error().toString();

  std::unique_ptr<Conn> serverConn;
  std::thread acceptThread([&]() { serverConn = server.accept().get(); });

  TcpClient client(cfg);
  auto clientConn =
      client.connect("127.0.0.1:" + std::to_string(server.getPort()));
  EXPECT_NE(clientConn, nullptr);

  acceptThread.join();
  EXPECT_NE(serverConn, nullptr);
}

TEST_F(TcpSocketConfigTest, ExplicitKeepaliveDisableIsValid) {
  auto cfg = TcpSocketConfig{};
  cfg.enableKeepalive = false;
  EXPECT_TRUE(cfg.validate().hasValue());

  TcpServer server("127.0.0.1:0", cfg);
  auto status = server.init();
  ASSERT_TRUE(status.hasValue()) << status.error().toString();

  std::unique_ptr<Conn> serverConn;
  std::thread acceptThread([&]() { serverConn = server.accept().get(); });

  TcpClient client(cfg);
  auto clientConn =
      client.connect("127.0.0.1:" + std::to_string(server.getPort()));
  EXPECT_NE(clientConn, nullptr);

  acceptThread.join();
  EXPECT_NE(serverConn, nullptr);
}
