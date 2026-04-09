// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/controller/TcpController.h"

#include <thread>

#include <gtest/gtest.h>

using namespace uniflow;
using namespace uniflow::controller;

// ---------------------------------------------------------------------------
// Parameterized fixture: runs server/client lifecycle tests over both
// IPv4 (127.0.0.1) and IPv6 (::1). IPv6 tests are skipped if not available.
// ---------------------------------------------------------------------------

struct AddrFamily {
  std::string serverAddr; // e.g., "127.0.0.1:0" or ":::0"
  std::string clientHost; // e.g., "127.0.0.1" or "::1"
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
  std::thread acceptThread([&]() { serverConn = server.accept(); });

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
  std::thread acceptThread([&]() { serverConn = server->accept(); });

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
  std::thread acceptThread([&]() { serverConn = server.accept(); });

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

// ---------------------------------------------------------------------------
// Non-parameterized tests: address parsing, wildcards, and edge cases that
// are not address-family-specific.
// ---------------------------------------------------------------------------

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

  // Client parsing
  TcpClient client(0, std::chrono::milliseconds(0));
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
  TcpClient client(0, std::chrono::milliseconds(0));

  // IPv4
  EXPECT_EQ(client.connect("127.0.0.1:59999"), nullptr);

  // IPv6
  EXPECT_EQ(client.connect("::1:59999"), nullptr);
}

TEST_F(TcpServerClientMiscTest, AcceptReturnsNullBeforeInit) {
  // IPv4
  {
    TcpServer server("127.0.0.1:0");
    EXPECT_EQ(server.accept(), nullptr);
  }

  // IPv6
  {
    TcpServer server(":::0");
    EXPECT_EQ(server.accept(), nullptr);
  }
}

TEST_F(TcpServerClientMiscTest, WildcardAddresses) {
  // IPv4 wildcard
  {
    TcpServer server("0.0.0.0:0");
    Status status = server.init();
    EXPECT_TRUE(status.hasValue())
        << "init() failed with 0.0.0.0: " << status.error().toString();
  }

  // Asterisk wildcard (maps to IPv6 dual-stack)
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
