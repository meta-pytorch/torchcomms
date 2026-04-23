// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/controller/TcpController.h"

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "comms/uniflow/executor/ScopedEventBaseThread.h"

using namespace uniflow;
using namespace uniflow::controller;

// ---------------------------------------------------------------------------
// Parameterized fixture: async connect tests over both IPv4 and IPv6.
// ---------------------------------------------------------------------------

struct AddrFamily {
  std::string serverAddr; // e.g., "127.0.0.1:0" or ":::0"
  std::string clientHost; // e.g., "127.0.0.1" or "::1"
};

class TcpAsyncConnectTest : public ::testing::TestWithParam<AddrFamily> {
 protected:
  std::string clientAddr(int port) const {
    return GetParam().clientHost + ":" + std::to_string(port);
  }
};

TEST_P(TcpAsyncConnectTest, SingleAsyncConnect) {
  ScopedEventBaseThread evbThread("async-connect");
  TcpServer server(GetParam().serverAddr);
  auto status = server.init();
  if (status.hasError()) {
    GTEST_SKIP() << "Not available: " << status.error().toString();
  }
  int port = server.getPort();

  std::unique_ptr<Conn> serverConn;
  std::thread acceptThread([&]() { serverConn = server.accept().get(); });

  AsyncTcpClient client({}, *evbThread.getEventBase());
  auto conn = client.connect(clientAddr(port)).get();
  EXPECT_NE(conn, nullptr);

  acceptThread.join();
  EXPECT_NE(serverConn, nullptr);
}

TEST_P(TcpAsyncConnectTest, MultipleAsyncConnects) {
  ScopedEventBaseThread evbThread("async-connect");
  TcpServer server(GetParam().serverAddr);
  auto status = server.init();
  if (status.hasError()) {
    GTEST_SKIP() << "Not available: " << status.error().toString();
  }
  int port = server.getPort();

  constexpr int kNumClients = 3;

  std::vector<std::unique_ptr<Conn>> serverConns(kNumClients);
  std::thread acceptThread([&]() {
    for (int i = 0; i < kNumClients; ++i) {
      serverConns[i] = server.accept().get();
    }
  });

  AsyncTcpClient client({}, *evbThread.getEventBase());
  std::vector<std::unique_ptr<Conn>> clientConns;
  for (int i = 0; i < kNumClients; ++i) {
    auto c = client.connect(clientAddr(port)).get();
    ASSERT_NE(c, nullptr) << "Client " << i << " failed to connect";
    clientConns.push_back(std::move(c));
  }

  acceptThread.join();

  for (int i = 0; i < kNumClients; ++i) {
    EXPECT_NE(serverConns[i], nullptr) << "Server conn " << i << " null";
  }
}

TEST_P(TcpAsyncConnectTest, SendRecvAfterAsyncConnect) {
  ScopedEventBaseThread evbThread("async-connect");
  TcpServer server(GetParam().serverAddr);
  auto status = server.init();
  if (status.hasError()) {
    GTEST_SKIP() << "Not available: " << status.error().toString();
  }
  int port = server.getPort();

  std::unique_ptr<Conn> serverConn;
  std::thread acceptThread([&]() { serverConn = server.accept().get(); });

  AsyncTcpClient client({}, *evbThread.getEventBase());
  auto clientConn = client.connect(clientAddr(port)).get();
  ASSERT_NE(clientConn, nullptr);

  acceptThread.join();
  ASSERT_NE(serverConn, nullptr);

  const std::vector<uint8_t> msg = {0x01, 0x02, 0x03, 0x04};
  auto sendResult = clientConn->send(msg);
  ASSERT_TRUE(sendResult.hasValue()) << sendResult.error().toString();

  std::vector<uint8_t> buf;
  auto recvResult = serverConn->recv(buf);
  ASSERT_TRUE(recvResult.hasValue()) << recvResult.error().toString();
  EXPECT_EQ(buf, msg);

  const std::vector<uint8_t> reply = {0x05, 0x06};
  sendResult = serverConn->send(reply);
  ASSERT_TRUE(sendResult.hasValue()) << sendResult.error().toString();

  buf.clear();
  recvResult = clientConn->recv(buf);
  ASSERT_TRUE(recvResult.hasValue()) << recvResult.error().toString();
  EXPECT_EQ(buf, reply);
}

TEST_P(TcpAsyncConnectTest, AsyncConnectFailure) {
  ScopedEventBaseThread evbThread("async-connect");
  TcpSocketConfig noRetryCfg;
  noRetryCfg.connectRetries = 0;
  noRetryCfg.retryTimeout = std::chrono::milliseconds(0);
  AsyncTcpClient client(noRetryCfg, *evbThread.getEventBase());
  auto conn = client.connect(clientAddr(59999)).get();
  EXPECT_EQ(conn, nullptr);
}

// ---------------------------------------------------------------------------
// Non-parameterized tests.
// ---------------------------------------------------------------------------

class TcpAsyncConnectMiscTest : public ::testing::Test {};

TEST_F(TcpAsyncConnectMiscTest, AsyncConnectRejectsNonUniflowServer) {
  ScopedEventBaseThread evbThread("async-connect");

  // Create a raw TCP server that sends garbage instead of the uniflow magic.
  int listenSock = ::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
  ASSERT_GE(listenSock, 0);

  int reuse = 1;
  ::setsockopt(listenSock, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = 0;
  ASSERT_EQ(
      ::bind(listenSock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);
  ASSERT_EQ(::listen(listenSock, 1), 0);

  socklen_t addrLen = sizeof(addr);
  ::getsockname(listenSock, reinterpret_cast<sockaddr*>(&addr), &addrLen);
  int port = ntohs(addr.sin_port);

  std::thread fakeServer([listenSock]() {
    int clientSock = ::accept(listenSock, nullptr, nullptr);
    if (clientSock >= 0) {
      uint32_t garbage = 0xDEADBEEF;
      ::send(clientSock, &garbage, sizeof(garbage), 0);
      // Drain the peer's magic to prevent send-side blocking
      uint32_t buf = 0;
      ::recv(clientSock, &buf, sizeof(buf), 0);
      ::close(clientSock);
    }
  });

  TcpSocketConfig noRetryCfg;
  noRetryCfg.connectRetries = 0;
  noRetryCfg.retryTimeout = std::chrono::milliseconds(0);
  AsyncTcpClient client(noRetryCfg, *evbThread.getEventBase());
  std::string id = "127.0.0.1:" + std::to_string(port);
  auto conn = client.connect(id).get();
  EXPECT_EQ(conn, nullptr);

  fakeServer.join();
  ::close(listenSock);
}

INSTANTIATE_TEST_SUITE_P(
    AddrFamilies,
    TcpAsyncConnectTest,
    ::testing::Values(
        AddrFamily{"127.0.0.1:0", "127.0.0.1"},
        AddrFamily{":::0", "::1"}),
    [](const ::testing::TestParamInfo<AddrFamily>& info) {
      return info.param.clientHost == "127.0.0.1" ? "IPv4" : "IPv6";
    });
