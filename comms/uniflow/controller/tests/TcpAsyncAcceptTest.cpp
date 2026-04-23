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
// Parameterized fixture: async accept tests over both IPv4 and IPv6.
// ---------------------------------------------------------------------------

struct AddrFamily {
  std::string serverAddr;
  std::string clientHost;
};

class TcpAsyncAcceptTest : public ::testing::TestWithParam<AddrFamily> {
 protected:
  std::string clientAddr(int port) const {
    return GetParam().clientHost + ":" + std::to_string(port);
  }
};

TEST_P(TcpAsyncAcceptTest, SingleAsyncAccept) {
  ScopedEventBaseThread evbThread("async-accept");
  AsyncTcpServer server(GetParam().serverAddr, {}, *evbThread.getEventBase());
  auto status = server.init();
  if (status.hasError()) {
    GTEST_SKIP() << "Not available: " << status.error().toString();
  }
  int port = server.getPort();

  auto future = server.accept();

  TcpClient client;
  auto clientConn = client.connect(clientAddr(port)).get();
  ASSERT_NE(clientConn, nullptr) << "Client failed to connect";

  auto conn = future.get();
  EXPECT_NE(conn, nullptr);
}

TEST_P(TcpAsyncAcceptTest, MultipleAsyncAccepts) {
  ScopedEventBaseThread evbThread("async-accept");
  AsyncTcpServer server(GetParam().serverAddr, {}, *evbThread.getEventBase());
  auto status = server.init();
  if (status.hasError()) {
    GTEST_SKIP() << "Not available: " << status.error().toString();
  }
  int port = server.getPort();

  constexpr int kNumClients = 3;

  std::vector<std::future<std::unique_ptr<Conn>>> futures;
  futures.reserve(kNumClients);
  for (int i = 0; i < kNumClients; ++i) {
    futures.push_back(server.accept());
  }

  TcpClient client;
  std::vector<std::unique_ptr<Conn>> clientConns;
  for (int i = 0; i < kNumClients; ++i) {
    auto c = client.connect(clientAddr(port)).get();
    ASSERT_NE(c, nullptr) << "Client " << i << " failed to connect";
    clientConns.push_back(std::move(c));
  }

  for (int i = 0; i < kNumClients; ++i) {
    auto conn = futures[i].get();
    EXPECT_NE(conn, nullptr) << "Async accept " << i << " returned nullptr";
  }
}

TEST_P(TcpAsyncAcceptTest, ConnectionQueueing) {
  ScopedEventBaseThread evbThread("async-accept");
  auto* evb = evbThread.getEventBase();
  AsyncTcpServer server(GetParam().serverAddr, {}, *evb);
  auto status = server.init();
  if (status.hasError()) {
    GTEST_SKIP() << "Not available: " << status.error().toString();
  }
  int port = server.getPort();

  // First accept triggers fd registration setup
  auto future1 = server.accept();

  TcpClient client;
  auto c1 = client.connect(clientAddr(port)).get();
  auto c2 = client.connect(clientAddr(port)).get();
  auto c3 = client.connect(clientAddr(port)).get();
  ASSERT_NE(c1, nullptr);
  ASSERT_NE(c2, nullptr);
  ASSERT_NE(c3, nullptr);

  auto conn1 = future1.get();
  ASSERT_NE(conn1, nullptr);

  // Wait for IO callback to process remaining connections into readyConns_
  evb->dispatchAndWait([]() noexcept {});

  auto conn2 = server.accept().get();
  auto conn3 = server.accept().get();
  EXPECT_NE(conn2, nullptr);
  EXPECT_NE(conn3, nullptr);
}

TEST_P(TcpAsyncAcceptTest, PromiseQueueing) {
  ScopedEventBaseThread evbThread("async-accept");
  AsyncTcpServer server(GetParam().serverAddr, {}, *evbThread.getEventBase());
  auto status = server.init();
  if (status.hasError()) {
    GTEST_SKIP() << "Not available: " << status.error().toString();
  }
  int port = server.getPort();

  auto future1 = server.accept();
  auto future2 = server.accept();

  TcpClient client;
  auto c1 = client.connect(clientAddr(port)).get();
  auto c2 = client.connect(clientAddr(port)).get();
  ASSERT_NE(c1, nullptr);
  ASSERT_NE(c2, nullptr);

  auto conn1 = future1.get();
  auto conn2 = future2.get();
  EXPECT_NE(conn1, nullptr);
  EXPECT_NE(conn2, nullptr);
}

TEST_P(TcpAsyncAcceptTest, ShutdownResolvesPromises) {
  ScopedEventBaseThread evbThread("async-accept");
  AsyncTcpServer server(GetParam().serverAddr, {}, *evbThread.getEventBase());
  auto status = server.init();
  if (status.hasError()) {
    GTEST_SKIP() << "Not available: " << status.error().toString();
  }

  auto future1 = server.accept();
  auto future2 = server.accept();

  server.shutdown();

  auto conn1 = future1.get();
  auto conn2 = future2.get();
  EXPECT_EQ(conn1, nullptr);
  EXPECT_EQ(conn2, nullptr);
}

TEST_P(TcpAsyncAcceptTest, ShutdownDuringAsyncAccept) {
  ScopedEventBaseThread evbThread("async-accept");
  AsyncTcpServer server(GetParam().serverAddr, {}, *evbThread.getEventBase());
  auto status = server.init();
  if (status.hasError()) {
    GTEST_SKIP() << "Not available: " << status.error().toString();
  }

  auto future = server.accept();

  // Shutdown from another thread while future.get() is blocking
  std::thread shutdownThread([&]() {
    // NOLINTNEXTLINE(facebook-hte-BadCall-sleep_for)
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    server.shutdown();
  });

  auto conn = future.get();
  EXPECT_EQ(conn, nullptr);

  shutdownThread.join();
}

TEST_P(TcpAsyncAcceptTest, AsyncAcceptRejectsNonUniflowClient) {
  ScopedEventBaseThread evbThread("async-accept");
  AsyncTcpServer server(GetParam().serverAddr, {}, *evbThread.getEventBase());
  auto status = server.init();
  if (status.hasError()) {
    GTEST_SKIP() << "Not available: " << status.error().toString();
  }
  int port = server.getPort();

  auto future = server.accept();

  {
    int sock = ::socket(
        GetParam().clientHost == "127.0.0.1" ? AF_INET : AF_INET6,
        SOCK_STREAM | SOCK_CLOEXEC,
        0);
    ASSERT_GE(sock, 0);

    sockaddr_storage addr{};
    if (GetParam().clientHost == "127.0.0.1") {
      auto* sa = reinterpret_cast<sockaddr_in*>(&addr);
      sa->sin_family = AF_INET;
      sa->sin_port = htons(static_cast<uint16_t>(port));
      ::inet_pton(AF_INET, "127.0.0.1", &sa->sin_addr);
      ::connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(sockaddr_in));
    } else {
      auto* sa = reinterpret_cast<sockaddr_in6*>(&addr);
      sa->sin6_family = AF_INET6;
      sa->sin6_port = htons(static_cast<uint16_t>(port));
      ::inet_pton(AF_INET6, "::1", &sa->sin6_addr);
      ::connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(sockaddr_in6));
    }
    uint32_t garbage = 0xDEADBEEF;
    ::send(sock, &garbage, sizeof(garbage), 0);
    ::close(sock);
  }

  // Now connect a valid uniflow client — should still be accepted
  TcpClient client;
  auto clientConn = client.connect(clientAddr(port)).get();
  ASSERT_NE(clientConn, nullptr) << "Valid client failed after bad client";

  auto conn = future.get();
  EXPECT_NE(conn, nullptr);
}

TEST_P(TcpAsyncAcceptTest, SendRecvAfterAsyncAccept) {
  ScopedEventBaseThread evbThread("async-accept");
  AsyncTcpServer server(GetParam().serverAddr, {}, *evbThread.getEventBase());
  auto status = server.init();
  if (status.hasError()) {
    GTEST_SKIP() << "Not available: " << status.error().toString();
  }
  int port = server.getPort();

  auto future = server.accept();

  TcpClient client;
  auto clientConn = client.connect(clientAddr(port)).get();
  ASSERT_NE(clientConn, nullptr);

  auto serverConn = future.get();
  ASSERT_NE(serverConn, nullptr);

  const std::vector<uint8_t> msg = {0x01, 0x02, 0x03, 0x04};
  auto sendResult = clientConn->send(msg).get();
  ASSERT_TRUE(sendResult.hasValue()) << sendResult.error().toString();

  std::vector<uint8_t> buf;
  auto recvResult = serverConn->recv(buf).get();
  ASSERT_TRUE(recvResult.hasValue()) << recvResult.error().toString();
  EXPECT_EQ(buf, msg);

  const std::vector<uint8_t> reply = {0x05, 0x06};
  sendResult = serverConn->send(reply).get();
  ASSERT_TRUE(sendResult.hasValue()) << sendResult.error().toString();

  buf.clear();
  recvResult = clientConn->recv(buf).get();
  ASSERT_TRUE(recvResult.hasValue()) << recvResult.error().toString();
  EXPECT_EQ(buf, reply);
}

TEST_P(TcpAsyncAcceptTest, ShutdownCleanupBeforeEventBaseDestroy) {
  auto evbThread = std::make_unique<ScopedEventBaseThread>("async-accept");
  AsyncTcpServer server(GetParam().serverAddr, {}, *evbThread->getEventBase());
  auto status = server.init();
  if (status.hasError()) {
    GTEST_SKIP() << "Not available: " << status.error().toString();
  }

  auto future = server.accept();

  server.shutdown();

  auto conn = future.get();
  EXPECT_EQ(conn, nullptr);

  // Destroying evbThread should exit cleanly — no dangling fd registrations
  evbThread.reset();
}

// ---------------------------------------------------------------------------
// Non-parameterized tests: error cases that are not address-family-specific.
// ---------------------------------------------------------------------------

class TcpAsyncAcceptMiscTest : public ::testing::Test {};

TEST_F(TcpAsyncAcceptMiscTest, AsyncAcceptBeforeInit) {
  ScopedEventBaseThread evbThread("async-accept");
  AsyncTcpServer server("127.0.0.1:0", {}, *evbThread.getEventBase());
  // Do NOT call init()

  // Before init, listenSock_ < 0, so accept returns nullptr
  auto conn = server.accept().get();
  EXPECT_EQ(conn, nullptr);
}

INSTANTIATE_TEST_SUITE_P(
    AddrFamilies,
    TcpAsyncAcceptTest,
    ::testing::Values(
        AddrFamily{"127.0.0.1:0", "127.0.0.1"},
        AddrFamily{":::0", "::1"}),
    [](const ::testing::TestParamInfo<AddrFamily>& info) {
      return info.param.clientHost == "127.0.0.1" ? "IPv4" : "IPv6";
    });
