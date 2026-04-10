// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/controller/TcpController.h"

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

using namespace uniflow;
using namespace uniflow::controller;

// ---------------------------------------------------------------------------
// Parameterized fixture: runs connection-level tests over both IPv4 and IPv6.
// IPv6 tests are skipped if not available on the host.
// ---------------------------------------------------------------------------

struct AddrFamily {
  std::string serverAddr; // e.g., "127.0.0.1:0" or ":::0"
  std::string clientHost; // e.g., "127.0.0.1" or "::1"
};

class TcpConnTest : public ::testing::TestWithParam<AddrFamily> {
 protected:
  int port_{};
  std::unique_ptr<TcpServer> server_;

  void SetUp() override {
    server_ = std::make_unique<TcpServer>(GetParam().serverAddr);
    auto status = server_->init();
    if (status.hasError()) {
      GTEST_SKIP() << "Not available: " << status.error().toString();
    }
    port_ = server_->getPort();
    ASSERT_GT(port_, 0) << "Failed to resolve bound port";
  }

  std::string clientAddr() const {
    return GetParam().clientHost + ":" + std::to_string(port_);
  }

  // Connect a raw TCP socket to the server (bypassing TcpClient/TcpConn).
  // Used to test handshake rejection with non-uniflow peers.
  int connectRawSocket() const {
    bool isIPv6 = (GetParam().clientHost != "127.0.0.1");
    int domain = isIPv6 ? AF_INET6 : AF_INET;
    int sock = ::socket(domain, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (sock < 0) {
      return -1;
    }

    sockaddr_storage addr{};
    socklen_t addrLen;
    if (isIPv6) {
      auto* sa = reinterpret_cast<sockaddr_in6*>(&addr);
      sa->sin6_family = AF_INET6;
      sa->sin6_port = htons(static_cast<uint16_t>(port_));
      ::inet_pton(AF_INET6, GetParam().clientHost.c_str(), &sa->sin6_addr);
      addrLen = sizeof(sockaddr_in6);
    } else {
      auto* sa = reinterpret_cast<sockaddr_in*>(&addr);
      sa->sin_family = AF_INET;
      sa->sin_port = htons(static_cast<uint16_t>(port_));
      ::inet_pton(AF_INET, GetParam().clientHost.c_str(), &sa->sin_addr);
      addrLen = sizeof(sockaddr_in);
    }

    if (::connect(sock, reinterpret_cast<sockaddr*>(&addr), addrLen) < 0) {
      ::close(sock);
      return -1;
    }
    return sock;
  }

  // Establish a connected client+server Conn pair.
  // No sleep needed: listen() backlog accepts connections before accept().
  std::pair<std::unique_ptr<Conn>, std::unique_ptr<Conn>> connectPair() {
    std::unique_ptr<Conn> serverConn;
    std::thread acceptThread(
        [this, &serverConn]() { serverConn = server_->accept(); });

    TcpClient client;
    auto clientConn = client.connect(clientAddr());

    acceptThread.join();
    return {std::move(clientConn), std::move(serverConn)};
  }
};

TEST_P(TcpConnTest, SendRecvBidirectional) {
  auto [clientConn, serverConn] = connectPair();
  ASSERT_NE(clientConn, nullptr);
  ASSERT_NE(serverConn, nullptr);

  // Client → Server (small message)
  std::vector<uint8_t> ping = {'P', 'I', 'N', 'G'};
  ASSERT_TRUE(clientConn->send(ping).hasValue());

  std::vector<uint8_t> recv1;
  auto result1 = serverConn->recv(recv1);
  ASSERT_TRUE(result1.hasValue()) << result1.error().toString();
  EXPECT_EQ(recv1, ping);

  // Server → Client (small message)
  std::vector<uint8_t> pong = {'P', 'O', 'N', 'G'};
  ASSERT_TRUE(serverConn->send(pong).hasValue());

  std::vector<uint8_t> recv2;
  auto result2 = clientConn->recv(recv2);
  ASSERT_TRUE(result2.hasValue()) << result2.error().toString();
  EXPECT_EQ(recv2, pong);

  // Empty message
  std::vector<uint8_t> empty;
  ASSERT_TRUE(clientConn->send(empty).hasValue());

  std::vector<uint8_t> recv3;
  auto result3 = serverConn->recv(recv3);
  ASSERT_TRUE(result3.hasValue());
  EXPECT_TRUE(recv3.empty());
}

TEST_P(TcpConnTest, SendRecvLargeData) {
  auto [clientConn, serverConn] = connectPair();
  ASSERT_NE(clientConn, nullptr);
  ASSERT_NE(serverConn, nullptr);

  // 1MB message exercises the full send/recv loop with partial writes/reads
  constexpr size_t kSize = 1 << 20;
  std::vector<uint8_t> sendData(kSize);
  for (size_t i = 0; i < kSize; ++i) {
    sendData[i] = static_cast<uint8_t>(i & 0xFF);
  }

  auto sendResult = clientConn->send(sendData);
  ASSERT_TRUE(sendResult.hasValue()) << sendResult.error().toString();
  EXPECT_EQ(sendResult.value(), kSize);

  std::vector<uint8_t> recvData;
  auto recvResult = serverConn->recv(recvData);
  ASSERT_TRUE(recvResult.hasValue()) << recvResult.error().toString();
  EXPECT_EQ(recvData.size(), kSize);
  EXPECT_EQ(recvData, sendData);
}

TEST_P(TcpConnTest, MultipleClientsConnect) {
  constexpr int kNumClients = 5;
  std::vector<std::unique_ptr<Conn>> serverConns(kNumClients);
  std::vector<std::unique_ptr<Conn>> clientConns(kNumClients);

  std::thread acceptThread([this, &serverConns]() {
    for (int i = 0; i < kNumClients; ++i) {
      serverConns[i] = server_->accept();
    }
  });

  TcpClient client;
  for (int i = 0; i < kNumClients; ++i) {
    clientConns[i] = client.connect(clientAddr());
    ASSERT_NE(clientConns[i], nullptr) << "Client " << i << " failed";
  }

  acceptThread.join();

  // Verify all connections are valid and can exchange data
  for (int i = 0; i < kNumClients; ++i) {
    ASSERT_NE(serverConns[i], nullptr) << "Server conn " << i << " null";
    std::vector<uint8_t> msg = {static_cast<uint8_t>(i)};
    ASSERT_TRUE(clientConns[i]->send(msg).hasValue());

    std::vector<uint8_t> recv;
    ASSERT_TRUE(serverConns[i]->recv(recv).hasValue());
    EXPECT_EQ(recv, msg);
  }
}

TEST_P(TcpConnTest, ConnectionClosedByPeer) {
  auto [clientConn, serverConn] = connectPair();
  ASSERT_NE(clientConn, nullptr);
  ASSERT_NE(serverConn, nullptr);

  // Close client side
  clientConn.reset();

  // Server recv should detect the closed connection
  std::vector<uint8_t> buffer;
  auto result = serverConn->recv(buffer);
  EXPECT_TRUE(result.hasError())
      << "recv should fail when peer has closed the connection";
}

TEST_P(TcpConnTest, SendOnClosedConnection) {
  auto [clientConn, serverConn] = connectPair();
  ASSERT_NE(clientConn, nullptr);
  ASSERT_NE(serverConn, nullptr);

  // Close server side
  serverConn.reset();

  // Give the FIN time to propagate
  std::vector<uint8_t> data(1024, 0xAA);

  // First send may succeed (kernel buffers), but repeated sends on a
  // connection with a closed peer will eventually fail with EPIPE/ECONNRESET.
  bool gotError = false;
  for (int i = 0; i < 100 && !gotError; ++i) {
    auto result = clientConn->send(data);
    if (result.hasError()) {
      gotError = true;
      EXPECT_EQ(result.error().code(), ErrCode::ConnectionFailed);
    }
  }
  EXPECT_TRUE(gotError) << "send should eventually fail on a closed connection";
}

TEST_P(TcpConnTest, RecvMultipleMessagesThenClose) {
  auto [clientConn, serverConn] = connectPair();
  ASSERT_NE(clientConn, nullptr);
  ASSERT_NE(serverConn, nullptr);

  // Send 3 messages, only recv 2, then close — verify no hang
  for (int i = 0; i < 3; ++i) {
    std::vector<uint8_t> msg = {static_cast<uint8_t>(i), 0xAA, 0xBB};
    ASSERT_TRUE(clientConn->send(msg).hasValue());
  }

  // Only drain 2 of 3
  for (int i = 0; i < 2; ++i) {
    std::vector<uint8_t> recv;
    auto result = serverConn->recv(recv);
    ASSERT_TRUE(result.hasValue()) << result.error().toString();
    EXPECT_EQ(recv[0], static_cast<uint8_t>(i));
  }

  // Close without draining the third message — should not hang or crash
  serverConn.reset();
  clientConn.reset();
}

TEST_P(TcpConnTest, SendRecvMultipleMessages) {
  auto [clientConn, serverConn] = connectPair();
  ASSERT_NE(clientConn, nullptr);
  ASSERT_NE(serverConn, nullptr);

  // Send multiple messages and verify FIFO ordering
  for (int i = 0; i < 10; ++i) {
    std::vector<uint8_t> msg = {static_cast<uint8_t>(i)};
    ASSERT_TRUE(clientConn->send(msg).hasValue());
  }

  for (int i = 0; i < 10; ++i) {
    std::vector<uint8_t> recv;
    ASSERT_TRUE(serverConn->recv(recv).hasValue());
    ASSERT_EQ(recv.size(), 1u);
    EXPECT_EQ(recv[0], static_cast<uint8_t>(i));
  }
}

TEST_P(TcpConnTest, RejectsWrongMagic) {
  std::unique_ptr<Conn> serverConn;
  std::thread acceptThread(
      [this, &serverConn]() { serverConn = server_->accept(); });

  // Connect with a raw socket (bypassing TcpClient/TcpConn)
  int rawSock = connectRawSocket();
  ASSERT_GE(rawSock, 0);

  // Send wrong magic — server's exchangeMagic() should reject
  uint32_t wrongMagic = htonl(0xDEADBEEF);
  ::send(rawSock, &wrongMagic, sizeof(wrongMagic), MSG_NOSIGNAL);

  // Drain the server's outbound magic to prevent send-side blocking
  uint32_t buf = 0;
  ::recv(rawSock, &buf, sizeof(buf), 0);

  // accept() continues after magic failure — shutdown to unblock it
  server_->shutdown();

  acceptThread.join();
  EXPECT_EQ(serverConn, nullptr)
      << "accept() should reject connection with wrong magic";

  ::close(rawSock);
}

TEST_P(TcpConnTest, RejectsPeerClosedBeforeMagic) {
  std::unique_ptr<Conn> serverConn;
  std::thread acceptThread(
      [this, &serverConn]() { serverConn = server_->accept(); });

  // Connect then immediately close — no magic sent
  int rawSock = connectRawSocket();
  ASSERT_GE(rawSock, 0);
  ::close(rawSock);

  // accept() continues after magic failure — shutdown to unblock it
  server_->shutdown();

  acceptThread.join();
  EXPECT_EQ(serverConn, nullptr)
      << "accept() should reject when peer closes before magic exchange";
}

INSTANTIATE_TEST_SUITE_P(
    AddrFamilies,
    TcpConnTest,
    ::testing::Values(
        AddrFamily{"127.0.0.1:0", "127.0.0.1"},
        AddrFamily{":::0", "::1"}),
    [](const ::testing::TestParamInfo<AddrFamily>& info) {
      return info.param.clientHost == "127.0.0.1" ? "IPv4" : "IPv6";
    });
