// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <chrono>
#include <optional>
#include <semaphore>
#include <thread>

#include <sys/socket.h>

#include <gtest/gtest.h>

#include <folly/logging/xlog.h>

#include "comms/ctran/bootstrap/AbortableSocket.h"

using namespace ::testing;
using namespace std::literals::chrono_literals;

class AbortableServerSocketTest : public ::testing::Test {
 public:
  void SetUp() override {
    createAbortableServerSocket();
  }
  void TearDown() override {
    // Close the sockets
    server->shutdown();
    EXPECT_EQ(-1, server->getFd());
  }

 protected:
  void createAbortableServerSocket() {
    serverAbort = ctran::utils::createAbort(/*enabled=*/true);

    server.emplace(1, serverAbort);
    EXPECT_EQ(server->getFd(), -1);
    EXPECT_EQ(0, server->bindAndListen(folly::SocketAddress("::1", 0), "lo"));
    EXPECT_NE(server->getFd(), -1);

    const auto maybeServerAddr = server->getListenAddress();
    EXPECT_FALSE(maybeServerAddr.hasError());
    serverAddr = maybeServerAddr.value();
    EXPECT_EQ(serverAddr.getFamily(), AF_INET6);
    EXPECT_NE(serverAddr.getPort(), 0);
    EXPECT_EQ(serverAddr.getIPAddress().str(), "::1");
  }

  std::shared_ptr<ctran::utils::Abort> serverAbort;
  std::optional<ctran::bootstrap::AbortableServerSocket> server;
  folly::SocketAddress serverAddr;
};

class AbortableSocketTest : public AbortableServerSocketTest {
 public:
  void SetUp() override {
    AbortableServerSocketTest::SetUp();
    createClientSocket();
  }
  void TearDown() override {
    AbortableServerSocketTest::TearDown();
    client.close();
    EXPECT_EQ(-1, client.getFd());
    acceptedClient->close();
    EXPECT_EQ(-1, acceptedClient->getFd());
  }

 protected:
  void createClientSocket() {
    clientAbort = ctran::utils::createAbort(/*enabled=*/true);
    client = ctran::bootstrap::AbortableSocket(clientAbort);

    EXPECT_EQ(0, client.connect(serverAddr, "lo"));
    auto maybeAcceptedClient = server->acceptSocket();
    EXPECT_FALSE(maybeAcceptedClient.hasError());
    acceptedClient = std::move(maybeAcceptedClient.value());
    EXPECT_NE(acceptedClient, nullptr);

    EXPECT_EQ(
        client.getPeerAddress().describe(),
        acceptedClient->getLocalAddress().describe());
    EXPECT_EQ(
        client.getLocalAddress().describe(),
        acceptedClient->getPeerAddress().describe());
  }

  std::shared_ptr<ctran::utils::Abort> clientAbort;
  ctran::bootstrap::AbortableSocket client;
  std::unique_ptr<ctran::bootstrap::ISocket> acceptedClient;
};

//
// Test Group #1: Basic Socket Operations
//

TEST_F(AbortableSocketTest, BasicLifecycleSendAllRecvAll) {
  const std::string request = "ping";
  ASSERT_EQ(0, client.send(request.data(), request.size()));
  char rcvdRequest[request.size()];
  ASSERT_EQ(0, acceptedClient->recv(rcvdRequest, request.size()));
  EXPECT_EQ(0, std::memcmp(request.data(), rcvdRequest, request.size()));

  const std::string response = "pong";
  ASSERT_EQ(0, acceptedClient->send(response.data(), response.size()));
  char rcvdResponse[response.size()];
  ASSERT_EQ(0, client.recv(rcvdResponse, response.size()));
  EXPECT_EQ(0, std::memcmp(response.data(), rcvdResponse, response.size()));
}

TEST_F(AbortableSocketTest, MoveSemantics) {
  ctran::bootstrap::AbortableSocket client1;
  EXPECT_EQ(0, client1.connect(serverAddr, "lo"));
  auto maybeAcceptedClient = server->acceptSocket();
  EXPECT_FALSE(maybeAcceptedClient.hasError());
  acceptedClient = std::move(maybeAcceptedClient.value());
  EXPECT_NE(acceptedClient, nullptr);

  int fd1 = client1.getFd();
  EXPECT_NE(fd1, -1);

  ctran::bootstrap::AbortableSocket client2 = std::move(client1);
  EXPECT_EQ(client1.getFd(), -1);
  EXPECT_EQ(client2.getFd(), fd1);
}
