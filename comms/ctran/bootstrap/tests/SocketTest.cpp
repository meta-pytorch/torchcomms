// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <chrono>

#include <sys/socket.h>

#include <gtest/gtest.h>

#include <folly/logging/xlog.h>

#include "comms/ctran/bootstrap/Socket.h"
#include "comms/utils/cvars/nccl_cvars.h"

using namespace ::testing;
using namespace std::literals::chrono_literals;

TEST(Socket, GetInterfaceAddress) {
  const bool kPreferV6{true};
  const bool kPreferV4{false};

  // Test for loopback interface "lo" v6
  auto maybeLoAddrV6 =
      ctran::bootstrap::getInterfaceAddress("lo", "", kPreferV6);
  ASSERT_FALSE(maybeLoAddrV6.hasError());
  EXPECT_EQ(maybeLoAddrV6->str(), std::string("::1"));
  EXPECT_FALSE(maybeLoAddrV6->isLinkLocal());

  auto maybeLoAddrV4 =
      ctran::bootstrap::getInterfaceAddress("lo", "", kPreferV4);
  ASSERT_FALSE(maybeLoAddrV4.hasError());
  EXPECT_EQ(maybeLoAddrV4->str(), std::string("127.0.0.1"));
  EXPECT_FALSE(maybeLoAddrV4->isLinkLocal());

  // Test for ethernet interface "eth0"
  auto maybeEth0Addr6 =
      ctran::bootstrap::getInterfaceAddress("eth0", "", kPreferV6);
  ASSERT_FALSE(maybeEth0Addr6.hasError());
  EXPECT_TRUE(maybeEth0Addr6->isV6());
  EXPECT_FALSE(maybeEth0Addr6->isLinkLocal());

  auto maybeEth0Addr4 =
      ctran::bootstrap::getInterfaceAddress("eth0", "", kPreferV4);
  ASSERT_FALSE(maybeEth0Addr4.hasError());
  // NOTE: At meta our hosts do not have v4 addresses at all. Ensure we get
  // v6 address
  EXPECT_TRUE(maybeEth0Addr4->isV6());
  EXPECT_FALSE(maybeEth0Addr4->isLinkLocal());

  maybeEth0Addr6 =
      ctran::bootstrap::getInterfaceAddress("eth0", "2401", kPreferV6);
  if (maybeEth0Addr6.hasValue()) {
    XLOG(INFO) << "Found v6 address: " << maybeEth0Addr6->str();
    EXPECT_TRUE(maybeEth0Addr6->isV6());
    EXPECT_EQ(0, maybeEth0Addr6->str().find("2401"));
  }
}

TEST(Socket, SocketLifeCycle) {
  ctran::bootstrap::ServerSocket server{1};
  ctran::bootstrap::Socket client;
  EXPECT_EQ(server.getFd(), -1);
  EXPECT_EQ(client.getFd(), -1);

  // Start server on the loopback interface
  ASSERT_EQ(0, server.bindAndListen(folly::SocketAddress("::1", 0), "lo"));
  EXPECT_NE(server.getFd(), -1);

  // Get the server address and validate it
  const auto& maybeServerAddr = server.getListenAddress();
  ASSERT_FALSE(maybeServerAddr.hasError());
  auto& serverAddr = maybeServerAddr.value();
  EXPECT_EQ(serverAddr.getFamily(), AF_INET6);
  EXPECT_NE(serverAddr.getPort(), 0);
  EXPECT_EQ(serverAddr.getIPAddress().str(), "::1");

  // Connect client to the server
  ASSERT_EQ(0, client.connect(serverAddr, "lo"));
  EXPECT_NE(client.getFd(), -1);

  // Accept client connection on server
  auto maybeClient = server.accept();
  ASSERT_FALSE(maybeClient.hasError()) << maybeClient.error();
  auto& acceptedClient = maybeClient.value();
  EXPECT_NE(acceptedClient.getFd(), -1);

  // Validate connection addresses
  EXPECT_EQ(
      client.getPeerAddress().describe(),
      acceptedClient.getLocalAddress().describe());
  EXPECT_EQ(
      client.getLocalAddress().describe(),
      acceptedClient.getPeerAddress().describe());

  // Send request from client to acceptedClient
  const std::string request = "ping";
  ASSERT_EQ(0, client.send(request.data(), request.size()));
  char rcvdRequest[request.size()];
  ASSERT_EQ(0, acceptedClient.recv(rcvdRequest, request.size()));
  EXPECT_EQ(0, std::memcmp(request.data(), rcvdRequest, request.size()));

  // Send reply from acceptedClient to client
  const std::string response = "pong";
  ASSERT_EQ(0, acceptedClient.send(response.data(), response.size()));
  char rcvdResponse[response.size()];
  ASSERT_EQ(0, client.recv(rcvdResponse, response.size()));
  EXPECT_EQ(0, std::memcmp(response.data(), rcvdResponse, response.size()));

  // Close the sockets
  EXPECT_EQ(0, server.shutdown());
  EXPECT_EQ(-1, server.getFd());
  EXPECT_EQ(0, client.close());
  EXPECT_EQ(-1, client.getFd());
  EXPECT_EQ(0, acceptedClient.close());
  EXPECT_EQ(-1, acceptedClient.getFd());
}

TEST(Socket, ConnectRetries) {
  ctran::bootstrap::ServerSocket server{1};

  // Bind server on the loopback interface but do not listen
  XLOG(INFO) << "Binding server..";
  ASSERT_EQ(0, server.bind(folly::SocketAddress("::1", 0), "lo"));
  const auto& maybeServerAddr = server.getListenAddress();
  ASSERT_FALSE(maybeServerAddr.hasError());
  auto& serverAddr = maybeServerAddr.value();

  // Connect client to the server. It may experience few connect errors but
  // retry will eventually make it succeed
  XLOG(INFO) << "Connecting to server..";
  ctran::bootstrap::Socket client;
  ASSERT_EQ(
      ECONNREFUSED, client.connect(serverAddr, "lo", 100ms, 5 /* retries */));
  EXPECT_EQ(client.getFd(), -1);

  // Delay the listen in a separate thread to simulate a connect error
  std::thread listenThread([&]() {
    std::this_thread::sleep_for(500ms);
    XLOG(INFO) << "Starting to listen on server";
    ASSERT_EQ(0, server.listen());
  });

  // Attempt to connect to server again .. we may succeed after few more retries
  XLOG(INFO) << "Attempting to connect to server again..";
  ASSERT_EQ(0, client.connect(serverAddr, "lo", 100ms, 10 /* retries */));
  EXPECT_NE(client.getFd(), -1);

  // Accept client connection on server
  listenThread.join();
  auto maybeClient = server.accept();
  ASSERT_FALSE(maybeClient.hasError()) << maybeClient.error();
  auto& acceptedClient = maybeClient.value();
  EXPECT_NE(acceptedClient.getFd(), -1);

  // Close the sockets
  EXPECT_EQ(0, server.shutdown());
  EXPECT_EQ(0, client.close());
  EXPECT_EQ(0, acceptedClient.close());
}

TEST(Socket, SocketNonBlocking) {
  ctran::bootstrap::ServerSocket server{1};
  ctran::bootstrap::Socket client1;
  ctran::bootstrap::Socket client2;
  EXPECT_EQ(server.getFd(), -1);
  EXPECT_EQ(client1.getFd(), -1);
  EXPECT_EQ(client2.getFd(), -1);

  // Start server on the loopback interface
  ASSERT_EQ(0, server.bindAndListen(folly::SocketAddress("::1", 0), "lo"));
  EXPECT_NE(server.getFd(), -1);

  // Get the server address and validate it
  const auto& maybeServerAddr = server.getListenAddress();
  ASSERT_FALSE(maybeServerAddr.hasError());
  auto& serverAddr = maybeServerAddr.value();
  EXPECT_EQ(serverAddr.getFamily(), AF_INET6);
  EXPECT_NE(serverAddr.getPort(), 0);
  EXPECT_EQ(serverAddr.getIPAddress().str(), "::1");

  // Connect client to the server
  ASSERT_EQ(
      0,
      client1.connect(
          serverAddr, "lo", std::chrono::milliseconds(1000), 10, true));
  EXPECT_NE(client1.getFd(), -1);

  ASSERT_EQ(
      0,
      client2.connect(
          serverAddr, "lo", std::chrono::milliseconds(1000), 10, true));
  EXPECT_NE(client2.getFd(), -1);

  std::thread pollthread([&server]() {
    // Accept client connection on server
    auto maybeClient1 = server.accept(true);
    auto& acceptedClient1 = maybeClient1.value();
    EXPECT_NE(acceptedClient1.getFd(), -1);

    auto maybeClient2 = server.accept(true);
    auto& acceptedClient2 = maybeClient2.value();
    EXPECT_NE(acceptedClient2.getFd(), -1);

    struct pollfd fds[2];
    ctran::bootstrap::Socket* sockets[2];
    fds[0].fd = acceptedClient1.getFd();
    fds[0].events = POLLIN;
    fds[1].fd = acceptedClient2.getFd();
    fds[1].events = POLLIN;
    sockets[0] = &acceptedClient1;
    sockets[1] = &acceptedClient2;
    char buffer[100];
    std::string expected = "ping";
    int timeout_counter = -1;
    int data_recv = 0;
    while (true) {
      if (data_recv == 2) {
        break;
      }
      int ret = 0;
      timeout_counter += 1;
      ret = poll(fds, 2, 500);
      if (ret == 0) {
        XLOG(INFO) << "Poll timeout counter: " << timeout_counter;
        continue;
      } else if (ret < 0) {
        XLOG(ERR) << "Poll error" << std::endl;
        break;
      } else {
        XLOG(INFO) << "Successfully polled " << ret << " fds";
        for (int fid = 0; fid < 2; fid++) {
          if (fds[fid].revents & POLLIN) {
            XLOG(INFO) << "fid: " << fid << " fd: " << fds[fid].fd
                       << " revents: " << fds[fid].revents;
            int rcvd = sockets[fid]->recvAsync(buffer, 100);
            if (rcvd < 0) {
              XLOG(ERR) << "Read error" << std::endl;
              break;
            } else if (rcvd == 0) {
              XLOG(INFO) << "Server closed the connection";
            } else {
              XLOG(INFO) << "fd " << fds[fid].fd << " successfully read "
                         << rcvd << " bytes";

              EXPECT_EQ(
                  0, std::memcmp(buffer, expected.data(), expected.size()));
              data_recv += 1;
              const std::string response = "pong";
              sockets[fid]->send(response.data(), response.size());
            }
          }
        }
      }
    }
    EXPECT_EQ(0, acceptedClient1.close());
    EXPECT_EQ(-1, acceptedClient1.getFd());
    EXPECT_EQ(0, acceptedClient2.close());
    EXPECT_EQ(-1, acceptedClient2.getFd());
  });
  // Send request from client to acceptedClient
  const std::string request = "ping";
  ASSERT_EQ(0, client1.send(request.data(), request.size()));
  ASSERT_EQ(0, client2.send(request.data(), request.size()));
  char buffer[100];
  const std::string expected = "pong";
  ASSERT_EQ(0, client1.recv(buffer, expected.size()));
  EXPECT_EQ(0, std::memcmp(expected.data(), buffer, expected.size()));
  ASSERT_EQ(0, client2.recv(buffer, expected.size()));
  EXPECT_EQ(0, std::memcmp(expected.data(), buffer, expected.size()));

  pollthread.join();
  // Close the sockets
  EXPECT_EQ(0, server.shutdown());
  EXPECT_EQ(-1, server.getFd());
  EXPECT_EQ(0, client1.close());
  EXPECT_EQ(-1, client1.getFd());
  EXPECT_EQ(0, client2.close());
  EXPECT_EQ(-1, client2.getFd());
}

TEST(Socket, BindAndUnbind) {
  ctran::bootstrap::ServerSocket server1{1};
  ctran::bootstrap::ServerSocket server2{1};

  // Bind server1 on the loopback interface
  ASSERT_EQ(0, server1.bind(folly::SocketAddress("::1", 0), "lo", true));
  EXPECT_NE(server1.getFd(), -1);

  // Get the server1 address and validate it
  const auto maybeServer1Addr = server1.getListenAddress();
  ASSERT_FALSE(maybeServer1Addr.hasError());
  auto server1Addr = maybeServer1Addr.value();
  EXPECT_EQ(server1Addr.getFamily(), AF_INET6);
  EXPECT_NE(server1Addr.getPort(), 0);
  EXPECT_EQ(server1Addr.getIPAddress().str(), "::1");

  // Attempt to bind server2 to the same address as server1
  ASSERT_EQ(0, server2.bind(server1Addr, "lo"));
  EXPECT_NE(server2.getFd(), -1);
  EXPECT_EQ(server2.getListenAddress().value(), server1Addr);

  // Shutdown server1 and attempt to bind to the same address
  ASSERT_EQ(0, server1.shutdown());
  ASSERT_EQ(0, server1.bind(server1Addr, "lo"));
  EXPECT_NE(server1.getFd(), -1);
  EXPECT_EQ(server1.getListenAddress().value(), server1Addr);

  // Close the sockets
  EXPECT_EQ(0, server1.shutdown());
  EXPECT_EQ(-1, server1.getFd());
  EXPECT_EQ(0, server2.shutdown());
  EXPECT_EQ(-1, server2.getFd());
}

TEST(Socket, OverrideTosValue) {
  const int kExpectedTos = 96;
  NCCL_SOCKET_TOS_CONFIG = kExpectedTos;
  ctran::bootstrap::ServerSocket server1{1};

  // Bind server1 on the loopback interface
  auto addr = folly::SocketAddress("::1", 0);
  ASSERT_EQ(0, server1.bindAndListen(addr, "lo"));
  EXPECT_NE(server1.getFd(), -1);

  // validate the TOS value has been set
  int socketTos = 0;
  socklen_t rlen = sizeof(int);
  getsockopt(server1.getFd(), IPPROTO_IPV6, IPV6_TCLASS, &socketTos, &rlen);
  EXPECT_EQ(socketTos, kExpectedTos);

  // Close the sockets
  EXPECT_EQ(0, server1.shutdown());
  EXPECT_EQ(-1, server1.getFd());
}

TEST(Socket, AcceptErrorOnShutdown) {
  ctran::bootstrap::ServerSocket server{1};
  ASSERT_EQ(0, server.bindAndListen(folly::SocketAddress("::1", 0), "lo"));
  std::thread listenThread([&server]() {
    auto maybeClient = server.accept();
    ASSERT_TRUE(maybeClient.hasError());
    EXPECT_TRUE(maybeClient.error() == EBADF || maybeClient.error() == EINVAL)
        << "Error: " << maybeClient.error();
  });
  server.shutdown();
  listenThread.join();
}

// Tests for ISocket interface implementation

TEST(Socket, ISocketInterfaceSendAll) {
  ctran::bootstrap::ServerSocket server{1};
  ctran::bootstrap::Socket client;

  // Start server on the loopback interface
  ASSERT_EQ(0, server.bindAndListen(folly::SocketAddress("::1", 0), "lo"));
  const auto& maybeServerAddr = server.getListenAddress();
  ASSERT_FALSE(maybeServerAddr.hasError());
  auto& serverAddr = maybeServerAddr.value();

  // Connect client to the server
  ASSERT_EQ(0, client.connect(serverAddr, "lo"));

  // Accept client connection on server
  auto maybeClient = server.accept();
  ASSERT_FALSE(maybeClient.hasError());
  auto& acceptedClient = maybeClient.value();

  // Test sendAll/recvAll via ISocket interface
  const std::string request = "test message for sendAll";
  std::unique_ptr<ctran::bootstrap::ISocket> clientSocket =
      std::make_unique<ctran::bootstrap::Socket>(std::move(client));
  ASSERT_EQ(0, clientSocket->send(request.data(), request.size()));

  char rcvdRequest[request.size()];
  ASSERT_EQ(0, acceptedClient.recv(rcvdRequest, request.size()));
  EXPECT_EQ(0, std::memcmp(request.data(), rcvdRequest, request.size()));

  // Cleanup
  EXPECT_EQ(0, server.shutdown());
  EXPECT_EQ(0, clientSocket->close());
  EXPECT_EQ(0, acceptedClient.close());
}
