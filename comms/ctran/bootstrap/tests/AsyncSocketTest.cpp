// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <chrono>

#include <gtest/gtest.h>

#include <folly/io/async/ScopedEventBaseThread.h>
#include <folly/logging/xlog.h>

#include "comms/ctran/bootstrap/AsyncSocket.h"
#include "comms/ctran/bootstrap/Socket.h"

using namespace ::testing;
using namespace std::literals::chrono_literals;

namespace {
enum ReqStatus {
  INCOMPLETE,
  COMPLETE,
  ERROR,
};
}

TEST(AsyncSocket, AsyncSocketLifeCycle) {
  const std::string request = "ping";
  auto eventThread = std::make_unique<folly::ScopedEventBaseThread>();
  std::string recv_str;
  std::atomic<ReqStatus> recvStatus(INCOMPLETE);

  ctran::bootstrap::AsyncServerSocket server(*eventThread->getEventBase());

  // Start server on the loopback interface and set up callback
  auto serverAddrFuture = server.start(
      folly::SocketAddress("::1", 0),
      request.size(),
      [&recv_str, &recvStatus](std::unique_ptr<folly::IOBuf> buf) {
        // Extract the received data into recv_str
        recv_str = buf->moveToFbString().toStdString();
        recvStatus.store(COMPLETE);
      });

  // Get the server address and validate it
  auto serverAddr = std::move(serverAddrFuture).get();
  EXPECT_EQ(serverAddr.getFamily(), AF_INET6);
  EXPECT_NE(serverAddr.getPort(), 0);
  EXPECT_EQ(serverAddr.getIPAddress().str(), "::1");

  // Send request from client to server
  std::atomic<ReqStatus> sendStatus(INCOMPLETE);
  ctran::bootstrap::AsyncClientSocket::send(
      *eventThread->getEventBase(),
      serverAddr,
      request.data(),
      request.size(),
      [&sendStatus](const folly::AsyncSocketException* err) {
        if (err != nullptr) {
          sendStatus.store(ERROR);
        } else {
          sendStatus.store(COMPLETE);
        }
      });

  // Wait for send to complete
  while (sendStatus.load() == INCOMPLETE) {
  }
  EXPECT_EQ(sendStatus.load(), COMPLETE);

  // Wait for receive to complete
  while (recvStatus.load() == INCOMPLETE) {
  }

  // Verify recvStatus and recv_str
  EXPECT_EQ(recvStatus.load(), COMPLETE);
  EXPECT_EQ(recv_str, request);

  // Stop the server and wait for cleanup
  auto fut = server.stop();
  EXPECT_EQ(folly::unit, std::move(fut).get());
}

TEST(AsyncSocket, AsyncServerSocketAcceptMultiMessagesInOrder) {
  struct Msg {
    char data;
    int seqNum;
  };
  const std::string sent_str = "abcdefghijk";
  auto eventThread = std::make_unique<folly::ScopedEventBaseThread>();
  // Total received message so far
  std::atomic_int recv_cnt = 0;
  // The seqNum of next expected message
  int recv_nxt = 0;
  // Received message map to ensure ordering [seqNum, data]
  std::unordered_map<int, char> recvd_map;
  std::string recvd_str = "";

  ctran::bootstrap::AsyncServerSocket server(*eventThread->getEventBase());

  // Start server on the loopback interface and set up callback
  auto serverAddrFuture = server.start(
      folly::SocketAddress("::1", 0),
      sizeof(Msg) /*msgSize*/,
      [&recv_cnt, &recv_nxt, &recvd_map, &recvd_str](
          std::unique_ptr<folly::IOBuf> buf) {
        Msg msg;
        std::memcpy(&msg, buf->data(), sizeof(Msg));
        recvd_map[msg.seqNum] = msg.data;
        while (recvd_map.find(recv_nxt) != recvd_map.end()) {
          recvd_str += recvd_map[recv_nxt];
          recvd_map.erase(recv_nxt);
          recv_nxt++;
        }
        recv_cnt.fetch_add(1);
      });

  // Get the server address and validate it
  auto serverAddr = std::move(serverAddrFuture).get();
  EXPECT_EQ(serverAddr.getFamily(), AF_INET6);
  EXPECT_NE(serverAddr.getPort(), 0);
  EXPECT_EQ(serverAddr.getIPAddress().str(), "::1");

  std::vector<std::atomic<ReqStatus>> sendStatus(sent_str.size());
  for (auto& status : sendStatus) {
    status.store(INCOMPLETE);
  }
  std::vector<Msg> msgs(sent_str.size());
  for (int i = 0; i < sent_str.size(); i++) {
    msgs[i] = Msg{.data = sent_str[i], .seqNum = i};
    ctran::bootstrap::AsyncClientSocket::send(
        *eventThread->getEventBase(),
        serverAddr,
        &msgs[i],
        sizeof(Msg),
        [statusPtr = &sendStatus[i]](const folly::AsyncSocketException* err) {
          if (err != nullptr) {
            statusPtr->store(ERROR);
          } else {
            statusPtr->store(COMPLETE);
          }
        });
  }

  // Wait for all sends to complete
  for (size_t i = 0; i < sent_str.size(); i++) {
    while (sendStatus[i].load() == INCOMPLETE) {
    }
    EXPECT_EQ(sendStatus[i].load(), COMPLETE);
  }

  // Wait for receive to complete
  while (recv_cnt.load() < sent_str.size()) {
  }
  EXPECT_EQ(recvd_str, sent_str);

  // Stop the server and wait for cleanup
  auto fut = server.stop();
  EXPECT_EQ(folly::unit, std::move(fut).get());
}

TEST(AsyncSocket, AsyncClientSocketConnectFailure) {
  const std::string request = "ping";
  auto eventThread = std::make_unique<folly::ScopedEventBaseThread>();
  auto dummyServerAddr = folly::SocketAddress();

  // Send request from client to server
  std::atomic<ReqStatus> sendStatus(INCOMPLETE);
  ctran::bootstrap::AsyncClientSocket::send(
      *eventThread->getEventBase(),
      dummyServerAddr,
      request.data(),
      request.size(),
      [&sendStatus](const folly::AsyncSocketException* err) {
        if (err != nullptr) {
          sendStatus.store(ERROR);
        } else {
          sendStatus.store(COMPLETE);
        }
      });

  // Wait for send to complete or error
  while (sendStatus.load() == INCOMPLETE) {
  }
  EXPECT_EQ(sendStatus.load(), ERROR);
}

TEST(AsyncSocket, AsyncServerSocketReceiveTimeout) {
  const std::string request = "ping";
  auto eventThread = std::make_unique<folly::ScopedEventBaseThread>();

  std::atomic<bool> recvCallbackInvoked(false);

  ctran::bootstrap::AsyncServerSocket server(*eventThread->getEventBase());

  // Start server with short timeout (500ms)
  auto serverAddrFuture = server.start(
      folly::SocketAddress("::1", 0),
      request.size(),
      [&recvCallbackInvoked](std::unique_ptr<folly::IOBuf> buf) {
        // This callback should NOT be invoked if timeout happens before data
        // arrives
        XLOGF(INFO, "Server received {} bytes", buf->computeChainDataLength());
        recvCallbackInvoked.store(true);
      },
      std::chrono::milliseconds(500)); // 500ms timeout

  auto serverAddr = std::move(serverAddrFuture).get();

  // Connect a client but don't send any data (simulating hung client)
  ctran::bootstrap::Socket silentClient;
  ASSERT_EQ(0, silentClient.connect(serverAddr, "lo"));
  XLOG(INFO) << "Client connected to server at " << serverAddr.describe()
             << " but not sending data";

  // Wait for timeout to occur (timeout is 500ms, wait a bit longer to ensure
  // timeout fires)
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));

  // Verify the receive callback was NOT invoked (timeout happened before data
  // arrived)
  EXPECT_FALSE(recvCallbackInvoked.load())
      << "Receive callback should not be invoked when timeout occurs";

  // Cleanup
  silentClient.close();
  auto fut = server.stop();
  std::move(fut).get();
}

TEST(AsyncSocket, AsyncServerSocketReceivePartialDataTimeout) {
  const size_t expectedSize = 100;
  auto eventThread = std::make_unique<folly::ScopedEventBaseThread>();

  std::atomic<bool> recvCallbackInvoked(false);

  ctran::bootstrap::AsyncServerSocket server(*eventThread->getEventBase());

  // Start server expecting 100 bytes with 500ms timeout
  auto serverAddrFuture = server.start(
      folly::SocketAddress("::1", 0),
      expectedSize,
      [&recvCallbackInvoked](std::unique_ptr<folly::IOBuf> buf) {
        // This callback should NOT be invoked if we only send partial data
        XLOGF(INFO, "Server received {} bytes", buf->computeChainDataLength());
        recvCallbackInvoked.store(true);
      },
      std::chrono::milliseconds(500)); // 500ms timeout

  auto serverAddr = std::move(serverAddrFuture).get();

  // Connect client and send only partial data
  ctran::bootstrap::Socket partialClient;
  ASSERT_EQ(0, partialClient.connect(serverAddr, "lo"));

  // Send only 50 bytes when server expects 100 bytes
  const std::string partialData(50, 'x');
  ASSERT_EQ(0, partialClient.send(partialData.data(), partialData.size()));
  XLOGF(
      INFO,
      "Client sent {} bytes (partial), server expects {} bytes",
      partialData.size(),
      expectedSize);

  // Wait for timeout to occur
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));

  // Verify the receive callback was NOT invoked (incomplete data received
  // before timeout)
  EXPECT_FALSE(recvCallbackInvoked.load())
      << "Receive callback should not be invoked when incomplete data is "
         "received and timeout occurs";

  // Cleanup
  partialClient.close();
  auto fut = server.stop();
  std::move(fut).get();
}
