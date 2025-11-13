// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <chrono>
#include <optional>
#include <semaphore>
#include <thread>

#include <sys/socket.h>

#include <gtest/gtest.h>

#include <folly/logging/xlog.h>
#include <folly/synchronization/Baton.h>
#include <folly/synchronization/Latch.h>

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

//
// Test Group #2: Multi-Client and Connection Management
//

TEST_F(AbortableServerSocketTest, MultipleClients) {
  ctran::bootstrap::AbortableSocket client1;
  ctran::bootstrap::AbortableSocket client2;

  ASSERT_EQ(0, client1.connect(serverAddr, "lo"));
  ASSERT_EQ(0, client2.connect(serverAddr, "lo"));

  auto maybeAccepted1 = server->acceptSocket();
  ASSERT_FALSE(maybeAccepted1.hasError());
  auto& accepted1 = maybeAccepted1.value();

  auto maybeAccepted2 = server->acceptSocket();
  ASSERT_FALSE(maybeAccepted2.hasError());
  auto& accepted2 = maybeAccepted2.value();

  const std::string msg1 = "client1";
  const std::string msg2 = "client2";

  ASSERT_EQ(0, client1.send(msg1.data(), msg1.size()));
  ASSERT_EQ(0, client2.send(msg2.data(), msg2.size()));

  char buffer1[msg1.size()];
  char buffer2[msg2.size()];

  int err1 = accepted1->recv(buffer1, sizeof(buffer1));
  EXPECT_EQ(err1, 0);

  int err2 = accepted2->recv(buffer2, sizeof(buffer2));
  EXPECT_EQ(err2, 0);

  EXPECT_EQ(std::string(buffer1), msg1);
  EXPECT_EQ(std::string(buffer2), msg2);

  EXPECT_EQ(0, client1.close());
  EXPECT_EQ(0, client2.close());
  EXPECT_EQ(0, accepted1->close());
  EXPECT_EQ(0, accepted2->close());
}

TEST_F(AbortableSocketTest, ConnectionRefused) {
  std::binary_semaphore sem{0};

  std::thread connectThread([&]() {
    folly::SocketAddress unreachableAddr("::1", 9999);
    int result = client.connect(unreachableAddr, "lo", 100ms, 1);
    EXPECT_EQ(result, ECONNABORTED);
    sem.release();
  });

  for (int i = 0; i < 10; i++) {
    std::this_thread::sleep_for(100ms);
    if (sem.try_acquire_for(1ms)) {
      ASSERT_FALSE(true) << "Connection should not have succeeded";
    }
  }

  clientAbort->Set();
  connectThread.join();
  EXPECT_TRUE(sem.try_acquire_for(1ms));
  EXPECT_TRUE(clientAbort->Test());
}

TEST_F(AbortableSocketTest, MultipleConnectionAttempts) {
  ctran::bootstrap::AbortableServerSocket server{1};

  // Bind server on the loopback interface but do not listen
  XLOG(INFO) << "Binding server..";
  ASSERT_EQ(0, server.bind(folly::SocketAddress("::1", 0), "lo"));
  const auto& maybeServerAddr = server.getListenAddress();
  ASSERT_FALSE(maybeServerAddr.hasError());
  auto& serverAddr = maybeServerAddr.value();

  // Connect client to the server. It may experience few connect errors but
  // retry will eventually make it succeed
  XLOG(INFO) << "Connecting to server..";
  auto abortCtrl = ctran::utils::createAbort(/*enabled=*/true);
  ctran::bootstrap::AbortableSocket client1(abortCtrl);
  std::atomic<int> result{-1};
  std::thread connectThread([&]() {
    result = client1.connect(serverAddr, "lo", 0ms /*ignored*/, 0 /*ignored*/);
  });

  std::this_thread::sleep_for(500ms);
  EXPECT_EQ(result.load(), -1);
  abortCtrl->Set();
  connectThread.join();
  EXPECT_EQ(result.load(), ECONNABORTED);
}

TEST_F(AbortableSocketTest, ConnectionEventuallySucceeds) {
  ctran::bootstrap::AbortableServerSocket server{1};

  // Bind server on the loopback interface but do not listen
  XLOG(INFO) << "Binding server..";
  ASSERT_EQ(0, server.bind(folly::SocketAddress("::1", 0), "lo"));
  const auto& maybeServerAddr = server.getListenAddress();
  ASSERT_FALSE(maybeServerAddr.hasError());
  auto& serverAddr = maybeServerAddr.value();

  folly::Baton connecting;

  // Delay the listen in a separate thread to simulate a connect error
  std::thread listenThread([&]() {
    connecting.wait();
    std::this_thread::sleep_for(100ms);
    XLOG(INFO) << "Starting to listen on server";
    ASSERT_EQ(0, server.listen());
  });

  ctran::bootstrap::AbortableSocket client2;
  std::this_thread::sleep_for(100ms);
  connecting.post();

  // Attempt to connect to server again .. we may succeed after few more retries
  XLOG(INFO) << "Attempting to connect to server again..";
  ASSERT_EQ(
      0, client2.connect(serverAddr, "lo", 0ms /*ignored*/, 0 /*ignored*/));
  EXPECT_NE(client2.getFd(), -1);

  // Accept client connection on server
  listenThread.join();
  auto maybeClient = server.acceptSocket();
  ASSERT_FALSE(maybeClient.hasError()) << maybeClient.error();
  auto& acceptedClient = maybeClient.value();
  EXPECT_NE(acceptedClient, nullptr);
  EXPECT_NE(acceptedClient->getFd(), -1);

  // Close the sockets
  EXPECT_EQ(0, server.shutdown());
  EXPECT_EQ(0, client2.close());
  EXPECT_EQ(0, acceptedClient->close());
}

TEST_F(AbortableSocketTest, LargeDataTransfer) {
  const size_t dataSize = 1 * 1024 * 1024;
  std::vector<uint8_t> sendData(dataSize);
  for (size_t i = 0; i < dataSize; i++) {
    sendData[i] = i % 256;
  }

  std::vector<uint8_t> recvData(dataSize);

  std::thread sendThread(
      [&]() { ASSERT_EQ(0, client.send(sendData.data(), sendData.size())); });

  ASSERT_EQ(0, acceptedClient->recv(recvData.data(), recvData.size()));

  sendThread.join();

  EXPECT_EQ(sendData, recvData);
}

//
// Test Group #3: Socket Binding and Port Management
//

TEST_F(AbortableSocketTest, ReusePort) {
  ctran::bootstrap::AbortableServerSocket server1{1};
  ctran::bootstrap::AbortableServerSocket server2{1};

  ASSERT_EQ(0, server1.bind(folly::SocketAddress("::1", 0), "lo", true));
  const auto maybeAddr1 = server1.getListenAddress();
  ASSERT_FALSE(maybeAddr1.hasError());
  auto addr1 = maybeAddr1.value();

  ASSERT_EQ(0, server2.bind(addr1, "lo"));
  const auto maybeAddr2 = server2.getListenAddress();
  ASSERT_FALSE(maybeAddr2.hasError());
  auto addr2 = maybeAddr2.value();

  EXPECT_EQ(addr1.getPort(), addr2.getPort());

  EXPECT_EQ(0, server1.shutdown());
  EXPECT_EQ(0, server2.shutdown());
}

TEST_F(AbortableSocketTest, BindAndUnbind) {
  ctran::bootstrap::AbortableServerSocket server1{1};
  ctran::bootstrap::AbortableServerSocket server2{1};

  // Bind server1 on the loopback interface with SO_REUSEPORT
  ASSERT_EQ(0, server1.bind(folly::SocketAddress("::1", 0), "lo", true));
  EXPECT_NE(server1.getFd(), -1);

  // Get the server1 address and validate it
  const auto maybeServer1Addr = server1.getListenAddress();
  ASSERT_FALSE(maybeServer1Addr.hasError());
  auto server1Addr = maybeServer1Addr.value();
  EXPECT_EQ(server1Addr.getFamily(), AF_INET6);
  EXPECT_NE(server1Addr.getPort(), 0);
  EXPECT_EQ(server1Addr.getIPAddress().str(), "::1");

  // Bind server2 to the same address as server1 (works due to SO_REUSEPORT)
  ASSERT_EQ(0, server2.bind(server1Addr, "lo"));
  EXPECT_NE(server2.getFd(), -1);
  EXPECT_EQ(server2.getListenAddress().value(), server1Addr);

  // Shutdown server1 and create a new server3 to bind to the same address
  ASSERT_EQ(0, server1.shutdown());
  EXPECT_EQ(-1, server1.getFd());

  ctran::bootstrap::AbortableServerSocket server3{1};
  ASSERT_EQ(0, server3.bind(server1Addr, "lo"));
  EXPECT_NE(server3.getFd(), -1);
  EXPECT_EQ(server3.getListenAddress().value(), server1Addr);

  // Close the remaining sockets
  EXPECT_EQ(0, server2.shutdown());
  EXPECT_EQ(-1, server2.getFd());
  EXPECT_EQ(0, server3.shutdown());
  EXPECT_EQ(-1, server3.getFd());
}

TEST_F(AbortableServerSocketTest, AcceptTimeout) {
  std::chrono::milliseconds timeout{250ms};
  serverAbort->SetTimeout(timeout);

  auto startTime = std::chrono::steady_clock::now();
  auto maybeClient = server->acceptSocket();
  auto elapsedMilliseconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - startTime);

  EXPECT_TRUE(maybeClient.hasError());
  EXPECT_TRUE(
      maybeClient.error() == ETIMEDOUT || maybeClient.error() == ECONNABORTED);
  ASSERT_GE(elapsedMilliseconds, 100ms)
      << "Elapsed milliseconds: " << elapsedMilliseconds.count();
  ASSERT_LT(elapsedMilliseconds, timeout + 100ms)
      << "Elapsed milliseconds: " << elapsedMilliseconds.count();
}

//
// Test Group #4: Timeout Functionality
//

// Helper function to test timeout on blocking I/O operations
// Sets a timeout on the abort object, runs the operation, measures elapsed
// time, and verifies the operation timed out within the expected bounds
template <typename OperationFn>
void testTimeoutOperation(
    std::shared_ptr<ctran::utils::Abort> abortObj,
    std::chrono::milliseconds timeout,
    OperationFn operation,
    std::optional<std::chrono::milliseconds> minElapsed = std::nullopt,
    std::optional<std::chrono::milliseconds> maxElapsed = std::nullopt) {
  abortObj->SetTimeout(timeout);
  ASSERT_TRUE(abortObj->HasTimeout());

  auto startTime = std::chrono::steady_clock::now();
  int result = operation();
  auto elapsedMilliseconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - startTime);

  EXPECT_TRUE(result == ECONNABORTED || result == ETIMEDOUT)
      << "Result: " << result;

  if (minElapsed) {
    EXPECT_GE(elapsedMilliseconds, *minElapsed)
        << "Elapsed milliseconds: " << elapsedMilliseconds.count();
  }

  if (maxElapsed) {
    EXPECT_LT(elapsedMilliseconds, *maxElapsed)
        << "Elapsed milliseconds: " << elapsedMilliseconds.count();
  }
}

TEST_F(AbortableSocketTest, SendTimeout) {
  const size_t largeSize = 10 * 1024 * 1024;
  std::vector<uint8_t> largeBuffer(largeSize, 0xAB);

  std::atomic<bool> continueRecv{true};
  std::thread clientThread([&]() {
    while (continueRecv.load()) {
      char buffer[1024];
      acceptedClient->recv(buffer, sizeof(buffer));
      std::this_thread::sleep_for(25ms);
    }
  });

  testTimeoutOperation(
      clientAbort,
      50ms,
      [&]() { return client.send(largeBuffer.data(), largeBuffer.size()); },
      std::nullopt,
      500ms);

  continueRecv = false;
  clientThread.join();
}

TEST_F(AbortableSocketTest, RecvTimeout) {
  auto timeout = 100ms;

  testTimeoutOperation(
      serverAbort,
      timeout,
      [&]() {
        char buffer[1024];
        return acceptedClient->recv(buffer, sizeof(buffer));
      },
      timeout,
      timeout * 2);
}

//
// Test Group #5: Close, Shutdown, and Resource Management
//

TEST_F(AbortableSocketTest, CloseWhileOperationPending) {
  std::thread recvThread([&]() {
    char buffer[100];
    int result = acceptedClient->recv(buffer, sizeof(buffer));
    EXPECT_NE(result, 0);
    XLOG(INFO, "RecvThread is exiting");
  });

  std::this_thread::sleep_for(100ms);
  client.close();

  recvThread.join();
}

TEST_F(AbortableServerSocketTest, AcceptErrorOnShutdown) {
  folly::Baton listenStarted;

  std::thread listenThread([this, &listenStarted]() {
    listenStarted.post();
    auto maybeClient = server->acceptSocket();
    ASSERT_TRUE(maybeClient.hasError());
    EXPECT_TRUE(
        maybeClient.error() == EBADF || maybeClient.error() == EINVAL ||
        maybeClient.error() == EIO || maybeClient.error() == ETIMEDOUT)
        << "Error: " << maybeClient.error();
  });
  listenStarted.wait();
  std::this_thread::sleep_for(50ms);
  server->shutdown();
  listenThread.join();
}

TEST_F(AbortableSocketTest, CloseThreadSafety) {
  EXPECT_NE(acceptedClient->getFd(), -1);

  // Launch multiple threads that all try to close the same socket
  constexpr int kNumThreads = 10;
  folly::Latch sync_point(kNumThreads);
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  std::vector<int> results(kNumThreads);
  std::atomic<int> successCount{0};

  for (int i = 0; i < kNumThreads; i++) {
    threads.emplace_back([this, &results, i, &successCount, &sync_point]() {
      sync_point.arrive_and_wait();
      results[i] = acceptedClient->close();
      XLOGF(INFO, "Thread #{} closed socket with result={}", i, results[i]);
      if (results[i] == 0) {
        successCount++;
      }
    });
  }

  // Wait for all threads to complete
  for (auto& thread : threads) {
    thread.join();
  }

  // Verify thread safety:
  // At least one thread should successfully close (return 0)
  EXPECT_GE(successCount.load(), 1)
      << "Expected at least one thread to successfully close";

  // The socket should be closed (fd == -1)
  EXPECT_EQ(acceptedClient->getFd(), -1);
}

TEST_F(AbortableServerSocketTest, ShutdownThreadSafety) {
  EXPECT_FALSE(server->hasShutDown());

  // Launch multiple threads that all try to shutdown the same server
  constexpr int kNumThreads = 10;
  folly::Latch sync_point(kNumThreads);
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  std::vector<int> results(kNumThreads);
  std::atomic<int> successCount{0};

  for (int i = 0; i < kNumThreads; i++) {
    threads.emplace_back([this, &results, i, &successCount, &sync_point]() {
      sync_point.arrive_and_wait();
      results[i] = server->shutdown();
      if (results[i] == 0) {
        successCount++;
      }
    });
  }

  // Wait for all threads to complete
  for (auto& thread : threads) {
    thread.join();
  }

  // Verify thread safety:
  // 1. Exactly one thread should have succeeded (returned 0)
  EXPECT_GE(successCount.load(), 1)
      << "Expected at least one thread to successfully shutdown";

  // 2. The server should be marked as shut down
  EXPECT_TRUE(server->hasShutDown());

  // 3. The file descriptor should be invalidated
  EXPECT_EQ(server->getFd(), -1);
}

//
// Test Group #6: Abort Functionality
//

// Helper function to test aborting blocking operations
// Takes an operation function and an abort object, runs the operation in a
// separate thread, aborts it after a delay, and verifies it was aborted
template <typename OperationFn>
void testAbortBlockingOperation(
    std::shared_ptr<ctran::utils::Abort> abortObj,
    OperationFn operation,
    std::chrono::milliseconds delayBeforeAbort = 150ms) {
  std::atomic<int> operationResult{-1};
  std::binary_semaphore operationStarted{0};

  // Start operation in a separate thread
  std::thread operationThread([&]() {
    operationStarted.release();
    operationResult = operation();
  });

  // Wait for operation to start
  operationStarted.acquire();

  // Give it some time to get into blocking state
  std::this_thread::sleep_for(delayBeforeAbort);

  EXPECT_FALSE(abortObj->Test());

  // Abort the operation
  abortObj->Set();

  // Wait for operation thread to complete
  operationThread.join();

  // Verify that operation was aborted
  EXPECT_TRUE(
      operationResult.load() == ECONNABORTED ||
      operationResult.load() == ETIMEDOUT);
}

// Test aborting connect operation with retries
TEST_F(AbortableSocketTest, AbortOnConnectWaiting) {
  auto abortObj = ctran::utils::createAbort(/*enabled=*/true);
  ctran::bootstrap::AbortableSocket client(abortObj);
  folly::SocketAddress unreachableAddr("::1", 9999);

  testAbortBlockingOperation(abortObj, [&]() {
    return client.connect(unreachableAddr, "lo", 0ms, 0);
  });

  EXPECT_EQ(0, client.close());
  EXPECT_EQ(client.getFd(), -1);
}

TEST_F(AbortableSocketTest, AbortOnAcceptWaiting) {
  testAbortBlockingOperation(
      serverAbort,
      [&]() {
        auto maybeClient = server->acceptSocket();
        return maybeClient.hasError() ? maybeClient.error() : 0;
      },
      250ms);
}

TEST_F(AbortableSocketTest, AbortAcceptConcurrent) {
  std::atomic<int> acceptError{-1};

  // Start accept in a separate thread
  std::thread acceptThread([&]() {
    auto maybeClient = server->acceptSocket();
    if (maybeClient.hasError()) {
      acceptError = maybeClient.error();
    } else {
      acceptError = 0;
    }
  });

  std::this_thread::sleep_for(250ms);
  serverAbort->Set();

  // Wait for threads to complete
  acceptThread.join();

  // Verify that accept was aborted
  EXPECT_TRUE(
      acceptError.load() == ECONNABORTED || acceptError.load() == ETIMEDOUT);
}

// Test timeout during connect operation
// This validates that the timeout mechanism properly triggers abort during
// connection establishment attempts
TEST_F(AbortableSocketTest, ConnectTimeout) {
  auto abort = ctran::utils::createAbort(/*enabled=*/true);
  ctran::bootstrap::AbortableSocket client(abort);

  // Try to connect to unreachable address with short timeout
  folly::SocketAddress unreachableAddr("::1", 9999);

  // Set timeout that will expire during connection attempts
  abort->SetTimeout(100ms);

  auto startTime = std::chrono::steady_clock::now();
  int result = client.connect(unreachableAddr, "lo");
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - startTime);

  // Should have timed out via the abort mechanism
  EXPECT_TRUE(result == ECONNABORTED || result == ETIMEDOUT);
  EXPECT_TRUE(abort->Test());
  EXPECT_TRUE(abort->TimedOut());

  // Should complete within reasonable time after timeout
  EXPECT_GE(elapsed, 100ms);
  EXPECT_LT(elapsed, 500ms);
}

TEST_F(AbortableSocketTest, AbortOnSend) {
  const size_t largeSize = 10 * 1024 * 1024;
  std::vector<uint8_t> largeBuffer(largeSize, 0xAB);

  testAbortBlockingOperation(
      clientAbort,
      [&]() { return client.send(largeBuffer.data(), largeBuffer.size()); },
      50ms);
}

TEST_F(AbortableSocketTest, AbortOnRecv) {
  const size_t bufferSize = 2 * 1024 * 1024;
  std::vector<uint8_t> buffer(bufferSize);

  testAbortBlockingOperation(
      serverAbort,
      [&]() { return acceptedClient->recv(buffer.data(), buffer.size()); },
      50ms);
}

// Test aborting send/recv operation concurrently
TEST_F(AbortableSocketTest, AbortSendRecvConcurrent) {
  std::atomic<int> recvResult{0};
  std::atomic<int> numSends{0};
  std::atomic<int> numRecvs{0};
  std::binary_semaphore recvStarted{0};
  std::binary_semaphore sendStarted{0};
  int messageLimit = 25;

  // Start send in separate thread
  std::thread sendThread([&]() {
    int sendResult = 0;
    sendStarted.release();

    do {
      const size_t sendSize = 2048;
      std::vector<uint8_t> sendBuffer(sendSize, 0xAB);
      sendResult = acceptedClient->send(sendBuffer.data(), sendBuffer.size());
      if (sendResult != 0 || numSends.load() >= messageLimit) {
        break;
      }
      std::this_thread::sleep_for(10ms);
      numSends += 1;
      XLOG(INFO) << "Sent message #" << numSends.load()
                 << ". sendResult=" << sendResult;
    } while (true);
  });

  // Start recv in separate thread
  std::thread recvThread([&]() {
    recvStarted.release();
    do {
      char buffer[2048];
      recvResult = client.recv(buffer, sizeof(buffer));
      numRecvs += 1;
      XLOG(INFO) << "Received message #" << numRecvs.load()
                 << ". recvResult=" << recvResult;
      if (recvResult.load() != 0 || numRecvs.load() >= messageLimit) {
        break;
      }
      std::this_thread::sleep_for(10ms);
    } while (true);
  });

  recvStarted.acquire();
  sendStarted.acquire();

  std::this_thread::sleep_for(50ms);

  clientAbort->Set();
  serverAbort->Set();
  EXPECT_TRUE(clientAbort->Test());

  if (recvThread.joinable()) {
    recvThread.join();
  }
  if (sendThread.joinable()) {
    sendThread.join();
  }

  ASSERT_GT(numSends.load(), 0);
  ASSERT_GT(numRecvs.load(), 0);

  ASSERT_LT(
      numRecvs.load(),
      messageLimit); // Should have aborted before receiving all
                     // 'messageLimit' messages.

  ASSERT_LT(
      numSends.load(), messageLimit); // Should have aborted before sending
                                      // all 'messageLimit' messages.

  // Verify recv was affected by abort
  EXPECT_TRUE(recvResult.load() == ECONNABORTED || recvResult.load() == EBADF)
      << "recvResult: " << recvResult.load();
}

// Helper function to test that operations fail immediately after abort is set
// Sets the abort, runs the operation, and verifies it fails quickly with
// ECONNABORTED
template <typename OperationFn>
void testOperationAfterAbort(
    std::shared_ptr<ctran::utils::Abort> abortObj,
    OperationFn operation,
    std::chrono::milliseconds maxElapsed = 500ms) {
  abortObj->Set();

  auto startTime = std::chrono::steady_clock::now();
  int result = operation();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - startTime);

  EXPECT_EQ(result, ECONNABORTED);
  EXPECT_LT(elapsed, maxElapsed)
      << "Operation should fail quickly after abort, elapsed: "
      << elapsed.count() << "ms";
}

TEST_F(AbortableSocketTest, SendAfterAbort) {
  const std::string message = "test";
  testOperationAfterAbort(clientAbort, [&]() {
    return client.send(message.data(), message.size());
  });
}

TEST_F(AbortableSocketTest, RecvAfterAbort) {
  char buffer[1024];
  testOperationAfterAbort(serverAbort, [&]() {
    return acceptedClient->recv(buffer, sizeof(buffer));
  });
}

TEST_F(AbortableSocketTest, ConnectAfterAbort) {
  auto abortObj = ctran::utils::createAbort(/*enabled=*/true);
  ctran::bootstrap::AbortableSocket client(abortObj);
  folly::SocketAddress unreachableAddr("::1", 9999);

  testOperationAfterAbort(abortObj, [&]() {
    return client.connect(unreachableAddr, "lo", 0ms, 0);
  });

  EXPECT_EQ(0, client.close());
}

TEST_F(AbortableSocketTest, AcceptAfterAbort) {
  testOperationAfterAbort(serverAbort, [&]() {
    auto maybeClient = server->acceptSocket();
    return maybeClient.hasError() ? maybeClient.error() : 0;
  });
}
