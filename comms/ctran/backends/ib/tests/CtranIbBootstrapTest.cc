// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

#include <folly/SocketAddress.h>
#include <folly/futures/Future.h>
#include <folly/synchronization/Baton.h>

#include "comms/ctran/backends/ib/CtranIb.h"
#include "comms/ctran/bootstrap/AbortableSocket.h"
#include "comms/ctran/bootstrap/ISocketFactory.h"
#include "comms/ctran/bootstrap/Socket.h"
#include "comms/ctran/bootstrap/tests/MockIServerSocket.h"
#include "comms/ctran/bootstrap/tests/MockISocket.h"
#include "comms/ctran/bootstrap/tests/MockInjectorSocketFactory.h"
#include "comms/ctran/tests/CtranStandaloneUTUtils.h"
#include "comms/ctran/utils/Abort.h"
#include "comms/utils/cvars/nccl_cvars.h"

using AbortPtr = std::shared_ptr<ctran::utils::Abort>;
using ::testing::_;
using ::testing::StrictMock;

// Type alias for socket factory creation function
using SocketFactoryCreator =
    std::function<std::shared_ptr<ctran::bootstrap::ISocketFactory>()>;

struct TestParam {
  std::string name;
  SocketFactoryCreator socketFactoryCreator;
};

// Helper to validate and return listen address
folly::SocketAddress getAndValidateListenAddr(CtranIb* ctranIb) {
  auto maybeListenAddr = ctranIb->getListenSocketListenAddr();
  EXPECT_FALSE(maybeListenAddr.hasError());
  auto listenAddr = maybeListenAddr.value();
  EXPECT_GT(listenAddr.getPort(), 0);
  return listenAddr;
}

// Helper to create a SocketServerAddr with default localhost settings
SocketServerAddr getSocketServerAddress(
    const int port = 0, // Let OS assign port
    const char* ipv4 = "127.0.0.1",
    const char* ifName = "lo") {
  SocketServerAddr serverAddr;
  serverAddr.port = port;
  serverAddr.ipv4 = ipv4;
  serverAddr.ifName = ifName;
  return serverAddr;
}

// Base test class without parameterization for non-parameterized tests
class CtranIbBootstrapTestBase : public ::testing::Test {
 public:
  CtranIbBootstrapTestBase() = default;

 protected:
  void SetUp() override {
    ncclCvarInit();

    EXPECT_EQ(cudaSetDevice(0), cudaSuccess); // Initialize CUDA devices

    int deviceCount;
    EXPECT_EQ(cudaGetDeviceCount(&deviceCount), cudaSuccess);
    ASSERT_FALSE(deviceCount <= 1)
        << "Test requires at least 2 CUDA devices, found " << deviceCount;
  }

  void TearDown() override {
    cudaDeviceReset(); // Reset CUDA device
  }

  // Helper to wait for VC to be established
  bool waitForVcEstablished(
      CtranIb* ctranIb,
      int peerRank,
      std::chrono::milliseconds timeout = std::chrono::seconds(5)) {
    auto start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start < timeout) {
      auto vc = ctranIb->getVc(peerRank);
      if (vc != nullptr) {
        return true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return false;
  }

  // Helper to create CtranIb for all tests
  std::unique_ptr<CtranIb> createCtranIb(
      int rank,
      CtranIb::BootstrapMode mode,
      AbortPtr abortCtrl,
      std::optional<const SocketServerAddr*> qpServerAddr = std::nullopt,
      std::shared_ptr<ctran::bootstrap::ISocketFactory> socketFactory =
          nullptr) {
    const uint64_t commHash = 0x12345678;
    const std::string commDesc = "test";

    if (socketFactory == nullptr) {
      socketFactory =
          std::make_shared<ctran::bootstrap::AbortableSocketFactory>();
    }

    return std::make_unique<CtranIb>(
        rank,
        rank, // Use rank as CUDA device identifier
        commHash,
        commDesc,
        nullptr, // ctrlMgr
        false, // enableLocalFlush
        mode,
        qpServerAddr,
        abortCtrl,
        socketFactory);
  }
};

// Parameterized test fixture
class CtranIbBootstrapParameterizedTest
    : public CtranIbBootstrapTestBase,
      public ::testing::WithParamInterface<TestParam> {
 protected:
  std::pair<std::unique_ptr<CtranIb>, AbortPtr> createCtranIbAndAbort(
      int rank,
      CtranIb::BootstrapMode mode,
      bool abortEnabled = true,
      std::optional<const SocketServerAddr*> qpServerAddr = std::nullopt) {
    auto abortCtrl = ctran::utils::createAbort(abortEnabled);

    auto param = GetParam();
    auto socketFactory = param.socketFactoryCreator();

    auto ctranIb =
        createCtranIb(rank, mode, abortCtrl, qpServerAddr, socketFactory);

    return std::pair<std::unique_ptr<CtranIb>, AbortPtr>(
        std::move(ctranIb), abortCtrl);
  }
};

class CtranIbBootstrapCommonTest : public CtranIbBootstrapTestBase {
 protected:
  void SetUp() override {
    CtranIbBootstrapTestBase::SetUp();
  }

  void TearDown() override {
    CtranIbBootstrapTestBase::TearDown();
  }
};

// Test basic bootstrapStart functionality
TEST_P(CtranIbBootstrapParameterizedTest, BootstrapStartDefaultServer) {
  auto [ctranIb, abortCtrl] = createCtranIbAndAbort(
      /*rank=*/0, CtranIb::BootstrapMode::kDefaultServer, true);

  getAndValidateListenAddr(ctranIb.get());
}

// Test bootstrapStart with specified server address
TEST_P(CtranIbBootstrapParameterizedTest, BootstrapStartSpecifiedServer) {
  SocketServerAddr serverAddr = getSocketServerAddress();
  auto [ctranIb, abortCtrl] = createCtranIbAndAbort(
      /*rank=*/0, CtranIb::BootstrapMode::kSpecifiedServer, true, &serverAddr);

  auto listenAddr = getAndValidateListenAddr(ctranIb.get());
  EXPECT_GT(listenAddr.getPort(), 0);
  EXPECT_EQ(listenAddr.getIPAddress().str(), "127.0.0.1");
}

TEST_P(CtranIbBootstrapParameterizedTest, BootstrapSendRecvCtrlMsg) {
  auto [listenAddrPromise0, listenAddrFuture0] =
      folly::makePromiseContract<folly::SocketAddress>();
  auto [listenAddrPromise1, listenAddrFuture1] =
      folly::makePromiseContract<folly::SocketAddress>();

  std::thread rank0Thread([&]() {
    EXPECT_EQ(cudaSetDevice(0), cudaSuccess);

    SocketServerAddr serverAddr0 = getSocketServerAddress(0, "127.0.0.1", "lo");

    auto [ctranIb0, abortCtrl] = createCtranIbAndAbort(
        /*rank=*/0,
        CtranIb::BootstrapMode::kSpecifiedServer,
        false,
        &serverAddr0);

    auto listenAddr = getAndValidateListenAddr(ctranIb0.get());
    listenAddrPromise0.setValue(listenAddr);

    // Wait for peer's listen address
    auto peerAddr = std::move(listenAddrFuture1).get();

    // Since rank 0 < rank 1, rank 0 should initiate the connection
    SocketServerAddr peerServerAddr = getSocketServerAddress(
        peerAddr.getPort(), peerAddr.getIPAddress().str().c_str(), "lo");

    ControlMsg msg;
    CtranIbRequest ctrlReq;
    CtranIbEpochRAII epochRAII(ctranIb0.get());

    commResult_t result = ctranIb0->isendCtrlMsg(
        msg.type, &msg, sizeof(msg), 1, ctrlReq, &peerServerAddr);

    EXPECT_EQ(result, commSuccess);

    bool established =
        waitForVcEstablished(ctranIb0.get(), 1, std::chrono::seconds(5));
    EXPECT_TRUE(established);

    do {
      auto res = ctranIb0->progress();
      EXPECT_EQ(res, commSuccess);
    } while (!ctrlReq.isComplete());
    EXPECT_TRUE(ctrlReq.isComplete());
  });

  std::thread rank1Thread([&]() {
    EXPECT_EQ(cudaSetDevice(1), cudaSuccess);

    // Create CtranIb for rank 1 with specified server
    SocketServerAddr serverAddr1 = getSocketServerAddress(0, "127.0.0.1", "lo");

    auto [ctranIb1, abortCtrl] = createCtranIbAndAbort(
        /*rank=*/1,
        CtranIb::BootstrapMode::kSpecifiedServer,
        false,
        &serverAddr1);

    auto listenAddr = getAndValidateListenAddr(ctranIb1.get());
    listenAddrPromise1.setValue(listenAddr);

    // Wait for peer's listen address (needed for synchronization)
    auto peerListenAddr = std::move(listenAddrFuture0).get();
    SocketServerAddr peerServerAddr = getSocketServerAddress(
        peerListenAddr.getPort(),
        peerListenAddr.getIPAddress().str().c_str(),
        "lo");

    ControlMsg msg;
    CtranIbRequest ctrlReq;
    ctranIb1->irecvCtrlMsg(&msg, sizeof(msg), 0, ctrlReq, &peerServerAddr);

    // Rank 1 waits for rank 0 to connect (handled by bootstrapAccept
    // thread) Wait for connection to be established
    bool established =
        waitForVcEstablished(ctranIb1.get(), 0, std::chrono::seconds(5));
    EXPECT_TRUE(established);

    do {
      auto res = ctranIb1->progress();
      EXPECT_EQ(res, commSuccess);
    } while (!ctrlReq.isComplete());
  });

  rank0Thread.join();
  rank1Thread.join();
}

TEST_F(CtranIbBootstrapCommonTest, AbortExplicitSendCtrlMsg) {
  auto abortCtrl = ctran::utils::createAbort(/*enabled=*/true);
  SocketServerAddr serverAddr = getSocketServerAddress();
  auto ctranIb = createCtranIb(
      /*rank=*/0,
      CtranIb::BootstrapMode::kSpecifiedServer,
      abortCtrl,
      &serverAddr);

  folly::Baton abortThreadStarted;

  // Set up a timer to abort after a short delay
  std::thread abortThread([&]() {
    abortThreadStarted.post();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    abortCtrl->Set();
  });

  // Try to connect to a non-existent server
  SocketServerAddr invalidServerAddr = getSocketServerAddress(
      12346 /* no server listening */, "127.0.0.1", "lo");

  abortThreadStarted.wait();
  auto start = std::chrono::steady_clock::now();

  ControlMsg msg;
  CtranIbRequest ctrlReq;
  CtranIbEpochRAII epochRAII(ctranIb.get());

  commResult_t result = ctranIb->isendCtrlMsg(
      msg.type, &msg, sizeof(msg), 1, ctrlReq, &invalidServerAddr);

  auto elapsed = std::chrono::steady_clock::now() - start;

  // Should abort quickly
  EXPECT_LT(elapsed, std::chrono::seconds(5));
  EXPECT_NE(result, commSuccess);

  // Verify abort flag is set
  EXPECT_TRUE(abortCtrl->Test());

  abortThread.join();
}

TEST_F(CtranIbBootstrapCommonTest, AbortTimeoutSendCtrlMsg) {
  auto abortCtrl = ctran::utils::createAbort(/*enabled=*/true);
  SocketServerAddr serverAddr = getSocketServerAddress();
  auto ctranIb = createCtranIb(
      /*rank=*/0,
      CtranIb::BootstrapMode::kSpecifiedServer,
      abortCtrl,
      &serverAddr);

  abortCtrl->SetTimeout(std::chrono::milliseconds(250));

  // Try to connect to a non-existent server
  SocketServerAddr invalidServerAddr = getSocketServerAddress(
      12346 /* no server listening */, "127.0.0.1", "lo");

  auto start = std::chrono::steady_clock::now();

  ControlMsg msg;
  CtranIbRequest ctrlReq;
  CtranIbEpochRAII epochRAII(ctranIb.get());

  commResult_t result = ctranIb->isendCtrlMsg(
      msg.type, &msg, sizeof(msg), 1, ctrlReq, &invalidServerAddr);

  auto elapsed = std::chrono::steady_clock::now() - start;

  // Should abort quickly
  EXPECT_LT(elapsed, std::chrono::seconds(5));
  EXPECT_NE(result, commSuccess);
  EXPECT_TRUE(abortCtrl->Test());
}

// Test that bootstrap respects magic number validation
TEST_P(CtranIbBootstrapParameterizedTest, InvalidMagicNumberRejection) {
  SocketServerAddr serverAddr = getSocketServerAddress();

  auto [ctranIb, abortCtrl] = createCtranIbAndAbort(
      /*rank=*/0, CtranIb::BootstrapMode::kSpecifiedServer, false, &serverAddr);

  auto listenAddr = getAndValidateListenAddr(ctranIb.get());

  // Create a client socket that sends invalid magic number
  std::thread clientThread([&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    auto param = GetParam();
    auto socketFactory = param.socketFactoryCreator();

    auto clientSocket = socketFactory->createClientSocket(abortCtrl);

    folly::SocketAddress serverAddrFolly;
    serverAddrFolly.setFromIpPort(
        listenAddr.getIPAddress().str(), listenAddr.getPort());

    int result = clientSocket->connect(
        serverAddrFolly, "lo", std::chrono::milliseconds(1000), 5);

    if (result == 0) {
      // Send invalid magic number
      uint64_t invalidMagic = 0xDEADBEEFCAFEBABE;
      clientSocket->send(&invalidMagic, sizeof(uint64_t));

      // Send rank (though connection should be rejected)
      int rank = 1;
      clientSocket->send(&rank, sizeof(int));

      // Give server time to process and reject
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
  });

  // Listen thread should receive connection but reject it due to invalid magic
  // The VC should not be established
  std::this_thread::sleep_for(std::chrono::seconds(1));

  // Verify VC was not established
  auto vc = ctranIb->getVc(1);
  EXPECT_EQ(vc, nullptr);

  clientThread.join();
}

// Enum to specify which control message operation to test
enum class CtrlMsgOperation { Send, Recv };

// Parameterized test fixture for control message abort testing
class CtranIbAbortCtrlMsgTest
    : public CtranIbBootstrapTestBase,
      public ::testing::WithParamInterface<CtrlMsgOperation> {
 protected:
  static constexpr int kPeerRank = 1;
  static constexpr int kPeerPort = 12346;

  void SetUp() override {
    CtranIbBootstrapTestBase::SetUp();
    EXPECT_EQ(cudaSetDevice(0), cudaSuccess);
    abortCtrl_ = ctran::utils::createAbort(/*enabled=*/true);
  }

  void TearDown() override {
    CtranIbBootstrapTestBase::TearDown();

    acceptSocketBaton_.post();
    mockServerSockets_.clear();
    mockSockets_.clear();
    ctranIb_.reset();
  }

  std::unique_ptr<StrictMock<ctran::bootstrap::testing::MockIServerSocket>>
  prepareMockIServerSocket(
      std::unique_ptr<StrictMock<ctran::bootstrap::testing::MockIServerSocket>>
          mockServerSocket = nullptr) {
    if (!mockServerSocket) {
      mockServerSocket = std::make_unique<
          StrictMock<ctran::bootstrap::testing::MockIServerSocket>>();
    }

    EXPECT_CALL(*mockServerSocket, shutdown()).WillOnce([]() { return 0; });

    EXPECT_CALL(*mockServerSocket, hasShutDown()).WillOnce([]() {
      return false;
    });

    EXPECT_CALL(*mockServerSocket, acceptSocket()).WillOnce([this]() {
      acceptSocketBaton_.wait(); // Shouldn't return immediately.
      // Return some error because this call shouldn't return a valid socket.
      return folly::makeUnexpected(EAGAIN);
    });

    EXPECT_CALL(*mockServerSocket, bindAndListen(::testing::_, ::testing::_))
        .WillOnce([](const folly::SocketAddress& addr,
                     const std::string& ifName) { return 0; });

    return mockServerSocket;
  }

  // Helper to create a valid remote bus card for testing
  std::string createValidRemoteBusCard() {
    // BusCard structure matches the one in CtranIbVc.cc
    // CTRAN_HARDCODED_MAX_QPS is defined in CtranIbVc.cc as 128
    // NOTE: This struct is intentionally duplicated from CtranIbVc.cc
    // because BusCard is an internal implementation detail not exposed
    // in any header. This must be kept in sync manually...
    // See CtranIbVc.cc for the canonical definition.
    constexpr int kCtranHardcodedMaxQps = 128;

    struct BusCard {
      enum ibverbx::ibv_mtu mtu;
      uint32_t controlQpn;
      uint32_t notifQpn;
      uint32_t atomicQpn;
      uint32_t dataQpn[kCtranHardcodedMaxQps];
      uint8_t ports[CTRAN_MAX_IB_DEVICES_PER_RANK];
      union {
        struct {
          uint64_t spns[CTRAN_MAX_IB_DEVICES_PER_RANK];
          uint64_t iids[CTRAN_MAX_IB_DEVICES_PER_RANK];
        } eth;
        struct {
          uint16_t lids[CTRAN_MAX_IB_DEVICES_PER_RANK];
        } ib;
      } u;
    };

    BusCard busCard{};
    busCard.mtu = ibverbx::IBV_MTU_4096;
    busCard.controlQpn = 100;
    busCard.notifQpn = 101;
    busCard.atomicQpn = 102;

    for (int i = 0; i < kCtranHardcodedMaxQps; i++) {
      busCard.dataQpn[i] = 200 + i;
    }

    for (int i = 0; i < NCCL_CTRAN_IB_DEVICES_PER_RANK; i++) {
      busCard.ports[i] = 1; // Port 1 is typical for IB devices
    }

    for (int i = 0; i < NCCL_CTRAN_IB_DEVICES_PER_RANK; i++) {
      busCard.u.eth.spns[i] = 0xfe80000000000000ULL; // Link-local subnet prefix
      busCard.u.eth.iids[i] = 0x0000000000000001ULL + i; // Interface ID
    }

    return std::string(
        reinterpret_cast<const char*>(&busCard), sizeof(BusCard));
  }

  // Execute a test for control message operations (send or recv) with abort
  //
  // CtranIb throws an std::runtime_error when a socket operation returns a
  // non-zero error code and the abortCtrl_ is unset.
  void testAbortedCtrlMsg(
      std::unique_ptr<StrictMock<ctran::bootstrap::testing::MockISocket>>
          mockSocket,
      bool shouldFail = true) {
    // Setup
    SocketServerAddr serverAddr = getSocketServerAddress();
    mockSockets_.push_back(std::move(mockSocket));
    auto mockServerSocket = prepareMockIServerSocket();
    mockServerSockets_.push_back(std::move(mockServerSocket));

    auto socketFactory =
        std::make_shared<ctran::bootstrap::testing::MockInjectorSocketFactory>(
            std::move(mockSockets_), std::move(mockServerSockets_));

    ctranIb_ = createCtranIb(
        0,
        CtranIb::BootstrapMode::kSpecifiedServer,
        abortCtrl_,
        &serverAddr,
        socketFactory);

    // Execute
    ControlMsg msg;
    CtranIbRequest req;
    CtranIbEpochRAII epochRAII(ctranIb_.get());

    auto start = std::chrono::steady_clock::now();

    SocketServerAddr peerServerAddr =
        getSocketServerAddress(kPeerPort, "127.0.0.1", "lo");

    commResult_t result;
    auto operation = GetParam();
    if (operation == CtrlMsgOperation::Send) {
      result = ctranIb_->isendCtrlMsg(
          msg.type, &msg, sizeof(msg), kPeerRank, req, &peerServerAddr);
    } else {
      result = ctranIb_->irecvCtrlMsg(
          &msg, sizeof(msg), kPeerRank, req, &peerServerAddr);
    }

    XLOGF(INFO, "i(send|recv)CtrlMsg received result={}", result);

    auto elapsed = std::chrono::steady_clock::now() - start;

    // Verify
    if (shouldFail) {
      EXPECT_NE(result, commSuccess);
    } else {
      EXPECT_EQ(result, commSuccess);
    }

    EXPECT_LT(elapsed, std::chrono::seconds(5)) << "Abort should fail quickly";

    XLOGF(
        INFO,
        "Control message {} {}",
        operation == CtrlMsgOperation::Send ? "send" : "recv",
        shouldFail ? "aborted as expected" : "completed successfully");
  }

  std::unique_ptr<CtranIb> ctranIb_;
  folly::Baton<> acceptSocketBaton_;
  std::shared_ptr<ctran::utils::Abort> abortCtrl_;
  std::vector<std::unique_ptr<ctran::bootstrap::testing::MockISocket>>
      mockSockets_;
  std::vector<std::unique_ptr<ctran::bootstrap::testing::MockIServerSocket>>
      mockServerSockets_;
};

TEST_P(CtranIbAbortCtrlMsgTest, SocketConnectError) {
  auto mockSocket =
      std::make_unique<StrictMock<ctran::bootstrap::testing::MockISocket>>();

  EXPECT_CALL(*mockSocket, connect(_, _, _, _, _))
      .WillRepeatedly([](const folly::SocketAddress& addr,
                         const std::string& ifName,
                         const std::chrono::milliseconds timeout,
                         size_t numRetries,
                         bool async) { return ECONNABORTED; });

  testAbortedCtrlMsg(std::move(mockSocket), true);
  EXPECT_FALSE(abortCtrl_->Test());
}

TEST_P(CtranIbAbortCtrlMsgTest, AbortDuringSocketConnect) {
  auto mockSocket =
      std::make_unique<StrictMock<ctran::bootstrap::testing::MockISocket>>();

  EXPECT_CALL(*mockSocket, connect(_, _, _, _, _))
      .WillOnce([this](
                    const folly::SocketAddress& addr,
                    const std::string& ifName,
                    const std::chrono::milliseconds timeout,
                    size_t numRetries,
                    bool async) {
        abortCtrl_->Set();
        return 0; // Only trigger abort; don't return error code here...
      });

  EXPECT_CALL(*mockSocket, send(_, _))
      .WillRepeatedly(
          [&](const void* buf, const size_t len) { return ECONNABORTED; });

  EXPECT_CALL(*mockSocket, recv(_, _))
      .WillRepeatedly(
          [&](const void* buf, const size_t len) { return ECONNABORTED; });

  testAbortedCtrlMsg(std::move(mockSocket), true);
  EXPECT_TRUE(abortCtrl_->Test());
}

TEST_P(CtranIbAbortCtrlMsgTest, AbortOnSocketSend) {
  auto mockSocket =
      std::make_unique<StrictMock<ctran::bootstrap::testing::MockISocket>>();

  EXPECT_CALL(*mockSocket, connect(_, _, _, _, _))
      .WillRepeatedly([&](const folly::SocketAddress& addr,
                          const std::string& ifName,
                          const std::chrono::milliseconds timeout,
                          size_t numRetries,
                          bool async) { return 0; });

  EXPECT_CALL(*mockSocket, send(_, _))
      .WillRepeatedly([&](const void* buf, const size_t len) {
        if (abortCtrl_->Test()) {
          // Once abort is triggered, should return errcode.
          return ECONNABORTED;
        }
        abortCtrl_->Set();
        return 0; // Only trigger abort; don't return error code here...
      });

  EXPECT_CALL(*mockSocket, recv(_, _))
      .WillRepeatedly([&](const void* buf, const size_t len) {
        if (abortCtrl_->Test()) {
          // Once abort is triggered, should return errcode.
          return ECONNABORTED;
        }
        return 0;
      });

  testAbortedCtrlMsg(std::move(mockSocket), true);
  EXPECT_TRUE(abortCtrl_->Test());
}

TEST_P(CtranIbAbortCtrlMsgTest, SocketSendError) {
  auto mockSocket =
      std::make_unique<StrictMock<ctran::bootstrap::testing::MockISocket>>();

  EXPECT_CALL(*mockSocket, connect(_, _, _, _, _))
      .WillOnce([&](const folly::SocketAddress& addr,
                    const std::string& ifName,
                    const std::chrono::milliseconds timeout,
                    size_t numRetries,
                    bool async) { return 0; });

  EXPECT_CALL(*mockSocket, recv(_, _))
      .WillRepeatedly([&](const void* buf, const size_t len) { return 0; });

  EXPECT_CALL(*mockSocket, send(_, _))
      .WillRepeatedly(
          [&](const void* buf, const size_t len) { return ECONNABORTED; });

  testAbortedCtrlMsg(std::move(mockSocket), true);
  EXPECT_FALSE(abortCtrl_->Test());
}

TEST_P(CtranIbAbortCtrlMsgTest, SocketRecvError) {
  auto mockSocket =
      std::make_unique<StrictMock<ctran::bootstrap::testing::MockISocket>>();

  EXPECT_CALL(*mockSocket, connect(_, _, _, _, _))
      .WillRepeatedly([&](const folly::SocketAddress& addr,
                          const std::string& ifName,
                          const std::chrono::milliseconds timeout,
                          size_t numRetries,
                          bool async) { return 0; });

  EXPECT_CALL(*mockSocket, send(_, _))
      .WillRepeatedly([&](const void* buf, const size_t len) { return 0; });

  EXPECT_CALL(*mockSocket, recv(_, _))
      .WillRepeatedly(
          [&](const void* buf, const size_t len) { return ECONNABORTED; });

  testAbortedCtrlMsg(std::move(mockSocket), true);
  EXPECT_FALSE(abortCtrl_->Test());
}

TEST_P(CtranIbAbortCtrlMsgTest, NoAbortNoError) {
  auto mockSocket =
      std::make_unique<StrictMock<ctran::bootstrap::testing::MockISocket>>();

  // Create a valid remote bus card
  std::string remoteBusCard = createValidRemoteBusCard();

  EXPECT_CALL(*mockSocket, connect(_, _, _, _, _))
      .WillOnce([&](const folly::SocketAddress& addr,
                    const std::string& ifName,
                    const std::chrono::milliseconds timeout,
                    size_t numRetries,
                    bool async) { return 0; });

  // Send operations: magic number, rank, local bus card, final ack
  EXPECT_CALL(*mockSocket, send(_, _))
      .WillRepeatedly([&](const void* buf, const size_t len) { return 0; });

  // Recv operation: populate buffer with valid remote bus card
  EXPECT_CALL(*mockSocket, recv(_, _))
      .WillRepeatedly([&](void* buf, const size_t len) {
        // Copy the remote bus card into the provided buffer
        std::memcpy(
            buf, remoteBusCard.data(), std::min(len, remoteBusCard.size()));
        return 0;
      });

  testAbortedCtrlMsg(std::move(mockSocket), false);
  EXPECT_FALSE(abortCtrl_->Test());
}

// Instantiate parameterized tests with both Socket and AbortableSocket
// implementations
INSTANTIATE_TEST_SUITE_P(
    SocketTypes,
    CtranIbBootstrapParameterizedTest,
    ::testing::Values(
        TestParam{
            "Socket_Test",
            // Test with SocketFactory (blocking Socket)
            []() -> std::shared_ptr<ctran::bootstrap::ISocketFactory> {
              return std::make_shared<ctran::bootstrap::SocketFactory>();
            }},
        // Test with AbortableSocketFactory (AbortableSocket)
        TestParam{
            "AbortableSocket_Test",
            []() -> std::shared_ptr<ctran::bootstrap::ISocketFactory> {
              return std::make_shared<
                  ctran::bootstrap::AbortableSocketFactory>();
            }}),
    // Test name generator
    [](const ::testing::TestParamInfo<TestParam>& info) {
      return info.param.name;
    });

// Instantiate control message abort tests for both Send and Recv operations
INSTANTIATE_TEST_SUITE_P(
    CtrlMsgOperations,
    CtranIbAbortCtrlMsgTest,
    ::testing::Values(CtrlMsgOperation::Send, CtrlMsgOperation::Recv),
    // Test name generator
    [](const ::testing::TestParamInfo<CtrlMsgOperation>& info) {
      return info.param == CtrlMsgOperation::Send ? "Send" : "Recv";
    });

using StrictMockIServerSocketPtr =
    std::unique_ptr<ctran::bootstrap::testing::MockIServerSocket>;
using MockInjectorSocketFactoryPtr =
    std::shared_ptr<ctran::bootstrap::testing::MockInjectorSocketFactory>;

struct PreparedMockedServerSocket {
  AbortPtr abortCtrl;
  std::shared_ptr<folly::Baton<>> acceptCalledBaton;
  std::shared_ptr<folly::Baton<>> unblockAcceptBaton;
  std::shared_ptr<std::atomic_flag> hasShutDown;
  MockInjectorSocketFactoryPtr socketFactory;
};

// Prepare a MockIServerSocket for use in the AbortDuringListenThreadAccept,
// RapidShutdownNoConnections, and ListenThreadTerminatesOnShutdown unit tests.
std::shared_ptr<PreparedMockedServerSocket> prepareMockIServerSocket(
    bool unblockAcceptInShutdown,
    bool abortEnabled = false) {
  auto prepared = std::make_shared<PreparedMockedServerSocket>();
  prepared->abortCtrl = ctran::utils::createAbort(/*enabled=*/abortEnabled);
  prepared->acceptCalledBaton = std::make_shared<folly::Baton<>>();
  prepared->unblockAcceptBaton = std::make_shared<folly::Baton<>>();
  prepared->hasShutDown = std::make_shared<std::atomic_flag>();

  auto mockedServerSocket = std::make_unique<
      StrictMock<ctran::bootstrap::testing::MockIServerSocket>>();

  // Capture shared pointers for use in in lambdas
  auto acceptCalledBaton = prepared->acceptCalledBaton;
  auto unblockAcceptBaton = prepared->unblockAcceptBaton;
  auto hasShutDown = prepared->hasShutDown;

  EXPECT_CALL(*mockedServerSocket, bindAndListen(_, _))
      .WillOnce([](const folly::SocketAddress& addr,
                   const std::string& ifName) { return 0; });

  EXPECT_CALL(*mockedServerSocket, hasShutDown()).WillOnce([hasShutDown]() {
    return hasShutDown->test();
  });

  // Accept should block until shutdown is called
  EXPECT_CALL(*mockedServerSocket, acceptSocket())
      .WillOnce([acceptCalledBaton, unblockAcceptBaton]() {
        acceptCalledBaton->post();
        unblockAcceptBaton->wait(); // Wait for signal to continue
        return folly::makeUnexpected(ECONNABORTED);
      });

  EXPECT_CALL(*mockedServerSocket, shutdown())
      .WillOnce([hasShutDown, unblockAcceptBaton, unblockAcceptInShutdown]() {
        hasShutDown->test_and_set();
        // Note that the internals here are tested by the AbortableSocket UTs.
        if (unblockAcceptInShutdown) {
          // Shutdown should unblock accept
          unblockAcceptBaton->post();
        }
        return 0;
      });

  std::vector<StrictMockIServerSocketPtr> mockServerSockets;
  mockServerSockets.push_back(std::move(mockedServerSocket));

  prepared->socketFactory =
      std::make_shared<ctran::bootstrap::testing::MockInjectorSocketFactory>(
          std::move(mockServerSockets));

  return prepared;
}

// Test that abort during listen thread's accept loop exits cleanly
TEST_F(CtranIbBootstrapCommonTest, AbortDuringListenThreadAccept) {
  auto preparedServerSocket = prepareMockIServerSocket(
      /*unblockAcceptInShutdown=*/false, /*abortEnabled=*/true);

  SocketServerAddr serverAddr = getSocketServerAddress();

  // Create CtranIb - this starts the listen thread
  auto ctranIb = createCtranIb(
      /*rank=*/0,
      CtranIb::BootstrapMode::kSpecifiedServer,
      preparedServerSocket->abortCtrl,
      &serverAddr,
      preparedServerSocket->socketFactory);

  // Wait for accept to be called
  preparedServerSocket->acceptCalledBaton->wait();
  preparedServerSocket->unblockAcceptBaton->post(); // Allow accept to continue

  // The HANDLE_SOCKET_ERROR macro will call abort().
  // So, wait up to 10sec for this to occur. Should happen quickly.
  auto start = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start);
  while (!preparedServerSocket->abortCtrl->Test() && elapsed.count() < 10000) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
  }

  EXPECT_TRUE(preparedServerSocket->abortCtrl->Test());

  // Destroy CtranIb
  auto startDestroy = std::chrono::steady_clock::now();
  ctranIb.reset();

  // Verify thread joined quickly (not hanging)
  auto destroyTime = std::chrono::steady_clock::now() - startDestroy;
  EXPECT_LT(destroyTime, std::chrono::seconds(2))
      << "Listen thread should terminate quickly after abort";
}

// Test that listen thread terminates cleanly via shutdown
TEST_F(CtranIbBootstrapCommonTest, ListenThreadTerminatesOnShutdown) {
  auto preparedMockIServerSocket = prepareMockIServerSocket(
      /*unblockAcceptInShutdown=*/true, /*abortEnabled=*/true);

  SocketServerAddr serverAddr = getSocketServerAddress();

  {
    auto ctranIb = createCtranIb(
        /*rank=*/0,
        CtranIb::BootstrapMode::kSpecifiedServer,
        preparedMockIServerSocket->abortCtrl,
        &serverAddr,
        preparedMockIServerSocket->socketFactory);

    // Wait for accept to be called
    preparedMockIServerSocket->acceptCalledBaton->wait();

    // Now destroy CtranIb - this triggers shutdown and should join thread
    auto startDestroy = std::chrono::steady_clock::now();
    ctranIb.reset();
    auto elapsed = std::chrono::steady_clock::now() - startDestroy;

    // Verify destruction completed quickly
    EXPECT_LT(elapsed, std::chrono::seconds(2))
        << "Destructor should complete quickly after shutdown";
  }

  // If we get here without hanging, the test passed
  EXPECT_TRUE(preparedMockIServerSocket->unblockAcceptBaton->ready())
      << "Shutdown should have been called";
  EXPECT_FALSE(preparedMockIServerSocket->abortCtrl->Test());
}
