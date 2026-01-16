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
#include "comms/ctran/tests/CtranTestUtils.h"
#include "comms/ctran/utils/Abort.h"
#include "comms/utils/cvars/nccl_cvars.h"

using AbortPtr = std::shared_ptr<ctran::utils::Abort>;
using namespace std::literals::chrono_literals;
using ::testing::_;
using ::testing::StrictMock;

// Type alias for socket factory creation function
using SocketFactoryCreator =
    std::function<std::shared_ptr<ctran::bootstrap::ISocketFactory>()>;

struct TestParam {
  std::string name;
  SocketFactoryCreator socketFactoryCreator;
};

// Helper to wait for VC to be established
bool waitForVcEstablished(
    CtranIb* ctranIb,
    int peerRank,
    std::chrono::milliseconds timeout = 5s) {
  auto start = std::chrono::steady_clock::now();
  while (std::chrono::steady_clock::now() - start < timeout) {
    auto vc = ctranIb->getVc(peerRank);
    if (vc != nullptr) {
      return true;
    }
    std::this_thread::sleep_for(10ms);
  }
  return false;
}

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

commResult_t sendCtrlMsg(
    CtranIb* ctranIb,
    int peerRank = 1,
    std::optional<const SocketServerAddr*> peerServerAddr = std::nullopt) {
  ControlMsg msg;
  CtranIbRequest ctrlReq;
  CtranIbEpochRAII epochRAII(ctranIb);
  commResult_t result = ctranIb->isendCtrlMsg(
      msg.type, &msg, sizeof(msg), peerRank, ctrlReq, peerServerAddr);

  EXPECT_EQ(result, commSuccess);

  do {
    auto res = ctranIb->progress();
    EXPECT_EQ(res, commSuccess);
  } while (!ctrlReq.isComplete());

  bool established = waitForVcEstablished(ctranIb, peerRank, 5s);
  EXPECT_TRUE(established);

  return result;
}

commResult_t recvCtrlMsg(
    CtranIb* ctranIb,
    int peerRank = 1,
    std::optional<const SocketServerAddr*> peerServerAddr = std::nullopt) {
  ControlMsg msg;
  CtranIbRequest ctrlReq;
  CtranIbEpochRAII epochRAII(ctranIb);
  commResult_t result = ctranIb->irecvCtrlMsg(
      &msg, sizeof(msg), peerRank, ctrlReq, peerServerAddr);

  EXPECT_EQ(result, commSuccess);

  do {
    auto res = ctranIb->progress();
    EXPECT_EQ(res, commSuccess);
  } while (!ctrlReq.isComplete());

  bool established = waitForVcEstablished(ctranIb, peerRank, 5s);
  EXPECT_TRUE(established);

  return result;
}

// Helper to create CtranIb for all tests
std::unique_ptr<CtranIb> createCtranIb(
    int rank,
    CtranIb::BootstrapMode mode,
    AbortPtr abortCtrl,
    std::optional<const SocketServerAddr*> qpServerAddr = std::nullopt,
    std::shared_ptr<ctran::bootstrap::ISocketFactory> socketFactory = nullptr) {
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
// Helper class to run two-rank tests with address exchange
class TwoRankTestHelper {
 public:
  using RankAction = std::function<void(
      CtranIb* ctranIb,
      const folly::SocketAddress& peerAddr,
      AbortPtr abortCtrl)>;

  TwoRankTestHelper(RankAction rank0Action, RankAction rank1Action)
      : rank0Action_(std::move(rank0Action)),
        rank1Action_(std::move(rank1Action)) {}

  void run() {
    auto [listenAddrPromise0, listenAddrFuture0] =
        folly::makePromiseContract<folly::SocketAddress>();
    auto [listenAddrPromise1, listenAddrFuture1] =
        folly::makePromiseContract<folly::SocketAddress>();

    std::thread rank0Thread([&]() {
      EXPECT_EQ(cudaSetDevice(0), cudaSuccess);
      SocketServerAddr serverAddr = getSocketServerAddress();
      auto abortCtrl = ctran::utils::createAbort(/*enabled=*/true);
      auto ctranIb = createCtranIb(
          0, CtranIb::BootstrapMode::kSpecifiedServer, abortCtrl, &serverAddr);

      auto listenAddr = getAndValidateListenAddr(ctranIb.get());
      listenAddrPromise0.setValue(listenAddr);

      auto peerAddr = std::move(listenAddrFuture1).get();
      rank0Action_(ctranIb.get(), peerAddr, abortCtrl);
    });

    std::thread rank1Thread([&]() {
      EXPECT_EQ(cudaSetDevice(1), cudaSuccess);
      SocketServerAddr serverAddr = getSocketServerAddress();
      auto abortCtrl = ctran::utils::createAbort(/*enabled=*/true);
      auto ctranIb = createCtranIb(
          1, CtranIb::BootstrapMode::kSpecifiedServer, abortCtrl, &serverAddr);

      auto listenAddr = getAndValidateListenAddr(ctranIb.get());
      listenAddrPromise1.setValue(listenAddr);

      auto peerAddr = std::move(listenAddrFuture0).get();
      rank1Action_(ctranIb.get(), peerAddr, abortCtrl);
    });

    rank0Thread.join();
    rank1Thread.join();
  }

 private:
  RankAction rank0Action_;
  RankAction rank1Action_;
};

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
      std::chrono::milliseconds timeout = 5000ms) {
    auto start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start < timeout) {
      auto vc = ctranIb->getVc(peerRank);
      if (vc != nullptr) {
        return true;
      }
      std::this_thread::sleep_for(10ms);
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

// Test that NCCL_SOCKET_IFNAME with a single interface works correctly
TEST_F(CtranIbBootstrapCommonTest, SingleInterfaceInSocketIfnameSucceeds) {
  std::vector<std::string> originalIfname = NCCL_SOCKET_IFNAME;
  SCOPE_EXIT {
    NCCL_SOCKET_IFNAME = originalIfname;
  };

  NCCL_SOCKET_IFNAME = {"lo"}; // Single interface in vector
  auto abortCtrl = ctran::utils::createAbort(/*enabled=*/true);

  // Should not throw; single interface is valid
  auto ctranIb = createCtranIb(
      /*rank=*/0, CtranIb::BootstrapMode::kDefaultServer, abortCtrl);
  getAndValidateListenAddr(ctranIb.get());
}

// Test that empty NCCL_SOCKET_IFNAME does not trigger the multi-interface error
// (it should fail later with "No socket interfaces found" instead)
TEST_F(
    CtranIbBootstrapCommonTest,
    EmptySocketIfnameDoesNotTriggerMultiIfError) {
  std::vector<std::string> originalIfname = NCCL_SOCKET_IFNAME;
  SCOPE_EXIT {
    NCCL_SOCKET_IFNAME = originalIfname;
  };

  NCCL_SOCKET_IFNAME = {}; // Empty vector
  auto abortCtrl = ctran::utils::createAbort(/*enabled=*/true);

  // Empty vector does not have more than 1 interface, so the multi-interface
  // check passes.
  try {
    auto ctranIb = createCtranIb(
        /*rank=*/0, CtranIb::BootstrapMode::kDefaultServer, abortCtrl);
    getAndValidateListenAddr(ctranIb.get()); // Creation succeeded
  } catch (const ::ctran::utils::Exception& e) {
    // If it throws, verify it's NOT the multi-interface error
    std::string errorMsg = e.what();
    EXPECT_EQ(
        errorMsg.find("should specify only one interface"), std::string::npos)
        << "Empty NCCL_SOCKET_IFNAME should not trigger multi-interface error";
  }
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
  auto rank0Action = [this](
                         CtranIb* ctranIb,
                         const folly::SocketAddress& peerAddr,
                         AbortPtr abortCtrl) {
    SocketServerAddr peerServerAddr = getSocketServerAddress(
        peerAddr.getPort(), peerAddr.getIPAddress().str().c_str(), "lo");

    sendCtrlMsg(ctranIb, /*peerRank=*/1, &peerServerAddr);
  };

  auto rank1Action = [this](
                         CtranIb* ctranIb,
                         const folly::SocketAddress& peerAddr,
                         AbortPtr abortCtrl) {
    SocketServerAddr peerServerAddr = getSocketServerAddress(
        peerAddr.getPort(), peerAddr.getIPAddress().str().c_str(), "lo");

    recvCtrlMsg(ctranIb, 0, &peerServerAddr);
  };

  TwoRankTestHelper(rank0Action, rank1Action).run();
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
    std::this_thread::sleep_for(100ms);
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
      msg.type, &msg, sizeof(msg), /*peerRank=*/1, ctrlReq, &invalidServerAddr);

  auto elapsed = std::chrono::steady_clock::now() - start;

  // Should abort quickly
  EXPECT_LT(elapsed, 5s);
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

  abortCtrl->SetTimeout(250ms);

  // Try to connect to a non-existent server
  SocketServerAddr invalidServerAddr = getSocketServerAddress(
      12346 /* no server listening */, "127.0.0.1", "lo");

  auto start = std::chrono::steady_clock::now();

  ControlMsg msg;
  CtranIbRequest ctrlReq;
  CtranIbEpochRAII epochRAII(ctranIb.get());
  commResult_t result = ctranIb->isendCtrlMsg(
      msg.type, &msg, sizeof(msg), /*peerRank=*/1, ctrlReq, &invalidServerAddr);
  auto elapsed = std::chrono::steady_clock::now() - start;

  // Should abort quickly
  EXPECT_LT(elapsed, 5s);
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
    std::this_thread::sleep_for(100ms);

    auto param = GetParam();
    auto socketFactory = param.socketFactoryCreator();

    auto clientSocket = socketFactory->createClientSocket(abortCtrl);

    folly::SocketAddress serverAddrFolly;
    serverAddrFolly.setFromIpPort(
        listenAddr.getIPAddress().str(), listenAddr.getPort());

    int result = clientSocket->connect(serverAddrFolly, "lo", 1s, 5);

    if (result == 0) {
      // Send invalid magic number
      uint64_t invalidMagic = 0xDEADBEEFCAFEBABE;
      clientSocket->send(&invalidMagic, sizeof(uint64_t));

      // Send rank (though connection should be rejected)
      int rank = 1;
      clientSocket->send(&rank, sizeof(int));

      // Give server time to process and reject
      std::this_thread::sleep_for(500ms);
    }
  });

  // Listen thread should receive connection but reject it due to invalid magic
  // The VC should not be established
  std::this_thread::sleep_for(1s);

  // Verify VC was not established
  auto vc = ctranIb->getVc(1);
  EXPECT_EQ(vc, nullptr);

  clientThread.join();
}

// Enum to specify which control message operation to test
enum class CtrlMsgOperation { Send, Recv };

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

  return std::string(reinterpret_cast<const char*>(&busCard), sizeof(BusCard));
}

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

    EXPECT_LT(elapsed, 5s) << "Abort should fail quickly";

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
    std::this_thread::sleep_for(10ms);
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
  }

  EXPECT_TRUE(preparedServerSocket->abortCtrl->Test());

  // Destroy CtranIb
  auto startDestroy = std::chrono::steady_clock::now();
  ctranIb.reset();

  // Verify thread joined quickly (not hanging)
  auto destroyTime = std::chrono::steady_clock::now() - startDestroy;
  EXPECT_LT(destroyTime, 2s)
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
    EXPECT_LT(elapsed, 2s)
        << "Destructor should complete quickly after shutdown";
  }

  // If we get here without hanging, the test passed
  EXPECT_TRUE(preparedMockIServerSocket->unblockAcceptBaton->ready())
      << "Shutdown should have been called";
  EXPECT_FALSE(preparedMockIServerSocket->abortCtrl->Test());
}

// Test that abort during bus card exchange fails gracefully
TEST_F(CtranIbBootstrapCommonTest, AbortDuringBusCardExchange) {
  auto abortCtrl = ctran::utils::createAbort(/*enabled=*/true);
  folly::Baton<> acceptSocketBaton;

  auto mockSocket =
      std::make_unique<StrictMock<ctran::bootstrap::testing::MockISocket>>();

  // Socket connects successfully
  EXPECT_CALL(*mockSocket, connect(_, _, _, _, _))
      .WillOnce([](const folly::SocketAddress& addr,
                   const std::string& ifName,
                   const std::chrono::milliseconds timeout,
                   size_t numRetries,
                   bool async) { return 0; });

  EXPECT_CALL(*mockSocket, send(_, _))
      .WillOnce([&](const void* buf, const size_t len) { return 0; })
      .WillOnce([&](const void* buf, const size_t len) { return 0; })
      .WillOnce([&](const void* buf, const size_t len) {
        // Third send (bus card) triggers abort and fails.
        abortCtrl->Set();
        return ECONNABORTED;
      })
      .WillRepeatedly([&](const void* buf, const size_t len) {
        // Any additional sends should also fail.
        return ECONNABORTED;
      });

  EXPECT_CALL(*mockSocket, recv(_, _))
      .WillRepeatedly(
          [&](void* buf, const size_t len) { return ECONNABORTED; });

  auto mockServerSocket = std::make_unique<
      StrictMock<ctran::bootstrap::testing::MockIServerSocket>>();

  EXPECT_CALL(*mockServerSocket, bindAndListen(_, _))
      .WillOnce([](const folly::SocketAddress& addr,
                   const std::string& ifName) { return 0; });

  EXPECT_CALL(*mockServerSocket, hasShutDown()).WillOnce([]() {
    return false;
  });

  EXPECT_CALL(*mockServerSocket, acceptSocket()).WillOnce([&]() {
    acceptSocketBaton.wait();
    return folly::makeUnexpected(ECONNABORTED);
  });

  EXPECT_CALL(*mockServerSocket, shutdown()).WillOnce([&]() {
    acceptSocketBaton.post();
    return 0;
  });

  std::vector<std::unique_ptr<ctran::bootstrap::testing::MockISocket>>
      mockSockets;
  mockSockets.push_back(std::move(mockSocket));

  std::vector<std::unique_ptr<ctran::bootstrap::testing::MockIServerSocket>>
      mockServerSockets;
  mockServerSockets.push_back(std::move(mockServerSocket));

  auto socketFactory =
      std::make_shared<ctran::bootstrap::testing::MockInjectorSocketFactory>(
          std::move(mockSockets), std::move(mockServerSockets));

  SocketServerAddr serverAddr = getSocketServerAddress();
  auto ctranIb = createCtranIb(
      /*rank=*/0,
      CtranIb::BootstrapMode::kSpecifiedServer,
      abortCtrl,
      &serverAddr,
      socketFactory);

  // Try to connect to peer - should fail during bus card exchange
  SocketServerAddr peerServerAddr =
      getSocketServerAddress(12347, "127.0.0.1", "lo");

  ControlMsg msg;
  CtranIbRequest req;
  CtranIbEpochRAII epochRAII(ctranIb.get());

  auto start = std::chrono::steady_clock::now();
  commResult_t result = ctranIb->isendCtrlMsg(
      msg.type, &msg, sizeof(msg), 1, req, &peerServerAddr);

  while (!req.isComplete()) {
    auto res = ctranIb->progress();
    EXPECT_EQ(res, commSuccess);

    auto elapsed = std::chrono::steady_clock::now() - start;

    // Keep trying to progress request for ~5 seconds before giving up.
    // Just need to confirm that it doesn't complete.
    if (elapsed > 1s) {
      break;
    }
  }

  auto elapsed = std::chrono::steady_clock::now() - start;

  // Verify abort happened quickly
  EXPECT_LT(elapsed, 10s);
  EXPECT_NE(result, commSuccess);
  EXPECT_TRUE(abortCtrl->Test());

  // Verify VC was not established
  auto vc = ctranIb->getVc(1);
  EXPECT_EQ(vc, nullptr) << "VC should not be created after abort during "
                            "bus card exchange";
}

// Test aborting waitNotify operation
TEST_F(CtranIbBootstrapCommonTest, AbortWaitNotify) {
  auto rank0Action = [this](
                         CtranIb* ctranIb,
                         const folly::SocketAddress& peerAddr,
                         AbortPtr abortCtrl) {
    SocketServerAddr peerServerAddr = getSocketServerAddress(
        peerAddr.getPort(), peerAddr.getIPAddress().str().c_str(), "lo");

    sendCtrlMsg(ctranIb, /*peerRank=*/1, &peerServerAddr);

    // Set timeout to abort waitNotify
    abortCtrl->SetTimeout(std::chrono::milliseconds(500));

    XLOG(INFO) << "Rank 0 calling ctranIb->waitNotify(1, 1)";

    // Try to wait for a notify that will never come
    auto startWaitNotify = std::chrono::steady_clock::now();
    EXPECT_THROW(ctranIb->waitNotify(1, 1), ::ctran::utils::Exception);
    auto elapsed = std::chrono::steady_clock::now() - startWaitNotify;

    // Should abort after timeout
    EXPECT_LT(elapsed, std::chrono::seconds(5));
    EXPECT_TRUE(abortCtrl->Test());
    XLOGF(INFO, "Rank 0 waitNotify aborted by timeout as expected");
  };

  auto rank1Action = [this](
                         CtranIb* ctranIb,
                         const folly::SocketAddress& peerAddr,
                         AbortPtr abortCtrl) {
    SocketServerAddr peerServerAddr = getSocketServerAddress(
        peerAddr.getPort(), peerAddr.getIPAddress().str().c_str(), "lo");

    recvCtrlMsg(ctranIb, 0, &peerServerAddr);

    // Don't send notifications - let rank 0 timeout
    std::this_thread::sleep_for(std::chrono::seconds(2));

    XLOGF(INFO, "Rank 1 completed without sending notify");
  };

  TwoRankTestHelper(rank0Action, rank1Action).run();
}

// Test multiple sequential control messages over the same connection
TEST_P(CtranIbBootstrapParameterizedTest, MultipleSequentialCtrlMsg) {
  auto rank0Action = [this](
                         CtranIb* ctranIb,
                         const folly::SocketAddress& peerAddr,
                         AbortPtr abortCtrl) {
    SocketServerAddr peerServerAddr = getSocketServerAddress(
        peerAddr.getPort(), peerAddr.getIPAddress().str().c_str(), "lo");

    // Send first control message
    sendCtrlMsg(ctranIb, /*peerRank=*/1, &peerServerAddr);

    // Send second control message over the same established connection
    sendCtrlMsg(ctranIb, /*peerRank=*/1, &peerServerAddr);

    // Send third control message
    sendCtrlMsg(ctranIb, /*peerRank=*/1, &peerServerAddr);

    XLOG(INFO) << "Rank 0 successfully sent 3 sequential control messages";
  };

  auto rank1Action = [this](
                         CtranIb* ctranIb,
                         const folly::SocketAddress& peerAddr,
                         AbortPtr abortCtrl) {
    SocketServerAddr peerServerAddr = getSocketServerAddress(
        peerAddr.getPort(), peerAddr.getIPAddress().str().c_str(), "lo");

    // Receive three control messages
    for (int i = 0; i < 3; ++i) {
      recvCtrlMsg(ctranIb, 0, &peerServerAddr);

      XLOGF(INFO, "Rank 1 received control message {}", i + 1);
    }
  };

  TwoRankTestHelper(rank0Action, rank1Action).run();
}

// Test bidirectional communication (both ranks send and receive)
TEST_P(CtranIbBootstrapParameterizedTest, BidirectionalCtrlMsg) {
  folly::Baton rank0Complete, rank1Complete;

  auto rank0Action = [this, &rank0Complete, &rank1Complete](
                         CtranIb* ctranIb,
                         const folly::SocketAddress& peerAddr,
                         AbortPtr abortCtrl) {
    SocketServerAddr peerServerAddr = getSocketServerAddress(
        peerAddr.getPort(), peerAddr.getIPAddress().str().c_str(), "lo");

    // Send control message to rank 1
    sendCtrlMsg(ctranIb, /*peerRank=*/1, &peerServerAddr);

    // Receive control message from rank 1
    recvCtrlMsg(ctranIb, /*peerRank=*/1, &peerServerAddr);

    XLOG(INFO) << "Rank 0 completed bidirectional communication";

    rank0Complete.post();
    rank1Complete.wait();
  };

  auto rank1Action = [this, &rank1Complete, &rank0Complete](
                         CtranIb* ctranIb,
                         const folly::SocketAddress& peerAddr,
                         AbortPtr abortCtrl) {
    SocketServerAddr peerServerAddr = getSocketServerAddress(
        peerAddr.getPort(), peerAddr.getIPAddress().str().c_str(), "lo");

    // Receive control message from rank 0
    recvCtrlMsg(ctranIb, /*peerRank=*/0, &peerServerAddr);

    // Send control message to rank 0
    sendCtrlMsg(ctranIb, /*peerRank=*/0, &peerServerAddr);

    XLOG(INFO) << "Rank 1 completed bidirectional communication";

    rank1Complete.post();
    rank0Complete.wait();
  };

  TwoRankTestHelper(rank0Action, rank1Action).run();
}

// Test that getVc() returns valid VC after connection establishment
TEST_P(CtranIbBootstrapParameterizedTest, GetVcAfterConnection) {
  auto rank0Action = [this](
                         CtranIb* ctranIb,
                         const folly::SocketAddress& peerAddr,
                         AbortPtr abortCtrl) {
    SocketServerAddr peerServerAddr = getSocketServerAddress(
        peerAddr.getPort(), peerAddr.getIPAddress().str().c_str(), "lo");

    // Before connection, getVc should return nullptr
    auto vcBeforeConnection = ctranIb->getVc(1);
    EXPECT_EQ(vcBeforeConnection, nullptr);

    // Send control message to establish connection
    sendCtrlMsg(ctranIb, /*peerRank=*/1, &peerServerAddr);

    // Wait for VC to be established
    bool established = waitForVcEstablished(ctranIb, 1, 5s);
    EXPECT_TRUE(established);

    // After connection, getVc should return valid VC
    auto vcAfterConnection = ctranIb->getVc(1);
    EXPECT_NE(vcAfterConnection, nullptr);

    // Multiple calls should return the same VC
    auto vcSecondCall = ctranIb->getVc(1);
    EXPECT_EQ(vcAfterConnection, vcSecondCall);

    // Verify VC is functional by calling getter methods
    EXPECT_GT(vcAfterConnection->getControlQpNum(), 0u);
    EXPECT_GT(vcAfterConnection->getNotifyQpNum(), 0u);
    EXPECT_GT(vcAfterConnection->getAtomicQpNum(), 0u);
    EXPECT_GT(vcAfterConnection->getMaxNumQp(), 0);

    // Send notifications over the established connection to verify it works
    constexpr int kNumNotifications = 3;
    {
      std::lock_guard<std::mutex> lock(vcAfterConnection->mutex);
      for (int i = 0; i < kNumNotifications; ++i) {
        auto result = vcAfterConnection->notify(nullptr);
        EXPECT_EQ(result, commSuccess);
      }
    }

    // Progress to send all notifications
    auto start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start < 5s) {
      auto res = ctranIb->progress();
      EXPECT_EQ(res, commSuccess);
      std::this_thread::sleep_for(10ms);
    }

    XLOG(INFO) << "Rank 0 verified VC state and sent " << kNumNotifications
               << " notifications";
  };

  auto rank1Action = [this](
                         CtranIb* ctranIb,
                         const folly::SocketAddress& peerAddr,
                         AbortPtr abortCtrl) {
    SocketServerAddr peerServerAddr = getSocketServerAddress(
        peerAddr.getPort(), peerAddr.getIPAddress().str().c_str(), "lo");

    // Before connection, getVc should return nullptr
    auto vcBeforeConnection = ctranIb->getVc(0);
    EXPECT_EQ(vcBeforeConnection, nullptr);

    // Receive control message
    ControlMsg msg;
    CtranIbRequest ctrlReq;
    CtranIbEpochRAII epochRAII(ctranIb);
    ctranIb->irecvCtrlMsg(&msg, sizeof(msg), 0, ctrlReq, &peerServerAddr);

    do {
      auto res = ctranIb->progress();
      EXPECT_EQ(res, commSuccess);
    } while (!ctrlReq.isComplete());

    // Wait for VC to be established
    bool established = waitForVcEstablished(ctranIb, 0, 5s);
    EXPECT_TRUE(established);

    // After connection, getVc should return valid VC
    auto vcAfterConnection = ctranIb->getVc(0);
    EXPECT_NE(vcAfterConnection, nullptr);

    // Verify VC is functional by calling getter methods
    EXPECT_GT(vcAfterConnection->getControlQpNum(), 0u);
    EXPECT_GT(vcAfterConnection->getNotifyQpNum(), 0u);
    EXPECT_GT(vcAfterConnection->getAtomicQpNum(), 0u);
    EXPECT_GT(vcAfterConnection->getMaxNumQp(), 0);

    // Wait to receive notifications from rank 0
    constexpr int kNumNotifications = 3;
    int notificationsReceived = 0;
    auto start = std::chrono::steady_clock::now();
    while (notificationsReceived < kNumNotifications &&
           std::chrono::steady_clock::now() - start < 5s) {
      auto res = ctranIb->progress();
      EXPECT_EQ(res, commSuccess);

      {
        std::lock_guard<std::mutex> lock(vcAfterConnection->mutex);
        bool hasNotify = false;
        vcAfterConnection->checkNotify(&hasNotify);
        if (hasNotify) {
          ++notificationsReceived;
        }
      }

      std::this_thread::sleep_for(10ms);
    }

    EXPECT_EQ(notificationsReceived, kNumNotifications)
        << "Should have received all notifications from rank 0";

    XLOG(INFO) << "Rank 1 verified VC state and received "
               << notificationsReceived << " notifications";
  };

  TwoRankTestHelper(rank0Action, rank1Action).run();
}

// Test that getListenSocketListenAddr() returns consistent address
TEST_P(CtranIbBootstrapParameterizedTest, ListenAddrConsistency) {
  SocketServerAddr serverAddr = getSocketServerAddress();
  auto [ctranIb, abortCtrl] = createCtranIbAndAbort(
      /*rank=*/0, CtranIb::BootstrapMode::kSpecifiedServer, true, &serverAddr);

  // Get listen address multiple times
  auto addr1 = ctranIb->getListenSocketListenAddr();
  auto addr2 = ctranIb->getListenSocketListenAddr();
  auto addr3 = ctranIb->getListenSocketListenAddr();

  // All calls should succeed
  EXPECT_FALSE(addr1.hasError());
  EXPECT_FALSE(addr2.hasError());
  EXPECT_FALSE(addr3.hasError());

  // All addresses should be identical
  EXPECT_EQ(addr1.value().getPort(), addr2.value().getPort());
  EXPECT_EQ(addr1.value().getPort(), addr3.value().getPort());
  EXPECT_EQ(
      addr1.value().getIPAddress().str(), addr2.value().getIPAddress().str());
  EXPECT_EQ(
      addr1.value().getIPAddress().str(), addr3.value().getIPAddress().str());
}
