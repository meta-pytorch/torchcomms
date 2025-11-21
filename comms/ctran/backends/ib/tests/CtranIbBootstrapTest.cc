// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <gtest/gtest.h>
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
#include "comms/ctran/tests/CtranStandaloneUTUtils.h"
#include "comms/ctran/utils/Abort.h"
#include "comms/utils/cvars/nccl_cvars.h"

using AbortPtr = std::shared_ptr<ctran::utils::Abort>;

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
