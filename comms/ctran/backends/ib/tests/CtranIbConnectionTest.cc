// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <chrono>
#include <memory>
#include <thread>

#include <folly/futures/Future.h>

#include "comms/ctran/backends/ib/CtranIb.h"
#include "comms/ctran/backends/ib/CtranIbBase.h"

class CtranIbConnectionTest : public ::testing::Test {
 public:
  CtranIbConnectionTest() = default;

 protected:
  void SetUp() override {
    ncclCvarInit();

    // Initialize CUDA devices
    EXPECT_EQ(cudaSetDevice(0), cudaSuccess);

    // Check if we have at least 2 CUDA devices for this test
    int deviceCount;
    EXPECT_EQ(cudaGetDeviceCount(&deviceCount), cudaSuccess);
    if (deviceCount < 2) {
      GTEST_SKIP() << "Test requires at least 2 CUDA devices, found "
                   << deviceCount;
    }
  }

  void TearDown() override {
    // Reset CUDA device
    cudaDeviceReset();
  }

  // Structure to hold synchronization objects for thread communication
  struct ThreadSyncObjects {
    folly::Promise<std::string> myVcIdPromise;
    folly::SemiFuture<std::string> peerVcIdFuture;
    folly::Promise<ControlMsg> memoryExportPromise;
    folly::SemiFuture<ControlMsg> memoryExportFuture;
    folly::Promise<bool> communicationResult;
  };

  // Common thread function that handles both sender and receiver logic
  void runCtranIbThread(
      bool isSender,
      bool doTransfer,
      ThreadSyncObjects& syncObjects) {
    const size_t bufferSize = 8192;
    const uint64_t commHash = 0x12345678;
    const std::string commDesc = "test";
    const int rank = isSender ? 0 : 1;
    const int peerRank = isSender ? 1 : 0;

    // Set CUDA device based on rank
    EXPECT_EQ(cudaSetDevice(rank), cudaSuccess);

    // ensure abort is not enabled for normal path
    auto abortCtrl = ctran::utils::createAbort(/*enabled=*/!doTransfer);

    // Create CtranIb instance
    auto ctranIb = std::make_unique<CtranIb>(
        rank,
        rank, // Use rank as CUDA device identifier
        commHash,
        commDesc,
        nullptr, // ctrlMgr
        false, // enableLocalFlush
        CtranIb::BootstrapMode::kExternal,
        /*qpServerAddr=*/std::nullopt,
        /*abortCtrl=*/abortCtrl);

    // Get and provide VC identifier
    std::string myVcId = ctranIb->getLocalVcIdentifier(peerRank);
    syncObjects.myVcIdPromise.setValue(myVcId);

    // Wait for peer's VC identifier
    std::string peerVcId = std::move(syncObjects.peerVcIdFuture).get();

    // Connect to peer
    commResult_t connectResult = ctranIb->connectVcDirect(peerVcId, peerRank);
    EXPECT_EQ(connectResult, commSuccess);

    // Allow time for connection establishment
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Allocate and register buffer
    void* buffer = nullptr;
    void* regElem = nullptr;
    EXPECT_EQ(cudaMalloc(&buffer, bufferSize), cudaSuccess);
    EXPECT_EQ(CtranIb::regMem(buffer, bufferSize, rank, &regElem), commSuccess);

    if (isSender) {
      // Sender: Initialize buffer with test data
      std::vector<uint8_t> testData(bufferSize, 0xAB);
      EXPECT_EQ(
          cudaMemcpy(
              buffer, testData.data(), bufferSize, cudaMemcpyHostToDevice),
          cudaSuccess);

      // Wait for remote memory information from receiver
      ControlMsg exportMsg = std::move(syncObjects.memoryExportFuture).get();

      // Import remote memory
      void* remoteRecvBuf = nullptr;
      CtranIbRemoteAccessKey remoteKey;
      EXPECT_EQ(
          CtranIb::importMem(&remoteRecvBuf, &remoteKey, exportMsg),
          commSuccess);

      // Lock epoch before operations
      EXPECT_EQ(ctranIb->epochLock(), commSuccess);

      // Perform RDMA put
      CtranIbRequest putRequest;
      CtranIbConfig config;

      if (doTransfer) {
        commResult_t putResult = ctranIb->iput(
            buffer, // source buffer
            remoteRecvBuf, // remote destination buffer
            bufferSize, // length
            peerRank, // peer rank
            regElem, // local registration element
            remoteKey, // remote access key
            true, // notify
            &config, // config
            &putRequest, // request
            false // fast
        );

        EXPECT_EQ(putResult, commSuccess);

        // Progress the operation and wait for completion
        while (!putRequest.isComplete()) {
          EXPECT_EQ(ctranIb->progress(), commSuccess);
          std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
      }

      EXPECT_EQ(ctranIb->epochUnlock(), commSuccess);
      syncObjects.communicationResult.setValue(true);

    } else {
      // Receiver: Initialize buffer with zero data
      std::vector<uint8_t> zeroData(bufferSize, 0x00);
      EXPECT_EQ(
          cudaMemcpy(
              buffer, zeroData.data(), bufferSize, cudaMemcpyHostToDevice),
          cudaSuccess);

      // Export receive buffer memory and send to sender
      ControlMsg exportMsg;
      EXPECT_EQ(CtranIb::exportMem(buffer, regElem, exportMsg), commSuccess);
      syncObjects.memoryExportPromise.setValue(exportMsg);

      // Lock epoch for operations
      EXPECT_EQ(ctranIb->epochLock(), commSuccess);

      if (doTransfer) {
        EXPECT_EQ(ctranIb->waitNotify(peerRank, 1), commSuccess);

        EXPECT_EQ(ctranIb->epochUnlock(), commSuccess);

        // Verify data was transferred correctly
        std::vector<uint8_t> receivedData(bufferSize);
        EXPECT_EQ(
            cudaMemcpy(
                receivedData.data(),
                buffer,
                bufferSize,
                cudaMemcpyDeviceToHost),
            cudaSuccess);

        // Check that received data matches sent data
        bool dataValid = true;
        for (size_t i = 0; i < bufferSize; ++i) {
          if (receivedData[i] != 0xAB) {
            dataValid = false;
            break;
          }
        }
        EXPECT_TRUE(dataValid) << "Data verification failed";
      } else {
        // !doTransfer, emulating send stuck case
        std::thread timer([&]() {
          std::this_thread::sleep_for(std::chrono::seconds(2));
          abortCtrl->Set();
        });
        SCOPE_EXIT {
          timer.join();
        };

        // Wait for notification from sender
        EXPECT_THROW(
            ctranIb->waitNotify(peerRank, 1), ::ctran::utils::Exception);
        EXPECT_THROW(
            ctranIb->waitNotify(peerRank, 1), ::ctran::utils::Exception);

        EXPECT_EQ(ctranIb->epochUnlock(), commSuccess);
      }
    }

    // Cleanup
    EXPECT_EQ(CtranIb::deregMem(regElem), commSuccess);
    EXPECT_EQ(cudaFree(buffer), cudaSuccess);
  }

  void runTest(bool testAbort) {
    // Promise/future pairs for exchanging VC identifiers between threads
    auto [vcIdPromise0, vcIdFuture0] =
        folly::makePromiseContract<std::string>();
    auto [vcIdPromise1, vcIdFuture1] =
        folly::makePromiseContract<std::string>();
    auto [memoryExportPromise, memoryExportFuture] =
        folly::makePromiseContract<ControlMsg>();
    auto [communicationResult, communicationFuture] =
        folly::makePromiseContract<bool>();

    // Setup synchronization objects for rank 0 (sender)
    ThreadSyncObjects syncObjects0{
        std::move(vcIdPromise0),
        std::move(vcIdFuture1),
        folly::Promise<ControlMsg>(), // sender doesn't export memory
        std::move(memoryExportFuture),
        std::move(communicationResult)};

    // Setup synchronization objects for rank 1 (receiver)
    ThreadSyncObjects syncObjects1{
        std::move(vcIdPromise1),
        std::move(vcIdFuture0),
        std::move(memoryExportPromise),
        folly::SemiFuture<ControlMsg>::makeEmpty(), // receiver doesn't import
                                                    // memory
        folly::Promise<bool>()}; // receiver doesn't set communication result

    // Launch both threads using the common function
    std::thread senderThread([&]() {
      runCtranIbThread(
          /*isSender=*/true, /*doTransfer=*/!testAbort, syncObjects0);
    });
    std::thread recverThread([&]() {
      runCtranIbThread(
          /*isSender=*/false, /*doTransfer=*/!testAbort, syncObjects1);
    });

    // Wait for both threads to complete
    senderThread.join();
    recverThread.join();

    // Verify the communication was successful
    bool success = std::move(communicationFuture).get();
    EXPECT_TRUE(success);
  }
};

TEST_F(CtranIbConnectionTest, BaseNoAbort) {
  this->runTest(/*testAbort=*/false);
}

TEST_F(CtranIbConnectionTest, TestAbortCtrl) {
  this->runTest(/*testAbort=*/true);
}
