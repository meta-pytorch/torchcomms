// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <chrono>
#include <memory>
#include <thread>

#include <folly/futures/Future.h>
#include <folly/io/async/EventBase.h>
#include <folly/io/async/ScopedEventBaseThread.h>

#include "comms/torchcomms/transport/RdmaTransport.h"
#include "comms/utils/cvars/nccl_cvars.h"

using namespace torch::comms;

class RdmaTransportTest : public ::testing::Test {
 public:
  RdmaTransportTest() = default;

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

    // Create event base thread
    evbThread_ = std::make_unique<folly::ScopedEventBaseThread>();
  }

  void TearDown() override {
    // Reset CUDA device
    cudaDeviceReset();
  }

  // Structure to hold synchronization objects for thread communication
  struct ThreadSyncObjects {
    folly::Promise<std::string> myUrlPromise;
    folly::SemiFuture<std::string> peerUrlFuture;
    folly::Promise<RdmaRemoteBuffer> memoryInfoPromise;
    folly::SemiFuture<RdmaRemoteBuffer> memoryInfoFuture;
    folly::Promise<bool> communicationResult;
  };

  // Common thread function that handles both server and client logic for write
  void runRdmaTransportThreadWrite(
      bool isServer,
      ThreadSyncObjects& syncObjects) {
    const size_t bufferSize = 8192;
    const int cudaDev = isServer ? 0 : 1;
    // Set CUDA device
    EXPECT_EQ(cudaSetDevice(cudaDev), cudaSuccess);

    // Create RdmaTransport instance
    auto transport = std::make_unique<torch::comms::RdmaTransport>(
        cudaDev, evbThread_->getEventBase());

    // Bind and get URL
    std::string myUrl = transport->bind();
    EXPECT_FALSE(myUrl.empty());
    syncObjects.myUrlPromise.setValue(myUrl);

    // Wait for peer's URL
    std::string peerUrl = std::move(syncObjects.peerUrlFuture).get();
    EXPECT_FALSE(peerUrl.empty());

    // Connect to peer
    commResult_t connectResult = transport->connect(peerUrl);
    EXPECT_EQ(connectResult, commSuccess);
    EXPECT_TRUE(transport->connected());

    // Allocate and register memory buffer
    void* buffer = nullptr;
    EXPECT_EQ(cudaMalloc(&buffer, bufferSize), cudaSuccess);
    void* testData;
    EXPECT_EQ(cudaMallocHost(&testData, bufferSize), cudaSuccess);

    torch::comms::RdmaMemory rdmaMemory(buffer, bufferSize, cudaDev);

    if (isServer) {
      // Server: Initialize buffer with test data
      memset(testData, 0xCD, bufferSize);
      EXPECT_EQ(
          cudaMemcpy(buffer, testData, bufferSize, cudaMemcpyHostToDevice),
          cudaSuccess);

      // Wait for client's memory information
      auto clientMemInfo = std::move(syncObjects.memoryInfoFuture).get();

      // Perform iput to transfer data to client's buffer
      auto putFuture = transport->write(
          rdmaMemory.createView(buffer, bufferSize),
          clientMemInfo,
          true // notify
      );
      EXPECT_EQ(commSuccess, std::move(putFuture).get());

      syncObjects.communicationResult.setValue(true);

    } else {
      // Client: Initialize buffer with different data
      memset(testData, 0x00, bufferSize);
      EXPECT_EQ(
          cudaMemcpy(buffer, testData, bufferSize, cudaMemcpyHostToDevice),
          cudaSuccess);

      // Export client's buffer information to server
      RdmaRemoteBuffer myMemInfo{
          .ptr = buffer,
          .len = bufferSize,
          .accessKey = rdmaMemory.remoteKey()};
      syncObjects.memoryInfoPromise.setValue(myMemInfo);

      // Wait for incoming put transfer from server
      auto waitFuture = transport->waitForWrite();
      EXPECT_EQ(commSuccess, std::move(waitFuture).get());

      // Verify data was transferred correctly
      std::vector<uint8_t> receivedData(bufferSize);
      EXPECT_EQ(
          cudaMemcpy(
              receivedData.data(), buffer, bufferSize, cudaMemcpyDeviceToHost),
          cudaSuccess);

      // Check that received data matches sent data
      bool dataValid = true;
      for (size_t i = 0; i < bufferSize; ++i) {
        if (receivedData[i] != 0xCD) {
          dataValid = false;
          break;
        }
      }
      EXPECT_TRUE(dataValid)
          << "Data verification failed - received data does not match sent data";
    }

    // Cleanup
    EXPECT_EQ(cudaFreeHost(testData), cudaSuccess);
    EXPECT_EQ(cudaFree(buffer), cudaSuccess);
  }

  // Common thread function that handles both server and client logic for read
  void runRdmaTransportThreadRead(
      bool isServer,
      ThreadSyncObjects& syncObjects,
      folly::Promise<bool>& readDonePromise,
      folly::SemiFuture<bool>& readDoneFuture) {
    const size_t bufferSize = 8192;
    const int cudaDev = isServer ? 0 : 1;
    // Set CUDA device
    EXPECT_EQ(cudaSetDevice(cudaDev), cudaSuccess);

    // Create RdmaTransport instance
    auto transport = std::make_unique<torch::comms::RdmaTransport>(
        cudaDev, evbThread_->getEventBase());

    // Bind and get URL
    std::string myUrl = transport->bind();
    EXPECT_FALSE(myUrl.empty());
    syncObjects.myUrlPromise.setValue(myUrl);

    // Wait for peer's URL
    std::string peerUrl = std::move(syncObjects.peerUrlFuture).get();
    EXPECT_FALSE(peerUrl.empty());

    // Connect to peer
    commResult_t connectResult = transport->connect(peerUrl);
    EXPECT_EQ(connectResult, commSuccess);
    EXPECT_TRUE(transport->connected());

    // Wait for connection to be established
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    if (isServer) {
      // Server: Allocate and initialize data buffer for client to read
      void* dataBuffer = nullptr;
      EXPECT_EQ(cudaMalloc(&dataBuffer, bufferSize), cudaSuccess);
      torch::comms::RdmaMemory dataMemory(dataBuffer, bufferSize, cudaDev);

      std::vector<uint8_t> testData(bufferSize, 0xAB);
      EXPECT_EQ(
          cudaMemcpy(
              dataBuffer, testData.data(), bufferSize, cudaMemcpyHostToDevice),
          cudaSuccess);

      // Export buffer for client to read
      RdmaRemoteBuffer myMemInfo{
          .ptr = dataBuffer, .accessKey = dataMemory.remoteKey()};
      syncObjects.memoryInfoPromise.setValue(myMemInfo);

      // Wait for client to complete read before cleaning up
      std::move(readDoneFuture).get();

      // Cleanup
      EXPECT_EQ(cudaFree(dataBuffer), cudaSuccess);
    } else {
      // Client: Get server's buffer info
      auto serverMemInfo = std::move(syncObjects.memoryInfoFuture).get();

      // Allocate local buffer to read into
      void* readBuffer = nullptr;
      EXPECT_EQ(cudaMalloc(&readBuffer, bufferSize), cudaSuccess);
      torch::comms::RdmaMemory readMemory(readBuffer, bufferSize, cudaDev);

      // Initialize read buffer with zeros
      std::vector<uint8_t> zeroData(bufferSize, 0x00);
      EXPECT_EQ(
          cudaMemcpy(
              readBuffer, zeroData.data(), bufferSize, cudaMemcpyHostToDevice),
          cudaSuccess);

      // Perform RDMA read from server's buffer using WriteableView
      auto readView = readMemory.createMutableView(readBuffer, bufferSize);
      auto readFuture = transport->read(readView, serverMemInfo);
      EXPECT_EQ(commSuccess, std::move(readFuture).get());
      readDonePromise.setValue(true);

      // Verify read data matches server's data
      std::vector<uint8_t> receivedData(bufferSize);
      EXPECT_EQ(
          cudaMemcpy(
              receivedData.data(),
              readBuffer,
              bufferSize,
              cudaMemcpyDeviceToHost),
          cudaSuccess);

      uint8_t expectedValue = 0xAB; // Client reads 0xAB from server
      bool dataValid = true;
      for (size_t i = 0; i < bufferSize; ++i) {
        if (receivedData[i] != expectedValue) {
          dataValid = false;
          break;
        }
      }
      EXPECT_TRUE(dataValid)
          << "Read data verification failed - received data does not match server data";

      // Signal server that read is complete
      syncObjects.communicationResult.setValue(true);

      // Cleanup
      EXPECT_EQ(cudaFree(readBuffer), cudaSuccess);
    }
  }

 private:
  std::unique_ptr<folly::ScopedEventBaseThread> evbThread_;
};

class RdmaMemoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ncclCvarInit();
    EXPECT_EQ(cudaSetDevice(0), cudaSuccess);

    // Allocate test buffer
    bufferSize_ = 8192; // > 4097 bytes as required
    EXPECT_EQ(cudaMalloc(&buffer_, bufferSize_), cudaSuccess);
    EXPECT_NE(buffer_, nullptr);
  }

  void TearDown() override {
    if (buffer_) {
      EXPECT_EQ(cudaFree(buffer_), cudaSuccess);
    }
    cudaDeviceReset();
  }

  void* buffer_{nullptr};
  size_t bufferSize_{0};
  int cudaDev_{0};
};

TEST_F(RdmaMemoryTest, BasicConstruction) {
  RdmaMemory memory(buffer_, bufferSize_, cudaDev_);

  // Test basic getters
  EXPECT_EQ(memory.getDevice(), cudaDev_);
  EXPECT_NE(memory.localKey(), nullptr);
  EXPECT_FALSE(memory.remoteKey().empty());

  // Test contains method
  EXPECT_TRUE(memory.contains(buffer_, bufferSize_));
  EXPECT_TRUE(memory.contains(buffer_, bufferSize_ / 2));
  EXPECT_TRUE(memory.contains(static_cast<uint8_t*>(buffer_) + 100, 100));

  // Test boundary cases
  EXPECT_FALSE(memory.contains(static_cast<uint8_t*>(buffer_) - 1, 1));
  EXPECT_FALSE(memory.contains(buffer_, bufferSize_ + 1));
}

TEST_F(RdmaMemoryTest, ViewCreation) {
  RdmaMemory memory(buffer_, bufferSize_, cudaDev_);

  // Test valid view creation
  auto view = memory.createView();
  EXPECT_EQ(view.data(), buffer_);
  EXPECT_EQ(view.size(), bufferSize_);
  EXPECT_EQ(view->getDevice(), cudaDev_);
  EXPECT_EQ(view->remoteKey(), memory.remoteKey());
}

TEST_F(RdmaMemoryTest, ViewCreationWithOffsetLength) {
  RdmaMemory memory(buffer_, bufferSize_, cudaDev_);

  // Test valid view creation - use explicit size_t cast to avoid ambiguity
  auto view = memory.createView(static_cast<size_t>(0), bufferSize_);
  EXPECT_EQ(view.data(), buffer_);
  EXPECT_EQ(view.size(), bufferSize_);

  // Test partial view
  size_t offset = 1024;
  size_t length = 2048;
  auto partialView = memory.createView(offset, length);
  EXPECT_EQ(partialView.data(), static_cast<uint8_t*>(buffer_) + offset);
  EXPECT_EQ(partialView.size(), length);

  // Test accessing parent through view
  EXPECT_EQ(partialView->getDevice(), cudaDev_);
  EXPECT_EQ(partialView->remoteKey(), memory.remoteKey());
}

TEST_F(RdmaMemoryTest, ViewCreationWithBufferPointer) {
  RdmaMemory memory(buffer_, bufferSize_, cudaDev_);

  // Test view creation with buffer pointer
  size_t length = 1024;
  auto view = memory.createView(buffer_, length);
  EXPECT_EQ(view.data(), buffer_);
  EXPECT_EQ(view.size(), length);

  // Test view with offset buffer
  uint8_t* offsetBuffer = static_cast<uint8_t*>(buffer_) + 512;
  auto offsetView = memory.createView(offsetBuffer, length);
  EXPECT_EQ(offsetView.data(), offsetBuffer);
  EXPECT_EQ(offsetView.size(), length);
}

TEST_F(RdmaMemoryTest, ViewBoundsChecking) {
  RdmaMemory memory(buffer_, bufferSize_, cudaDev_);

  // These should work fine - use explicit size_t casts
  EXPECT_NO_THROW(memory.createView(static_cast<size_t>(0), bufferSize_));
  EXPECT_NO_THROW(memory.createView(bufferSize_ - 1, static_cast<size_t>(1)));
  EXPECT_NO_THROW(
      memory.createView(static_cast<size_t>(1000), bufferSize_ - 1000));

  // These should trigger CHECK failures
  EXPECT_THROW(
      memory.createView(bufferSize_ + 1, static_cast<size_t>(1)),
      std::invalid_argument);
  EXPECT_THROW(
      memory.createView(static_cast<size_t>(0), bufferSize_ + 1),
      std::invalid_argument);
  EXPECT_THROW(
      memory.createView(bufferSize_ - 100, static_cast<size_t>(200)),
      std::invalid_argument);
}

TEST_F(RdmaMemoryTest, ViewEdgeCases) {
  RdmaMemory memory(buffer_, bufferSize_, cudaDev_);

  // Test zero-length view (should be allowed based on current implementation)
  EXPECT_NO_THROW(
      memory.createView(static_cast<size_t>(0), static_cast<size_t>(0)));
  auto zeroView =
      memory.createView(static_cast<size_t>(0), static_cast<size_t>(0));
  EXPECT_EQ(zeroView.size(), 0);
  EXPECT_EQ(zeroView.data(), buffer_);

  // Test view at the very end
  auto endView = memory.createView(bufferSize_, static_cast<size_t>(0));
  EXPECT_EQ(endView.size(), 0);
  EXPECT_EQ(endView.data(), static_cast<uint8_t*>(buffer_) + bufferSize_);
}

TEST_F(RdmaMemoryTest, MultipleViews) {
  RdmaMemory memory(buffer_, bufferSize_, cudaDev_);

  // Create multiple non-overlapping views - use explicit size_t casts
  auto view1 =
      memory.createView(static_cast<size_t>(0), static_cast<size_t>(1024));
  auto view2 =
      memory.createView(static_cast<size_t>(1024), static_cast<size_t>(1024));
  auto view3 = memory.createView(static_cast<size_t>(2048), bufferSize_ - 2048);

  // Verify they don't interfere with each other
  EXPECT_EQ(view1.size(), 1024);
  EXPECT_EQ(view2.size(), 1024);
  EXPECT_EQ(view3.size(), bufferSize_ - 2048);

  // Verify correct data pointers
  EXPECT_EQ(view1.data(), buffer_);
  EXPECT_EQ(view2.data(), static_cast<uint8_t*>(buffer_) + 1024);
  EXPECT_EQ(view3.data(), static_cast<uint8_t*>(buffer_) + 2048);

  // All should access the same parent
  EXPECT_EQ(view1->remoteKey(), view2->remoteKey());
  EXPECT_EQ(view2->remoteKey(), view3->remoteKey());
}

TEST_F(RdmaMemoryTest, ViewOperatorArrow) {
  RdmaMemory memory(buffer_, bufferSize_, cudaDev_);
  auto view = memory.createView(static_cast<size_t>(0), bufferSize_);

  // Test that operator-> provides access to parent methods
  EXPECT_EQ(view->getDevice(), memory.getDevice());
  EXPECT_EQ(view->remoteKey(), memory.remoteKey());
  EXPECT_EQ(view->localKey(), memory.localKey());
  EXPECT_TRUE(view->contains(buffer_, bufferSize_));
}

TEST_F(RdmaMemoryTest, MoveOnlySemantics) {
  // Test that RdmaMemory is move-only by verifying it can be constructed
  // and used properly, and that copy construction is deleted
  RdmaMemory memory(buffer_, bufferSize_, cudaDev_);
  std::string originalKey = memory.remoteKey();

  // Test that we can create views and access methods
  auto view =
      memory.createView(static_cast<size_t>(0), static_cast<size_t>(1024));
  EXPECT_EQ(view.size(), 1024);
  EXPECT_EQ(view->remoteKey(), originalKey);
  EXPECT_EQ(view->getDevice(), cudaDev_);

  // Test that the memory object maintains its state
  EXPECT_EQ(memory.remoteKey(), originalKey);
  EXPECT_EQ(memory.getDevice(), cudaDev_);
  EXPECT_TRUE(memory.contains(buffer_, bufferSize_));
}

TEST_F(RdmaTransportTest, ServerClientDataTransferWrite) {
  // Promise/future pairs for exchanging URLs between threads
  auto [urlPromise0, urlFuture0] = folly::makePromiseContract<std::string>();
  auto [urlPromise1, urlFuture1] = folly::makePromiseContract<std::string>();
  auto [memoryInfoPromise, memoryInfoFuture] =
      folly::makePromiseContract<RdmaRemoteBuffer>();
  auto [communicationResult, communicationFuture] =
      folly::makePromiseContract<bool>();

  // Setup synchronization objects for server (CUDA device 0)
  ThreadSyncObjects serverSyncObjects{
      std::move(urlPromise0),
      std::move(urlFuture1),
      folly::Promise<RdmaRemoteBuffer>(), // server doesn't export memory
      std::move(memoryInfoFuture),
      std::move(communicationResult)};

  // Setup synchronization objects for client (CUDA device 1)
  ThreadSyncObjects clientSyncObjects{
      std::move(urlPromise1),
      std::move(urlFuture0),
      std::move(memoryInfoPromise),
      folly::SemiFuture<RdmaRemoteBuffer>::makeEmpty(), // client doesn't import
                                                        // memory
      folly::Promise<bool>()}; // client doesn't set communication result

  // Launch both threads using the common function
  std::thread serverThread(
      [&]() { runRdmaTransportThreadWrite(true, serverSyncObjects); });
  std::thread clientThread(
      [&]() { runRdmaTransportThreadWrite(false, clientSyncObjects); });

  // Wait for both threads to complete
  serverThread.join();
  clientThread.join();

  // Verify the communication was successful
  bool success = std::move(communicationFuture).get();
  EXPECT_TRUE(success);
}

TEST_F(RdmaTransportTest, ServerClientDataTransferRead) {
  // Promise/future pairs for exchanging URLs between threads
  auto [urlPromise0, urlFuture0] = folly::makePromiseContract<std::string>();
  auto [urlPromise1, urlFuture1] = folly::makePromiseContract<std::string>();
  auto [memoryInfoPromise, memoryInfoFuture] =
      folly::makePromiseContract<RdmaRemoteBuffer>();
  auto [communicationResult, communicationFuture] =
      folly::makePromiseContract<bool>();

  auto [readDonePromise, readDoneFuture] = folly::makePromiseContract<bool>();

  // Setup synchronization objects for server (CUDA device 0)
  ThreadSyncObjects serverSyncObjects{
      std::move(urlPromise0),
      std::move(urlFuture1),
      std::move(memoryInfoPromise),
      folly::SemiFuture<RdmaRemoteBuffer>::makeEmpty(), // server doesn't import
                                                        // memory
      folly::Promise<bool>(), // server doesn't set comm result
  };

  // Setup synchronization objects for client (CUDA device 1)
  ThreadSyncObjects clientSyncObjects{
      std::move(urlPromise1),
      std::move(urlFuture0),
      folly::Promise<RdmaRemoteBuffer>(), // client doesn't export memory
      std::move(memoryInfoFuture),
      std::move(communicationResult)};

  // Launch both threads using the common function
  std::thread serverThread([&]() {
    runRdmaTransportThreadRead(
        true, serverSyncObjects, readDonePromise, readDoneFuture);
  });
  std::thread clientThread([&]() {
    runRdmaTransportThreadRead(
        false, clientSyncObjects, readDonePromise, readDoneFuture);
  });

  // Wait for both threads to complete
  serverThread.join();
  clientThread.join();

  // Verify the communication was successful
  bool success = std::move(communicationFuture).get();
  EXPECT_TRUE(success);
}
