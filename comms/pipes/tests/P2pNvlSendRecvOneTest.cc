// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include <memory>
#include <string>
#include <vector>

#include "comms/pipes/MultiPeerNvlTransport.h"
#include "comms/pipes/P2pNvlTransportDevice.cuh"
#include "comms/pipes/tests/P2pNvlSendRecvOneTest.cuh"

#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/CudaRAII.h"

using namespace meta::comms;

namespace comms::pipes::tests {

class P2pNvlSendRecvMultipleTestFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }

  void TearDown() override {
    MpiBaseTestFixture::TearDown();
  }
};

// Parameterized test for send_one/recv_one with various configurations
struct SendOneTestParams {
  std::string testName;
  size_t nbytes; // Number of bytes to send
  size_t offsetInOutput; // Offset in receiver's buffer
  int numBlocks; // Number of thread blocks (warps)
  int blockSize; // Threads per block
};

class P2pNvlSendRecvOneSingleCallTest
    : public P2pNvlSendRecvMultipleTestFixture,
      public ::testing::WithParamInterface<SendOneTestParams> {};

// Test send_one/recv_one with various data sizes and warp configurations
TEST_P(P2pNvlSendRecvOneSingleCallTest, SendRecvOne) {
  auto params = GetParam();
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const int peerRank = (globalRank == 0) ? 1 : 0;
  const size_t chunkSize = params.nbytes;
  const bool hasMore = false; // Single call: no more chunks coming
  const size_t offsetInOutput = params.offsetInOutput;

  // Allocate buffer large enough for offset + data
  const size_t totalBufferSize = offsetInOutput + chunkSize;

  XLOGF(
      INFO,
      "Rank {}: Testing {} - {} bytes at offset {} with {} blocks × {} threads",
      globalRank,
      params.testName,
      chunkSize,
      offsetInOutput,
      params.numBlocks,
      params.blockSize);

  // Transport configuration
  const size_t dataBufferSize = 4096;
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = 512,
      .pipelineDepth = 4,
  };

  auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
  MultiPeerNvlTransport transport(globalRank, numRanks, bootstrap, config);
  transport.exchange();
  XLOGF(INFO, "Rank {}: Transport initialized", globalRank);

  auto p2p = transport.getP2pTransportDevice(peerRank);

  // Allocate device buffers
  DeviceBuffer sendDataBuffer(chunkSize);
  DeviceBuffer recvDataBuffer(
      totalBufferSize); // Large enough for offset + data
  DeviceBuffer hasMoreBuffer(sizeof(bool));
  DeviceBuffer offsetBuffer(sizeof(size_t));
  DeviceBuffer nbytesBuffer(sizeof(size_t));

  auto sendData_d = static_cast<uint8_t*>(sendDataBuffer.get());
  auto recvData_d = static_cast<uint8_t*>(recvDataBuffer.get());
  auto hasMore_d = static_cast<bool*>(hasMoreBuffer.get());
  auto offset_d = static_cast<size_t*>(offsetBuffer.get());
  auto nbytes_d = static_cast<size_t*>(nbytesBuffer.get());

  // Initialize send data with test pattern
  std::vector<uint8_t> sendData(chunkSize);
  for (size_t i = 0; i < chunkSize; i++) {
    sendData[i] = static_cast<uint8_t>((i * 7) % 256); // Varied pattern
  }

  // Initialize receive buffer with sentinel value (0xFF)
  constexpr uint8_t SENTINEL = 0xFF;
  std::vector<uint8_t> recvDataInit(totalBufferSize, SENTINEL);
  CUDACHECK_TEST(cudaMemcpy(
      recvData_d,
      recvDataInit.data(),
      totalBufferSize,
      cudaMemcpyHostToDevice));

  // Copy send data to device
  CUDACHECK_TEST(cudaMemcpy(
      sendData_d, sendData.data(), chunkSize, cudaMemcpyHostToDevice));

  if (globalRank == 0) {
    // Sender: Send single chunk via send_one
    XLOGF(INFO, "Rank {}: Sending chunk via send_one", globalRank);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    test::testSendOne(
        p2p,
        sendData_d,
        chunkSize,
        offsetInOutput,
        hasMore,
        params.numBlocks,
        params.blockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    XLOGF(INFO, "Rank {}: Sent chunk via send_one", globalRank);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  } else {
    // Receiver: Receive chunk via recv_one
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    test::testRecvOne(
        p2p,
        recvData_d,
        nbytes_d,
        offset_d,
        hasMore_d,
        params.numBlocks,
        params.blockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    XLOGF(INFO, "Rank {}: Received chunk via recv_one", globalRank);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Verify metadata
    bool receivedHasMore = true; // Initialize to opposite of expected
    size_t receivedOffset = 0;
    size_t receivedNbytes = 0;

    CUDACHECK_TEST(cudaMemcpy(
        &receivedHasMore, hasMore_d, sizeof(bool), cudaMemcpyDeviceToHost));
    CUDACHECK_TEST(cudaMemcpy(
        &receivedOffset, offset_d, sizeof(size_t), cudaMemcpyDeviceToHost));
    CUDACHECK_TEST(cudaMemcpy(
        &receivedNbytes, nbytes_d, sizeof(size_t), cudaMemcpyDeviceToHost));

    XLOGF(
        INFO,
        "Rank {}: Received metadata - has_more={}, offset={}, nbytes={}",
        globalRank,
        receivedHasMore,
        receivedOffset,
        receivedNbytes);

    EXPECT_EQ(receivedHasMore, hasMore)
        << "has_more mismatch: expected " << hasMore << ", got "
        << receivedHasMore;
    EXPECT_EQ(receivedOffset, offsetInOutput)
        << "offset mismatch: expected " << offsetInOutput << ", got "
        << receivedOffset;
    EXPECT_EQ(receivedNbytes, chunkSize)
        << "nbytes mismatch: expected " << chunkSize << ", got "
        << receivedNbytes;

    // Verify received data
    std::vector<uint8_t> receivedData(totalBufferSize);
    CUDACHECK_TEST(cudaMemcpy(
        receivedData.data(),
        recvData_d,
        totalBufferSize,
        cudaMemcpyDeviceToHost));

    XLOGF(INFO, "Rank {}: Verifying received data", globalRank);

    // Verify data before offset remains as SENTINEL (untouched)
    for (size_t i = 0; i < offsetInOutput; i++) {
      EXPECT_EQ(receivedData[i], SENTINEL)
          << "Data before offset should remain SENTINEL at byte " << i
          << ": expected 0xFF, got " << static_cast<int>(receivedData[i]);
      if (receivedData[i] != SENTINEL) {
        break; // Stop after first mismatch
      }
    }

    // Verify data at offset matches sent data
    for (size_t i = 0; i < chunkSize; i++) {
      EXPECT_EQ(receivedData[offsetInOutput + i], sendData[i])
          << "Data mismatch at offset " << offsetInOutput << " + byte " << i
          << ": expected " << static_cast<int>(sendData[i]) << ", got "
          << static_cast<int>(receivedData[offsetInOutput + i]);
      if (receivedData[offsetInOutput + i] != sendData[i]) {
        break; // Stop after first mismatch
      }
    }

    XLOGF(
        INFO,
        "Rank {}: Verification complete - data and metadata match!",
        globalRank);
  }

  XLOGF(INFO, "Rank {}: Test completed successfully", globalRank);
}

// Test structure for calling send_one/recv_one multiple times in one kernel
struct SendOneMultipleCallsParams {
  std::string testName;
  std::vector<size_t> nbytes; // Number of bytes for each call
  std::vector<size_t> offsetInOutput; // Offset for each call
  int numBlocks; // Number of thread blocks (warps)
  int blockSize; // Threads per block
};

class P2pNvlSendRecvOneMultipleCallsTest
    : public P2pNvlSendRecvMultipleTestFixture,
      public ::testing::WithParamInterface<SendOneMultipleCallsParams> {};

// Test send_one/recv_one called multiple times in one kernel
TEST_P(P2pNvlSendRecvOneMultipleCallsTest, SendRecvOneMultipleCalls) {
  auto params = GetParam();
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const int peerRank = (globalRank == 0) ? 1 : 0;
  const size_t num_calls = params.nbytes.size();

  // Calculate total buffer size needed
  size_t totalBufferSize = 0;
  for (size_t i = 0; i < num_calls; i++) {
    size_t end_offset = params.offsetInOutput[i] + params.nbytes[i];
    if (end_offset > totalBufferSize) {
      totalBufferSize = end_offset;
    }
  }

  XLOGF(
      INFO,
      "Rank {}: Testing {} - {} calls with total buffer size {}",
      globalRank,
      params.testName,
      num_calls,
      totalBufferSize);

  // Transport configuration
  const size_t dataBufferSize = 4096;
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = 512,
      .pipelineDepth = 4,
  };

  auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
  MultiPeerNvlTransport transport(globalRank, numRanks, bootstrap, config);
  transport.exchange();
  XLOGF(INFO, "Rank {}: Transport initialized", globalRank);

  auto p2p = transport.getP2pTransportDevice(peerRank);

  // Allocate device buffers for send data (one buffer per call)
  std::vector<DeviceBuffer> sendDataBuffers;
  std::vector<void*> sendData_d_ptrs;
  std::vector<std::vector<uint8_t>> sendDataHost;

  for (size_t i = 0; i < num_calls; i++) {
    sendDataBuffers.emplace_back(params.nbytes[i]);
    sendData_d_ptrs.push_back(sendDataBuffers.back().get());

    // Initialize send data with unique pattern for each call
    std::vector<uint8_t> sendData(params.nbytes[i]);
    for (size_t j = 0; j < params.nbytes[i]; j++) {
      sendData[j] = static_cast<uint8_t>((i * 31 + j * 7) % 256);
    }
    sendDataHost.push_back(sendData);

    CUDACHECK_TEST(cudaMemcpy(
        sendData_d_ptrs[i],
        sendData.data(),
        params.nbytes[i],
        cudaMemcpyHostToDevice));
  }

  // Copy send data pointers to device
  DeviceBuffer sendDataPtrsBuffer(num_calls * sizeof(void*));
  auto sendDataPtrs_d = static_cast<void**>(sendDataPtrsBuffer.get());
  CUDACHECK_TEST(cudaMemcpy(
      sendDataPtrs_d,
      sendData_d_ptrs.data(),
      num_calls * sizeof(void*),
      cudaMemcpyHostToDevice));

  // Allocate device buffers for parameters
  DeviceBuffer nbytesBuffer(num_calls * sizeof(size_t));
  DeviceBuffer offsetBuffer(num_calls * sizeof(size_t));
  DeviceBuffer hasMoreBuffer(num_calls * sizeof(bool));

  auto nbytes_d = static_cast<size_t*>(nbytesBuffer.get());
  auto offset_d = static_cast<size_t*>(offsetBuffer.get());
  auto hasMore_d = static_cast<bool*>(hasMoreBuffer.get());

  // Initialize parameter arrays
  // has_more is true for all calls except the last one
  // Note: std::vector<bool> is a special case that doesn't provide contiguous
  // storage, so we use a unique_ptr<bool[]> instead
  auto hasMore = std::make_unique<bool[]>(num_calls);
  for (size_t i = 0; i < num_calls - 1; i++) {
    hasMore[i] = true;
  }
  hasMore[num_calls - 1] = false; // Last call: no more chunks
  CUDACHECK_TEST(cudaMemcpy(
      nbytes_d,
      params.nbytes.data(),
      num_calls * sizeof(size_t),
      cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemcpy(
      offset_d,
      params.offsetInOutput.data(),
      num_calls * sizeof(size_t),
      cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemcpy(
      hasMore_d,
      hasMore.get(),
      num_calls * sizeof(bool),
      cudaMemcpyHostToDevice));

  // Allocate receive buffer
  DeviceBuffer recvDataBuffer(totalBufferSize);
  auto recvData_d = static_cast<uint8_t*>(recvDataBuffer.get());

  // Initialize receive buffer with sentinel value
  constexpr uint8_t SENTINEL = 0xFF;
  std::vector<uint8_t> recvDataInit(totalBufferSize, SENTINEL);
  CUDACHECK_TEST(cudaMemcpy(
      recvData_d,
      recvDataInit.data(),
      totalBufferSize,
      cudaMemcpyHostToDevice));

  // Allocate buffers for received metadata
  DeviceBuffer recvNbytesBuffer(num_calls * sizeof(size_t));
  DeviceBuffer recvOffsetBuffer(num_calls * sizeof(size_t));
  DeviceBuffer recvHasMoreBuffer(num_calls * sizeof(bool));

  auto recvNbytes_d = static_cast<size_t*>(recvNbytesBuffer.get());
  auto recvOffset_d = static_cast<size_t*>(recvOffsetBuffer.get());
  auto recvHasMore_d = static_cast<bool*>(recvHasMoreBuffer.get());

  if (globalRank == 0) {
    // Sender: Send multiple chunks via send_one
    XLOGF(
        INFO, "Rank {}: Sending {} chunks via send_one", globalRank, num_calls);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    test::testSendOneMultipleTimes(
        p2p,
        const_cast<const void**>(sendDataPtrs_d),
        nbytes_d,
        offset_d,
        hasMore_d,
        num_calls,
        params.numBlocks,
        params.blockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    XLOGF(INFO, "Rank {}: Sent all chunks via send_one", globalRank);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  } else {
    // Receiver: Receive chunks via recv_one
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    test::testRecvOneMultipleTimes(
        p2p,
        recvData_d,
        recvNbytes_d,
        recvOffset_d,
        recvHasMore_d,
        num_calls,
        params.numBlocks,
        params.blockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    XLOGF(INFO, "Rank {}: Received all chunks via recv_one", globalRank);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Verify metadata for each call
    std::vector<size_t> recvNbytes(num_calls);
    std::vector<size_t> recvOffset(num_calls);
    auto recvHasMore = std::make_unique<bool[]>(num_calls);

    CUDACHECK_TEST(cudaMemcpy(
        recvNbytes.data(),
        recvNbytes_d,
        num_calls * sizeof(size_t),
        cudaMemcpyDeviceToHost));
    CUDACHECK_TEST(cudaMemcpy(
        recvOffset.data(),
        recvOffset_d,
        num_calls * sizeof(size_t),
        cudaMemcpyDeviceToHost));
    CUDACHECK_TEST(cudaMemcpy(
        recvHasMore.get(),
        recvHasMore_d,
        num_calls * sizeof(bool),
        cudaMemcpyDeviceToHost));

    XLOGF(
        INFO,
        "Rank {}: Verifying metadata for {} calls",
        globalRank,
        num_calls);

    for (size_t i = 0; i < num_calls; i++) {
      XLOGF(
          INFO,
          "Rank {}: Call {} metadata - nbytes={}, offset={}, has_more={}",
          globalRank,
          i,
          recvNbytes[i],
          recvOffset[i],
          recvHasMore[i]);

      EXPECT_EQ(recvNbytes[i], params.nbytes[i])
          << "Call " << i << " nbytes mismatch";
      EXPECT_EQ(recvOffset[i], params.offsetInOutput[i])
          << "Call " << i << " offset mismatch";
      EXPECT_EQ(recvHasMore[i], hasMore[i])
          << "Call " << i << " has_more mismatch";
    }

    // Verify received data
    std::vector<uint8_t> recvData(totalBufferSize);
    CUDACHECK_TEST(cudaMemcpy(
        recvData.data(), recvData_d, totalBufferSize, cudaMemcpyDeviceToHost));

    XLOGF(INFO, "Rank {}: Verifying received data", globalRank);

    // Track which bytes should have been written
    std::vector<bool> byteWritten(totalBufferSize, false);
    for (size_t i = 0; i < num_calls; i++) {
      size_t offset = params.offsetInOutput[i];
      size_t nbytes = params.nbytes[i];
      for (size_t j = 0; j < nbytes; j++) {
        byteWritten[offset + j] = true;
      }
    }

    // Verify each call's data
    for (size_t i = 0; i < num_calls; i++) {
      size_t offset = params.offsetInOutput[i];
      size_t nbytes = params.nbytes[i];

      XLOGF(
          INFO,
          "Rank {}: Verifying call {} data at offset {} ({} bytes)",
          globalRank,
          i,
          offset,
          nbytes);

      for (size_t j = 0; j < nbytes; j++) {
        uint8_t expected = sendDataHost[i][j];
        uint8_t actual = recvData[offset + j];
        EXPECT_EQ(actual, expected)
            << "Call " << i << " data mismatch at byte " << j << " (offset "
            << offset + j << "): expected " << static_cast<int>(expected)
            << ", got " << static_cast<int>(actual);
        if (actual != expected) {
          break; // Stop after first mismatch
        }
      }
    }

    // Verify unwritten bytes remain as SENTINEL
    size_t unwrittenCount = 0;
    for (size_t i = 0; i < totalBufferSize; i++) {
      if (!byteWritten[i]) {
        unwrittenCount++;
        EXPECT_EQ(recvData[i], SENTINEL) << "Unwritten byte at offset " << i
                                         << " should be SENTINEL (0xFF), got "
                                         << static_cast<int>(recvData[i]);
        if (recvData[i] != SENTINEL && unwrittenCount <= 5) {
          // Only show first few errors
          XLOGF(
              ERR,
              "Rank {}: Unwritten byte at offset {} is not SENTINEL: {}",
              globalRank,
              i,
              static_cast<int>(recvData[i]));
        }
      }
    }

    XLOGF(
        INFO,
        "Rank {}: Verification complete - all {} calls verified!",
        globalRank,
        num_calls);
  }

  XLOGF(INFO, "Rank {}: Test completed successfully", globalRank);
}

// Parameterized test cases covering different scenarios:
// - Data sizes: 1KB, 2KB, 10KB, 1MB
// - Offsets: 0, 4096
// - Warp configurations:
//   - 1 warp (1 block × 32 threads)
//   - 8 warps in 1 block (1 block × 256 threads)
//   - 4 blocks with 1 warp/block (4 blocks × 32 threads)
//   - 4 blocks with 4 warps/block (4 blocks × 128 threads)
INSTANTIATE_TEST_SUITE_P(
    SendOneVariations,
    P2pNvlSendRecvOneSingleCallTest,
    ::testing::Values(
        // ======== 1 warp (1 block × 32 threads) ========
        // Offset 0
        SendOneTestParams{
            .testName = "0KB_1Block1Warp_Offset0",
            .nbytes = 0,
            .offsetInOutput = 0,
            .numBlocks = 1,
            .blockSize = 32},
        SendOneTestParams{
            .testName = "1KB_1Block1Warp_Offset0",
            .nbytes = 1024,
            .offsetInOutput = 0,
            .numBlocks = 1,
            .blockSize = 32},
        SendOneTestParams{
            .testName = "2KB_1Block1Warp_Offset0",
            .nbytes = 2048,
            .offsetInOutput = 0,
            .numBlocks = 1,
            .blockSize = 32},
        SendOneTestParams{
            .testName = "10KB_1Block1Warp_Offset0",
            .nbytes = 10240,
            .offsetInOutput = 0,
            .numBlocks = 1,
            .blockSize = 32},
        SendOneTestParams{
            .testName = "1MB_1Block1Warp_Offset0",
            .nbytes = 1048576,
            .offsetInOutput = 0,
            .numBlocks = 1,
            .blockSize = 32},
        // Offset 4096
        SendOneTestParams{
            .testName = "1KB_1Block1Warp_Offset4096",
            .nbytes = 1024,
            .offsetInOutput = 4096,
            .numBlocks = 1,
            .blockSize = 32},
        SendOneTestParams{
            .testName = "10KB_1Block1Warp_Offset4096",
            .nbytes = 10240,
            .offsetInOutput = 4096,
            .numBlocks = 1,
            .blockSize = 32},

        // ======== 4 warps in 1 block (1 block × 128 threads) ========
        // Matches Send1Chunk_4WarpsPerBlock from SendMultiple test
        SendOneTestParams{
            .testName = "640B_1Block4Warp_Offset960",
            .nbytes = 640,
            .offsetInOutput = 960,
            .numBlocks = 1,
            .blockSize = 128},

        // ======== 8 warps in 1 block (1 block × 256 threads) ========
        // Offset 0
        SendOneTestParams{
            .testName = "0KB_1Block8Warp_Offset0",
            .nbytes = 0,
            .offsetInOutput = 0,
            .numBlocks = 1,
            .blockSize = 256},
        SendOneTestParams{
            .testName = "1KB_1Block8Warp_Offset0",
            .nbytes = 1024,
            .offsetInOutput = 0,
            .numBlocks = 1,
            .blockSize = 256},
        SendOneTestParams{
            .testName = "2KB_1Block8Warp_Offset0",
            .nbytes = 2048,
            .offsetInOutput = 0,
            .numBlocks = 1,
            .blockSize = 256},
        SendOneTestParams{
            .testName = "10KB_1Block8Warp_Offset0",
            .nbytes = 10240,
            .offsetInOutput = 0,
            .numBlocks = 1,
            .blockSize = 256},
        SendOneTestParams{
            .testName = "1MB_1Block8Warp_Offset0",
            .nbytes = 1048576,
            .offsetInOutput = 0,
            .numBlocks = 1,
            .blockSize = 256},
        // Offset 4096
        SendOneTestParams{
            .testName = "1KB_1Block8Warp_Offset4096",
            .nbytes = 1024,
            .offsetInOutput = 4096,
            .numBlocks = 1,
            .blockSize = 256},
        SendOneTestParams{
            .testName = "10KB_1Block8Warp_Offset4096",
            .nbytes = 10240,
            .offsetInOutput = 4096,
            .numBlocks = 1,
            .blockSize = 256},

        // ======== 4 blocks with 1 warp/block (4 blocks × 32 threads) ========
        // Offset 0
        SendOneTestParams{
            .testName = "0KB_4Block1Warp_Offset0",
            .nbytes = 0,
            .offsetInOutput = 0,
            .numBlocks = 4,
            .blockSize = 32},
        SendOneTestParams{
            .testName = "1KB_4Block1Warp_Offset0",
            .nbytes = 1024,
            .offsetInOutput = 0,
            .numBlocks = 4,
            .blockSize = 32},
        SendOneTestParams{
            .testName = "2KB_4Block1Warp_Offset0",
            .nbytes = 2048,
            .offsetInOutput = 0,
            .numBlocks = 4,
            .blockSize = 32},
        SendOneTestParams{
            .testName = "10KB_4Block1Warp_Offset0",
            .nbytes = 10240,
            .offsetInOutput = 0,
            .numBlocks = 4,
            .blockSize = 32},
        SendOneTestParams{
            .testName = "1MB_4Block1Warp_Offset0",
            .nbytes = 1048576,
            .offsetInOutput = 0,
            .numBlocks = 4,
            .blockSize = 32},
        // Offset 4096
        SendOneTestParams{
            .testName = "1KB_4Block1Warp_Offset4096",
            .nbytes = 1024,
            .offsetInOutput = 4096,
            .numBlocks = 4,
            .blockSize = 32},
        SendOneTestParams{
            .testName = "10KB_4Block1Warp_Offset4096",
            .nbytes = 10240,
            .offsetInOutput = 4096,
            .numBlocks = 4,
            .blockSize = 32},

        // ======== 4 blocks with 4 warps/block (4 blocks × 128 threads)
        // ======== Offset 0
        SendOneTestParams{
            .testName = "0KB_4Block4Warp_Offset0",
            .nbytes = 0,
            .offsetInOutput = 0,
            .numBlocks = 4,
            .blockSize = 128},
        SendOneTestParams{
            .testName = "1KB_4Block4Warp_Offset0",
            .nbytes = 1024,
            .offsetInOutput = 0,
            .numBlocks = 4,
            .blockSize = 128},
        SendOneTestParams{
            .testName = "2KB_4Block4Warp_Offset0",
            .nbytes = 2048,
            .offsetInOutput = 0,
            .numBlocks = 4,
            .blockSize = 128},
        SendOneTestParams{
            .testName = "10KB_4Block4Warp_Offset0",
            .nbytes = 10240,
            .offsetInOutput = 0,
            .numBlocks = 4,
            .blockSize = 128},
        SendOneTestParams{
            .testName = "1MB_4Block4Warp_Offset0",
            .nbytes = 1048576,
            .offsetInOutput = 0,
            .numBlocks = 4,
            .blockSize = 128},
        // Offset 4096
        SendOneTestParams{
            .testName = "1KB_4Block4Warp_Offset4096",
            .nbytes = 1024,
            .offsetInOutput = 4096,
            .numBlocks = 4,
            .blockSize = 128},
        SendOneTestParams{
            .testName = "10KB_4Block4Warp_Offset4096",
            .nbytes = 10240,
            .offsetInOutput = 4096,
            .numBlocks = 4,
            .blockSize = 128}),
    [](const ::testing::TestParamInfo<SendOneTestParams>& info) {
      return info.param.testName;
    });

// Parameterized test cases for multiple send_one/recv_one calls:
// - 2, 3, 4 calls with different data sizes (1KB, 2KB, 10KB, 1MB)
// - Fixed offsets per chunk: 0, 16384, 32768, 49152
// - Warp configurations:
//   - 1 block × 256 threads (8 warps in 1 block)
//   - 4 blocks × 256 threads (8 warps per block, multiple blocks)
INSTANTIATE_TEST_SUITE_P(
    SendOneMultipleCalls,
    P2pNvlSendRecvOneMultipleCallsTest,
    ::testing::Values(
        // ======== 2 calls ========
        // 1 block × 256 threads (8 warps)
        SendOneMultipleCallsParams{
            .testName = "2Calls_1KB_2KB_1Block8Warp",
            .nbytes = {1024, 2048},
            .offsetInOutput = {0, 16384},
            .numBlocks = 1,
            .blockSize = 256},
        SendOneMultipleCallsParams{
            .testName = "2Calls_10KB_1MB_1Block8Warp",
            .nbytes = {10240, 1048576},
            .offsetInOutput = {0, 16384},
            .numBlocks = 1,
            .blockSize = 256},
        // 4 blocks × 256 threads (8 warps per block)
        SendOneMultipleCallsParams{
            .testName = "2Calls_1KB_2KB_4Block8Warp",
            .nbytes = {1024, 2048},
            .offsetInOutput = {0, 16384},
            .numBlocks = 4,
            .blockSize = 256},
        SendOneMultipleCallsParams{
            .testName = "2Calls_10KB_1MB_4Block8Warp",
            .nbytes = {10240, 1048576},
            .offsetInOutput = {0, 16384},
            .numBlocks = 4,
            .blockSize = 256},

        // ======== 3 calls ========
        // 1 block × 256 threads (8 warps)
        SendOneMultipleCallsParams{
            .testName = "3Calls_1KB_2KB_10KB_1Block8Warp",
            .nbytes = {1024, 2048, 10240},
            .offsetInOutput = {0, 16384, 32768},
            .numBlocks = 1,
            .blockSize = 256},
        SendOneMultipleCallsParams{
            .testName = "3Calls_2KB_10KB_1MB_1Block8Warp",
            .nbytes = {2048, 10240, 1048576},
            .offsetInOutput = {0, 16384, 32768},
            .numBlocks = 1,
            .blockSize = 256},
        // 4 blocks × 256 threads (8 warps per block)
        SendOneMultipleCallsParams{
            .testName = "3Calls_1KB_2KB_10KB_4Block8Warp",
            .nbytes = {1024, 2048, 10240},
            .offsetInOutput = {0, 16384, 32768},
            .numBlocks = 4,
            .blockSize = 256},
        SendOneMultipleCallsParams{
            .testName = "3Calls_2KB_10KB_1MB_4Block8Warp",
            .nbytes = {2048, 10240, 1048576},
            .offsetInOutput = {0, 16384, 32768},
            .numBlocks = 4,
            .blockSize = 256},

        // ======== 4 calls ========
        // 1 block × 256 threads (8 warps)
        SendOneMultipleCallsParams{
            .testName = "4Calls_1KB_2KB_10KB_1MB_1Block8Warp",
            .nbytes = {1024, 2048, 10240, 1048576},
            .offsetInOutput = {0, 16384, 32768, 49152},
            .numBlocks = 1,
            .blockSize = 256},
        SendOneMultipleCallsParams{
            .testName = "4Calls_Mixed_1Block8Warp",
            .nbytes = {2048, 1024, 10240, 1048576},
            .offsetInOutput = {0, 16384, 32768, 49152},
            .numBlocks = 1,
            .blockSize = 256},
        // 4 blocks × 256 threads (8 warps per block)
        SendOneMultipleCallsParams{
            .testName = "4Calls_1KB_2KB_10KB_1MB_4Block8Warp",
            .nbytes = {1024, 2048, 10240, 1048576},
            .offsetInOutput = {0, 16384, 32768, 49152},
            .numBlocks = 4,
            .blockSize = 256},
        SendOneMultipleCallsParams{
            .testName = "4Calls_Mixed_4Block8Warp",
            .nbytes = {2048, 1024, 10240, 1048576},
            .offsetInOutput = {0, 16384, 32768, 49152},
            .numBlocks = 4,
            .blockSize = 256}),
    [](const ::testing::TestParamInfo<SendOneMultipleCallsParams>& info) {
      return info.param.testName;
    });

} // namespace comms::pipes::tests

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  auto mpi_env = std::make_unique<MPIEnvironmentBase>();
  ::testing::AddGlobalTestEnvironment(mpi_env.get());
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
