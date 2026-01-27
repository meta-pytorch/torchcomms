// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include <string>
#include <vector>

#include "comms/pipes/MultiPeerNvlTransport.h"
#include "comms/pipes/P2pNvlTransportDevice.cuh"
#include "comms/pipes/tests/P2pNvlSendRecvMultipleTest.cuh"

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

// Parameterized test for send_multiple/recv_multiple with various
// configurations
struct SendMultipleTestParams {
  std::string testName;
  std::vector<size_t> chunkSizes; // Sizes of each chunk
  std::vector<size_t> chunkIndices; // Indices of chunks to send
  int numBlocks; // Number of thread blocks (warps)
  int blockSize; // Threads per block
};

class P2pNvlSendRecvMultipleTest
    : public P2pNvlSendRecvMultipleTestFixture,
      public ::testing::WithParamInterface<SendMultipleTestParams> {};

// Test send_multiple/recv_multiple with various chunk configurations
TEST_P(P2pNvlSendRecvMultipleTest, SendRecvMultiple) {
  auto params = GetParam();
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const int peerRank = (globalRank == 0) ? 1 : 0;
  const size_t numChunks = params.chunkSizes.size();
  const size_t numChunksToSend = params.chunkIndices.size();

  // Calculate total buffer size (sum of all chunk sizes)
  size_t totalBufferSize = 0;
  for (const auto& size : params.chunkSizes) {
    totalBufferSize += size;
  }

  XLOGF(
      INFO,
      "Rank {}: Testing {} - {} chunks, {} to send, total {} bytes, {} blocks × {} threads",
      globalRank,
      params.testName,
      numChunks,
      numChunksToSend,
      totalBufferSize,
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
  DeviceBuffer sendDataBuffer(totalBufferSize);
  DeviceBuffer recvDataBuffer(totalBufferSize);
  DeviceBuffer chunkSizesBuffer(numChunks * sizeof(size_t));
  DeviceBuffer chunkIndicesBuffer(
      numChunksToSend > 0 ? numChunksToSend * sizeof(size_t) : sizeof(size_t));
  DeviceBuffer recvChunkSizesBuffer(numChunks * sizeof(size_t));

  auto sendData_d = static_cast<uint8_t*>(sendDataBuffer.get());
  auto recvData_d = static_cast<uint8_t*>(recvDataBuffer.get());
  auto chunkSizes_d = static_cast<size_t*>(chunkSizesBuffer.get());
  auto chunkIndices_d = static_cast<size_t*>(chunkIndicesBuffer.get());
  auto recvChunkSizes_d = static_cast<size_t*>(recvChunkSizesBuffer.get());

  // Initialize send data with test pattern
  std::vector<uint8_t> sendData(totalBufferSize);
  for (size_t i = 0; i < totalBufferSize; i++) {
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

  // Copy send data and chunk sizes to device
  CUDACHECK_TEST(cudaMemcpy(
      sendData_d, sendData.data(), totalBufferSize, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemcpy(
      chunkSizes_d,
      params.chunkSizes.data(),
      numChunks * sizeof(size_t),
      cudaMemcpyHostToDevice));
  if (numChunksToSend > 0) {
    CUDACHECK_TEST(cudaMemcpy(
        chunkIndices_d,
        params.chunkIndices.data(),
        numChunksToSend * sizeof(size_t),
        cudaMemcpyHostToDevice));
  }

  if (globalRank == 0) {
    // Sender: Send chunks via send_multiple
    XLOGF(
        INFO,
        "Rank {}: Sending {} chunks via send_multiple",
        globalRank,
        numChunksToSend);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    test::testSendMultiple(
        p2p,
        sendData_d,
        chunkSizes_d,
        numChunks,
        chunkIndices_d,
        numChunksToSend,
        params.numBlocks,
        params.blockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    XLOGF(INFO, "Rank {}: Sent chunks via send_multiple", globalRank);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  } else {
    // Receiver: Receive chunks via recv_multiple
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    test::testRecvMultiple(
        p2p,
        recvData_d,
        recvChunkSizes_d,
        numChunks,
        params.numBlocks,
        params.blockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    XLOGF(INFO, "Rank {}: Received chunks via recv_multiple", globalRank);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Verify received chunk sizes
    std::vector<size_t> receivedChunkSizes(numChunks);
    CUDACHECK_TEST(cudaMemcpy(
        receivedChunkSizes.data(),
        recvChunkSizes_d,
        numChunks * sizeof(size_t),
        cudaMemcpyDeviceToHost));

    XLOGF(INFO, "Rank {}: Verifying chunk sizes", globalRank);
    for (size_t i = 0; i < numChunks; i++) {
      EXPECT_EQ(receivedChunkSizes[i], params.chunkSizes[i])
          << "Chunk " << i << " size mismatch: expected "
          << params.chunkSizes[i] << ", got " << receivedChunkSizes[i];
    }

    // Verify received data
    std::vector<uint8_t> receivedData(totalBufferSize);
    CUDACHECK_TEST(cudaMemcpy(
        receivedData.data(),
        recvData_d,
        totalBufferSize,
        cudaMemcpyDeviceToHost));

    XLOGF(INFO, "Rank {}: Verifying received data", globalRank);

    // Calculate which bytes should have been received
    std::vector<bool> byteReceived(totalBufferSize, false);
    for (size_t idx : params.chunkIndices) {
      size_t offset = 0;
      for (size_t j = 0; j < idx; j++) {
        offset += params.chunkSizes[j];
      }
      for (size_t j = 0; j < params.chunkSizes[idx]; j++) {
        byteReceived[offset + j] = true;
      }
    }

    // Verify each chunk that was sent
    for (size_t idx : params.chunkIndices) {
      size_t offset = 0;
      for (size_t j = 0; j < idx; j++) {
        offset += params.chunkSizes[j];
      }
      size_t chunkSize = params.chunkSizes[idx];

      XLOGF(
          INFO,
          "Rank {}: Verifying chunk {} at offset {} ({} bytes)",
          globalRank,
          idx,
          offset,
          chunkSize);

      for (size_t j = 0; j < chunkSize; j++) {
        EXPECT_EQ(receivedData[offset + j], sendData[offset + j])
            << "Chunk " << idx << " data mismatch at byte " << j << " (offset "
            << offset + j << "): expected "
            << static_cast<int>(sendData[offset + j]) << ", got "
            << static_cast<int>(receivedData[offset + j]);
        if (receivedData[offset + j] != sendData[offset + j]) {
          break; // Stop after first mismatch
        }
      }
    }

    // Verify bytes that should not have been received remain as SENTINEL
    if (numChunksToSend < numChunks) {
      size_t unreceivedCount = 0;
      for (size_t i = 0; i < totalBufferSize; i++) {
        if (!byteReceived[i]) {
          unreceivedCount++;
          EXPECT_EQ(receivedData[i], SENTINEL)
              << "Unreceived byte at offset " << i
              << " should be SENTINEL (0xFF), got "
              << static_cast<int>(receivedData[i]);
          if (receivedData[i] != SENTINEL && unreceivedCount <= 5) {
            XLOGF(
                ERR,
                "Rank {}: Unreceived byte at offset {} is not SENTINEL: {}",
                globalRank,
                i,
                static_cast<int>(receivedData[i]));
          }
        }
      }
    }

    XLOGF(
        INFO,
        "Rank {}: Verification complete - data and chunk sizes match!",
        globalRank);
  }

  XLOGF(INFO, "Rank {}: Test completed successfully", globalRank);
}

// Parameterized test cases covering different scenarios:
// - Number of chunks: 1, 2, 4, 8
// - Chunk sizes: uniform and varying
// - Number of chunks to send: all, subset, none
// - Warp configurations:
//   - 1 warp (1 block × 32 threads)
//   - 8 warps in 1 block (1 block × 256 threads)
//   - 4 blocks with 4 warps/block (4 blocks × 128 threads)
INSTANTIATE_TEST_SUITE_P(
    SendMultipleVariations,
    P2pNvlSendRecvMultipleTest,
    ::testing::Values(
        // ======== Send 0 chunks (edge case) ========
        SendMultipleTestParams{
            .testName = "Send0Chunks_1Block8Warp",
            .chunkSizes = {1024, 2048, 512},
            .chunkIndices = {},
            .numBlocks = 1,
            .blockSize = 256},
        SendMultipleTestParams{
            .testName = "Send0Chunks_4Block4Warp",
            .chunkSizes = {1024, 2048, 512},
            .chunkIndices = {},
            .numBlocks = 4,
            .blockSize = 128},

        // ======== Single chunk ========
        SendMultipleTestParams{
            .testName = "Send1Chunk_1Block1Warp",
            .chunkSizes = {1024},
            .chunkIndices = {0},
            .numBlocks = 1,
            .blockSize = 32},
        SendMultipleTestParams{
            .testName = "Send1Chunk_1Block4Warp",
            .chunkSizes = {320, 640, 960},
            .chunkIndices = {1},
            .numBlocks = 1,
            .blockSize = 128},
        SendMultipleTestParams{
            .testName = "Send1Chunk_1Block8Warp",
            .chunkSizes = {10240},
            .chunkIndices = {0},
            .numBlocks = 1,
            .blockSize = 256},
        SendMultipleTestParams{
            .testName = "Send1Chunk_4Block4Warp",
            .chunkSizes = {1048576},
            .chunkIndices = {0},
            .numBlocks = 4,
            .blockSize = 128},

        // ======== 2 chunks ========
        // Send all
        SendMultipleTestParams{
            .testName = "Send2ChunksAll_Uniform_1Block8Warp",
            .chunkSizes = {1024, 1024},
            .chunkIndices = {0, 1},
            .numBlocks = 1,
            .blockSize = 256},
        SendMultipleTestParams{
            .testName = "Send2ChunksAll_Varying_1Block8Warp",
            .chunkSizes = {1024, 10240},
            .chunkIndices = {0, 1},
            .numBlocks = 1,
            .blockSize = 256},
        // Send subset
        SendMultipleTestParams{
            .testName = "Send2ChunksFirst_1Block8Warp",
            .chunkSizes = {2048, 4096},
            .chunkIndices = {0},
            .numBlocks = 1,
            .blockSize = 256},
        SendMultipleTestParams{
            .testName = "Send2ChunksSecond_1Block8Warp",
            .chunkSizes = {2048, 4096},
            .chunkIndices = {1},
            .numBlocks = 1,
            .blockSize = 256},

        // ======== 4 chunks ========
        // Send all
        SendMultipleTestParams{
            .testName = "Send4ChunksAll_Uniform_1Block8Warp",
            .chunkSizes = {1024, 1024, 1024, 1024},
            .chunkIndices = {0, 1, 2, 3},
            .numBlocks = 1,
            .blockSize = 256},
        SendMultipleTestParams{
            .testName = "Send4ChunksAll_Varying_1Block8Warp",
            .chunkSizes = {512, 1024, 2048, 4096},
            .chunkIndices = {0, 1, 2, 3},
            .numBlocks = 1,
            .blockSize = 256},
        SendMultipleTestParams{
            .testName = "Send4ChunksAll_Large_4Block4Warp",
            .chunkSizes = {10240, 20480, 10240, 1048576},
            .chunkIndices = {0, 1, 2, 3},
            .numBlocks = 4,
            .blockSize = 128},
        // Send subset
        SendMultipleTestParams{
            .testName = "Send4ChunksSubset_EvenIndices_1Block8Warp",
            .chunkSizes = {1024, 2048, 1024, 2048},
            .chunkIndices = {0, 2},
            .numBlocks = 1,
            .blockSize = 256},
        SendMultipleTestParams{
            .testName = "Send4ChunksSubset_OddIndices_1Block8Warp",
            .chunkSizes = {1024, 2048, 1024, 2048},
            .chunkIndices = {1, 3},
            .numBlocks = 1,
            .blockSize = 256},
        SendMultipleTestParams{
            .testName = "Send4ChunksSubset_First3_4Block4Warp",
            .chunkSizes = {10240, 10240, 10240, 10240},
            .chunkIndices = {0, 1, 2},
            .numBlocks = 4,
            .blockSize = 128},

        // ======== 8 chunks ========
        SendMultipleTestParams{
            .testName = "Send8ChunksAll_Uniform_1Block8Warp",
            .chunkSizes = {512, 512, 512, 512, 512, 512, 512, 512},
            .chunkIndices = {0, 1, 2, 3, 4, 5, 6, 7},
            .numBlocks = 1,
            .blockSize = 256},
        SendMultipleTestParams{
            .testName = "Send8ChunksSubset_Every2nd_4Block4Warp",
            .chunkSizes = {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024},
            .chunkIndices = {0, 2, 4, 6},
            .numBlocks = 4,
            .blockSize = 128},
        SendMultipleTestParams{
            .testName = "Send8ChunksAll_Varying_4Block4Warp",
            .chunkSizes = {256, 512, 1024, 2048, 4096, 8192, 16384, 32768},
            .chunkIndices = {0, 1, 2, 3, 4, 5, 6, 7},
            .numBlocks = 4,
            .blockSize = 128}),
    [](const ::testing::TestParamInfo<SendMultipleTestParams>& info) {
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
