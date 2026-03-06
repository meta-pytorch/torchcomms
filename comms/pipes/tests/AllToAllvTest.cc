// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include "comms/pipes/MultiPeerNvlTransport.h"
#include "comms/pipes/P2pSelfTransportDevice.cuh"
#include "comms/pipes/collectives/AllToAllv.cuh"
#include "comms/pipes/tests/AllToAllvTest.cuh"
#include "comms/pipes/tests/Utils.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/CudaRAII.h"

using namespace meta::comms;

namespace comms::pipes {

namespace {
// Helper function to print device buffer contents
void printDeviceBuffer(
    const char* label,
    void* deviceBuffer,
    int rank,
    int numRanks,
    size_t numIntsPerRank,
    bool showAll = false) {
  size_t totalInts = numRanks * numIntsPerRank;
  std::vector<int32_t> h_buffer(totalInts);
  CUDACHECK_TEST(cudaMemcpy(
      h_buffer.data(),
      deviceBuffer,
      totalInts * sizeof(int32_t),
      cudaMemcpyDeviceToHost));

  XLOGF(
      DBG1,
      "Rank {}: {} (showing {} values for each peer):",
      rank,
      label,
      showAll ? numIntsPerRank : std::min(size_t(4), numIntsPerRank));

  for (int peer = 0; peer < numRanks; peer++) {
    std::string line = "  Peer " + std::to_string(peer) + ": ";
    size_t numToShow =
        showAll ? numIntsPerRank : std::min(size_t(8), numIntsPerRank);
    for (size_t i = 0; i < numToShow; i++) {
      line += std::to_string(h_buffer[peer * numIntsPerRank + i]) + " ";
    }
    if (!showAll && numIntsPerRank > 8) {
      line += "...";
    }
    XLOG(DBG1) << line;
  }
}
} // namespace

class AllToAllvTestFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }

  void TearDown() override {
    MpiBaseTestFixture::TearDown();
  }
};

// Test parameters for AllToAllv tests
struct AllToAllvTestParams {
  int numBlocks;
  int blockSize;
  size_t numIntsPerRank;
  std::string testName;
};

// Parameterized test fixture for AllToAllv tests
class AllToAllvEqualSizeTest
    : public AllToAllvTestFixture,
      public ::testing::WithParamInterface<AllToAllvTestParams> {};

// Test all_to_allv with actual data transfer and verification
TEST_P(AllToAllvEqualSizeTest, AllToAllvEqualSize) {
  const auto& params = GetParam();
  const size_t numIntsPerRank = params.numIntsPerRank;
  const int numBlocks = params.numBlocks;
  const int blockSize = params.blockSize;

  XLOGF(
      DBG1,
      "Rank {}: Running {} with numBlocks={}, blockSize={}, numIntsPerRank={}",
      globalRank,
      params.testName,
      numBlocks,
      blockSize,
      numIntsPerRank);

  // Configuration for P2pNvlTransport - dynamically sized based on test params
  const size_t totalInts = numIntsPerRank * numRanks;
  const size_t bufferSize = totalInts * sizeof(int32_t);

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = std::max(size_t(2048), bufferSize), // At least 2KB
      .chunkSize = 512, // 512 byte chunk size
      .pipelineDepth = 4,
  };

  // Create transport and exchange IPC handles
  auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
  MultiPeerNvlTransport transport(globalRank, numRanks, bootstrap, config);
  transport.exchange();
  XLOGF(DBG1, "Rank {} created transport and exchanged IPC", globalRank);

  // Get transport devices for all peer ranks
  // Use preallocated Transport array from MultiPeerNvlTransport
  // (includes P2pSelfTransportDevice for self and P2pNvlTransportDevice for
  // peers)
  DeviceSpan<Transport> transports_span(
      transport.getTransportsArray(), numRanks);

  // Allocate send and recv buffers based on test parameters
  DeviceBuffer sendBuffer(bufferSize);
  DeviceBuffer recvBuffer(bufferSize);

  // Initialize recv buffer with -1
  test::fillBuffer(reinterpret_cast<int*>(recvBuffer.get()), -1, totalInts);

  // Fill send buffer: each rank fills with pattern based on rank ID and
  // position For rank R, sending to peer P at position i: value = R * 1000 + P
  // * 100 + i
  // Rank 0: To peer 0: 0,1,2,3  To peer 1: 100,101,102,103
  // Rank 1: To peer 0: 1000,1001,1002,1003  To peer 1: 1100,1101,1102,1103
  // Rank 2: To peer 0: 2000,2001,2002,2003  To peer 1: 2100,2101,2102,2103
  std::vector<int32_t> h_send_init(totalInts);
  for (int peer = 0; peer < numRanks; peer++) {
    for (size_t i = 0; i < numIntsPerRank; i++) {
      int32_t value = globalRank * 1000 + peer * 100 + static_cast<int32_t>(i);
      h_send_init[peer * numIntsPerRank + i] = value;
    }
  }
  CUDACHECK_TEST(cudaMemcpy(
      sendBuffer.get(),
      h_send_init.data(),
      bufferSize,
      cudaMemcpyHostToDevice));

  // Setup ChunkInfo: each rank sends numIntsPerRank int32_t to each peer
  std::vector<ChunkInfo> h_send_chunk_infos;
  std::vector<ChunkInfo> h_recv_chunk_infos;

  for (int rank = 0; rank < numRanks; rank++) {
    size_t offset = rank * numIntsPerRank * sizeof(int32_t);
    size_t nbytes = numIntsPerRank * sizeof(int32_t);
    h_send_chunk_infos.emplace_back(offset, nbytes);
    h_recv_chunk_infos.emplace_back(offset, nbytes);
  }

  // Copy chunk infos to device
  DeviceBuffer d_send_chunk_infos(sizeof(ChunkInfo) * numRanks);
  DeviceBuffer d_recv_chunk_infos(sizeof(ChunkInfo) * numRanks);
  CUDACHECK_TEST(cudaMemcpy(
      d_send_chunk_infos.get(),
      h_send_chunk_infos.data(),
      sizeof(ChunkInfo) * numRanks,
      cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemcpy(
      d_recv_chunk_infos.get(),
      h_recv_chunk_infos.data(),
      sizeof(ChunkInfo) * numRanks,
      cudaMemcpyHostToDevice));

  // Create DeviceSpans
  DeviceSpan<ChunkInfo> send_chunk_infos(
      static_cast<ChunkInfo*>(d_send_chunk_infos.get()), numRanks);
  DeviceSpan<ChunkInfo> recv_chunk_infos(
      static_cast<ChunkInfo*>(d_recv_chunk_infos.get()), numRanks);

  // Barrier to ensure all ranks are ready
  MPI_Barrier(MPI_COMM_WORLD);

  // Debug: Print send and recv buffers before all_to_allv
  printDeviceBuffer(
      "Send buffer BEFORE",
      sendBuffer.get(),
      globalRank,
      numRanks,
      numIntsPerRank);
  printDeviceBuffer(
      "Recv buffer BEFORE",
      recvBuffer.get(),
      globalRank,
      numRanks,
      numIntsPerRank);

  XLOGF(DBG1, "Rank {}: calling all_to_allv", globalRank);

  // Call all_to_allv with actual data transfer
  test::testAllToAllv(
      recvBuffer.get(),
      sendBuffer.get(),
      globalRank,
      numRanks,
      transports_span,
      send_chunk_infos,
      recv_chunk_infos,
      numBlocks,
      blockSize);

  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Debug: Print recv buffer after all_to_allv
  printDeviceBuffer(
      "Recv buffer AFTER",
      recvBuffer.get(),
      globalRank,
      numRanks,
      numIntsPerRank);

  // Verify received data
  // Each rank should receive numIntsPerRank int32_t from each peer
  // Expected: from peer P at position i, value = P * 1000 + myRank * 100 + i
  std::vector<int32_t> h_recv_after(totalInts);
  CUDACHECK_TEST(cudaMemcpy(
      h_recv_after.data(),
      recvBuffer.get(),
      bufferSize,
      cudaMemcpyDeviceToHost));

  int h_errorCount = 0;

  // Verify on host for easier debugging
  for (int peer = 0; peer < numRanks; peer++) {
    for (size_t i = 0; i < numIntsPerRank; i++) {
      int32_t expected =
          peer * 1000 + globalRank * 100 + static_cast<int32_t>(i);
      int32_t actual = h_recv_after[peer * numIntsPerRank + i];
      if (expected != actual) {
        h_errorCount++;
        if (h_errorCount <= 10) { // Only print first 10 errors
          XLOGF(
              ERR,
              "Rank {}: Error at peer {} position {}: expected {}, got {}",
              globalRank,
              peer,
              i,
              expected,
              actual);
        }
      }
    }
  }

  XLOGF(
      DBG1,
      "Rank {}: verification completed, errors = {}",
      globalRank,
      h_errorCount);
  EXPECT_EQ(h_errorCount, 0) << "Rank " << globalRank << " found "
                             << h_errorCount << " verification errors";

  // Barrier to ensure all ranks have completed
  MPI_Barrier(MPI_COMM_WORLD);
}

INSTANTIATE_TEST_SUITE_P(
    EqualSizeConfigs,
    AllToAllvEqualSizeTest,
    ::testing::Values(
        // Format: #blocks_#threads_#msgsize
        AllToAllvTestParams{
            .numBlocks = 4,
            .blockSize = 256,
            .numIntsPerRank = 16,
            .testName = "4b_256t_64B"},
        AllToAllvTestParams{
            .numBlocks = 7,
            .blockSize = 256,
            .numIntsPerRank = 64,
            .testName = "7b_256t_256B"},
        AllToAllvTestParams{
            .numBlocks = 8,
            .blockSize = 512,
            .numIntsPerRank = 256,
            .testName = "8b_512t_1KB"},
        AllToAllvTestParams{
            .numBlocks = 16,
            .blockSize = 128,
            .numIntsPerRank = 32,
            .testName = "16b_128t_128B"}),
    [](const ::testing::TestParamInfo<AllToAllvTestParams>& info) {
      return info.param.testName;
    });

// Test parameters for AllToAllv unequal size tests
struct AllToAllvUnequalSizeParams {
  int numBlocks;
  int blockSize;
  size_t base_ints; // Base number of ints per chunk
  std::string testName;
};

// Parameterized test fixture for AllToAllv unequal size tests
class AllToAllvUnequalSizeTest
    : public AllToAllvTestFixture,
      public ::testing::WithParamInterface<AllToAllvUnequalSizeParams> {};

// Test all_to_allv with variable chunk sizes per peer
// Sizes are symmetric: rank i→j size == rank j→i size
TEST_P(AllToAllvUnequalSizeTest, AllToAllvUnequalSize) {
  const auto& params = GetParam();
  const int numBlocks = params.numBlocks;
  const int blockSize = params.blockSize;
  const size_t base_ints = params.base_ints;

  XLOGF(
      DBG1,
      "Rank {}: Running {} with numBlocks={}, blockSize={}, base_ints={}",
      globalRank,
      params.testName,
      numBlocks,
      blockSize,
      base_ints);

  // Calculate max buffer size needed for any rank
  // Max size happens at the highest rank pair: (numRanks-1 + numRanks-1 + 1) *
  // base_ints * sizeof(int32_t) Total for one rank = sum of (globalRank + rank
  // + 1) for rank in [0, numRanks)
  //                    = numRanks * globalRank + sum(rank) + numRanks
  //                    = numRanks * globalRank + numRanks*(numRanks-1)/2 +
  //                    numRanks
  // Worst case (highest rank): numRanks * (numRanks-1) +
  // numRanks*(numRanks-1)/2 + numRanks
  size_t max_total_ints = numRanks * (2 * numRanks - 1 + 1) / 2 * base_ints;
  size_t max_buffer_size = max_total_ints * sizeof(int32_t);

  // Configuration for P2pNvlTransport - use dynamic buffer size
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = std::max(size_t(2048), max_buffer_size), // At least 2KB
      .chunkSize = 512,
      .pipelineDepth = 4,
  };

  // Create transport and exchange IPC handles
  auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
  MultiPeerNvlTransport transport(globalRank, numRanks, bootstrap, config);
  transport.exchange();
  XLOGF(DBG1, "Rank {} created transport and exchanged IPC", globalRank);

  // Get transport devices for all peer ranks
  // Use preallocated Transport array from MultiPeerNvlTransport
  // (includes P2pSelfTransportDevice for self and P2pNvlTransportDevice for
  // peers)
  DeviceSpan<Transport> transports_span(
      transport.getTransportsArray(), numRanks);

  // Calculate variable chunk sizes using symmetric formula
  // Rank i sends to rank j: (i + j + 1) * base_ints * sizeof(int32_t) bytes
  // This ensures: rank i→j size == rank j→i size
  std::vector<ChunkInfo> h_send_chunk_infos;
  std::vector<ChunkInfo> h_recv_chunk_infos;

  size_t send_offset = 0;
  size_t recv_offset = 0;

  for (int rank = 0; rank < numRanks; rank++) {
    // Symmetric size calculation
    size_t num_ints = (globalRank + rank + 1) * base_ints;
    size_t nbytes = num_ints * sizeof(int32_t);

    h_send_chunk_infos.emplace_back(send_offset, nbytes);
    h_recv_chunk_infos.emplace_back(recv_offset, nbytes);

    send_offset += nbytes;
    recv_offset += nbytes;
  }

  const size_t sendBufferSize = send_offset;
  const size_t recvBufferSize = recv_offset;

  XLOGF(
      DBG1,
      "Rank {}: sendBufferSize = {}, recvBufferSize = {}",
      globalRank,
      sendBufferSize,
      recvBufferSize);

  DeviceBuffer sendBuffer(sendBufferSize);
  DeviceBuffer recvBuffer(recvBufferSize);

  // Initialize recv buffer with -1
  std::vector<int32_t> h_recv_init(recvBufferSize / sizeof(int32_t), -1);
  CUDACHECK_TEST(cudaMemcpy(
      recvBuffer.get(),
      h_recv_init.data(),
      recvBufferSize,
      cudaMemcpyHostToDevice));

  // Fill send buffer with pattern: rank * 1000 + peer * 100 + position
  std::vector<int32_t> h_send_init(sendBufferSize / sizeof(int32_t));
  size_t int_offset = 0;
  for (int peer = 0; peer < numRanks; peer++) {
    size_t num_ints = (globalRank + peer + 1) * base_ints;
    for (size_t i = 0; i < num_ints; i++) {
      h_send_init[int_offset + i] =
          globalRank * 1000 + peer * 100 + static_cast<int32_t>(i);
    }
    int_offset += num_ints;
  }
  CUDACHECK_TEST(cudaMemcpy(
      sendBuffer.get(),
      h_send_init.data(),
      sendBufferSize,
      cudaMemcpyHostToDevice));

  // Copy chunk infos to device
  DeviceBuffer d_send_chunk_infos(sizeof(ChunkInfo) * numRanks);
  DeviceBuffer d_recv_chunk_infos(sizeof(ChunkInfo) * numRanks);
  CUDACHECK_TEST(cudaMemcpy(
      d_send_chunk_infos.get(),
      h_send_chunk_infos.data(),
      sizeof(ChunkInfo) * numRanks,
      cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemcpy(
      d_recv_chunk_infos.get(),
      h_recv_chunk_infos.data(),
      sizeof(ChunkInfo) * numRanks,
      cudaMemcpyHostToDevice));

  DeviceSpan<ChunkInfo> send_chunk_infos(
      static_cast<ChunkInfo*>(d_send_chunk_infos.get()), numRanks);
  DeviceSpan<ChunkInfo> recv_chunk_infos(
      static_cast<ChunkInfo*>(d_recv_chunk_infos.get()), numRanks);

  // Barrier to ensure all ranks are ready
  MPI_Barrier(MPI_COMM_WORLD);

  XLOGF(DBG1, "Rank {}: calling all_to_allv with variable sizes", globalRank);

  // Call all_to_allv with variable sizes
  test::testAllToAllv(
      recvBuffer.get(),
      sendBuffer.get(),
      globalRank,
      numRanks,
      transports_span,
      send_chunk_infos,
      recv_chunk_infos,
      numBlocks,
      blockSize);

  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Verify received data
  // Expected: from peer P at position i, value = P * 1000 + myRank * 100 + i
  std::vector<int32_t> h_recv_after(recvBufferSize / sizeof(int32_t));
  CUDACHECK_TEST(cudaMemcpy(
      h_recv_after.data(),
      recvBuffer.get(),
      recvBufferSize,
      cudaMemcpyDeviceToHost));

  int h_errorCount = 0;
  int_offset = 0;

  for (int peer = 0; peer < numRanks; peer++) {
    size_t num_ints = (globalRank + peer + 1) * base_ints;
    for (size_t i = 0; i < num_ints; i++) {
      int32_t expected =
          peer * 1000 + globalRank * 100 + static_cast<int32_t>(i);
      int32_t actual = h_recv_after[int_offset + i];
      if (expected != actual) {
        h_errorCount++;
        if (h_errorCount <= 10) { // Only print first 10 errors
          XLOGF(
              ERR,
              "Rank {}: Error at peer {} position {}: expected {}, got {}",
              globalRank,
              peer,
              i,
              expected,
              actual);
        }
      }
    }
    int_offset += num_ints;
  }

  XLOGF(
      DBG1,
      "Rank {}: verification completed, errors = {}",
      globalRank,
      h_errorCount);
  EXPECT_EQ(h_errorCount, 0) << "Rank " << globalRank << " found "
                             << h_errorCount << " verification errors";

  // Barrier to ensure all ranks have completed
  MPI_Barrier(MPI_COMM_WORLD);
}

INSTANTIATE_TEST_SUITE_P(
    UnequalSizeConfigs,
    AllToAllvUnequalSizeTest,
    ::testing::Values(
        // Format: #blocks_#threads_#maxmsgsize (max is for highest rank pair)
        // base_ints = 16, max = (7+7+1)*16*4 = 960 bytes
        AllToAllvUnequalSizeParams{
            .numBlocks = 7,
            .blockSize = 256,
            .base_ints = 16,
            .testName = "7b_256t_960B"},
        // base_ints = 64, max = (7+7+1)*64*4 = 3840 bytes
        AllToAllvUnequalSizeParams{
            .numBlocks = 8,
            .blockSize = 256,
            .base_ints = 64,
            .testName = "8b_256t_3840B"},
        // base_ints = 256, max = (7+7+1)*256*4 = 15KB
        AllToAllvUnequalSizeParams{
            .numBlocks = 8,
            .blockSize = 512,
            .base_ints = 256,
            .testName = "8b_512t_15KB"},
        // base_ints = 512, max = (7+7+1)*512*4 = 30KB
        AllToAllvUnequalSizeParams{
            .numBlocks = 16,
            .blockSize = 256,
            .base_ints = 512,
            .testName = "16b_256t_30KB"}),
    [](const ::testing::TestParamInfo<AllToAllvUnequalSizeParams>& info) {
      return info.param.testName;
    });

} // namespace comms::pipes

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
