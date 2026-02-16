// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include "comms/pipes/MultiPeerNvlTransport.h"
#include "comms/pipes/P2pSelfTransportDevice.cuh"
#include "comms/pipes/collectives/AllGather.cuh"
#include "comms/pipes/tests/AllGatherTest.cuh"
#include "comms/pipes/tests/Utils.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::DeviceBuffer;
using meta::comms::MpiBaseTestFixture;
using meta::comms::MPIEnvironmentBase;

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
    std::string line = "  From rank " + std::to_string(peer) + ": ";
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

class AllGatherTestFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }

  void TearDown() override {
    MpiBaseTestFixture::TearDown();
  }
};

// Test parameters for AllGather tests
struct AllGatherTestParams {
  int numBlocks;
  int blockSize;
  size_t numIntsPerRank;
  std::string testName;
};

// Parameterized test fixture for AllGather tests
class AllGatherTest
    : public AllGatherTestFixture,
      public ::testing::WithParamInterface<AllGatherTestParams> {};

// Test all_gather with actual data transfer and verification
TEST_P(AllGatherTest, AllGatherBasic) {
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

  // Configuration for P2pNvlTransport
  const size_t sendcount = numIntsPerRank * sizeof(int32_t);
  const size_t recvBufferSize = numRanks * sendcount;

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = std::max(size_t(2048), recvBufferSize), // At least 2KB
      .chunkSize = 512, // 512 byte chunk size
      .pipelineDepth = 4,
  };

  // Create transport and exchange IPC handles
  auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
  MultiPeerNvlTransport transport(globalRank, numRanks, bootstrap, config);
  transport.exchange();
  XLOGF(DBG1, "Rank {} created transport and exchanged IPC", globalRank);

  // Get transport devices for all peer ranks
  P2pSelfTransportDevice selfTransport;

  // Create transport array: rank 0, rank 1, ..., rank N-1
  // For my rank, use SelfTransportDevice; for others, use P2pNvlTransportDevice
  std::vector<Transport> h_transports;
  h_transports.reserve(numRanks);

  for (int rank = 0; rank < numRanks; rank++) {
    if (rank == globalRank) {
      h_transports.emplace_back(selfTransport);
    } else {
      h_transports.emplace_back(transport.getP2pTransportDevice(rank));
    }
  }

  // Copy transports to device
  DeviceBuffer d_transports(sizeof(Transport) * numRanks);
  CUDACHECK_TEST(cudaMemcpy(
      d_transports.get(),
      h_transports.data(),
      sizeof(Transport) * numRanks,
      cudaMemcpyHostToDevice));

  DeviceSpan<Transport> transports_span(
      static_cast<Transport*>(d_transports.get()), numRanks);

  // Allocate send and recv buffers
  // sendbuff: numIntsPerRank ints (my local data)
  // recvbuff: numRanks * numIntsPerRank ints (gathered data from all ranks)
  DeviceBuffer sendBuffer(sendcount);
  DeviceBuffer recvBuffer(recvBufferSize);

  // Initialize recv buffer with -1
  const size_t totalRecvInts = numRanks * numIntsPerRank;
  test::fillBuffer(reinterpret_cast<int*>(recvBuffer.get()), -1, totalRecvInts);

  // Fill send buffer: each rank fills with pattern based on rank ID and
  // position Rank R at position i: value = R * 1000 + i Rank 0: 0, 1, 2, 3, ...
  // Rank 1: 1000, 1001, 1002, 1003, ...
  // Rank 2: 2000, 2001, 2002, 2003, ...
  std::vector<int32_t> h_send_init(numIntsPerRank);
  for (size_t i = 0; i < numIntsPerRank; i++) {
    h_send_init[i] = globalRank * 1000 + static_cast<int32_t>(i);
  }
  CUDACHECK_TEST(cudaMemcpy(
      sendBuffer.get(), h_send_init.data(), sendcount, cudaMemcpyHostToDevice));

  // Barrier to ensure all ranks are ready
  MPI_Barrier(MPI_COMM_WORLD);

  // Debug: Print send buffer before all_gather
  XLOGF(DBG1, "Rank {}: Send buffer (my data): ", globalRank);
  std::vector<int32_t> h_send_debug(numIntsPerRank);
  CUDACHECK_TEST(cudaMemcpy(
      h_send_debug.data(),
      sendBuffer.get(),
      sendcount,
      cudaMemcpyDeviceToHost));
  std::string sendLine = "  ";
  for (size_t i = 0; i < std::min(size_t(8), numIntsPerRank); i++) {
    sendLine += std::to_string(h_send_debug[i]) + " ";
  }
  if (numIntsPerRank > 8) {
    sendLine += "...";
  }
  XLOG(DBG1) << sendLine;

  // Debug: Print recv buffer before all_gather
  printDeviceBuffer(
      "Recv buffer BEFORE",
      recvBuffer.get(),
      globalRank,
      numRanks,
      numIntsPerRank);

  XLOGF(DBG1, "Rank {}: calling all_gather", globalRank);

  // Call all_gather with actual data transfer
  test::testAllGather(
      recvBuffer.get(),
      sendBuffer.get(),
      sendcount,
      globalRank,
      numRanks,
      transports_span,
      numBlocks,
      blockSize);

  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Debug: Print recv buffer after all_gather
  printDeviceBuffer(
      "Recv buffer AFTER",
      recvBuffer.get(),
      globalRank,
      numRanks,
      numIntsPerRank);

  // Verify received data
  // After AllGather, each rank should have:
  // [data_from_rank0 | data_from_rank1 | ... | data_from_rankN]
  // Expected: from rank R at position i, value = R * 1000 + i
  std::vector<int32_t> h_recv_after(totalRecvInts);
  CUDACHECK_TEST(cudaMemcpy(
      h_recv_after.data(),
      recvBuffer.get(),
      recvBufferSize,
      cudaMemcpyDeviceToHost));

  int h_errorCount = 0;

  // Verify on host for easier debugging
  for (int sourceRank = 0; sourceRank < numRanks; sourceRank++) {
    for (size_t i = 0; i < numIntsPerRank; i++) {
      int32_t expected = sourceRank * 1000 + static_cast<int32_t>(i);
      int32_t actual = h_recv_after[sourceRank * numIntsPerRank + i];
      if (expected != actual) {
        h_errorCount++;
        if (h_errorCount <= 10) { // Only print first 10 errors
          XLOGF(
              ERR,
              "Rank {}: Error at source_rank {} position {}: expected {}, got {}",
              globalRank,
              sourceRank,
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
    AllGatherConfigs,
    AllGatherTest,
    ::testing::Values(
        // Format: #blocks_#threads_#msgsize
        AllGatherTestParams{
            .numBlocks = 4,
            .blockSize = 256,
            .numIntsPerRank = 16,
            .testName = "4b_256t_64B"},
        AllGatherTestParams{
            .numBlocks = 7,
            .blockSize = 256,
            .numIntsPerRank = 64,
            .testName = "7b_256t_256B"},
        AllGatherTestParams{
            .numBlocks = 8,
            .blockSize = 512,
            .numIntsPerRank = 256,
            .testName = "8b_512t_1KB"},
        AllGatherTestParams{
            .numBlocks = 16,
            .blockSize = 128,
            .numIntsPerRank = 32,
            .testName = "16b_128t_128B"},
        AllGatherTestParams{
            .numBlocks = 16,
            .blockSize = 512,
            .numIntsPerRank = 1024,
            .testName = "16b_512t_4KB"},
        AllGatherTestParams{
            .numBlocks = 8,
            .blockSize = 256,
            .numIntsPerRank = 4096,
            .testName = "8b_256t_16KB"}),
    [](const ::testing::TestParamInfo<AllGatherTestParams>& info) {
      return info.param.testName;
    });

// Test with larger message sizes
class AllGatherLargeTest
    : public AllGatherTestFixture,
      public ::testing::WithParamInterface<AllGatherTestParams> {};

TEST_P(AllGatherLargeTest, AllGatherLarge) {
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

  const size_t sendcount = numIntsPerRank * sizeof(int32_t);
  const size_t recvBufferSize = numRanks * sendcount;

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = std::max(size_t(8 * 1024 * 1024), recvBufferSize),
      .chunkSize = 64 * 1024, // 64KB chunk size for large messages
      .pipelineDepth = 4,
  };

  auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
  MultiPeerNvlTransport transport(globalRank, numRanks, bootstrap, config);
  transport.exchange();

  P2pSelfTransportDevice selfTransport;
  std::vector<Transport> h_transports;
  h_transports.reserve(numRanks);

  for (int rank = 0; rank < numRanks; rank++) {
    if (rank == globalRank) {
      h_transports.emplace_back(selfTransport);
    } else {
      h_transports.emplace_back(transport.getP2pTransportDevice(rank));
    }
  }

  DeviceBuffer d_transports(sizeof(Transport) * numRanks);
  CUDACHECK_TEST(cudaMemcpy(
      d_transports.get(),
      h_transports.data(),
      sizeof(Transport) * numRanks,
      cudaMemcpyHostToDevice));

  DeviceSpan<Transport> transports_span(
      static_cast<Transport*>(d_transports.get()), numRanks);

  DeviceBuffer sendBuffer(sendcount);
  DeviceBuffer recvBuffer(recvBufferSize);

  const size_t totalRecvInts = numRanks * numIntsPerRank;
  test::fillBuffer(reinterpret_cast<int*>(recvBuffer.get()), -1, totalRecvInts);

  // Fill send buffer with pattern
  std::vector<int32_t> h_send_init(numIntsPerRank);
  for (size_t i = 0; i < numIntsPerRank; i++) {
    h_send_init[i] = globalRank * 1000000 + static_cast<int32_t>(i);
  }
  CUDACHECK_TEST(cudaMemcpy(
      sendBuffer.get(), h_send_init.data(), sendcount, cudaMemcpyHostToDevice));

  MPI_Barrier(MPI_COMM_WORLD);

  test::testAllGather(
      recvBuffer.get(),
      sendBuffer.get(),
      sendcount,
      globalRank,
      numRanks,
      transports_span,
      numBlocks,
      blockSize);

  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Verify
  std::vector<int32_t> h_recv_after(totalRecvInts);
  CUDACHECK_TEST(cudaMemcpy(
      h_recv_after.data(),
      recvBuffer.get(),
      recvBufferSize,
      cudaMemcpyDeviceToHost));

  int h_errorCount = 0;
  for (int sourceRank = 0; sourceRank < numRanks; sourceRank++) {
    for (size_t i = 0; i < numIntsPerRank; i++) {
      int32_t expected = sourceRank * 1000000 + static_cast<int32_t>(i);
      int32_t actual = h_recv_after[sourceRank * numIntsPerRank + i];
      if (expected != actual) {
        h_errorCount++;
        if (h_errorCount <= 10) {
          XLOGF(
              ERR,
              "Rank {}: Error at source_rank {} position {}: expected {}, got {}",
              globalRank,
              sourceRank,
              i,
              expected,
              actual);
        }
      }
    }
  }

  EXPECT_EQ(h_errorCount, 0) << "Rank " << globalRank << " found "
                             << h_errorCount << " verification errors";

  MPI_Barrier(MPI_COMM_WORLD);
}

INSTANTIATE_TEST_SUITE_P(
    AllGatherLargeConfigs,
    AllGatherLargeTest,
    ::testing::Values(
        AllGatherTestParams{
            .numBlocks = 16,
            .blockSize = 512,
            .numIntsPerRank = 32768, // 128KB
            .testName = "16b_512t_128KB"},
        AllGatherTestParams{
            .numBlocks = 16,
            .blockSize = 512,
            .numIntsPerRank = 131072, // 512KB
            .testName = "16b_512t_512KB"},
        AllGatherTestParams{
            .numBlocks = 16,
            .blockSize = 512,
            .numIntsPerRank = 262144, // 1MB
            .testName = "16b_512t_1MB"}),
    [](const ::testing::TestParamInfo<AllGatherTestParams>& info) {
      return info.param.testName;
    });

} // namespace comms::pipes

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
