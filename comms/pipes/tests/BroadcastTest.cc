// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include "comms/pipes/MultiPeerNvlTransport.h"
#include "comms/pipes/P2pSelfTransportDevice.cuh"
#include "comms/pipes/tests/BroadcastTest.cuh"
#include "comms/pipes/tests/Utils.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::DeviceBuffer;
using meta::comms::MpiBaseTestFixture;
using meta::comms::MpiBootstrap;
using meta::comms::MPIEnvironmentBase;

namespace comms::pipes::collectives {

namespace {
// Helper function to print device buffer contents for debugging
void printDeviceBuffer(
    const char* label,
    void* deviceBuffer,
    int rank,
    size_t numInts,
    bool showAll = false) {
  std::vector<int32_t> h_buffer(numInts);
  CUDACHECK_TEST(cudaMemcpy(
      h_buffer.data(),
      deviceBuffer,
      numInts * sizeof(int32_t),
      cudaMemcpyDeviceToHost));

  size_t numToShow = showAll ? numInts : std::min(size_t(8), numInts);
  XLOGF(
      DBG1,
      "Rank {}: {} (showing {}/{} values):",
      rank,
      label,
      numToShow,
      numInts);

  std::string line = "  Values: ";
  for (size_t i = 0; i < numToShow; i++) {
    line += std::to_string(h_buffer[i]) + " ";
  }
  if (!showAll && numInts > 8) {
    line += "...";
  }
  XLOG(DBG1) << line;
}
} // namespace

// Test fixture for Broadcast tests using MPI for multi-rank coordination
class BroadcastTestFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();

    // Check GPU availability - tests require 4 GPUs (ppn: 4 in BUCK)
    int deviceCount = 0;
    CUDACHECK_TEST(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 4) {
      GTEST_SKIP() << "Test requires at least 4 GPUs";
    }

    CUDACHECK_TEST(cudaSetDevice(localRank));
  }

  void TearDown() override {
    MpiBaseTestFixture::TearDown();
  }
};

/**
 * Test single-rank broadcast edge case.
 *
 * When there's only one rank, broadcast should be a no-op since the root
 * already has the data and there are no peers to send to. This tests that
 * the implementation correctly handles nranks == 1.
 */
TEST_F(BroadcastTestFixture, SingleRankBroadcastIsNoop) {
  // This test simulates single-rank behavior by creating a transport array
  // with only one entry (self-transport) and verifying data is preserved.
  const size_t numBytes = 4096;
  const int numBlocks = 4;
  const int blockSize = 256;
  const int myRank = 0;
  const int rootRank = 0;

  XLOGF(
      DBG1,
      "Rank {}: Running single-rank broadcast test with numBytes={}",
      globalRank,
      numBytes);

  // Synchronize all ranks before test
  MPI_Barrier(MPI_COMM_WORLD);

  // Build transport array with only self-transport (single rank scenario)
  P2pSelfTransportDevice selfTransport;
  std::vector<Transport> h_transports;
  h_transports.emplace_back(selfTransport);

  // Copy transport to device
  DeviceBuffer d_transports(sizeof(Transport));
  CUDACHECK_TEST(cudaMemcpy(
      d_transports.get(),
      h_transports.data(),
      sizeof(Transport),
      cudaMemcpyHostToDevice));

  DeviceSpan<Transport> transports_span(
      static_cast<Transport*>(d_transports.get()), 1);

  // Allocate and initialize buffer with known data
  DeviceBuffer buffer(numBytes);
  const size_t numInts = numBytes / sizeof(int32_t);

  std::vector<int32_t> h_init(numInts);
  for (size_t i = 0; i < numInts; i++) {
    h_init[i] = rootRank * 1000 + static_cast<int32_t>(i);
  }
  CUDACHECK_TEST(cudaMemcpy(
      buffer.get(), h_init.data(), numBytes, cudaMemcpyHostToDevice));

  // Execute broadcast (should be no-op for single rank)
  test::testBroadcast(
      buffer.get(),
      myRank,
      rootRank,
      transports_span,
      numBytes,
      numBlocks,
      blockSize);

  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Verify data is unchanged (single-rank broadcast is no-op)
  std::vector<int32_t> h_result(numInts);
  CUDACHECK_TEST(cudaMemcpy(
      h_result.data(), buffer.get(), numBytes, cudaMemcpyDeviceToHost));

  EXPECT_EQ(h_result, h_init)
      << "Single-rank broadcast should preserve data unchanged";

  // Ensure all ranks complete before next test
  MPI_Barrier(MPI_COMM_WORLD);
}

// =============================================================================
// Parameterized Broadcast Tests
// =============================================================================

// Test parameters for parameterized Broadcast tests
struct BroadcastParams {
  std::size_t nbytes;
  int rootRank; // 0 = first, -1 = last, positive = specific rank
  int numBlocks;
  int blockSize;
  std::string name;
};

// Parameterized test fixture for broadcast tests
class BroadcastParamTest
    : public BroadcastTestFixture,
      public ::testing::WithParamInterface<BroadcastParams> {};

/**
 * Test broadcast with parameterized configurations.
 *
 * This test verifies that broadcast correctly transfers data from
 * root to all other ranks.
 *
 * Data pattern: root_rank * 1000 + position
 * This allows easy identification of data source and position for debugging.
 */
TEST_P(BroadcastParamTest, DataTransfer) {
  const auto& params = GetParam();
  const size_t numBytes = params.nbytes;
  const int numBlocks = params.numBlocks;
  const int blockSize = params.blockSize;

  // Resolve root rank: -1 means use last rank
  int rootRank = params.rootRank;
  if (rootRank < 0) {
    rootRank = numRanks - 1;
  } else {
    rootRank = rootRank % numRanks;
  }

  XLOGF(
      DBG1,
      "Rank {}: Running {} with numBlocks={}, blockSize={}, numBytes={}, root={}",
      globalRank,
      params.name,
      numBlocks,
      blockSize,
      numBytes,
      rootRank);

  // Configuration for P2pNvlTransport
  // Use larger staging buffer for large messages
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = 8 * 1024 * 1024, // 8MB
      .chunkSize = 32 * 1024, // 32KB
      .pipelineDepth = 2,
  };

  // Create transport and exchange IPC handles across all ranks
  auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
  MultiPeerNvlTransport transport(globalRank, numRanks, bootstrap, config);
  transport.exchange();
  XLOGF(DBG1, "Rank {} created transport and exchanged IPC", globalRank);

  // Build transport array: self-transport for my rank, P2P for peers
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

  // Copy transports to device
  DeviceBuffer d_transports(sizeof(Transport) * numRanks);
  CUDACHECK_TEST(cudaMemcpy(
      d_transports.get(),
      h_transports.data(),
      sizeof(Transport) * numRanks,
      cudaMemcpyHostToDevice));

  DeviceSpan<Transport> transports_span(
      static_cast<Transport*>(d_transports.get()), numRanks);

  // Handle zero-byte edge case: verify broadcast is a no-op
  if (numBytes == 0) {
    MPI_Barrier(MPI_COMM_WORLD);

    // Execute zero-byte broadcast (should be no-op)
    test::testBroadcast(
        nullptr,
        globalRank,
        rootRank,
        transports_span,
        0,
        numBlocks,
        blockSize);

    CUDACHECK_TEST(cudaDeviceSynchronize());
    XLOGF(DBG1, "Rank {}: zero-byte broadcast completed (no-op)", globalRank);

    MPI_Barrier(MPI_COMM_WORLD);
    return;
  }

  // Allocate broadcast buffer
  DeviceBuffer buffer(numBytes);

  // Calculate number of int32_t elements
  const size_t numInts = numBytes / sizeof(int32_t);

  // Initialize buffer: root has data, non-roots have -1 (sentinel)
  if (globalRank == rootRank) {
    // Root: fill with pattern root_rank * 1000 + position
    std::vector<int32_t> h_init(numInts);
    for (size_t i = 0; i < numInts; i++) {
      h_init[i] = rootRank * 1000 + static_cast<int32_t>(i % 1000);
    }
    CUDACHECK_TEST(cudaMemcpy(
        buffer.get(), h_init.data(), numBytes, cudaMemcpyHostToDevice));
  } else {
    // Non-root: initialize with -1 to detect missing writes
    ::comms::pipes::test::fillBuffer(
        reinterpret_cast<int*>(buffer.get()), -1, numInts);
  }

  // Debug: print buffer before broadcast
  printDeviceBuffer("Buffer BEFORE", buffer.get(), globalRank, numInts);

  // Synchronize all ranks before broadcast
  MPI_Barrier(MPI_COMM_WORLD);

  XLOGF(DBG1, "Rank {}: calling broadcast with root={}", globalRank, rootRank);

  // Execute broadcast
  test::testBroadcast(
      buffer.get(),
      globalRank,
      rootRank,
      transports_span,
      numBytes,
      numBlocks,
      blockSize);

  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Debug: print buffer after broadcast
  printDeviceBuffer("Buffer AFTER", buffer.get(), globalRank, numInts);

  // Verify all ranks have correct data
  // Expected: value[i] = rootRank * 1000 + (i % 1000)

  // Generate expected data
  std::vector<int32_t> h_expected(numInts);
  for (size_t i = 0; i < numInts; i++) {
    h_expected[i] = rootRank * 1000 + static_cast<int32_t>(i % 1000);
  }

  // Copy result from device
  std::vector<int32_t> h_result(numInts);
  CUDACHECK_TEST(cudaMemcpy(
      h_result.data(), buffer.get(), numBytes, cudaMemcpyDeviceToHost));

  XLOGF(DBG1, "Rank {}: verification completed", globalRank);
  EXPECT_EQ(h_result, h_expected)
      << "Rank " << globalRank << " verification failed";

  // Ensure all ranks complete before cleanup
  MPI_Barrier(MPI_COMM_WORLD);
}

INSTANTIATE_TEST_SUITE_P(
    BroadcastConfigs,
    BroadcastParamTest,
    ::testing::Values(
        // Zero/edge cases
        BroadcastParams{0, 0, 4, 256, "zero_bytes"},

        // Small messages
        BroadcastParams{64, 0, 4, 256, "small_64B"},
        BroadcastParams{1024, 0, 4, 256, "small_1KB"},
        BroadcastParams{4096, 0, 4, 256, "small_4KB"},

        // Medium messages
        BroadcastParams{64 * 1024, 0, 8, 256, "medium_64KB"},
        BroadcastParams{256 * 1024, 0, 8, 256, "medium_256KB"},
        BroadcastParams{512 * 1024, 0, 8, 512, "medium_512KB"},

        // Large messages
        BroadcastParams{1024 * 1024, 0, 8, 512, "large_1MB"},
        BroadcastParams{2 * 1024 * 1024, 0, 8, 512, "large_2MB"},
        BroadcastParams{4 * 1024 * 1024, 0, 8, 512, "large_4MB"},
        BroadcastParams{8 * 1024 * 1024, 0, 8, 512, "large_8MB"},

        // Different roots
        BroadcastParams{1024 * 1024, -1, 8, 512, "large_1MB_last_root"},
        BroadcastParams{2 * 1024 * 1024, 2, 8, 512, "large_2MB_middle_root"},

        // Non-aligned sizes
        BroadcastParams{1000, 0, 4, 256, "non_aligned_1000B"},
        BroadcastParams{65537 * 4, 0, 8, 256, "non_aligned_64KB_plus_1"}),
    [](const ::testing::TestParamInfo<BroadcastParams>& info) {
      return info.param.name;
    });

} // namespace comms::pipes::collectives

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
