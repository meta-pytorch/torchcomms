// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include "comms/pipes/MultiPeerNvlTransport.h"
#include "comms/pipes/P2pSelfTransportDevice.cuh"
#include "comms/pipes/collectives/BroadcastFlat.cuh"
#include "comms/pipes/tests/BroadcastTest.cuh"
#include "comms/pipes/tests/Utils.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/CudaRAII.h"

using namespace meta::comms;

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
  test::testBroadcastFlat(
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

// Broadcast algorithm types for parameterized algorithm testing
enum class BroadcastAlgorithm { FlatTree, BinomialTree, Ring };

// Test parameters for parameterized Broadcast tests
struct BroadcastTestParams {
  int numBlocks;
  int blockSize;
  size_t numBytes;
  int rootRank; // -1 means use numRanks/2 as root
  BroadcastAlgorithm algorithm;
  std::string testName;
};

// Parameterized test fixture for broadcast tests
class BroadcastParamTest
    : public BroadcastTestFixture,
      public ::testing::WithParamInterface<BroadcastTestParams> {};

/**
 * Test broadcast with parameterized algorithm and configurations.
 *
 * This test verifies that each broadcast algorithm (flat-tree,
 * later binomial tree and ring) correctly broadcasts data from
 * root to all other ranks.
 *
 * Data pattern: root_rank * 1000 + position
 * This allows easy identification of data source and position for debugging.
 */
TEST_P(BroadcastParamTest, DataTransfer) {
  const auto& params = GetParam();
  const size_t numBytes = params.numBytes;
  const int numBlocks = params.numBlocks;
  const int blockSize = params.blockSize;

  // Resolve root rank: -1 means use middle rank
  const int rootRank =
      params.rootRank < 0 ? numRanks / 2 : params.rootRank % numRanks;

  const char* algorithmName = params.algorithm == BroadcastAlgorithm::FlatTree
      ? "FlatTree"
      : (params.algorithm == BroadcastAlgorithm::BinomialTree ? "BinomialTree"
                                                              : "Ring");

  XLOGF(
      DBG1,
      "Rank {}: Running {} with algorithm={}, numBlocks={}, blockSize={}, numBytes={}, root={}",
      globalRank,
      params.testName,
      algorithmName,
      numBlocks,
      blockSize,
      numBytes,
      rootRank);

  // Configuration for P2pNvlTransport
  // Use larger staging buffer for large messages
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = 256 * 1024,
      .chunkSize = 512,
      .pipelineDepth = 4,
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
    switch (params.algorithm) {
      case BroadcastAlgorithm::FlatTree:
        test::testBroadcastFlat(
            nullptr,
            globalRank,
            rootRank,
            transports_span,
            0,
            numBlocks,
            blockSize);
        break;
      default:
        GTEST_SKIP() << "Broadcast algorithm " << algorithmName
                     << " not implemented yet";
        return;
    }

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
      h_init[i] = rootRank * 1000 + static_cast<int32_t>(i);
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

  XLOGF(
      DBG1,
      "Rank {}: calling {} broadcast with root={}",
      globalRank,
      algorithmName,
      rootRank);

  // Execute broadcast with the specified algorithm
  switch (params.algorithm) {
    case BroadcastAlgorithm::FlatTree:
      test::testBroadcastFlat(
          buffer.get(),
          globalRank,
          rootRank,
          transports_span,
          numBytes,
          numBlocks,
          blockSize);
      break;
      // TODO: add support for binomial tree and ring.
    default:
      GTEST_SKIP() << "Broadcast algorithm " << algorithmName
                   << " not implemented yet";
      return;
  }

  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Debug: print buffer after broadcast
  printDeviceBuffer("Buffer AFTER", buffer.get(), globalRank, numInts);

  // Verify all ranks have correct data
  // Expected: value[i] = rootRank * 1000 + i

  // Generate expected data
  std::vector<int32_t> h_expected(numInts);
  for (size_t i = 0; i < numInts; i++) {
    h_expected[i] = rootRank * 1000 + static_cast<int32_t>(i);
  }

  // Copy result from device
  std::vector<int32_t> h_result(numInts);
  CUDACHECK_TEST(cudaMemcpy(
      h_result.data(), buffer.get(), numBytes, cudaMemcpyDeviceToHost));

  // Direct comparison - Google Test provides detailed diff on failure
  XLOGF(DBG1, "Rank {}: verification completed", globalRank);
  EXPECT_EQ(h_result, h_expected)
      << "Rank " << globalRank << " verification failed using "
      << algorithmName;

  // Ensure all ranks complete before cleanup
  MPI_Barrier(MPI_COMM_WORLD);
}

// Flat Tree algorithm tests
// Tests the FlatTree algorithm with various message sizes, root ranks,
// thread configurations, and edge cases (zero bytes, non-aligned sizes)
INSTANTIATE_TEST_SUITE_P(
    FlatTreeConfigs,
    BroadcastParamTest,
    ::testing::Values(
        // Edge case: zero-byte broadcast (should be no-op)
        BroadcastTestParams{
            .numBlocks = 4,
            .blockSize = 256,
            .numBytes = 0,
            .rootRank = 0,
            .algorithm = BroadcastAlgorithm::FlatTree,
            .testName = "zero_bytes"},

        // Small message (64 bytes = 16 int32_t), root=0
        BroadcastTestParams{
            .numBlocks = 4,
            .blockSize = 256,
            .numBytes = 64,
            .rootRank = 0,
            .algorithm = BroadcastAlgorithm::FlatTree,
            .testName = "small_64B_root0"},

        // Non-power-of-2 message size (1000 bytes = 250 int32_t)
        // Tests handling of non-aligned buffer sizes
        BroadcastTestParams{
            .numBlocks = 4,
            .blockSize = 256,
            .numBytes = 1000,
            .rootRank = 0,
            .algorithm = BroadcastAlgorithm::FlatTree,
            .testName = "non_aligned_1000B"},

        // Medium message (4KB), root=0
        BroadcastTestParams{
            .numBlocks = 4,
            .blockSize = 256,
            .numBytes = 4096,
            .rootRank = 0,
            .algorithm = BroadcastAlgorithm::FlatTree,
            .testName = "medium_4KB_root0"},

        // Medium message at threshold (64KB), root=0
        BroadcastTestParams{
            .numBlocks = 8,
            .blockSize = 512,
            .numBytes = 64 * 1024,
            .rootRank = 0,
            .algorithm = BroadcastAlgorithm::FlatTree,
            .testName = "medium_64KB_root0"},

        // Large message (256KB), root=0
        BroadcastTestParams{
            .numBlocks = 8,
            .blockSize = 512,
            .numBytes = 256 * 1024,
            .rootRank = 0,
            .algorithm = BroadcastAlgorithm::FlatTree,
            .testName = "large_256KB_root0"},

        // Large message (512KB), middle root
        BroadcastTestParams{
            .numBlocks = 8,
            .blockSize = 512,
            .numBytes = 512 * 1024,
            .rootRank = -1,
            .algorithm = BroadcastAlgorithm::FlatTree,
            .testName = "large_512KB_middle_root"},

        // Large message (1MB), root=0
        BroadcastTestParams{
            .numBlocks = 8,
            .blockSize = 512,
            .numBytes = 1048576,
            .rootRank = 0,
            .algorithm = BroadcastAlgorithm::FlatTree,
            .testName = "large_1MB_root0"},

        // Near ring threshold (1MB - 1KB), root=0
        BroadcastTestParams{
            .numBlocks = 8,
            .blockSize = 512,
            .numBytes = 1024 * 1024 - 1024,
            .rootRank = 0,
            .algorithm = BroadcastAlgorithm::FlatTree,
            .testName = "large_1MB_minus_1KB_root0"},

        // Different thread configuration: more blocks, fewer threads
        BroadcastTestParams{
            .numBlocks = 16,
            .blockSize = 128,
            .numBytes = 4096,
            .rootRank = 0,
            .algorithm = BroadcastAlgorithm::FlatTree,
            .testName = "16b_128t_4KB"},

        // Different thread configuration: single block
        BroadcastTestParams{
            .numBlocks = 1,
            .blockSize = 256,
            .numBytes = 1024,
            .rootRank = 0,
            .algorithm = BroadcastAlgorithm::FlatTree,
            .testName = "1b_256t_1KB"},

        // Boundary test: last rank as root (root_rank = nranks - 1)
        // This verifies the peer mapping correctly handles the case where
        // peer_idx values 0 through nranks-2 all map to ranks < root_rank,
        // ensuring no off-by-one errors in the mapping logic.
        BroadcastTestParams{
            .numBlocks = 4,
            .blockSize = 256,
            .numBytes = 4096,
            .rootRank = 7, // Last rank in 8-rank configuration
            .algorithm = BroadcastAlgorithm::FlatTree,
            .testName = "medium_4KB_last_root"},

        // Boundary test: last rank as root with larger message
        BroadcastTestParams{
            .numBlocks = 8,
            .blockSize = 512,
            .numBytes = 128 * 1024,
            .rootRank = 7,
            .algorithm = BroadcastAlgorithm::FlatTree,
            .testName = "large_128KB_last_root"}),
    [](const ::testing::TestParamInfo<BroadcastTestParams>& info) {
      return info.param.testName;
    });

} // namespace comms::pipes::collectives

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
