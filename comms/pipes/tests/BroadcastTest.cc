// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include "comms/pipes/MultiPeerNvlTransport.h"
#include "comms/pipes/P2pSelfTransportDevice.cuh"
#include "comms/pipes/collectives/Broadcast.cuh"
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

// Test parameters for parameterized Broadcast tests
struct BroadcastTestParams {
  int numBlocks;
  int blockSize;
  size_t numBytes;
  int rootRank; // -1 means use numRanks/2 as root
  std::string testName;
};

// Parameterized test fixture
class BroadcastParamTest
    : public BroadcastTestFixture,
      public ::testing::WithParamInterface<BroadcastTestParams> {};

/**
 * Test broadcast with parameterized configurations.
 *
 * This test verifies that:
 * 1. Root rank's data is correctly broadcast to all other ranks
 * 2. All ranks have identical data after broadcast
 * 3. Data integrity is preserved (no corruption)
 *
 * Data pattern: root_rank * 1000 + position
 * This allows easy identification of data source and position for debugging.
 */
TEST_P(BroadcastParamTest, BroadcastDataTransfer) {
  const auto& params = GetParam();
  const size_t numBytes = params.numBytes;
  const int numBlocks = params.numBlocks;
  const int blockSize = params.blockSize;

  // Resolve root rank: -1 means use middle rank
  const int rootRank =
      params.rootRank < 0 ? numRanks / 2 : params.rootRank % numRanks;

  XLOGF(
      DBG1,
      "Rank {}: Running {} with numBlocks={}, blockSize={}, numBytes={}, root={}",
      globalRank,
      params.testName,
      numBlocks,
      blockSize,
      numBytes,
      rootRank);

  // Configuration for P2pNvlTransport
  // Use a fixed staging buffer size (64KB) - large transfers are handled via
  // pipelining. This keeps memory usage reasonable with 8 ranks (7 peers each).
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = 64 * 1024, // 64KB staging buffer per peer
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
      << "Rank " << globalRank << " verification failed";

  // Ensure all ranks complete before cleanup
  MPI_Barrier(MPI_COMM_WORLD);
}

// Broadcast algorithm types for explicit algorithm testing
enum class BroadcastAlgorithm { FlatTree, BinomialTree, Ring };

// Extended test parameters including algorithm selection
struct BroadcastAlgorithmTestParams {
  int numBlocks;
  int blockSize;
  size_t numBytes;
  int rootRank;
  BroadcastAlgorithm algorithm;
  std::string testName;
};

// Parameterized test fixture for explicit algorithm selection tests
class BroadcastAlgorithmParamTest
    : public BroadcastTestFixture,
      public ::testing::WithParamInterface<BroadcastAlgorithmTestParams> {};

/**
 * Test explicit algorithm selection with various configurations.
 *
 * This test verifies that each broadcast algorithm (flat-tree, binomial tree,
 * ring) correctly broadcasts data from root to all other ranks, regardless of
 * message size. This allows testing algorithms with message sizes outside their
 * typical adaptive selection thresholds.
 */
TEST_P(BroadcastAlgorithmParamTest, AlgorithmDataTransfer) {
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
      test::testBroadcast(
          buffer.get(),
          globalRank,
          rootRank,
          transports_span,
          numBytes,
          numBlocks,
          blockSize);
      break;
    case BroadcastAlgorithm::BinomialTree:
      test::testBroadcastBinomialTree(
          buffer.get(),
          globalRank,
          rootRank,
          transports_span,
          numBytes,
          numBlocks,
          blockSize);
      break;
    case BroadcastAlgorithm::Ring:
      test::testBroadcastRing(
          buffer.get(),
          globalRank,
          rootRank,
          transports_span,
          numBytes,
          numBlocks,
          blockSize);
      break;
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

// Test configurations covering various scenarios:
// - Different message sizes (small, medium, large)
// - Different thread configurations
// - Different root ranks
INSTANTIATE_TEST_SUITE_P(
    BroadcastConfigs,
    BroadcastParamTest,
    ::testing::Values(
        // Small message (64 bytes = 16 int32_t), root=0
        BroadcastTestParams{
            .numBlocks = 4,
            .blockSize = 256,
            .numBytes = 64,
            .rootRank = 0,
            .testName = "small_64B_root0"},

        // Medium message (4KB), root=0
        BroadcastTestParams{
            .numBlocks = 4,
            .blockSize = 256,
            .numBytes = 4096,
            .rootRank = 0,
            .testName = "medium_4KB_root0"},

        // Large message (1MB), root=0
        BroadcastTestParams{
            .numBlocks = 8,
            .blockSize = 512,
            .numBytes = 1048576,
            .rootRank = 0,
            .testName = "large_1MB_root0"},

        // Medium message with non-zero root (middle rank)
        BroadcastTestParams{
            .numBlocks = 4,
            .blockSize = 256,
            .numBytes = 4096,
            .rootRank = -1, // Will use numRanks/2
            .testName = "medium_4KB_middle_root"},

        // Different thread configuration: more blocks, fewer threads
        BroadcastTestParams{
            .numBlocks = 16,
            .blockSize = 128,
            .numBytes = 4096,
            .rootRank = 0,
            .testName = "16b_128t_4KB"},

        // Different thread configuration: single block
        BroadcastTestParams{
            .numBlocks = 1,
            .blockSize = 256,
            .numBytes = 1024,
            .rootRank = 0,
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
            .testName = "medium_4KB_last_root"}),
    [](const ::testing::TestParamInfo<BroadcastTestParams>& info) {
      return info.param.testName;
    });

// Binomial Tree algorithm tests (designed for messages >= 64KB)
// Tests the O(log N) rounds algorithm with various message sizes and root ranks
INSTANTIATE_TEST_SUITE_P(
    BinomialTreeConfigs,
    BroadcastAlgorithmParamTest,
    ::testing::Values(
        // Medium message at threshold (64KB), root=0
        BroadcastAlgorithmTestParams{
            .numBlocks = 8,
            .blockSize = 512,
            .numBytes = 64 * 1024,
            .rootRank = 0,
            .algorithm = BroadcastAlgorithm::BinomialTree,
            .testName = "binomial_64KB_root0"},

        // Large message (256KB), root=0
        BroadcastAlgorithmTestParams{
            .numBlocks = 8,
            .blockSize = 512,
            .numBytes = 256 * 1024,
            .rootRank = 0,
            .algorithm = BroadcastAlgorithm::BinomialTree,
            .testName = "binomial_256KB_root0"},

        // Large message (512KB), middle root
        BroadcastAlgorithmTestParams{
            .numBlocks = 8,
            .blockSize = 512,
            .numBytes = 512 * 1024,
            .rootRank = -1,
            .algorithm = BroadcastAlgorithm::BinomialTree,
            .testName = "binomial_512KB_middle_root"},

        // Near ring threshold (1MB - 1), root=0
        BroadcastAlgorithmTestParams{
            .numBlocks = 8,
            .blockSize = 512,
            .numBytes = 1024 * 1024 - 1024,
            .rootRank = 0,
            .algorithm = BroadcastAlgorithm::BinomialTree,
            .testName = "binomial_1MB_minus_1KB_root0"},

        // Boundary test: last rank as root
        BroadcastAlgorithmTestParams{
            .numBlocks = 8,
            .blockSize = 512,
            .numBytes = 128 * 1024,
            .rootRank = 7,
            .algorithm = BroadcastAlgorithm::BinomialTree,
            .testName = "binomial_128KB_last_root"}),
    [](const ::testing::TestParamInfo<BroadcastAlgorithmTestParams>& info) {
      return info.param.testName;
    });

// Ring algorithm tests (designed for messages >= 1MB)
// Tests the bandwidth-optimized ring algorithm with various message sizes
INSTANTIATE_TEST_SUITE_P(
    RingConfigs,
    BroadcastAlgorithmParamTest,
    ::testing::Values(
        // At ring threshold (1MB), root=0
        BroadcastAlgorithmTestParams{
            .numBlocks = 8,
            .blockSize = 512,
            .numBytes = 1024 * 1024,
            .rootRank = 0,
            .algorithm = BroadcastAlgorithm::Ring,
            .testName = "ring_1MB_root0"},

        // Large message (2MB), root=0
        BroadcastAlgorithmTestParams{
            .numBlocks = 16,
            .blockSize = 512,
            .numBytes = 2 * 1024 * 1024,
            .rootRank = 0,
            .algorithm = BroadcastAlgorithm::Ring,
            .testName = "ring_2MB_root0"},

        // Large message (4MB), middle root
        BroadcastAlgorithmTestParams{
            .numBlocks = 16,
            .blockSize = 512,
            .numBytes = 4 * 1024 * 1024,
            .rootRank = -1,
            .algorithm = BroadcastAlgorithm::Ring,
            .testName = "ring_4MB_middle_root"},

        // Very large message (8MB), root=0
        BroadcastAlgorithmTestParams{
            .numBlocks = 16,
            .blockSize = 512,
            .numBytes = 8 * 1024 * 1024,
            .rootRank = 0,
            .algorithm = BroadcastAlgorithm::Ring,
            .testName = "ring_8MB_root0"},

        // Boundary test: last rank as root
        BroadcastAlgorithmTestParams{
            .numBlocks = 16,
            .blockSize = 512,
            .numBytes = 2 * 1024 * 1024,
            .rootRank = 7,
            .algorithm = BroadcastAlgorithm::Ring,
            .testName = "ring_2MB_last_root"}),
    [](const ::testing::TestParamInfo<BroadcastAlgorithmTestParams>& info) {
      return info.param.testName;
    });

/**
 * Test edge case: single-rank broadcast (no-op).
 *
 * When there's only one rank, broadcast should be a no-op.
 * The data should remain unchanged.
 *
 * Note: This test only runs effectively when launched with ppn=1.
 * When launched with ppn>1, it tests that single-rank edge case
 * handling works even if we manually set nranks=1.
 */
TEST_F(BroadcastTestFixture, SingleRankNoOp) {
  // Skip if actually running multi-rank - this test is for single rank only
  if (numRanks > 1) {
    GTEST_SKIP() << "Skipping single-rank test in multi-rank environment";
  }

  const size_t numBytes = 256;
  const size_t numInts = numBytes / sizeof(int32_t);
  const int rootRank = 0;

  XLOGF(DBG1, "Rank {}: Running single-rank no-op test", globalRank);

  // Create minimal transport setup
  P2pSelfTransportDevice selfTransport;
  std::vector<Transport> h_transports;
  h_transports.emplace_back(selfTransport);

  DeviceBuffer d_transports(sizeof(Transport));
  CUDACHECK_TEST(cudaMemcpy(
      d_transports.get(),
      h_transports.data(),
      sizeof(Transport),
      cudaMemcpyHostToDevice));

  DeviceSpan<Transport> transports_span(
      static_cast<Transport*>(d_transports.get()), 1);

  // Initialize buffer with known pattern
  DeviceBuffer buffer(numBytes);
  std::vector<int32_t> h_init(numInts);
  for (size_t i = 0; i < numInts; i++) {
    h_init[i] = 42 + static_cast<int32_t>(i); // Arbitrary pattern
  }
  CUDACHECK_TEST(cudaMemcpy(
      buffer.get(), h_init.data(), numBytes, cudaMemcpyHostToDevice));

  // Execute broadcast (should be no-op for single rank)
  test::testBroadcast(
      buffer.get(),
      globalRank,
      rootRank,
      transports_span,
      numBytes,
      4, // numBlocks
      256); // blockSize

  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Verify data is unchanged
  std::vector<int32_t> h_result(numInts);
  CUDACHECK_TEST(cudaMemcpy(
      h_result.data(), buffer.get(), numBytes, cudaMemcpyDeviceToHost));

  EXPECT_EQ(h_result, h_init) << "Single-rank broadcast should not modify data";
}

/**
 * Test edge case: zero-byte broadcast (no-op).
 *
 * Broadcasting zero bytes should be a no-op and not crash.
 */
TEST_F(BroadcastTestFixture, ZeroByteBroadcast) {
  const int rootRank = 0;

  XLOGF(DBG1, "Rank {}: Running zero-byte broadcast test", globalRank);

  // Setup transport
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = 2048,
      .chunkSize = 512,
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

  // Allocate a small buffer (won't be used since nbytes=0)
  DeviceBuffer buffer(64);
  ::comms::pipes::test::fillBuffer(
      reinterpret_cast<int*>(buffer.get()), 99, 16);

  MPI_Barrier(MPI_COMM_WORLD);

  // Execute zero-byte broadcast
  test::testBroadcast(
      buffer.get(),
      globalRank,
      rootRank,
      transports_span,
      0, // zero bytes
      4,
      256);

  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Verify buffer is unchanged (should still have 99s)
  std::vector<int32_t> h_result(16);
  CUDACHECK_TEST(
      cudaMemcpy(h_result.data(), buffer.get(), 64, cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < 16; i++) {
    EXPECT_EQ(h_result[i], 99)
        << "Zero-byte broadcast should not modify buffer at index " << i;
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace comms::pipes::collectives

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
