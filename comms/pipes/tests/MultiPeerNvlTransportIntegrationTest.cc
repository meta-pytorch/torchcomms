// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include <string>
#include <vector>

#include "comms/pipes/MultiPeerDeviceTransport.cuh"
#include "comms/pipes/MultiPeerNvlTransport.h"
#include "comms/pipes/tests/MultiPeerNvlTransportIntegrationTest.cuh"
#include "comms/pipes/tests/Utils.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::DeviceBuffer;
using meta::comms::MpiBaseTestFixture;
using meta::comms::MpiBootstrap;
using meta::comms::MPIEnvironmentBase;

namespace comms::pipes::tests {

// =============================================================================
// Test Configuration Constants
// =============================================================================

namespace {
// Default configuration for transport setup
constexpr std::size_t kDefaultDataBufferSize = 1024 * 1024; // 1MB
constexpr std::size_t kDefaultChunkSize = 1024;
constexpr std::size_t kDefaultPipelineDepth = 4;

// Transfer sizes for data tests
constexpr std::size_t kSmallTransferSize = 1024 * 1024; // 1MB

// Stress test parameters
constexpr int kStressIterations = 50;

// Kernel launch parameters
constexpr int kDefaultNumBlocks = 4;
constexpr int kDefaultBlockSize = 128;
} // namespace

// =============================================================================
// Test Fixture
// =============================================================================

class MultiPeerNvlTransportIntegrationTestFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaDeviceSynchronize());
    MpiBaseTestFixture::TearDown();
  }

  // Helper to create a configured transport and get the
  // MultiPeerDeviceTransport
  std::pair<std::unique_ptr<MultiPeerNvlTransport>, MultiPeerDeviceTransport>
  createTransport(const MultiPeerNvlTransportConfig& config) {
    auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
    auto transport = std::make_unique<MultiPeerNvlTransport>(
        globalRank, numRanks, bootstrap, config);
    transport->exchange();
    auto device = transport->getMultiPeerDeviceTransport();
    return {std::move(transport), device};
  }
};

// =============================================================================
// getMultiPeerDeviceTransport() End-to-End Test
// =============================================================================

TEST_F(
    MultiPeerNvlTransportIntegrationTestFixture,
    GetMultiPeerDeviceTransport) {
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = kDefaultDataBufferSize,
      .chunkSize = kDefaultChunkSize,
      .pipelineDepth = kDefaultPipelineDepth,
  };

  auto [transport, device] = createTransport(config);

  // Allocate result buffer on device
  DeviceBuffer resultsBuffer(3 * sizeof(int));
  auto results_d = static_cast<int*>(resultsBuffer.get());

  // Call test kernel to verify accessors
  test::testMultiPeerDeviceTransportAccessors(device, results_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Copy results back to host
  std::vector<int> results_h(3);
  CUDACHECK_TEST(cudaMemcpy(
      results_h.data(), results_d, 3 * sizeof(int), cudaMemcpyDeviceToHost));

  EXPECT_EQ(results_h[0], globalRank) << "rank() should return " << globalRank;
  EXPECT_EQ(results_h[1], numRanks) << "nRanks() should return " << numRanks;
  EXPECT_EQ(results_h[2], numRanks - 1)
      << "numPeers() should return " << (numRanks - 1);

  XLOGF(
      INFO,
      "Rank {}: getMultiPeerDeviceTransport test completed (rank={}, nRanks={}, numPeers={})",
      globalRank,
      results_h[0],
      results_h[1],
      results_h[2]);
}

// =============================================================================
// Repeated getMultiPeerDeviceTransport() Calls
// =============================================================================

TEST_F(
    MultiPeerNvlTransportIntegrationTestFixture,
    GetMultiPeerDeviceTransportRepeated) {
  // Test that getMultiPeerDeviceTransport() can be called multiple times
  // and returns consistent results (tests lazy initialization)
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = kDefaultDataBufferSize,
      .chunkSize = kDefaultChunkSize,
      .pipelineDepth = kDefaultPipelineDepth,
  };

  auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
  MultiPeerNvlTransport transport(globalRank, numRanks, bootstrap, config);
  transport.exchange();

  // Call getMultiPeerDeviceTransport() multiple times
  auto device1 = transport.getMultiPeerDeviceTransport();
  auto device2 = transport.getMultiPeerDeviceTransport();
  auto device3 = transport.getMultiPeerDeviceTransport();

  // Allocate result buffers
  DeviceBuffer results1Buffer(3 * sizeof(int));
  DeviceBuffer results2Buffer(3 * sizeof(int));
  DeviceBuffer results3Buffer(3 * sizeof(int));

  auto results1_d = static_cast<int*>(results1Buffer.get());
  auto results2_d = static_cast<int*>(results2Buffer.get());
  auto results3_d = static_cast<int*>(results3Buffer.get());

  test::testMultiPeerDeviceTransportAccessors(device1, results1_d);
  test::testMultiPeerDeviceTransportAccessors(device2, results2_d);
  test::testMultiPeerDeviceTransportAccessors(device3, results3_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<int> results1_h(3), results2_h(3), results3_h(3);
  CUDACHECK_TEST(cudaMemcpy(
      results1_h.data(), results1_d, 3 * sizeof(int), cudaMemcpyDeviceToHost));
  CUDACHECK_TEST(cudaMemcpy(
      results2_h.data(), results2_d, 3 * sizeof(int), cudaMemcpyDeviceToHost));
  CUDACHECK_TEST(cudaMemcpy(
      results3_h.data(), results3_d, 3 * sizeof(int), cudaMemcpyDeviceToHost));

  // All calls should return the same values
  EXPECT_EQ(results1_h, results2_h)
      << "Repeated getMultiPeerDeviceTransport calls returned different values";
  EXPECT_EQ(results1_h, results3_h)
      << "Repeated getMultiPeerDeviceTransport calls returned different values";

  XLOGF(
      INFO,
      "Rank {}: Repeated getMultiPeerDeviceTransport test completed",
      globalRank);
}

// =============================================================================
// Multi-GPU Send/Recv Test
// =============================================================================

TEST_F(MultiPeerNvlTransportIntegrationTestFixture, SendRecv) {
  if (numRanks != 2) {
    GTEST_SKIP() << "Requires exactly 2 ranks, got " << numRanks;
  }

  const size_t dataBufferSize = 1024 * 1024;
  const size_t nbytes = 4 * 1024 * 1024; // 4MB transfer
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = 1024,
      .pipelineDepth = 4,
  };

  auto [transport, device] = createTransport(config);

  // In a 2-rank setup, each rank has exactly one peer at index 0
  int peerIndex = 0;
  int peerRank = (globalRank == 0) ? 1 : 0;
  const size_t numInts = nbytes / sizeof(int);

  DeviceBuffer srcBuffer(nbytes);
  DeviceBuffer dstBuffer(nbytes);

  auto src_d = static_cast<int*>(srcBuffer.get());
  auto dst_d = static_cast<int*>(dstBuffer.get());

  const int testValue = 42 + globalRank;
  const int expectedValue = 42 + peerRank;

  if (globalRank == 0) {
    // Rank 0: Fill source buffer and send
    test::fillBuffer(src_d, testValue, numInts);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    test::testSinglePeerSend(device, peerIndex, src_d, nbytes, 4, 128);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  } else {
    // Rank 1: Clear destination buffer and receive
    CUDACHECK_TEST(cudaMemset(dst_d, 0, nbytes));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    test::testSinglePeerRecv(device, peerIndex, dst_d, nbytes, 4, 128);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Verify received data
    std::vector<int> hostBuffer(numInts);
    CUDACHECK_TEST(
        cudaMemcpy(hostBuffer.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));

    std::vector<int> expected(numInts, expectedValue);
    EXPECT_EQ(hostBuffer, expected) << "Data mismatch in SendRecv transfer";
  }

  XLOGF(INFO, "Rank {}: Send/Recv test completed", globalRank);
}

// =============================================================================
// Bidirectional Send/Recv Test
// =============================================================================

TEST_F(MultiPeerNvlTransportIntegrationTestFixture, BidirectionalSendRecv) {
  if (numRanks != 2) {
    GTEST_SKIP() << "Requires exactly 2 ranks, got " << numRanks;
  }

  const size_t dataBufferSize = 1024 * 1024;
  const size_t nbytes = 2 * 1024 * 1024; // 2MB transfer each direction
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = 1024,
      .pipelineDepth = 4,
  };

  auto [transport, device] = createTransport(config);

  int peerIndex = 0; // In 2-rank setup, each rank has exactly one peer
  int peerRank = (globalRank == 0) ? 1 : 0;
  const size_t numInts = nbytes / sizeof(int);

  DeviceBuffer sendBuffer(nbytes);
  DeviceBuffer recvBuffer(nbytes);

  auto send_d = static_cast<int*>(sendBuffer.get());
  auto recv_d = static_cast<int*>(recvBuffer.get());

  const int sendValue = 100 + globalRank;
  const int expectedRecvValue = 100 + peerRank;

  // Fill send buffer and clear receive buffer
  test::fillBuffer(send_d, sendValue, numInts);
  CUDACHECK_TEST(cudaMemset(recv_d, 0, nbytes));
  CUDACHECK_TEST(cudaDeviceSynchronize());

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Rank 0 sends then receives, Rank 1 receives then sends
  if (globalRank == 0) {
    test::testSinglePeerSend(device, peerIndex, send_d, nbytes, 4, 128);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    test::testSinglePeerRecv(device, peerIndex, recv_d, nbytes, 4, 128);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  } else {
    test::testSinglePeerRecv(device, peerIndex, recv_d, nbytes, 4, 128);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    test::testSinglePeerSend(device, peerIndex, send_d, nbytes, 4, 128);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  }

  // Verify received data
  std::vector<int> hostBuffer(numInts);
  CUDACHECK_TEST(
      cudaMemcpy(hostBuffer.data(), recv_d, nbytes, cudaMemcpyDeviceToHost));

  std::vector<int> expected(numInts, expectedRecvValue);
  EXPECT_EQ(hostBuffer, expected)
      << "Rank " << globalRank << ": Data mismatch in BidirectionalSendRecv";

  XLOGF(INFO, "Rank {}: Bidirectional Send/Recv test completed", globalRank);
}

// =============================================================================
// Stress Test with Many Iterations
// =============================================================================

TEST_F(MultiPeerNvlTransportIntegrationTestFixture, SendRecvStress) {
  if (numRanks != 2) {
    GTEST_SKIP() << "Requires exactly 2 ranks, got " << numRanks;
  }

  constexpr std::size_t kStressDataBufferSize = 512 * 1024;
  constexpr std::size_t kStressChunkSize = 512;
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = kStressDataBufferSize,
      .chunkSize = kStressChunkSize,
      .pipelineDepth = kDefaultPipelineDepth,
  };

  auto [transport, device] = createTransport(config);

  int peerIndex = 0; // In 2-rank setup, each rank has exactly one peer
  const size_t numInts = kSmallTransferSize / sizeof(int);

  DeviceBuffer srcBuffer(kSmallTransferSize);
  DeviceBuffer dstBuffer(kSmallTransferSize);

  auto src_d = static_cast<int*>(srcBuffer.get());
  auto dst_d = static_cast<int*>(dstBuffer.get());

  for (int iter = 0; iter < kStressIterations; ++iter) {
    const int testValue = 1000 + iter;

    if (globalRank == 0) {
      test::fillBuffer(src_d, testValue, numInts);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      test::testSinglePeerSend(
          device,
          peerIndex,
          src_d,
          kSmallTransferSize,
          kDefaultNumBlocks,
          kDefaultBlockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    } else {
      CUDACHECK_TEST(cudaMemset(dst_d, 0, kSmallTransferSize));

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      test::testSinglePeerRecv(
          device,
          peerIndex,
          dst_d,
          kSmallTransferSize,
          kDefaultNumBlocks,
          kDefaultBlockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      // Verify a sample of received data
      std::vector<int> hostBuffer(numInts);
      CUDACHECK_TEST(cudaMemcpy(
          hostBuffer.data(),
          dst_d,
          kSmallTransferSize,
          cudaMemcpyDeviceToHost));

      EXPECT_EQ(hostBuffer[0], testValue)
          << "Iteration " << iter << ": first element mismatch";
      EXPECT_EQ(hostBuffer[numInts - 1], testValue)
          << "Iteration " << iter << ": last element mismatch";
    }
  }

  XLOGF(
      INFO,
      "Rank {}: Stress test completed ({} iterations)",
      globalRank,
      kStressIterations);
}

// =============================================================================
// Transport Accessor Verification Test
// =============================================================================

TEST_F(MultiPeerNvlTransportIntegrationTestFixture, TransportAccessorTypes) {
  // Test that get_peer_transport/get_self_transport return correct types
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = kDefaultDataBufferSize,
      .chunkSize = kDefaultChunkSize,
      .pipelineDepth = kDefaultPipelineDepth,
  };

  auto [transport, device] = createTransport(config);

  // Allocate result buffer: [numPeers, selfType, peer0Type, peer1Type, ...]
  DeviceBuffer resultsBuffer((1 + numRanks) * sizeof(int));
  auto results_d = static_cast<int*>(resultsBuffer.get());
  CUDACHECK_TEST(cudaMemset(results_d, 0, (1 + numRanks) * sizeof(int)));

  test::testTransportTypes(device, results_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<int> results_h(1 + numRanks);
  CUDACHECK_TEST(cudaMemcpy(
      results_h.data(),
      results_d,
      (1 + numRanks) * sizeof(int),
      cudaMemcpyDeviceToHost));

  EXPECT_EQ(results_h[0], numRanks - 1)
      << "numPeers should be " << (numRanks - 1);

  // Verify self transport is SELF type (0) and others are P2P_NVL (1)
  for (int peer = 0; peer < numRanks; ++peer) {
    int expectedType = (peer == globalRank) ? 0 : 1; // SELF=0, P2P_NVL=1
    EXPECT_EQ(results_h[1 + peer], expectedType)
        << "Transport type mismatch for peer " << peer;
  }

  XLOGF(INFO, "Rank {}: Transport accessor types test completed", globalRank);
}

} // namespace comms::pipes::tests

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  auto mpi_env = std::make_unique<MPIEnvironmentBase>();
  ::testing::AddGlobalTestEnvironment(mpi_env.get());
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
