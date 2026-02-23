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
#include "comms/pipes/window/WindowMemory.h"
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

// Signal and barrier slot counts
constexpr int kDefaultSignalCount = 2;
constexpr int kMultiSlotSignalCount = 4;

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

  // Bundle that keeps transport and window memory alive together.
  // WindowMemory must outlive any device objects extracted from it.
  struct TransportBundle {
    std::unique_ptr<MultiPeerNvlTransport> transport;
    std::unique_ptr<WindowMemory> windowMemory;
    MultiPeerDeviceTransport device;
  };

  // Helper to create a configured transport and get the
  // MultiPeerDeviceTransport
  TransportBundle createTransport(
      const MultiPeerNvlTransportConfig& config,
      const WindowMemoryConfig& wmConfig = {}) {
    auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
    auto transport = std::make_unique<MultiPeerNvlTransport>(
        globalRank, numRanks, bootstrap, config);
    auto wm = std::make_unique<WindowMemory>(
        globalRank, numRanks, bootstrap, wmConfig);
    transport->exchange();
    wm->exchange();
    auto device = transport->getMultiPeerDeviceTransport(*wm);
    return {std::move(transport), std::move(wm), device};
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

  auto [transport, wm, device] = createTransport(config);

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
  WindowMemory wm(globalRank, numRanks, bootstrap, WindowMemoryConfig{});
  transport.exchange();
  wm.exchange();

  // Call getMultiPeerDeviceTransport() multiple times
  auto device1 = transport.getMultiPeerDeviceTransport(wm);
  auto device2 = transport.getMultiPeerDeviceTransport(wm);
  auto device3 = transport.getMultiPeerDeviceTransport(wm);

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
// Multi-GPU Signal/Wait Test
// =============================================================================

TEST_F(MultiPeerNvlTransportIntegrationTestFixture, SignalWait) {
  if (numRanks != 2) {
    GTEST_SKIP() << "Requires exactly 2 ranks, got " << numRanks;
  }

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = kDefaultDataBufferSize,
      .chunkSize = kDefaultChunkSize,
      .pipelineDepth = kDefaultPipelineDepth,
  };

  auto [transport, wm, device] = createTransport(config);

  int peerRank = (globalRank == 0) ? 1 : 0;

  DeviceBuffer resultBuffer(sizeof(int));
  auto result_d = static_cast<int*>(resultBuffer.get());
  CUDACHECK_TEST(cudaMemset(result_d, 0, sizeof(int)));

  // Rank 0 signals, Rank 1 waits
  bool isSignaler = (globalRank == 0);

  // Synchronize before starting
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  test::testSignalWait(device, peerRank, 0, isSignaler, result_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  int result_h = 0;
  CUDACHECK_TEST(
      cudaMemcpy(&result_h, result_d, sizeof(int), cudaMemcpyDeviceToHost));

  EXPECT_EQ(result_h, 1) << "Signal/Wait operation failed";

  XLOGF(
      INFO,
      "Rank {}: Signal/Wait test completed (isSignaler={})",
      globalRank,
      isSignaler);
}

// =============================================================================
// Bidirectional Signal/Wait Test
// =============================================================================

TEST_F(MultiPeerNvlTransportIntegrationTestFixture, BidirectionalSignalWait) {
  if (numRanks != 2) {
    GTEST_SKIP() << "Requires exactly 2 ranks, got " << numRanks;
  }

  const size_t dataBufferSize = 1024 * 1024;
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = 1024,
      .pipelineDepth = 4,
  };
  WindowMemoryConfig wmConfig{
      .signalCount = kDefaultSignalCount, // Need 2 slots for bidirectional
  };

  auto [transport, wm, device] = createTransport(config, wmConfig);

  int peerRank = (globalRank == 0) ? 1 : 0;

  // Use different signal slots for each phase to avoid state accumulation
  constexpr int kPhase1SignalSlot = 0;
  constexpr int kPhase2SignalSlot = 1;

  DeviceBuffer result1Buffer(sizeof(int));
  DeviceBuffer result2Buffer(sizeof(int));
  auto result1_d = static_cast<int*>(result1Buffer.get());
  auto result2_d = static_cast<int*>(result2Buffer.get());
  CUDACHECK_TEST(cudaMemset(result1_d, 0, sizeof(int)));
  CUDACHECK_TEST(cudaMemset(result2_d, 0, sizeof(int)));

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Phase 1: Rank 0 signals slot 0, Rank 1 waits on slot 0
  test::testSignalWait(
      device, peerRank, kPhase1SignalSlot, globalRank == 0, result1_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Phase 2: Rank 1 signals slot 1, Rank 0 waits on slot 1
  // Using different slot to avoid signal value accumulation
  test::testSignalWait(
      device, peerRank, kPhase2SignalSlot, globalRank == 1, result2_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  int result1_h = 0, result2_h = 0;
  CUDACHECK_TEST(
      cudaMemcpy(&result1_h, result1_d, sizeof(int), cudaMemcpyDeviceToHost));
  CUDACHECK_TEST(
      cudaMemcpy(&result2_h, result2_d, sizeof(int), cudaMemcpyDeviceToHost));

  EXPECT_EQ(result1_h, 1) << "Phase 1 Signal/Wait failed";
  EXPECT_EQ(result2_h, 1) << "Phase 2 Signal/Wait failed";

  XLOGF(INFO, "Rank {}: Bidirectional Signal/Wait test completed", globalRank);
}

// =============================================================================
// Multi-GPU Barrier Test
// =============================================================================

TEST_F(MultiPeerNvlTransportIntegrationTestFixture, Barrier) {
  const size_t dataBufferSize = 1024 * 1024;
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = 1024,
      .pipelineDepth = 4,
  };

  auto [transport, wm, device] = createTransport(config);

  DeviceBuffer resultBuffer(sizeof(int));
  auto result_d = static_cast<int*>(resultBuffer.get());
  CUDACHECK_TEST(cudaMemset(result_d, 0, sizeof(int)));

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  test::testBarrier(device, 0, result_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  int result_h = 0;
  CUDACHECK_TEST(
      cudaMemcpy(&result_h, result_d, sizeof(int), cudaMemcpyDeviceToHost));

  EXPECT_EQ(result_h, 1) << "Barrier operation failed";

  XLOGF(INFO, "Rank {}: Barrier test completed", globalRank);
}

// =============================================================================
// Multi-GPU Barrier Peer Test (Two-Sided Barrier)
// =============================================================================

// Test barrier_peer() which synchronizes with a single peer.
TEST_F(MultiPeerNvlTransportIntegrationTestFixture, BarrierPeer) {
  if (numRanks != 2) {
    GTEST_SKIP() << "Requires exactly 2 ranks, got " << numRanks;
  }

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = kDefaultDataBufferSize,
      .chunkSize = kDefaultChunkSize,
      .pipelineDepth = kDefaultPipelineDepth,
  };

  auto [transport, wm, device] = createTransport(config);

  int peerRank = (globalRank == 0) ? 1 : 0;

  DeviceBuffer resultBuffer(sizeof(int));
  auto result_d = static_cast<int*>(resultBuffer.get());
  CUDACHECK_TEST(cudaMemset(result_d, 0, sizeof(int)));

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Both ranks call barrier_peer with each other's rank
  test::testBarrierPeer(device, peerRank, 0, result_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  int result_h = 0;
  CUDACHECK_TEST(
      cudaMemcpy(&result_h, result_d, sizeof(int), cudaMemcpyDeviceToHost));

  EXPECT_EQ(result_h, 1) << "barrier_peer() operation failed on rank "
                         << globalRank;

  XLOGF(
      INFO,
      "Rank {}: BarrierPeer test completed (peerRank={})",
      globalRank,
      peerRank);
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

  auto [transport, wm, device] = createTransport(config);

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

    test::testSinglePeerSend(device, peerRank, src_d, nbytes, 4, 128);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  } else {
    // Rank 1: Clear destination buffer and receive
    CUDACHECK_TEST(cudaMemset(dst_d, 0, nbytes));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    test::testSinglePeerRecv(device, peerRank, dst_d, nbytes, 4, 128);
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

  auto [transport, wm, device] = createTransport(config);

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
    test::testSinglePeerSend(device, peerRank, send_d, nbytes, 4, 128);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    test::testSinglePeerRecv(device, peerRank, recv_d, nbytes, 4, 128);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  } else {
    test::testSinglePeerRecv(device, peerRank, recv_d, nbytes, 4, 128);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    test::testSinglePeerSend(device, peerRank, send_d, nbytes, 4, 128);
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
// Multiple Barrier Iterations Test
// =============================================================================

TEST_F(MultiPeerNvlTransportIntegrationTestFixture, MultipleBarriers) {
  const size_t dataBufferSize = 1024 * 1024;
  // Use multiple barrier slots to avoid state accumulation issues
  constexpr int kNumBarrierSlots = 4;
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = 1024,
      .pipelineDepth = 4,
  };
  WindowMemoryConfig wmConfig{
      .barrierCount = kNumBarrierSlots,
  };

  auto [transport, wm, device] = createTransport(config, wmConfig);

  const int numIterations = 10;

  DeviceBuffer resultBuffer(sizeof(int));
  auto result_d = static_cast<int*>(resultBuffer.get());

  for (int iter = 0; iter < numIterations; ++iter) {
    // Use modular slot selection to avoid barrier state accumulation
    // Each iteration uses a different slot in round-robin fashion
    int slotIdx = iter % kNumBarrierSlots;

    CUDACHECK_TEST(cudaMemset(result_d, 0, sizeof(int)));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    test::testBarrier(device, slotIdx, result_d);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    int result_h = 0;
    CUDACHECK_TEST(
        cudaMemcpy(&result_h, result_d, sizeof(int), cudaMemcpyDeviceToHost));

    EXPECT_EQ(result_h, 1) << "Barrier iteration " << iter << " (slot "
                           << slotIdx << ") failed on rank " << globalRank;
  }

  XLOGF(
      INFO,
      "Rank {}: Multiple Barriers test completed ({} iterations, {} slots)",
      globalRank,
      numIterations,
      kNumBarrierSlots);
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

  auto [transport, wm, device] = createTransport(config);

  int peerRank = (globalRank == 0) ? 1 : 0;
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
          peerRank,
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
          peerRank,
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
// Tests with Custom signalCount Configuration
// =============================================================================

TEST_F(MultiPeerNvlTransportIntegrationTestFixture, MultipleSignalSlots) {
  if (numRanks != 2) {
    GTEST_SKIP() << "Requires exactly 2 ranks, got " << numRanks;
  }

  const size_t dataBufferSize = 1024 * 1024;
  const int numSignalSlots = 4;
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = 1024,
      .pipelineDepth = 4,
  };
  WindowMemoryConfig wmConfig{
      .signalCount = numSignalSlots,
  };

  auto [transport, wm, device] = createTransport(config, wmConfig);

  int peerRank = (globalRank == 0) ? 1 : 0;

  // Test signaling on each slot
  for (int slotIdx = 0; slotIdx < numSignalSlots; ++slotIdx) {
    DeviceBuffer resultBuffer(sizeof(int));
    auto result_d = static_cast<int*>(resultBuffer.get());
    CUDACHECK_TEST(cudaMemset(result_d, 0, sizeof(int)));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Alternate which rank signals vs waits for each slot
    bool isSignaler = ((globalRank + slotIdx) % 2 == 0);
    test::testSignalWait(device, peerRank, slotIdx, isSignaler, result_d);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    int result_h = 0;
    CUDACHECK_TEST(
        cudaMemcpy(&result_h, result_d, sizeof(int), cudaMemcpyDeviceToHost));

    EXPECT_EQ(result_h, 1) << "Signal slot " << slotIdx << " failed on rank "
                           << globalRank;
  }

  XLOGF(
      INFO,
      "Rank {}: Multiple signal slots test completed ({} slots)",
      globalRank,
      numSignalSlots);
}

TEST_F(MultiPeerNvlTransportIntegrationTestFixture, ConcurrentSignalSlots) {
  if (numRanks != 2) {
    GTEST_SKIP() << "Requires exactly 2 ranks, got " << numRanks;
  }

  const size_t dataBufferSize = 1024 * 1024;
  const int numSignalSlots = 4;
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = 1024,
      .pipelineDepth = 4,
  };
  WindowMemoryConfig wmConfig{
      .signalCount = numSignalSlots,
  };

  auto [transport, wm, device] = createTransport(config, wmConfig);

  int peerRank = (globalRank == 0) ? 1 : 0;

  // Test using all signal slots in sequence within a single phase
  // Rank 0 signals all slots, Rank 1 waits on all slots
  std::vector<DeviceBuffer> resultBuffers;
  std::vector<int*> results_d;
  for (int i = 0; i < numSignalSlots; ++i) {
    resultBuffers.emplace_back(sizeof(int));
    results_d.push_back(static_cast<int*>(resultBuffers.back().get()));
    CUDACHECK_TEST(cudaMemset(results_d.back(), 0, sizeof(int)));
  }

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Signal/wait on all slots
  for (int slotIdx = 0; slotIdx < numSignalSlots; ++slotIdx) {
    bool isSignaler = (globalRank == 0);
    test::testSignalWait(
        device, peerRank, slotIdx, isSignaler, results_d[slotIdx]);
  }
  CUDACHECK_TEST(cudaDeviceSynchronize());

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Verify all slots succeeded
  for (int slotIdx = 0; slotIdx < numSignalSlots; ++slotIdx) {
    int result_h = 0;
    CUDACHECK_TEST(cudaMemcpy(
        &result_h, results_d[slotIdx], sizeof(int), cudaMemcpyDeviceToHost));
    EXPECT_EQ(result_h, 1) << "Concurrent signal slot " << slotIdx
                           << " failed on rank " << globalRank;
  }

  XLOGF(
      INFO,
      "Rank {}: Concurrent signal slots test completed ({} slots)",
      globalRank,
      numSignalSlots);
}

// =============================================================================
// Tests with Custom barrierCount Configuration
// =============================================================================

TEST_F(MultiPeerNvlTransportIntegrationTestFixture, MultipleBarrierSlots) {
  const size_t dataBufferSize = 1024 * 1024;
  const int numBarrierSlots = 4;
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = 1024,
      .pipelineDepth = 4,
  };
  WindowMemoryConfig wmConfig{
      .barrierCount = numBarrierSlots,
  };

  auto [transport, wm, device] = createTransport(config, wmConfig);

  // Test each barrier slot
  for (int slotIdx = 0; slotIdx < numBarrierSlots; ++slotIdx) {
    DeviceBuffer resultBuffer(sizeof(int));
    auto result_d = static_cast<int*>(resultBuffer.get());
    CUDACHECK_TEST(cudaMemset(result_d, 0, sizeof(int)));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    test::testBarrier(device, slotIdx, result_d);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    int result_h = 0;
    CUDACHECK_TEST(
        cudaMemcpy(&result_h, result_d, sizeof(int), cudaMemcpyDeviceToHost));

    EXPECT_EQ(result_h, 1) << "Barrier slot " << slotIdx << " failed on rank "
                           << globalRank;
  }

  XLOGF(
      INFO,
      "Rank {}: Multiple barrier slots test completed ({} slots)",
      globalRank,
      numBarrierSlots);
}

TEST_F(MultiPeerNvlTransportIntegrationTestFixture, BarrierSlotStress) {
  const size_t dataBufferSize = 1024 * 1024;
  const int numBarrierSlots = 4;
  const int numIterations = 20;
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = 1024,
      .pipelineDepth = 4,
  };
  WindowMemoryConfig wmConfig{
      .barrierCount = numBarrierSlots,
  };

  auto [transport, wm, device] = createTransport(config, wmConfig);

  DeviceBuffer resultBuffer(sizeof(int));
  auto result_d = static_cast<int*>(resultBuffer.get());

  // Stress test: cycle through all barrier slots multiple times
  for (int iter = 0; iter < numIterations; ++iter) {
    int slotIdx = iter % numBarrierSlots;
    CUDACHECK_TEST(cudaMemset(result_d, 0, sizeof(int)));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    test::testBarrier(device, slotIdx, result_d);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    int result_h = 0;
    CUDACHECK_TEST(
        cudaMemcpy(&result_h, result_d, sizeof(int), cudaMemcpyDeviceToHost));

    EXPECT_EQ(result_h, 1) << "Barrier stress iter " << iter << " (slot "
                           << slotIdx << ") failed on rank " << globalRank;
  }

  XLOGF(
      INFO,
      "Rank {}: Barrier slot stress test completed ({} iterations, {} slots)",
      globalRank,
      numIterations,
      numBarrierSlots);
}

// =============================================================================
// Barrier Monotonic Counters Test
// =============================================================================

TEST_F(MultiPeerNvlTransportIntegrationTestFixture, BarrierMonotonicCounters) {
  const size_t dataBufferSize = 1024 * 1024;
  constexpr int kNumPhases = 3;
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = 1024,
      .pipelineDepth = 4,
  };
  WindowMemoryConfig wmConfig{
      .barrierCount = 1, // Single barrier slot, reused via monotonic counters
  };

  auto [transport, wm, device] = createTransport(config, wmConfig);

  DeviceBuffer resultBuffer(sizeof(int));
  auto result_d = static_cast<int*>(resultBuffer.get());
  CUDACHECK_TEST(cudaMemset(result_d, 0, sizeof(int)));

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Launch a single kernel that performs kNumPhases barrier synchronizations
  // on the same slot. Counters accumulate monotonically — no reset needed.
  test::testBarrierMonotonic(device, 0, kNumPhases, result_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  int result_h = 0;
  CUDACHECK_TEST(
      cudaMemcpy(&result_h, result_d, sizeof(int), cudaMemcpyDeviceToHost));

  EXPECT_EQ(result_h, 1) << "Barrier monotonic counters test failed on rank "
                         << globalRank;

  XLOGF(
      INFO,
      "Rank {}: Barrier monotonic counters test completed ({} phases)",
      globalRank,
      kNumPhases);
}

// =============================================================================
// Barrier Multi-Block Stress Test (Issue 3)
// =============================================================================

TEST_F(MultiPeerNvlTransportIntegrationTestFixture, BarrierMultiBlockStress) {
  constexpr int kNumBlocks = 8;
  constexpr int kNumBarrierSlots = 8;

  const size_t dataBufferSize = 1024 * 1024;
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = 1024,
      .pipelineDepth = 4,
  };
  WindowMemoryConfig wmConfig{
      .barrierCount = kNumBarrierSlots,
  };

  auto [transport, wm, device] = createTransport(config, wmConfig);

  DeviceBuffer resultsBuffer(kNumBlocks * sizeof(int));
  auto results_d = static_cast<int*>(resultsBuffer.get());
  CUDACHECK_TEST(cudaMemset(results_d, 0, kNumBlocks * sizeof(int)));

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  test::testBarrierMultiBlockStress(
      device, kNumBarrierSlots, results_d, kNumBlocks);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Verify all blocks completed successfully
  std::vector<int> results_h(kNumBlocks);
  CUDACHECK_TEST(cudaMemcpy(
      results_h.data(),
      results_d,
      kNumBlocks * sizeof(int),
      cudaMemcpyDeviceToHost));

  std::vector<int> expected(kNumBlocks, 1);
  EXPECT_EQ(results_h, expected)
      << "Not all blocks completed barrier successfully on rank " << globalRank;

  XLOGF(
      INFO,
      "Rank {}: Barrier multi-block stress test completed ({} blocks, {} slots)",
      globalRank,
      kNumBlocks,
      kNumBarrierSlots);
}

// =============================================================================
// Combined Signal and Barrier Configuration Test
// =============================================================================

TEST_F(
    MultiPeerNvlTransportIntegrationTestFixture,
    CombinedSignalBarrierConfig) {
  if (numRanks != 2) {
    GTEST_SKIP() << "Requires exactly 2 ranks, got " << numRanks;
  }

  const size_t dataBufferSize = 1024 * 1024;
  const int numSignalSlots = 4;
  const int numBarrierSlots = 2;
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = 1024,
      .pipelineDepth = 4,
  };
  WindowMemoryConfig wmConfig{
      .signalCount = numSignalSlots,
      .barrierCount = numBarrierSlots,
  };

  auto [transport, wm, device] = createTransport(config, wmConfig);

  int peerRank = (globalRank == 0) ? 1 : 0;

  // Phase 1: Signal on slot 0
  {
    DeviceBuffer resultBuffer(sizeof(int));
    auto result_d = static_cast<int*>(resultBuffer.get());
    CUDACHECK_TEST(cudaMemset(result_d, 0, sizeof(int)));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    test::testSignalWait(device, peerRank, 0, globalRank == 0, result_d);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    int result_h = 0;
    CUDACHECK_TEST(
        cudaMemcpy(&result_h, result_d, sizeof(int), cudaMemcpyDeviceToHost));
    EXPECT_EQ(result_h, 1) << "Phase 1 signal failed";
  }

  // Phase 2: Barrier on slot 0
  {
    DeviceBuffer resultBuffer(sizeof(int));
    auto result_d = static_cast<int*>(resultBuffer.get());
    CUDACHECK_TEST(cudaMemset(result_d, 0, sizeof(int)));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    test::testBarrier(device, 0, result_d);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    int result_h = 0;
    CUDACHECK_TEST(
        cudaMemcpy(&result_h, result_d, sizeof(int), cudaMemcpyDeviceToHost));
    EXPECT_EQ(result_h, 1) << "Phase 2 barrier failed";
  }

  // Phase 3: Signal on slot 3 (max slot)
  {
    DeviceBuffer resultBuffer(sizeof(int));
    auto result_d = static_cast<int*>(resultBuffer.get());
    CUDACHECK_TEST(cudaMemset(result_d, 0, sizeof(int)));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    test::testSignalWait(
        device, peerRank, numSignalSlots - 1, globalRank == 1, result_d);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    int result_h = 0;
    CUDACHECK_TEST(
        cudaMemcpy(&result_h, result_d, sizeof(int), cudaMemcpyDeviceToHost));
    EXPECT_EQ(result_h, 1) << "Phase 3 signal failed";
  }

  // Phase 4: Barrier on slot 1 (max slot)
  {
    DeviceBuffer resultBuffer(sizeof(int));
    auto result_d = static_cast<int*>(resultBuffer.get());
    CUDACHECK_TEST(cudaMemset(result_d, 0, sizeof(int)));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    test::testBarrier(device, numBarrierSlots - 1, result_d);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    int result_h = 0;
    CUDACHECK_TEST(
        cudaMemcpy(&result_h, result_d, sizeof(int), cudaMemcpyDeviceToHost));
    EXPECT_EQ(result_h, 1) << "Phase 4 barrier failed";
  }

  XLOGF(
      INFO,
      "Rank {}: Combined signal/barrier config test completed ({} signal slots, {} barrier slots)",
      globalRank,
      numSignalSlots,
      numBarrierSlots);
}

// =============================================================================
// Concurrent Signal Multi-Block Test (T3)
// =============================================================================

TEST_F(
    MultiPeerNvlTransportIntegrationTestFixture,
    ConcurrentSignalMultiBlock) {
  if (numRanks != 2) {
    GTEST_SKIP() << "Requires exactly 2 ranks, got " << numRanks;
  }

  constexpr int kNumBlocks = 4;
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = kDefaultDataBufferSize,
      .chunkSize = kDefaultChunkSize,
      .pipelineDepth = kDefaultPipelineDepth,
  };
  WindowMemoryConfig wmConfig{
      .signalCount = kMultiSlotSignalCount,
  };

  auto [transport, wm, device] = createTransport(config, wmConfig);

  int peerRank = (globalRank == 0) ? 1 : 0;

  DeviceBuffer resultsBuffer(kNumBlocks * sizeof(int));
  auto results_d = static_cast<int*>(resultsBuffer.get());
  CUDACHECK_TEST(cudaMemset(results_d, 0, kNumBlocks * sizeof(int)));

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Rank 0 signals on all slots, Rank 1 waits
  bool isSignaler = (globalRank == 0);
  test::testConcurrentSignalMultiBlock(
      device,
      peerRank,
      kMultiSlotSignalCount,
      isSignaler,
      results_d,
      kNumBlocks);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Verify all blocks completed successfully
  std::vector<int> results_h(kNumBlocks);
  CUDACHECK_TEST(cudaMemcpy(
      results_h.data(),
      results_d,
      kNumBlocks * sizeof(int),
      cudaMemcpyDeviceToHost));

  EXPECT_EQ(results_h, std::vector<int>(kNumBlocks, 1));

  XLOGF(
      INFO,
      "Rank {}: Concurrent signal multi-block test completed ({} blocks)",
      globalRank,
      kNumBlocks);
}

// =============================================================================
// Signal Reset Between Phases Test
// =============================================================================

TEST_F(MultiPeerNvlTransportIntegrationTestFixture, SignalResetBetweenPhases) {
  if (numRanks != 2) {
    GTEST_SKIP() << "Requires exactly 2 ranks, got " << numRanks;
  }

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = kDefaultDataBufferSize,
      .chunkSize = kDefaultChunkSize,
      .pipelineDepth = kDefaultPipelineDepth,
  };
  WindowMemoryConfig wmConfig{
      .signalCount = kMultiSlotSignalCount,
  };

  auto [transport, wm, device] = createTransport(config, wmConfig);

  int peerRank = (globalRank == 0) ? 1 : 0;
  constexpr int kSignalSlot = 0;
  constexpr int kNumPhases = 3;

  // Test that signal reuse with reset works correctly across multiple phases
  // Without reset, signal values would accumulate and waits might pass early
  for (int phase = 0; phase < kNumPhases; ++phase) {
    DeviceBuffer resultBuffer(sizeof(int));
    auto result_d = static_cast<int*>(resultBuffer.get());
    CUDACHECK_TEST(cudaMemset(result_d, 0, sizeof(int)));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Alternate which rank signals each phase
    bool isSignaler = ((globalRank + phase) % 2 == 0);
    test::testSignalWait(device, peerRank, kSignalSlot, isSignaler, result_d);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    int result_h = 0;
    CUDACHECK_TEST(
        cudaMemcpy(&result_h, result_d, sizeof(int), cudaMemcpyDeviceToHost));

    EXPECT_EQ(result_h, 1) << "Signal phase " << phase << " failed on rank "
                           << globalRank;

    // Reset signal between phases to test proper reset functionality
    // This is done implicitly by using different expected values in
    // testSignalWait For explicit reset testing, we would call resetSignalFrom
  }

  XLOGF(
      INFO,
      "Rank {}: Signal reset between phases test completed ({} phases)",
      globalRank,
      kNumPhases);
}

// =============================================================================
// Extended Concurrent Signal Stress Test
// =============================================================================

TEST_F(
    MultiPeerNvlTransportIntegrationTestFixture,
    ConcurrentSignalStressExtended) {
  if (numRanks != 2) {
    GTEST_SKIP() << "Requires exactly 2 ranks, got " << numRanks;
  }

  // Extended stress test with more blocks and warps
  constexpr int kNumBlocks = 8;
  constexpr int kNumSignalSlots = 8;
  constexpr int kNumIterations = 5;

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = kDefaultDataBufferSize,
      .chunkSize = kDefaultChunkSize,
      .pipelineDepth = kDefaultPipelineDepth,
  };
  WindowMemoryConfig wmConfig{
      .signalCount = kNumSignalSlots,
  };

  auto [transport, wm, device] = createTransport(config, wmConfig);

  int peerRank = (globalRank == 0) ? 1 : 0;

  for (int iter = 0; iter < kNumIterations; ++iter) {
    DeviceBuffer resultsBuffer(kNumBlocks * sizeof(int));
    auto results_d = static_cast<int*>(resultsBuffer.get());
    CUDACHECK_TEST(cudaMemset(results_d, 0, kNumBlocks * sizeof(int)));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Alternate sender/receiver each iteration
    bool isSignaler = ((globalRank + iter) % 2 == 0);
    test::testConcurrentSignalMultiBlock(
        device, peerRank, kNumSignalSlots, isSignaler, results_d, kNumBlocks);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Verify all blocks completed successfully
    std::vector<int> results_h(kNumBlocks);
    CUDACHECK_TEST(cudaMemcpy(
        results_h.data(),
        results_d,
        kNumBlocks * sizeof(int),
        cudaMemcpyDeviceToHost));

    std::vector<int> expected(kNumBlocks, 1);
    EXPECT_EQ(results_h, expected)
        << "Iteration " << iter << " failed on rank " << globalRank;
  }

  XLOGF(
      INFO,
      "Rank {}: Concurrent signal stress extended test completed ({} iterations, {} blocks)",
      globalRank,
      kNumIterations,
      kNumBlocks);
}

// =============================================================================
// Concurrent Signal Multi-Warp Stress Test (Issue 3)
// =============================================================================

TEST_F(
    MultiPeerNvlTransportIntegrationTestFixture,
    ConcurrentSignalWaitMultiWarp) {
  if (numRanks != 2) {
    GTEST_SKIP() << "Requires exactly 2 ranks, got " << numRanks;
  }

  // Test with multiple warps within a single block
  constexpr int kWarpsPerBlock = 4;
  constexpr int kNumSignalSlots = 8; // More slots than warps to test modulo

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = kDefaultDataBufferSize,
      .chunkSize = kDefaultChunkSize,
      .pipelineDepth = kDefaultPipelineDepth,
  };
  WindowMemoryConfig wmConfig{
      .signalCount = kNumSignalSlots,
  };

  auto [transport, wm, device] = createTransport(config, wmConfig);

  int peerRank = (globalRank == 0) ? 1 : 0;

  DeviceBuffer resultsBuffer(kWarpsPerBlock * sizeof(int));
  auto results_d = static_cast<int*>(resultsBuffer.get());
  CUDACHECK_TEST(cudaMemset(results_d, 0, kWarpsPerBlock * sizeof(int)));

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Rank 0 signals, Rank 1 waits - each warp uses different slot
  bool isSignaler = (globalRank == 0);
  test::testConcurrentSignalWaitMultiWarp(
      device, peerRank, kNumSignalSlots, isSignaler, results_d, kWarpsPerBlock);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Verify all warps completed successfully
  std::vector<int> results_h(kWarpsPerBlock);
  CUDACHECK_TEST(cudaMemcpy(
      results_h.data(),
      results_d,
      kWarpsPerBlock * sizeof(int),
      cudaMemcpyDeviceToHost));

  std::vector<int> expected(kWarpsPerBlock, 1);
  EXPECT_EQ(results_h, expected)
      << "Not all warps completed successfully on rank " << globalRank;

  XLOGF(
      INFO,
      "Rank {}: Concurrent signal multi-warp test completed ({} warps)",
      globalRank,
      kWarpsPerBlock);
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

  auto [transport, wm, device] = createTransport(config);

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

// =============================================================================
// signal_all() Test
// =============================================================================

TEST_F(MultiPeerNvlTransportIntegrationTestFixture, SignalAll) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Requires at least 2 ranks, got " << numRanks;
  }

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = kDefaultDataBufferSize,
      .chunkSize = kDefaultChunkSize,
      .pipelineDepth = kDefaultPipelineDepth,
  };

  auto [transport, wm, device] = createTransport(config);

  constexpr int kSignalerRank = 0;
  constexpr int kSignalIdx = 0;

  DeviceBuffer resultBuffer(sizeof(int));
  auto result_d = static_cast<int*>(resultBuffer.get());
  CUDACHECK_TEST(cudaMemset(result_d, 0, sizeof(int)));

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Rank 0 signals all peers, all other ranks wait for signal from rank 0
  test::testSignalAll(device, kSignalerRank, kSignalIdx, result_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  int result_h = 0;
  CUDACHECK_TEST(
      cudaMemcpy(&result_h, result_d, sizeof(int), cudaMemcpyDeviceToHost));

  EXPECT_EQ(result_h, 1) << "signal_all() operation failed on rank "
                         << globalRank;

  XLOGF(
      INFO,
      "Rank {}: signal_all() test completed (signalerRank={})",
      globalRank,
      kSignalerRank);
}

// =============================================================================
// wait_signal_from_all() Test
// =============================================================================

TEST_F(MultiPeerNvlTransportIntegrationTestFixture, WaitSignalFromAll) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Requires at least 2 ranks, got " << numRanks;
  }

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = kDefaultDataBufferSize,
      .chunkSize = kDefaultChunkSize,
      .pipelineDepth = kDefaultPipelineDepth,
  };

  auto [transport, wm, device] = createTransport(config);

  constexpr int kTargetRank = 0;
  constexpr int kSignalIdx = 0;

  DeviceBuffer resultBuffer(sizeof(int));
  auto result_d = static_cast<int*>(resultBuffer.get());
  CUDACHECK_TEST(cudaMemset(result_d, 0, sizeof(int)));

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // All peers signal rank 0, rank 0 waits for signals from all peers
  test::testWaitSignalFromAll(device, kTargetRank, kSignalIdx, result_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  int result_h = 0;
  CUDACHECK_TEST(
      cudaMemcpy(&result_h, result_d, sizeof(int), cudaMemcpyDeviceToHost));

  EXPECT_EQ(result_h, 1) << "wait_signal_from_all() operation failed on rank "
                         << globalRank;

  XLOGF(
      INFO,
      "Rank {}: wait_signal_from_all() test completed (targetRank={})",
      globalRank,
      kTargetRank);
}

// =============================================================================
// CmpOp::CMP_EQ Test
// =============================================================================

TEST_F(MultiPeerNvlTransportIntegrationTestFixture, WaitWithCmpEq) {
  if (numRanks != 2) {
    GTEST_SKIP() << "Requires exactly 2 ranks, got " << numRanks;
  }

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = kDefaultDataBufferSize,
      .chunkSize = kDefaultChunkSize,
      .pipelineDepth = kDefaultPipelineDepth,
  };

  auto [transport, wm, device] = createTransport(config);

  int peerRank = (globalRank == 0) ? 1 : 0;
  constexpr int kSignalIdx = 0;
  constexpr uint64_t kExpectedValue = 42;

  DeviceBuffer resultBuffer(sizeof(int));
  auto result_d = static_cast<int*>(resultBuffer.get());
  CUDACHECK_TEST(cudaMemset(result_d, 0, sizeof(int)));

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Rank 0 signals with exact value using SIGNAL_SET, Rank 1 waits with CMP_EQ
  bool isSignaler = (globalRank == 0);
  test::testWaitWithCmpEq(
      device, peerRank, kSignalIdx, kExpectedValue, isSignaler, result_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  int result_h = 0;
  CUDACHECK_TEST(
      cudaMemcpy(&result_h, result_d, sizeof(int), cudaMemcpyDeviceToHost));

  EXPECT_EQ(result_h, 1) << "CMP_EQ wait operation failed on rank "
                         << globalRank;

  XLOGF(
      INFO,
      "Rank {}: WaitWithCmpEq test completed (expectedValue={})",
      globalRank,
      kExpectedValue);
}

// =============================================================================
// Monotonic Wait Values Test
// =============================================================================

TEST_F(MultiPeerNvlTransportIntegrationTestFixture, MonotonicWaitValues) {
  if (numRanks != 2) {
    GTEST_SKIP() << "Requires exactly 2 ranks, got " << numRanks;
  }

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = kDefaultDataBufferSize,
      .chunkSize = kDefaultChunkSize,
      .pipelineDepth = kDefaultPipelineDepth,
  };

  auto [transport, wm, device] = createTransport(config);

  int peerRank = (globalRank == 0) ? 1 : 0;
  constexpr int kSignalIdx = 0;
  constexpr int kNumIterations = 5;

  DeviceBuffer resultBuffer(sizeof(int));
  auto result_d = static_cast<int*>(resultBuffer.get());
  CUDACHECK_TEST(cudaMemset(result_d, 0, sizeof(int)));

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Test monotonically increasing wait values pattern:
  // signal(1), wait_for(1), signal(1), wait_for(2), etc.
  bool isSignaler = (globalRank == 0);
  test::testMonotonicWaitValues(
      device, peerRank, kSignalIdx, kNumIterations, isSignaler, result_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  int result_h = 0;
  CUDACHECK_TEST(
      cudaMemcpy(&result_h, result_d, sizeof(int), cudaMemcpyDeviceToHost));

  EXPECT_EQ(result_h, 1) << "Monotonic wait values test failed on rank "
                         << globalRank;

  XLOGF(
      INFO,
      "Rank {}: MonotonicWaitValues test completed ({} iterations)",
      globalRank,
      kNumIterations);
}

// =============================================================================
// SIGNAL_SET Integration Test
// =============================================================================

TEST_F(MultiPeerNvlTransportIntegrationTestFixture, SignalWithSet) {
  if (numRanks != 2) {
    GTEST_SKIP() << "Requires exactly 2 ranks, got " << numRanks;
  }

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = kDefaultDataBufferSize,
      .chunkSize = kDefaultChunkSize,
      .pipelineDepth = kDefaultPipelineDepth,
  };

  auto [transport, wm, device] = createTransport(config);

  int peerRank = (globalRank == 0) ? 1 : 0;
  constexpr int kSignalIdx = 0;
  constexpr uint64_t kSetValue = 100;

  DeviceBuffer resultBuffer(sizeof(int));
  auto result_d = static_cast<int*>(resultBuffer.get());
  CUDACHECK_TEST(cudaMemset(result_d, 0, sizeof(int)));

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Rank 0 signals using SIGNAL_SET, Rank 1 waits for the set value
  bool isSignaler = (globalRank == 0);
  test::testSignalWithSet(
      device, peerRank, kSignalIdx, kSetValue, isSignaler, result_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  int result_h = 0;
  CUDACHECK_TEST(
      cudaMemcpy(&result_h, result_d, sizeof(int), cudaMemcpyDeviceToHost));

  EXPECT_EQ(result_h, 1) << "SIGNAL_SET operation failed on rank "
                         << globalRank;

  XLOGF(
      INFO,
      "Rank {}: SignalWithSet test completed (setValue={})",
      globalRank,
      kSetValue);
}

// =============================================================================
// Put Operation Test (P2P path) - Tests put_signal() and wait_signal()
// =============================================================================

TEST_F(MultiPeerNvlTransportIntegrationTestFixture, PutSignalOperation) {
  if (numRanks != 2) {
    GTEST_SKIP() << "Requires exactly 2 ranks, got " << numRanks;
  }

  // Use a transfer size that fits within the transport's data buffer
  constexpr std::size_t kTransferSize = 4096;
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = kDefaultDataBufferSize,
      .chunkSize = kDefaultChunkSize,
      .pipelineDepth = kDefaultPipelineDepth,
  };
  WindowMemoryConfig wmConfig{
      .signalCount = kDefaultSignalCount,
  };

  auto [transport, wm, device] = createTransport(config, wmConfig);

  int peerRank = (globalRank == 0) ? 1 : 0;
  const int testValue = 0xCD + globalRank;

  // Get the P2pNvlTransportDevice to access IPC-mapped remote buffers
  // The transport's remoteState_.dataBuffer is already IPC-mapped
  auto p2pTransport = transport->getP2pTransportDevice(peerRank);

  // POINTER RELATIONSHIP CLARIFICATION:
  //
  // remoteDataBuffer: This rank's IPC-mapped pointer to the PEER's local
  // buffer.
  //                   When this rank writes here, data appears in peer's
  //                   memory.
  //
  // localDataBuffer: This rank's local staging buffer. When the PEER writes to
  //                  their remoteDataBuffer, the data appears here.
  //
  // So for rank 0 writing to rank 1:
  //   - rank 0's remoteDataBuffer == IPC pointer to rank 1's localDataBuffer
  //   - rank 0 writes to its remoteDataBuffer
  //   - rank 1 reads from its localDataBuffer (same physical memory)
  //
  char* remoteDataBuffer = p2pTransport.getRemoteState().dataBuffer;
  char* localDataBuffer = p2pTransport.getLocalState().dataBuffer;

  // Allocate local source buffer and result buffer
  DeviceBuffer localSrcBuffer(kTransferSize);
  DeviceBuffer resultBuffer(sizeof(int));

  auto localSrc_d = static_cast<int*>(localSrcBuffer.get());
  auto result_d = static_cast<int*>(resultBuffer.get());

  // Rank 0: fill local source buffer with test pattern
  // Rank 1: the local data buffer (destination) will receive the data
  if (globalRank == 0) {
    test::fillBuffer(localSrc_d, testValue, kTransferSize / sizeof(int));
  }

  // Clear the local data buffer (this is where peer will write to via NVLink)
  CUDACHECK_TEST(cudaMemset(localDataBuffer, 0, kTransferSize));
  CUDACHECK_TEST(cudaMemset(result_d, 0, sizeof(int)));
  CUDACHECK_TEST(cudaDeviceSynchronize());

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Rank 0 writes to rank 1's buffer using put_signal (via IPC-mapped ptr)
  // Rank 1 waits for the signal
  bool isWriter = (globalRank == 0);
  constexpr int kSignalId = 0;

  // remoteDataBuffer points to peer's local data buffer (via IPC)
  // So rank 0's remoteDataBuffer points to rank 1's localDataBuffer
  test::testPutOperation(
      device,
      peerRank,
      isWriter ? remoteDataBuffer : nullptr, // Write to peer's buffer via IPC
      isWriter ? localSrc_d : nullptr, // Source is our local data
      kTransferSize,
      kSignalId,
      isWriter,
      result_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Verify operation completed
  int result_h = 0;
  CUDACHECK_TEST(
      cudaMemcpy(&result_h, result_d, sizeof(int), cudaMemcpyDeviceToHost));
  EXPECT_EQ(result_h, 1) << "Put/Signal operation failed on rank "
                         << globalRank;

  // Verify data on receiver side
  // Rank 1's localDataBuffer should now contain the data written by rank 0
  if (globalRank == 1) {
    std::vector<int> recvHost(kTransferSize / sizeof(int));
    CUDACHECK_TEST(cudaMemcpy(
        recvHost.data(),
        localDataBuffer,
        kTransferSize,
        cudaMemcpyDeviceToHost));

    const int expectedValue = 0xCD + 0; // Sender's testValue (rank 0)
    std::vector<int> expected(kTransferSize / sizeof(int), expectedValue);
    EXPECT_EQ(recvHost, expected) << "Data mismatch after put_signal";
  }

  XLOGF(
      INFO,
      "Rank {}: Put/Signal operation test completed (isWriter={})",
      globalRank,
      isWriter);
}

// =============================================================================
// wait_signal_from() Basic Per-Peer Signal/Wait Test
// =============================================================================

TEST_F(MultiPeerNvlTransportIntegrationTestFixture, WaitSignalFromPeer) {
  if (numRanks != 2) {
    GTEST_SKIP() << "Requires exactly 2 ranks, got " << numRanks;
  }

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = kDefaultDataBufferSize,
      .chunkSize = kDefaultChunkSize,
      .pipelineDepth = kDefaultPipelineDepth,
  };

  auto [transport, wm, device] = createTransport(config);

  int peerRank = (globalRank == 0) ? 1 : 0;
  constexpr int kSignalIdx = 0;

  DeviceBuffer resultBuffer(sizeof(int));
  auto result_d = static_cast<int*>(resultBuffer.get());
  CUDACHECK_TEST(cudaMemset(result_d, 0, sizeof(int)));

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Rank 0 signals, Rank 1 waits using wait_signal_from
  bool isSignaler = (globalRank == 0);
  test::testWaitSignalFromPeer(
      device, peerRank, kSignalIdx, isSignaler, result_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  int result_h = 0;
  CUDACHECK_TEST(
      cudaMemcpy(&result_h, result_d, sizeof(int), cudaMemcpyDeviceToHost));

  EXPECT_EQ(result_h, 1) << "wait_signal_from() failed on rank " << globalRank;

  XLOGF(
      INFO,
      "Rank {}: WaitSignalFromPeer test completed (isSignaler={})",
      globalRank,
      isSignaler);
}

// =============================================================================
// wait_signal_from() Multi-Peer Isolation Test
// =============================================================================

TEST_F(
    MultiPeerNvlTransportIntegrationTestFixture,
    WaitSignalFromMultiPeerIsolation) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Requires at least 2 ranks, got " << numRanks;
  }

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = kDefaultDataBufferSize,
      .chunkSize = kDefaultChunkSize,
      .pipelineDepth = kDefaultPipelineDepth,
  };

  auto [transport, wm, device] = createTransport(config);

  constexpr int kTargetRank = 0;
  constexpr int kSignalIdx = 0;

  DeviceBuffer resultBuffer(sizeof(int));
  auto result_d = static_cast<int*>(resultBuffer.get());
  CUDACHECK_TEST(cudaMemset(result_d, 0, sizeof(int)));

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // All peers signal rank 0 with different values, rank 0 verifies isolation
  test::testWaitSignalFromMultiPeerIsolation(
      device, kTargetRank, kSignalIdx, result_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  int result_h = 0;
  CUDACHECK_TEST(
      cudaMemcpy(&result_h, result_d, sizeof(int), cudaMemcpyDeviceToHost));

  EXPECT_EQ(result_h, 1) << "WaitSignalFromMultiPeerIsolation failed on rank "
                         << globalRank;

  XLOGF(
      INFO,
      "Rank {}: WaitSignalFromMultiPeerIsolation test completed (targetRank={})",
      globalRank,
      kTargetRank);
}

// =============================================================================
// wait_signal() and wait_signal_from() Both Work Test
// =============================================================================

TEST_F(
    MultiPeerNvlTransportIntegrationTestFixture,
    WaitSignalAndWaitSignalFromBothWork) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Requires at least 2 ranks, got " << numRanks;
  }

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = kDefaultDataBufferSize,
      .chunkSize = kDefaultChunkSize,
      .pipelineDepth = kDefaultPipelineDepth,
  };

  auto [transport, wm, device] = createTransport(config);

  constexpr int kTargetRank = 0;
  constexpr int kSignalIdx = 0;

  DeviceBuffer resultBuffer(sizeof(int));
  auto result_d = static_cast<int*>(resultBuffer.get());
  CUDACHECK_TEST(cudaMemset(result_d, 0, sizeof(int)));

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // All peers signal rank 0, rank 0 verifies both accumulated and per-peer
  test::testWaitSignalAndWaitSignalFromBothWork(
      device, kTargetRank, kSignalIdx, result_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  int result_h = 0;
  CUDACHECK_TEST(
      cudaMemcpy(&result_h, result_d, sizeof(int), cudaMemcpyDeviceToHost));

  EXPECT_EQ(result_h, 1)
      << "WaitSignalAndWaitSignalFromBothWork failed on rank " << globalRank;

  XLOGF(
      INFO,
      "Rank {}: WaitSignalAndWaitSignalFromBothWork test completed (targetRank={})",
      globalRank,
      kTargetRank);
}

// =============================================================================
// Multi-GPU Signal/Wait Test (BLOCK Scope - fallback path coverage)
// =============================================================================

TEST_F(MultiPeerNvlTransportIntegrationTestFixture, SignalWaitBlockScope) {
  if (numRanks != 2) {
    GTEST_SKIP() << "Requires exactly 2 ranks, got " << numRanks;
  }

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = kDefaultDataBufferSize,
      .chunkSize = kDefaultChunkSize,
      .pipelineDepth = kDefaultPipelineDepth,
  };

  auto [transport, wm, device] = createTransport(config);

  int peerRank = (globalRank == 0) ? 1 : 0;

  DeviceBuffer resultBuffer(sizeof(int));
  auto result_d = static_cast<int*>(resultBuffer.get());
  CUDACHECK_TEST(cudaMemset(result_d, 0, sizeof(int)));

  // Rank 0 signals, Rank 1 waits
  bool isSignaler = (globalRank == 0);

  // Synchronize before starting
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Uses BLOCK scope - exercises non-WARP fallback path in wait_signal()
  test::testSignalWaitBlockScope(device, peerRank, 0, isSignaler, result_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  int result_h = 0;
  CUDACHECK_TEST(
      cudaMemcpy(&result_h, result_d, sizeof(int), cudaMemcpyDeviceToHost));

  EXPECT_EQ(result_h, 1) << "Signal/Wait (BLOCK scope) operation failed";

  XLOGF(
      INFO,
      "Rank {}: Signal/Wait (BLOCK scope) test completed (isSignaler={})",
      globalRank,
      isSignaler);
}

} // namespace comms::pipes::tests

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
