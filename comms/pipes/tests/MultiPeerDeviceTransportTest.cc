// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <numeric>
#include <vector>

#include "comms/pipes/MultiPeerDeviceTransport.cuh"
#include "comms/pipes/Transport.cuh"
#include "comms/pipes/tests/MultiPeerDeviceTransportTest.cuh"
#include "comms/pipes/window/DeviceWindowBarrier.cuh"
#include "comms/pipes/window/DeviceWindowSignal.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::DeviceBuffer;

namespace comms::pipes {

class MultiPeerDeviceTransportTestFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    CUDACHECK_TEST(cudaSetDevice(0));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaDeviceSynchronize());
  }
};

// =============================================================================
// DeviceWindowSignal Tests
// =============================================================================

TEST_F(MultiPeerDeviceTransportTestFixture, DeviceWindowSignalConstruction) {
  const int myRank = 0;
  const int nRanks = 4;
  const int signalCount = 2;

  DeviceBuffer resultsBuffer(3 * sizeof(int));
  auto results_d = static_cast<int*>(resultsBuffer.get());

  test::testDeviceWindowSignalConstruction(
      myRank, nRanks, signalCount, results_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<int> results_h(3);
  CUDACHECK_TEST(cudaMemcpy(
      results_h.data(), results_d, 3 * sizeof(int), cudaMemcpyDeviceToHost));

  EXPECT_EQ(results_h[0], myRank) << "rank() should return " << myRank;
  EXPECT_EQ(results_h[1], nRanks) << "nRanks() should return " << nRanks;
  EXPECT_EQ(results_h[2], signalCount)
      << "signalCount() should return " << signalCount;
}

TEST_F(MultiPeerDeviceTransportTestFixture, SignalInboxBufferSize) {
  // Test buffer size calculation
  const int signalCount = 2;

  std::size_t bufferSize = getSignalInboxBufferSize(signalCount);

  // Expected: signalCount * sizeof(SignalState), aligned to 128 bytes
  std::size_t expectedSlots = signalCount;
  std::size_t expectedMinSize = expectedSlots * sizeof(SignalState);
  // Should be aligned to 128 bytes
  EXPECT_GE(bufferSize, expectedMinSize);
  EXPECT_EQ(bufferSize % 128, 0) << "Buffer size should be 128-byte aligned";
}

// =============================================================================
// DeviceWindowBarrier Tests
// =============================================================================

TEST_F(MultiPeerDeviceTransportTestFixture, DeviceWindowBarrierConstruction) {
  const int myRank = 1;
  const int nRanks = 4;

  DeviceBuffer resultsBuffer(2 * sizeof(int));
  auto results_d = static_cast<int*>(resultsBuffer.get());

  test::testDeviceWindowBarrierConstruction(myRank, nRanks, results_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<int> results_h(2);
  CUDACHECK_TEST(cudaMemcpy(
      results_h.data(), results_d, 2 * sizeof(int), cudaMemcpyDeviceToHost));

  EXPECT_EQ(results_h[0], myRank) << "rank() should return " << myRank;
  EXPECT_EQ(results_h[1], nRanks) << "nRanks() should return " << nRanks;
}

TEST_F(MultiPeerDeviceTransportTestFixture, BarrierBufferSize) {
  // Test buffer size calculation
  const int barrierCount = 4;

  std::size_t bufferSize = getMultiPeerBarrierBufferSize(barrierCount);

  // Expected: barrierCount * sizeof(BarrierState), aligned to 128 bytes
  std::size_t expectedMinSize = barrierCount * sizeof(BarrierState);
  // Should be aligned to 128 bytes
  EXPECT_GE(bufferSize, expectedMinSize);
  EXPECT_EQ(bufferSize % 128, 0) << "Buffer size should be 128-byte aligned";
}

// =============================================================================
// MultiPeerDeviceTransport Tests
// =============================================================================

TEST_F(
    MultiPeerDeviceTransportTestFixture,
    MultiPeerDeviceTransportConstruction) {
  const int myRank = 2;
  const int nRanks = 4;

  DeviceBuffer resultsBuffer(2 * sizeof(int));
  auto results_d = static_cast<int*>(resultsBuffer.get());

  test::testMultiPeerDeviceTransportConstruction(myRank, nRanks, results_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<int> results_h(2);
  CUDACHECK_TEST(cudaMemcpy(
      results_h.data(), results_d, 2 * sizeof(int), cudaMemcpyDeviceToHost));

  EXPECT_EQ(results_h[0], myRank) << "rank() should return " << myRank;
  EXPECT_EQ(results_h[1], nRanks) << "nRanks() should return " << nRanks;
}

// =============================================================================
// DeviceWindowMemory Tests
// =============================================================================

// Verify DeviceWindowMemory bundles signal and barrier correctly.
// Constructs DeviceWindowMemory from DeviceWindowSignal + DeviceWindowBarrier,
// then verifies signal() and barrier() return objects with matching metadata.
TEST_F(MultiPeerDeviceTransportTestFixture, DeviceWindowMemoryAccessors) {
  const int myRank = 1;
  const int nRanks = 4;
  const int signalCount = 2;

  DeviceBuffer resultsBuffer(5 * sizeof(int));
  auto results_d = static_cast<int*>(resultsBuffer.get());

  test::testDeviceWindowMemoryAccessors(myRank, nRanks, signalCount, results_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<int> results_h(5);
  CUDACHECK_TEST(cudaMemcpy(
      results_h.data(), results_d, 5 * sizeof(int), cudaMemcpyDeviceToHost));

  // Signal accessors should match input parameters
  EXPECT_EQ(results_h[0], myRank) << "signal().rank()";
  EXPECT_EQ(results_h[1], nRanks) << "signal().n_ranks()";
  EXPECT_EQ(results_h[2], signalCount) << "signal().signal_count()";

  // Barrier accessors should match input parameters
  EXPECT_EQ(results_h[3], myRank) << "barrier().rank()";
  EXPECT_EQ(results_h[4], nRanks) << "barrier().n_ranks()";
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(MultiPeerDeviceTransportTestFixture, SingleRankSignal) {
  // Test with single rank (edge case)
  const int myRank = 0;
  const int nRanks = 1;
  const int signalCount = 1;

  DeviceBuffer resultsBuffer(3 * sizeof(int));
  auto results_d = static_cast<int*>(resultsBuffer.get());

  test::testDeviceWindowSignalConstruction(
      myRank, nRanks, signalCount, results_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<int> results_h(3);
  CUDACHECK_TEST(cudaMemcpy(
      results_h.data(), results_d, 3 * sizeof(int), cudaMemcpyDeviceToHost));

  EXPECT_EQ(results_h[0], myRank);
  EXPECT_EQ(results_h[1], nRanks);
  EXPECT_EQ(results_h[2], signalCount);
}

TEST_F(MultiPeerDeviceTransportTestFixture, MaxRanksTransport) {
  // Test with a high rank count to verify construction works
  // (actual max depends on hardware: H100=8, GB200=72)
  const int myRank = 7;
  const int nRanks = 8;

  DeviceBuffer resultsBuffer(2 * sizeof(int));
  auto results_d = static_cast<int*>(resultsBuffer.get());

  test::testMultiPeerDeviceTransportConstruction(myRank, nRanks, results_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<int> results_h(2);
  CUDACHECK_TEST(cudaMemcpy(
      results_h.data(), results_d, 2 * sizeof(int), cudaMemcpyDeviceToHost));

  EXPECT_EQ(results_h[0], myRank) << "rank() should return " << myRank;
  EXPECT_EQ(results_h[1], nRanks) << "nRanks() should return " << nRanks;
}

// =============================================================================
// Self-Transport Tests (via get_self_transport())
// =============================================================================

TEST_F(MultiPeerDeviceTransportTestFixture, GetTransportReturnsCorrectType) {
  // Test that get_self_transport() returns a valid Transport with SELF type
  // when accessing myRank's transport

  // Allocate Transport on device with SELF type
  P2pSelfTransportDevice selfTransport;
  Transport hostTransport(selfTransport);

  // Copy Transport to device (need to use placement new due to deleted copy)
  Transport* transport_d = nullptr;
  CUDACHECK_TEST(cudaMalloc(&transport_d, sizeof(Transport)));

  // Use cudaMemcpy to copy the bytes (Transport has trivial layout for union)
  CUDACHECK_TEST(cudaMemcpy(
      transport_d, &hostTransport, sizeof(Transport), cudaMemcpyHostToDevice));

  // Allocate results buffer
  DeviceBuffer resultsBuffer(sizeof(int));
  auto results_d = static_cast<int*>(resultsBuffer.get());
  CUDACHECK_TEST(cudaMemset(results_d, 0, sizeof(int)));

  // Test that transport type is SELF
  test::testGetTransportType(transport_d, results_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  int result_h = 0;
  CUDACHECK_TEST(
      cudaMemcpy(&result_h, results_d, sizeof(int), cudaMemcpyDeviceToHost));

  EXPECT_EQ(result_h, 1) << "Transport type should be SELF";

  CUDACHECK_TEST(cudaFree(transport_d));
}

TEST_F(MultiPeerDeviceTransportTestFixture, SelfTransportPutCopiesData) {
  // Test that self-transport put() correctly copies data
  // This tests the path: get_self_transport()->self.put()

  const std::size_t nbytes = 4096; // 4KB transfer
  const std::size_t numInts = nbytes / sizeof(int);
  const int testValue = 42;

  // Allocate source and destination buffers
  DeviceBuffer srcBuffer(nbytes);
  DeviceBuffer dstBuffer(nbytes);
  auto src_d = static_cast<int*>(srcBuffer.get());
  auto dst_d = static_cast<int*>(dstBuffer.get());

  // Initialize source with test value, destination with zeros
  std::vector<int> srcHost(numInts, testValue);
  CUDACHECK_TEST(
      cudaMemcpy(src_d, srcHost.data(), nbytes, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(dst_d, 0, nbytes));

  // Create self-transport on device
  P2pSelfTransportDevice selfTransport;
  Transport hostTransport(selfTransport);

  Transport* transport_d = nullptr;
  CUDACHECK_TEST(cudaMalloc(&transport_d, sizeof(Transport)));
  CUDACHECK_TEST(cudaMemcpy(
      transport_d, &hostTransport, sizeof(Transport), cudaMemcpyHostToDevice));

  // Run the put operation
  const int numBlocks = 4;
  const int blockSize = 256;
  test::testSelfTransportPut(
      transport_d,
      reinterpret_cast<char*>(dst_d),
      reinterpret_cast<const char*>(src_d),
      nbytes,
      numBlocks,
      blockSize);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Verify destination has correct data
  std::vector<int> dstHost(numInts);
  CUDACHECK_TEST(
      cudaMemcpy(dstHost.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));

  int errorCount = 0;
  for (std::size_t i = 0; i < numInts; ++i) {
    if (dstHost[i] != testValue) {
      ++errorCount;
      if (errorCount <= 5) {
        ADD_FAILURE() << "Mismatch at index " << i << ": expected " << testValue
                      << ", got " << dstHost[i];
      }
    }
  }

  EXPECT_EQ(errorCount, 0) << "Total mismatches: " << errorCount;

  CUDACHECK_TEST(cudaFree(transport_d));
}

TEST_F(MultiPeerDeviceTransportTestFixture, SelfTransportPutLargeTransfer) {
  // Test larger transfer to exercise multi-chunk path
  const std::size_t nbytes = 1024 * 1024; // 1MB transfer
  const std::size_t numInts = nbytes / sizeof(int);
  const int testValue = 0xDEADBEEF;

  // Allocate source and destination buffers
  DeviceBuffer srcBuffer(nbytes);
  DeviceBuffer dstBuffer(nbytes);
  auto src_d = static_cast<int*>(srcBuffer.get());
  auto dst_d = static_cast<int*>(dstBuffer.get());

  // Initialize source with test pattern
  std::vector<int> srcHost(numInts, testValue);
  CUDACHECK_TEST(
      cudaMemcpy(src_d, srcHost.data(), nbytes, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(dst_d, 0, nbytes));

  // Create self-transport on device
  P2pSelfTransportDevice selfTransport;
  Transport hostTransport(selfTransport);

  Transport* transport_d = nullptr;
  CUDACHECK_TEST(cudaMalloc(&transport_d, sizeof(Transport)));
  CUDACHECK_TEST(cudaMemcpy(
      transport_d, &hostTransport, sizeof(Transport), cudaMemcpyHostToDevice));

  // Run the put operation with more parallelism
  const int numBlocks = 8;
  const int blockSize = 256;
  test::testSelfTransportPut(
      transport_d,
      reinterpret_cast<char*>(dst_d),
      reinterpret_cast<const char*>(src_d),
      nbytes,
      numBlocks,
      blockSize);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Verify destination has correct data (spot check for performance)
  std::vector<int> dstHost(numInts);
  CUDACHECK_TEST(
      cudaMemcpy(dstHost.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));

  // Check first, middle, and last values
  EXPECT_EQ(dstHost[0], testValue) << "First element mismatch";
  EXPECT_EQ(dstHost[numInts / 2], testValue) << "Middle element mismatch";
  EXPECT_EQ(dstHost[numInts - 1], testValue) << "Last element mismatch";

  // Full verification
  int errorCount = 0;
  for (std::size_t i = 0; i < numInts; ++i) {
    if (dstHost[i] != testValue) {
      ++errorCount;
    }
  }
  EXPECT_EQ(errorCount, 0) << "Total mismatches in 1MB transfer: "
                           << errorCount;

  CUDACHECK_TEST(cudaFree(transport_d));
}
// Peer Iteration Helper Tests
// =============================================================================

TEST_F(MultiPeerDeviceTransportTestFixture, PeerIterationHelpersRank0) {
  // Test peer iteration helpers with myRank=0, nRanks=4
  // Expected: numPeers=3, peerIndexToRank=[1, 2, 3]
  const int myRank = 0;
  const int nRanks = 4;
  const int numPeers = nRanks - 1;

  DeviceBuffer resultsBuffer((1 + numPeers) * sizeof(int));
  auto results_d = static_cast<int*>(resultsBuffer.get());

  test::testPeerIterationHelpers(myRank, nRanks, results_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<int> results_h(1 + numPeers);
  CUDACHECK_TEST(cudaMemcpy(
      results_h.data(),
      results_d,
      (1 + numPeers) * sizeof(int),
      cudaMemcpyDeviceToHost));

  EXPECT_EQ(results_h[0], numPeers) << "numPeers() should return " << numPeers;
  EXPECT_EQ(results_h[1], 1) << "peerIndexToRank(0) should return 1";
  EXPECT_EQ(results_h[2], 2) << "peerIndexToRank(1) should return 2";
  EXPECT_EQ(results_h[3], 3) << "peerIndexToRank(2) should return 3";
}

TEST_F(MultiPeerDeviceTransportTestFixture, PeerIterationHelpersRank2) {
  // Test peer iteration helpers with myRank=2, nRanks=4
  // Expected: numPeers=3, peerIndexToRank=[0, 1, 3] (skips self at 2)
  const int myRank = 2;
  const int nRanks = 4;
  const int numPeers = nRanks - 1;

  DeviceBuffer resultsBuffer((1 + numPeers) * sizeof(int));
  auto results_d = static_cast<int*>(resultsBuffer.get());

  test::testPeerIterationHelpers(myRank, nRanks, results_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<int> results_h(1 + numPeers);
  CUDACHECK_TEST(cudaMemcpy(
      results_h.data(),
      results_d,
      (1 + numPeers) * sizeof(int),
      cudaMemcpyDeviceToHost));

  EXPECT_EQ(results_h[0], numPeers) << "numPeers() should return " << numPeers;
  EXPECT_EQ(results_h[1], 0) << "peerIndexToRank(0) should return 0";
  EXPECT_EQ(results_h[2], 1) << "peerIndexToRank(1) should return 1";
  EXPECT_EQ(results_h[3], 3)
      << "peerIndexToRank(2) should return 3 (skip self)";
}

TEST_F(MultiPeerDeviceTransportTestFixture, PeerIterationHelpersRank7) {
  // Test peer iteration helpers with myRank=7, nRanks=8 (max rank)
  // Expected: numPeers=7, peerIndexToRank=[0, 1, 2, 3, 4, 5, 6]
  const int myRank = 7;
  const int nRanks = 8;
  const int numPeers = nRanks - 1;

  DeviceBuffer resultsBuffer((1 + numPeers) * sizeof(int));
  auto results_d = static_cast<int*>(resultsBuffer.get());

  test::testPeerIterationHelpers(myRank, nRanks, results_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<int> results_h(1 + numPeers);
  CUDACHECK_TEST(cudaMemcpy(
      results_h.data(),
      results_d,
      (1 + numPeers) * sizeof(int),
      cudaMemcpyDeviceToHost));

  EXPECT_EQ(results_h[0], numPeers) << "numPeers() should return " << numPeers;
  // For myRank=7, all indices map directly (0->0, 1->1, ..., 6->6)
  std::vector<int> expectedPeerRanks(numPeers);
  std::iota(expectedPeerRanks.begin(), expectedPeerRanks.end(), 0);
  std::vector<int> actualPeerRanks(results_h.begin() + 1, results_h.end());
  EXPECT_EQ(actualPeerRanks, expectedPeerRanks)
      << "peerIndexToRank mapping mismatch for myRank=7";
}

TEST_F(MultiPeerDeviceTransportTestFixture, PeerIterationHelpersSinglePeer) {
  // Test with nRanks=2 (minimal multi-peer case)
  // Expected: numPeers=1
  const int myRank = 0;
  const int nRanks = 2;
  const int numPeers = nRanks - 1;

  DeviceBuffer resultsBuffer((1 + numPeers) * sizeof(int));
  auto results_d = static_cast<int*>(resultsBuffer.get());

  test::testPeerIterationHelpers(myRank, nRanks, results_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<int> results_h(1 + numPeers);
  CUDACHECK_TEST(cudaMemcpy(
      results_h.data(),
      results_d,
      (1 + numPeers) * sizeof(int),
      cudaMemcpyDeviceToHost));

  EXPECT_EQ(results_h[0], 1) << "numPeers() should return 1";
  EXPECT_EQ(results_h[1], 1) << "peerIndexToRank(0) should return 1";
}

// =============================================================================
// Peer Index Conversion Roundtrip Tests
// =============================================================================

// Helper to run the roundtrip test kernel and verify results for a given
// myRank/nRanks configuration. Tests:
//   1. rank_to_peer_index() produces expected values
//   2. peer_index_to_rank(rank_to_peer_index(rank)) == rank (roundtrip)
//   3. rank_to_peer_index(peer_index_to_rank(i)) == i (roundtrip)
//   4. get_self_transport()->type == SELF
//   5. get_peer_transport(i)->type == P2P_NVL for all peers
void verifyPeerIndexConversionRoundtrip(int myRank, int nRanks) {
  const int numPeers = nRanks - 1;
  // Results layout: numPeers + 3*numPeers (conversions/roundtrips) + 1 (self)
  //                 + numPeers (peer types) = 4*numPeers + 2
  const int numResults = 4 * numPeers + 2;

  DeviceBuffer resultsBuffer(numResults * sizeof(int));
  auto results_d = static_cast<int*>(resultsBuffer.get());
  CUDACHECK_TEST(cudaMemset(results_d, 0xFF, numResults * sizeof(int)));

  test::testPeerIndexConversionRoundtrip(myRank, nRanks, results_d);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<int> r(numResults);
  CUDACHECK_TEST(cudaMemcpy(
      r.data(), results_d, numResults * sizeof(int), cudaMemcpyDeviceToHost));

  int idx = 0;

  // [0] = numPeers
  EXPECT_EQ(r[idx++], numPeers)
      << "numPeers() for myRank=" << myRank << ", nRanks=" << nRanks;

  // [1 .. numPeers] = rank_to_peer_index for each non-self rank
  // Compute expected: for rank in [0, nRanks) excluding myRank,
  //   peer_index = (rank < myRank) ? rank : (rank - 1)
  for (int rank = 0; rank < nRanks; ++rank) {
    if (rank == myRank) {
      continue;
    }
    int expectedPeerIndex = (rank < myRank) ? rank : (rank - 1);
    EXPECT_EQ(r[idx], expectedPeerIndex)
        << "rank_to_peer_index(" << rank << ") for myRank=" << myRank
        << ", nRanks=" << nRanks;
    idx++;
  }

  // [numPeers+1 .. 2*numPeers] = roundtrip rank->peer_index->rank
  for (int rank = 0; rank < nRanks; ++rank) {
    if (rank == myRank) {
      continue;
    }
    EXPECT_EQ(r[idx], rank)
        << "Roundtrip peer_index_to_rank(rank_to_peer_index(" << rank
        << ")) should equal " << rank << " for myRank=" << myRank
        << ", nRanks=" << nRanks;
    idx++;
  }

  // [2*numPeers+1 .. 3*numPeers] = roundtrip peer_index->rank->peer_index
  for (int i = 0; i < numPeers; ++i) {
    EXPECT_EQ(r[idx], i) << "Roundtrip rank_to_peer_index(peer_index_to_rank("
                         << i << ")) should equal " << i
                         << " for myRank=" << myRank << ", nRanks=" << nRanks;
    idx++;
  }

  // [3*numPeers+1] = get_self_transport()->type (expect SELF=0)
  EXPECT_EQ(r[idx++], 0)
      << "get_self_transport()->type should be SELF for myRank=" << myRank;

  // [3*numPeers+2 .. 4*numPeers+1] = get_peer_transport(i)->type
  // (expect P2P_NVL=1)
  for (int i = 0; i < numPeers; ++i) {
    EXPECT_EQ(r[idx], 1) << "get_peer_transport(" << i
                         << ")->type should be P2P_NVL for myRank=" << myRank;
    idx++;
  }
}

TEST_F(
    MultiPeerDeviceTransportTestFixture,
    PeerIndexConversionRoundtripRank0Of4) {
  verifyPeerIndexConversionRoundtrip(/*myRank=*/0, /*nRanks=*/4);
}

TEST_F(
    MultiPeerDeviceTransportTestFixture,
    PeerIndexConversionRoundtripRank2Of4) {
  // Middle rank: tests the skip-self logic in both directions
  verifyPeerIndexConversionRoundtrip(/*myRank=*/2, /*nRanks=*/4);
}

TEST_F(
    MultiPeerDeviceTransportTestFixture,
    PeerIndexConversionRoundtripRank3Of4) {
  // Last rank: peer indices map 1:1 to ranks [0, 1, 2]
  verifyPeerIndexConversionRoundtrip(/*myRank=*/3, /*nRanks=*/4);
}

TEST_F(
    MultiPeerDeviceTransportTestFixture,
    PeerIndexConversionRoundtripRank0Of2) {
  // Minimal 2-rank case: single peer
  verifyPeerIndexConversionRoundtrip(/*myRank=*/0, /*nRanks=*/2);
}

TEST_F(
    MultiPeerDeviceTransportTestFixture,
    PeerIndexConversionRoundtripRank1Of2) {
  // Minimal 2-rank case from the other side
  verifyPeerIndexConversionRoundtrip(/*myRank=*/1, /*nRanks=*/2);
}

TEST_F(
    MultiPeerDeviceTransportTestFixture,
    PeerIndexConversionRoundtripRank4Of8) {
  // 8-rank case with middle rank: exercises larger peer counts
  verifyPeerIndexConversionRoundtrip(/*myRank=*/4, /*nRanks=*/8);
}

// =============================================================================
// Bounds Checking Documentation
// =============================================================================
//
// MULTI_PEER_CHECK_RANK / MULTI_PEER_CHECK_NOT_SELF TESTING STRATEGY:
//
// These macros provide validation for rank-based operations. Testing this is
// challenging because:
//
// 1. Device code (__CUDA_ARCH__): The macros call __trap() which aborts the
//    kernel. This cannot be caught by standard gtest mechanisms. Testing
//    invalid ranks on device would require:
//    - Running the kernel with invalid input
//    - Checking cudaGetLastError() for cudaErrorLaunchFailure
//    - This is not done here because it would leave the CUDA context in a bad
//      state and contaminate subsequent tests
//
// 2. Host code: The macros use assert() which is:
//    - A no-op in release builds
//    - Calls abort() in debug builds, testable with EXPECT_DEBUG_DEATH
//
// IMPLICIT TESTING:
// All functional tests that pass valid target ranks indirectly test that
// MULTI_PEER_CHECK_RANK allows valid inputs through. The tests above
// (PeerIterationHelpers*) verify peer iteration for all valid ranks.
//
// For invalid rank detection during development, the macros print detailed
// debug information (file:line, block/thread indices) before aborting,
// making issues easy to diagnose.
//
// =============================================================================

} // namespace comms::pipes
