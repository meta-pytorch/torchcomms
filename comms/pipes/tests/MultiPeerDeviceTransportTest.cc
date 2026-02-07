// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <numeric>
#include <vector>

#include "comms/pipes/MultiPeerDeviceTransport.cuh"
#include "comms/pipes/Transport.cuh"
#include "comms/pipes/tests/MultiPeerDeviceTransportTest.cuh"
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
// Edge Cases
// =============================================================================

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
// Self-Transport Tests (via get_transport())
// =============================================================================

TEST_F(MultiPeerDeviceTransportTestFixture, GetTransportReturnsCorrectType) {
  // Test that get_transport() returns a valid Transport with SELF type
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
// Bounds Checking Documentation
// =============================================================================
//
// MULTI_PEER_CHECK_PEER TESTING STRATEGY:
//
// The MULTI_PEER_CHECK_PEER macro provides bounds validation for peer ranks.
// Testing this is challenging because:
//
// 1. Device code (__CUDA_ARCH__): The macro calls __trap() which aborts the
//    kernel. This cannot be caught by standard gtest mechanisms. Testing
//    invalid peer ranks on device would require:
//    - Running the kernel with invalid input
//    - Checking cudaGetLastError() for cudaErrorLaunchFailure
//    - This is not done here because it would leave the CUDA context in a bad
//      state and contaminate subsequent tests
//
// 2. Host code: The macro uses assert() which is:
//    - A no-op in release builds
//    - Calls abort() in debug builds, testable with EXPECT_DEBUG_DEATH
//
// IMPLICIT TESTING:
// All functional tests that pass valid peer ranks indirectly test that
// MULTI_PEER_CHECK_PEER allows valid inputs through. The tests above
// (PeerIterationHelpers*) verify peer iteration for all valid peer indices.
//
// For invalid peer detection during development, the macro prints detailed
// debug information (file:line, block/thread indices) before aborting,
// making issues easy to diagnose.
//
// =============================================================================

} // namespace comms::pipes
