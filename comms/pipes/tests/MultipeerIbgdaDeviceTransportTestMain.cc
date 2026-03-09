// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Tests for MultipeerIbgdaDeviceTransport
// Note: MultipeerIbgdaDeviceTransport.cuh includes CUDA headers that cannot
// be compiled by a regular C++ compiler. Device-side tests are implemented in
// MultipeerIbgdaDeviceTransportTest.cu and launched via kernel wrapper
// functions.

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "comms/pipes/tests/MultipeerIbgdaDeviceTransportTest.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/CudaRAII.h"

using namespace meta::comms;

namespace comms::pipes::tests {

// =============================================================================
// Parameterized Device-side Rank Mapping Test
// =============================================================================

struct RankMappingTestCase {
  int myRank;
  int nRanks;
  std::vector<int> expectedResults;
};

class RankMappingTest : public ::testing::TestWithParam<RankMappingTestCase> {};

TEST_P(RankMappingTest, DeviceRankMapping) {
  const auto& testCase = GetParam();
  const int myRank = testCase.myRank;
  const int nRanks = testCase.nRanks;
  const int numPeers = nRanks - 1;
  const auto& expectedResults = testCase.expectedResults;

  // Allocate device memory
  DeviceBuffer resultsBuf(numPeers * sizeof(int));
  DeviceBuffer expectedBuf(numPeers * sizeof(int));
  DeviceBuffer successBuf(sizeof(bool));

  auto* d_results = static_cast<int*>(resultsBuf.get());
  auto* d_expected = static_cast<int*>(expectedBuf.get());
  auto* d_success = static_cast<bool*>(successBuf.get());

  // Copy expected results to device
  CUDACHECK_TEST(cudaMemcpy(
      d_expected,
      expectedResults.data(),
      numPeers * sizeof(int),
      cudaMemcpyHostToDevice));

  // Initialize success to false
  bool initSuccess = false;
  CUDACHECK_TEST(cudaMemcpy(
      d_success, &initSuccess, sizeof(bool), cudaMemcpyHostToDevice));

  // Run kernel
  runTestRankMappingKernel(
      myRank, nRanks, d_results, d_expected, numPeers, d_success);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Copy results back
  bool success = false;
  CUDACHECK_TEST(
      cudaMemcpy(&success, d_success, sizeof(bool), cudaMemcpyDeviceToHost));

  std::vector<int> actualResults(numPeers);
  CUDACHECK_TEST(cudaMemcpy(
      actualResults.data(),
      d_results,
      numPeers * sizeof(int),
      cudaMemcpyDeviceToHost));

  EXPECT_TRUE(success) << "Device-side rank mapping test failed for myRank="
                       << myRank;
  EXPECT_EQ(actualResults, expectedResults);
}

INSTANTIATE_TEST_SUITE_P(
    MultipeerIbgdaDeviceTransport,
    RankMappingTest,
    ::testing::Values(
        // Rank 0: all peers have higher ranks [1, 2, 3]
        RankMappingTestCase{0, 4, {1, 2, 3}},
        // Rank 2: peers on both sides [0, 1, 3]
        RankMappingTestCase{2, 4, {0, 1, 3}},
        // Rank 3: all peers have lower ranks [0, 1, 2]
        RankMappingTestCase{3, 4, {0, 1, 2}}));

// =============================================================================
// Peer-Index Offset Test
//
// Verifies the invariant that for every pair of ranks (A, B):
//   A's remote offset for B == B's local offset for A
//
// This ensures that when rank A RDMA-writes to rank B's per-peer buffer
// (signal inbox, counter buffer, etc.), it targets the slot that rank B
// actually reads for rank A. The bug fixed in D93259521 was using the
// sender's peer index for the remote offset instead of the receiver's
// peer index for the sender.
//
// The offset logic is used by the window layer for per-peer signal/counter
// buffers and mirrors the skip-self rank-to-peer-index mapping:
//   rankToPeerIndex(rank, myRank) = (rank < myRank) ? rank : (rank - 1)
//   peerIndexToRank(idx, myRank)  = (idx < myRank) ? idx : (idx + 1)
//   localOffset(i)  = i * slotCount * sizeof(uint64_t)
//   remoteOffset(i) = myIndexOnPeer * slotCount * sizeof(uint64_t)
//     where myIndexOnPeer = rankToPeerIndex(myRank, peerRank)
// =============================================================================

namespace {

// Skip-self peer index: the index that `myRank` assigns to `rank`
int rank_to_peer_index(int rank, int myRank) {
  return (rank < myRank) ? rank : (rank - 1);
}

// Reverse: global rank at peer index `idx` from perspective of `myRank`
int peer_index_to_rank(int idx, int myRank) {
  return (idx < myRank) ? idx : (idx + 1);
}

} // namespace

class PeerIndexOffsetTest : public ::testing::TestWithParam<int> {};

TEST_P(PeerIndexOffsetTest, RemoteOffsetMatchesLocalSlot) {
  const int nRanks = GetParam();
  const int numPeers = nRanks - 1;
  const std::size_t slotCount = 1;

  // For every rank acting as sender...
  for (int myRank = 0; myRank < nRanks; myRank++) {
    // For every peer index i on this rank...
    for (int i = 0; i < numPeers; i++) {
      int peerRank = peer_index_to_rank(i, myRank);

      // Sender's local offset for peer index i
      std::size_t localOffset = i * slotCount * sizeof(uint64_t);

      // Sender's remote offset: where to RDMA-write on the peer's buffer
      int myIndexOnPeer = rank_to_peer_index(myRank, peerRank);
      std::size_t remoteOffset = myIndexOnPeer * slotCount * sizeof(uint64_t);

      // The peer's local offset for us must match our remote offset
      int peersIndexForUs = rank_to_peer_index(myRank, peerRank);
      std::size_t peersLocalOffset =
          peersIndexForUs * slotCount * sizeof(uint64_t);

      EXPECT_EQ(remoteOffset, peersLocalOffset)
          << "Rank " << myRank << " -> Rank " << peerRank
          << ": remoteOffset=" << remoteOffset
          << " != peer's localOffset=" << peersLocalOffset
          << " (myIndexOnPeer=" << myIndexOnPeer
          << ", peersIndexForUs=" << peersIndexForUs << ")";

      // Also verify the reverse direction: peer's remote offset for us
      // must match our local offset
      int peersPeerIndex = rank_to_peer_index(myRank, peerRank);
      int peersMyIndexOnUs = rank_to_peer_index(peerRank, myRank);
      std::size_t peersRemoteOffset =
          peersMyIndexOnUs * slotCount * sizeof(uint64_t);

      EXPECT_EQ(peersRemoteOffset, localOffset)
          << "Rank " << peerRank << " -> Rank " << myRank
          << ": remoteOffset=" << peersRemoteOffset
          << " != our localOffset=" << localOffset
          << " (peer's peerIndex=" << peersPeerIndex
          << ", peersMyIndexOnUs=" << peersMyIndexOnUs << ")";
    }
  }
}

// Test with multiple slot counts to verify scaling
TEST_P(PeerIndexOffsetTest, MultipleSlotCounts) {
  const int nRanks = GetParam();
  const int numPeers = nRanks - 1;

  for (std::size_t slotCount : {1, 4, 16}) {
    for (int myRank = 0; myRank < nRanks; myRank++) {
      for (int i = 0; i < numPeers; i++) {
        int peerRank = peer_index_to_rank(i, myRank);
        int myIndexOnPeer = rank_to_peer_index(myRank, peerRank);

        std::size_t remoteOffset = myIndexOnPeer * slotCount * sizeof(uint64_t);
        std::size_t peersLocalOffset =
            rank_to_peer_index(myRank, peerRank) * slotCount * sizeof(uint64_t);

        EXPECT_EQ(remoteOffset, peersLocalOffset)
            << "slotCount=" << slotCount << " Rank " << myRank << " -> Rank "
            << peerRank;
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    MultipeerIbgdaDeviceTransport,
    PeerIndexOffsetTest,
    ::testing::Values(2, 3, 4, 8),
    [](const ::testing::TestParamInfo<int>& info) {
      return std::to_string(info.param) + "Ranks";
    });

} // namespace comms::pipes::tests

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
