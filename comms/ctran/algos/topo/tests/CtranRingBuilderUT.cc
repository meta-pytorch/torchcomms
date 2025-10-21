// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <algorithm>
#include <set>
#include <vector>

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "comms/ctran/algos/topo/CtranRingBuilder.h"

namespace ctran::algos::topo {

class CtranRingBuilderTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(CtranRingBuilderTest, ValidateSpecificSingleRing) {
  // Test specific single ring configuration: 0 1 2 3 4 5 6 7
  int nNodes = 8;
  int nLocalRanks = 1; // Single rank per node
  int nRanks = 8;

  auto rings = getMultiFlatRing(nNodes, nLocalRanks, nRanks);

  EXPECT_EQ(rings.size(), nLocalRanks);
  EXPECT_EQ(rings[0].size(), nRanks);

  // Expected sequence: 0 1 2 3 4 5 6 7
  std::vector<int> expectedRing = {0, 1, 2, 3, 4, 5, 6, 7};

  EXPECT_EQ(rings[0], expectedRing)
      << "Ring should be exactly [0, 1, 2, 3, 4, 5, 6, 7]";

  // Validate each position explicitly
  for (int i = 0; i < 8; i++) {
    EXPECT_EQ(rings[0][i], i)
        << "Position " << i << " should contain rank " << i;
  }

  // Also test prev/next relationships for this ring
  int rank = 0;
  int nRings = 1;
  std::vector<std::vector<int>> ringPrev(nRings, std::vector<int>(nRanks));
  std::vector<std::vector<int>> ringNext(nRings, std::vector<int>(nRanks));
  std::vector<std::vector<int>> builtRings(nRings, std::vector<int>(nRanks));

  buildMultiFlatRing(
      rank,
      nLocalRanks,
      nRanks,
      nNodes,
      nRings,
      ringPrev,
      ringNext,
      builtRings);

  // Verify the built ring matches our expected sequence
  EXPECT_EQ(builtRings[0], expectedRing)
      << "Built ring should be exactly [0, 1, 2, 3, 4, 5, 6, 7]";

  // Verify prev/next relationships for the sequential ring
  for (int r = 0; r < nRanks; r++) {
    int expectedPrev = (r - 1 + nRanks) % nRanks;
    int expectedNext = (r + 1) % nRanks;

    EXPECT_EQ(ringPrev[0][r], expectedPrev)
        << "Rank " << r << " should have prev=" << expectedPrev;
    EXPECT_EQ(ringNext[0][r], expectedNext)
        << "Rank " << r << " should have next=" << expectedNext;
  }
}

TEST_F(CtranRingBuilderTest, ValidateLogCaseMultipleRings) {
  // Test the specific case from the log output:
  // 16 ranks (0-15), 2 nodes, 8 ranks per node, multiple ring channels
  int nNodes = 2;
  int nLocalRanks = 8; // 8 ranks per node
  int nRanks = 16; // Total 16 ranks (0-15)

  auto rings = getMultiFlatRing(nNodes, nLocalRanks, nRanks);

  EXPECT_EQ(rings.size(), nLocalRanks);
  for (const auto& ring : rings) {
    EXPECT_EQ(ring.size(), nRanks);
  }

  // Expected patterns based on the log output:
  // Ring 0: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
  std::vector<int> expectedRing0 = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  EXPECT_EQ(rings[0], expectedRing0) << "Ring 0 should be sequential [0-15]";

  // Ring 2: 0, 1, 2, 3, 4, 5, 14, 15, 8, 9, 10, 11, 12, 13, 6, 7
  // This shows swap between positions 6 and 14 (ranks 6 and 14)
  std::vector<int> expectedRing2 = {
      0, 1, 2, 3, 4, 5, 14, 15, 8, 9, 10, 11, 12, 13, 6, 7};
  EXPECT_EQ(rings[2], expectedRing2)
      << "Ring 2 should have swapped ranks 6 and 14";

  // Ring 3: 0, 1, 2, 3, 4, 13, 14, 15, 8, 9, 10, 11, 12, 5, 6, 7
  // This shows swap between positions 5 and 13 (ranks 5 and 13)
  std::vector<int> expectedRing3 = {
      0, 1, 2, 3, 4, 13, 14, 15, 8, 9, 10, 11, 12, 5, 6, 7};
  EXPECT_EQ(rings[3], expectedRing3)
      << "Ring 3 should have swapped ranks 5 and 13";

  // Ring 7: 0, 9, 10, 11, 12, 13, 14, 15, 8, 1, 2, 3, 4, 5, 6, 7
  // This shows swap between positions 1 and 9 (ranks 1 and 9)
  std::vector<int> expectedRing7 = {
      0, 9, 10, 11, 12, 13, 14, 15, 8, 1, 2, 3, 4, 5, 6, 7};
  EXPECT_EQ(rings[7], expectedRing7)
      << "Ring 7 should have swapped ranks 1 and 9";

  // Verify that all rings contain all ranks exactly once
  for (size_t ringIdx = 0; ringIdx < rings.size(); ringIdx++) {
    std::set<int> unique_ranks(rings[ringIdx].begin(), rings[ringIdx].end());
    EXPECT_EQ(unique_ranks.size(), nRanks)
        << "Ring " << ringIdx << " should contain all " << nRanks << " ranks";

    // Check that ranks are in valid range
    for (int rank : rings[ringIdx]) {
      EXPECT_GE(rank, 0) << "Invalid rank: " << rank << " in ring " << ringIdx;
      EXPECT_LT(rank, nRanks)
          << "Rank out of bounds: " << rank << " in ring " << ringIdx;
    }
  }

  // Test buildMultiRing with this configuration
  int rank = 0;
  int nRings = 8; // Test first 8 rings
  std::vector<std::vector<int>> ringPrev(nRings, std::vector<int>(nRanks));
  std::vector<std::vector<int>> ringNext(nRings, std::vector<int>(nRanks));
  std::vector<std::vector<int>> builtRings(nRings, std::vector<int>(nRanks));

  buildMultiFlatRing(
      rank,
      nLocalRanks,
      nRanks,
      nNodes,
      nRings,
      ringPrev,
      ringNext,
      builtRings);

  // Verify that built rings match our expected patterns
  EXPECT_EQ(builtRings[0], expectedRing0) << "Built ring 0 mismatch";
  EXPECT_EQ(builtRings[2], expectedRing2) << "Built ring 2 mismatch";
  EXPECT_EQ(builtRings[3], expectedRing3) << "Built ring 3 mismatch";
  EXPECT_EQ(builtRings[7], expectedRing7) << "Built ring 7 mismatch";

  // Verify prev/next relationships for ring 0 (sequential)
  for (int r = 0; r < nRanks; r++) {
    int expectedPrev = (r - 1 + nRanks) % nRanks;
    int expectedNext = (r + 1) % nRanks;

    EXPECT_EQ(ringPrev[0][r], expectedPrev)
        << "Ring 0: Rank " << r << " should have prev=" << expectedPrev;
    EXPECT_EQ(ringNext[0][r], expectedNext)
        << "Ring 0: Rank " << r << " should have next=" << expectedNext;
  }

  // Verify that ring 1 has correct prev/next for the swapped elements
  // In ring 1: [0, 1, 2, 3, 4, 5, 6, 15, 8, 9, 10, 11, 12, 13, 14, 7]
  // Rank 6 (position 6) -> prev=5, next=15
  // Rank 15 (position 7) -> prev=6, next=8
  // Rank 7 (position 15) -> prev=14, next=0
  // Rank 14 (position 14) -> prev=13, next=7
  EXPECT_EQ(ringPrev[1][6], 5) << "Ring 1: Rank 6 prev should be 5";
  EXPECT_EQ(ringNext[1][6], 15) << "Ring 1: Rank 6 next should be 15";
  EXPECT_EQ(ringPrev[1][15], 6) << "Ring 1: Rank 15 prev should be 6";
  EXPECT_EQ(ringNext[1][15], 8) << "Ring 1: Rank 15 next should be 8";
  EXPECT_EQ(ringPrev[1][7], 14) << "Ring 1: Rank 7 prev should be 14";
  EXPECT_EQ(ringNext[1][7], 0) << "Ring 1: Rank 7 next should be 0";
}

TEST_F(CtranRingBuilderTest, ValidateFourNodeThirtyTwoRankCase) {
  // Test the specific case from the log output:
  // 32 ranks (0-31), 4 nodes, 8 ranks per node, multiple ring configurations
  int nNodes = 4;
  int nLocalRanks = 8; // 8 ranks per node
  int nRanks = 32; // Total 32 ranks (0-31)

  auto rings = getMultiFlatRing(nNodes, nLocalRanks, nRanks);

  EXPECT_EQ(rings.size(), nLocalRanks);
  for (const auto& ring : rings) {
    EXPECT_EQ(ring.size(), nRanks);
  }

  // Expected patterns based on the log output:
  // Ring 0: Sequential 0, 1, 2, ..., 31
  std::vector<int> expectedRing0 = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                    11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                    22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
  EXPECT_EQ(rings[0], expectedRing0) << "Ring 0 should be sequential [0-31]";

  // Ring 1: Load balancing with swaps at position 7, 15, 23, 31
  // 0, 1, 2, 3, 4, 5, 6, 15, 8, 9, 10, 11, 12, 13, 14, 23, 16, 17, 18, 19, 20,
  // 21, 22, 31, 24, 25, 26, 27, 28, 29, 30, 7
  std::vector<int> expectedRing1 = {0,  1,  2,  3,  4,  5,  6,  15, 8,  9,  10,
                                    11, 12, 13, 14, 23, 16, 17, 18, 19, 20, 21,
                                    22, 31, 24, 25, 26, 27, 28, 29, 30, 7};
  EXPECT_EQ(rings[1], expectedRing1)
      << "Ring 1 should have load balancing swaps at node boundaries";

  // Ring 2: Load balancing with swaps at position 6, 14, 22, 30
  // 0, 1, 2, 3, 4, 5, 14, 15, 8, 9, 10, 11, 12, 13, 22, 23, 16, 17, 18, 19, 20,
  // 21, 30, 31, 24, 25, 26, 27, 28, 29, 6, 7
  std::vector<int> expectedRing2 = {0,  1,  2,  3,  4,  5,  14, 15, 8,  9,  10,
                                    11, 12, 13, 22, 23, 16, 17, 18, 19, 20, 21,
                                    30, 31, 24, 25, 26, 27, 28, 29, 6,  7};
  EXPECT_EQ(rings[2], expectedRing2)
      << "Ring 2 should have load balancing swaps at positions 6, 14, 22, 30";

  // Ring 3: Load balancing with swaps at position 5, 13, 21, 29
  // 0, 1, 2, 3, 4, 13, 14, 15, 8, 9, 10, 11, 12, 21, 22, 23, 16, 17, 18, 19,
  // 20, 29, 30, 31, 24, 25, 26, 27, 28, 5, 6, 7
  std::vector<int> expectedRing3 = {0,  1,  2,  3,  4,  13, 14, 15, 8,  9,  10,
                                    11, 12, 21, 22, 23, 16, 17, 18, 19, 20, 29,
                                    30, 31, 24, 25, 26, 27, 28, 5,  6,  7};
  EXPECT_EQ(rings[3], expectedRing3)
      << "Ring 3 should have load balancing swaps at positions 5, 13, 21, 29";

  // Ring 7: Load balancing with swaps at position 1, 9, 17, 25
  // 0, 9, 10, 11, 12, 13, 14, 15, 8, 17, 18, 19, 20, 21, 22, 23, 16, 25, 26,
  // 27, 28, 29, 30, 31, 24, 1, 2, 3, 4, 5, 6, 7
  std::vector<int> expectedRing7 = {0,  9,  10, 11, 12, 13, 14, 15, 8,  17, 18,
                                    19, 20, 21, 22, 23, 16, 25, 26, 27, 28, 29,
                                    30, 31, 24, 1,  2,  3,  4,  5,  6,  7};
  EXPECT_EQ(rings[7], expectedRing7)
      << "Ring 7 should have load balancing swaps at positions 1, 9, 17, 25";

  // Verify node boundaries are preserved (first rank of each node is always in
  // correct position)
  for (size_t ringIdx = 0; ringIdx < rings.size(); ringIdx++) {
    EXPECT_EQ(rings[ringIdx][0], 0)
        << "Ring " << ringIdx << " should start with rank 0";
    EXPECT_EQ(rings[ringIdx][8], 8)
        << "Ring " << ringIdx << " should have rank 8 at position 8";
    EXPECT_EQ(rings[ringIdx][16], 16)
        << "Ring " << ringIdx << " should have rank 16 at position 16";
    EXPECT_EQ(rings[ringIdx][24], 24)
        << "Ring " << ringIdx << " should have rank 24 at position 24";
  }

  // Verify that all rings contain all ranks exactly once
  for (size_t ringIdx = 0; ringIdx < rings.size(); ringIdx++) {
    std::set<int> unique_ranks(rings[ringIdx].begin(), rings[ringIdx].end());
    EXPECT_EQ(unique_ranks.size(), nRanks)
        << "Ring " << ringIdx << " should contain all " << nRanks << " ranks";

    // Check that ranks are in valid range
    for (int rank : rings[ringIdx]) {
      EXPECT_GE(rank, 0) << "Invalid rank: " << rank << " in ring " << ringIdx;
      EXPECT_LT(rank, nRanks)
          << "Rank out of bounds: " << rank << " in ring " << ringIdx;
    }
  }

  // Test buildMultiRing with this configuration
  int rank = 0;
  int nRings = 8; // Test first 8 rings
  std::vector<std::vector<int>> ringPrev(nRings, std::vector<int>(nRanks));
  std::vector<std::vector<int>> ringNext(nRings, std::vector<int>(nRanks));
  std::vector<std::vector<int>> builtRings(nRings, std::vector<int>(nRanks));

  buildMultiFlatRing(
      rank,
      nLocalRanks,
      nRanks,
      nNodes,
      nRings,
      ringPrev,
      ringNext,
      builtRings);

  // Verify that built rings match our expected patterns
  EXPECT_EQ(builtRings[0], expectedRing0) << "Built ring 0 mismatch";
  EXPECT_EQ(builtRings[1], expectedRing1) << "Built ring 1 mismatch";
  EXPECT_EQ(builtRings[2], expectedRing2) << "Built ring 2 mismatch";
  EXPECT_EQ(builtRings[3], expectedRing3) << "Built ring 3 mismatch";
  EXPECT_EQ(builtRings[7], expectedRing7) << "Built ring 7 mismatch";

  // Verify prev/next relationships for ring 0 (sequential)
  for (int r = 0; r < nRanks; r++) {
    int expectedPrev = (r - 1 + nRanks) % nRanks;
    int expectedNext = (r + 1) % nRanks;

    EXPECT_EQ(ringPrev[0][r], expectedPrev)
        << "Ring 0: Rank " << r << " should have prev=" << expectedPrev;
    EXPECT_EQ(ringNext[0][r], expectedNext)
        << "Ring 0: Rank " << r << " should have next=" << expectedNext;
  }

  // Verify specific prev/next relationships for ring 1 with swapped elements
  // In ring 1: [0, 1, 2, 3, 4, 5, 6, 15, 8, 9, 10, 11, 12, 13, 14, 23, 16, 17,
  // 18, 19, 20, 21, 22, 31, 24, 25, 26, 27, 28, 29, 30, 7] Rank 7 (position 31)
  // -> prev=30, next=0 Rank 15 (position 7) -> prev=6, next=8 Rank 23 (position
  // 15) -> prev=14, next=16 Rank 31 (position 23) -> prev=22, next=24
  EXPECT_EQ(ringPrev[1][7], 30) << "Ring 1: Rank 7 prev should be 30";
  EXPECT_EQ(ringNext[1][7], 0) << "Ring 1: Rank 7 next should be 0";
  EXPECT_EQ(ringPrev[1][15], 6) << "Ring 1: Rank 15 prev should be 6";
  EXPECT_EQ(ringNext[1][15], 8) << "Ring 1: Rank 15 next should be 8";
  EXPECT_EQ(ringPrev[1][23], 14) << "Ring 1: Rank 23 prev should be 14";
  EXPECT_EQ(ringNext[1][23], 16) << "Ring 1: Rank 23 next should be 16";
  EXPECT_EQ(ringPrev[1][31], 22) << "Ring 1: Rank 31 prev should be 22";
  EXPECT_EQ(ringNext[1][31], 24) << "Ring 1: Rank 31 next should be 24";
}

} // namespace ctran::algos::topo
