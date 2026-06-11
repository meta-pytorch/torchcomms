// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <set>

#include "comms/prims/collectives/RingUtils.h"

namespace comms::prims::test {

TEST(MakeStandardRings, SingleRing) {
  auto rings = make_standard_rings(8, 0, 1);
  ASSERT_TRUE(rings.has_value());
  ASSERT_EQ(rings->size(), 1);
  EXPECT_EQ((*rings)[0].prev_rank, 7);
  EXPECT_EQ((*rings)[0].next_rank, 1);
}

TEST(MakeStandardRings, NeighborsAreConsistent) {
  int W = 8;
  for (int num_rings = 1; num_rings <= 4; num_rings++) {
    for (int my_rank = 0; my_rank < W; my_rank++) {
      auto rings = make_standard_rings(W, my_rank, num_rings);
      ASSERT_TRUE(rings.has_value());
      for (int r = 0; r < num_rings; r++) {
        int next = (*rings)[r].next_rank;
        auto neighbor_rings = make_standard_rings(W, next, num_rings);
        ASSERT_TRUE(neighbor_rings.has_value());
        EXPECT_EQ((*neighbor_rings)[r].prev_rank, my_rank)
            << "ring=" << r << " my_rank=" << my_rank << " next=" << next;
      }
    }
  }
}

TEST(MakeStandardRings, EachRingVisitsAllRanks) {
  int W = 8;
  auto rings = make_standard_rings(W, 0, 4);
  ASSERT_TRUE(rings.has_value());
  for (int r = 0; r < 4; r++) {
    std::set<int> visited;
    int current = 0;
    for (int i = 0; i < W; i++) {
      visited.insert(current);
      auto cur_rings = make_standard_rings(W, current, 4);
      ASSERT_TRUE(cur_rings.has_value());
      current = (*cur_rings)[r].next_rank;
    }
    EXPECT_EQ(visited.size(), static_cast<std::size_t>(W))
        << "Ring " << r << " does not visit all ranks";
    EXPECT_EQ(current, 0) << "Ring " << r << " does not cycle back to start";
  }
}

TEST(MakeStandardRings, DistinctNeighborsAcrossRings) {
  auto rings = make_standard_rings(8, 0, 4);
  ASSERT_TRUE(rings.has_value());

  std::set<int> next_ranks;
  for (int r = 0; r < 4; r++) {
    next_ranks.insert((*rings)[r].next_rank);
  }
  EXPECT_EQ(next_ranks.size(), 4) << "All rings have same next neighbor";
}

TEST(MakeStandardRings, TooManyRingsReturnsNullopt) {
  auto rings = make_standard_rings(4, 0, 3);
  EXPECT_FALSE(rings.has_value());
}

TEST(MakeStandardRings, MaxRingsForPowerOfTwo) {
  auto rings = make_standard_rings(8, 0, 4);
  ASSERT_TRUE(rings.has_value());
  EXPECT_EQ(rings->size(), 4);

  auto too_many = make_standard_rings(8, 0, 5);
  EXPECT_FALSE(too_many.has_value());
}

TEST(MakeStandardRings, PrimeWorldSize) {
  auto rings = make_standard_rings(7, 0, 6);
  ASSERT_TRUE(rings.has_value());
  EXPECT_EQ(rings->size(), 6);
}

TEST(MakeStandardRings, TwoRanksOnlyOneRing) {
  auto rings = make_standard_rings(2, 0, 1);
  ASSERT_TRUE(rings.has_value());
  EXPECT_EQ(rings->size(), 1);
  EXPECT_EQ((*rings)[0].prev_rank, 1);
  EXPECT_EQ((*rings)[0].next_rank, 1);

  auto too_many = make_standard_rings(2, 0, 2);
  EXPECT_FALSE(too_many.has_value());
}

TEST(MakeStandardRings, IterativeRankWalkMatchesRingTraversal) {
  int W = 8;
  auto rings_r0 = make_standard_rings(W, 0, 4);
  ASSERT_TRUE(rings_r0.has_value());

  for (int r = 0; r < 4; r++) {
    for (int my_rank = 0; my_rank < W; my_rank++) {
      auto my_rings = make_standard_rings(W, my_rank, 4);
      ASSERT_TRUE(my_rings.has_value());
      int stride = (my_rank - (*my_rings)[r].prev_rank + W) % W;

      int current = my_rank;
      int walk_rank = my_rank;
      for (int step = 0; step < W - 1; step++) {
        auto cur_rings = make_standard_rings(W, current, 4);
        ASSERT_TRUE(cur_rings.has_value());
        current = (*cur_rings)[r].prev_rank;

        walk_rank = (walk_rank + W - stride) % W;

        EXPECT_EQ(walk_rank, current)
            << "ring=" << r << " my_rank=" << my_rank << " step=" << step;
      }
    }
  }
}

} // namespace comms::prims::test
