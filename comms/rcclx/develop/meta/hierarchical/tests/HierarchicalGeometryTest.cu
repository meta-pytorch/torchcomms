// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <vector>

#include "meta/hierarchical/hierarchical_geometry.h"

using rcclx::hier::chooseTiles;
using rcclx::hier::Geometry;
using rcclx::hier::geometryFeasible;
using rcclx::hier::InterAlgo;
using rcclx::hier::kAlignElems;
using rcclx::hier::Piece;

namespace {

// Every (tile, shard, sub-shard) piece in buffer order.
std::vector<Piece> allSubs(const Geometry& g) {
  std::vector<Piece> out;
  for (int t = 0; t < g.nTiles; t++) {
    for (int j = 0; j < g.nLocal; j++) {
      for (int m = 0; m < g.nNodes; m++) {
        out.push_back(g.sub(t, j, m));
      }
    }
  }
  return out;
}

struct Shape {
  size_t count;
  int nNodes;
  int nLocal;
  int nTiles;
};

const std::vector<Shape> kShapes = {
    {2 * 8 * kAlignElems, 2, 8, 1},
    {2 * 8 * kAlignElems + 1, 2, 8, 1},
    {10'000'019, 2, 8, 7},
    {10'000'019, 4, 8, 16},
    {10'000'019, 8, 8, 3},
    {8 * kAlignElems * 5 + 333, 8, 1, 5},
};

} // namespace

TEST(HierarchicalGeometryTest, SubShardsTileTheBufferExactly) {
  for (const Shape& s : kShapes) {
    const Geometry g{s.count, s.nNodes, s.nLocal, s.nTiles};
    size_t next = 0;
    for (const Piece& p : allSubs(g)) {
      EXPECT_EQ(p.offset, next) << "count=" << s.count;
      EXPECT_GE(p.count, kAlignElems) << "count=" << s.count;
      next = p.offset + p.count;
    }
    EXPECT_EQ(next, s.count);
  }
}

TEST(HierarchicalGeometryTest, InteriorBoundariesAreAligned) {
  for (const Shape& s : kShapes) {
    const Geometry g{s.count, s.nNodes, s.nLocal, s.nTiles};
    for (const Piece& p : allSubs(g)) {
      EXPECT_EQ(p.offset % kAlignElems, 0u) << "count=" << s.count;
    }
  }
}

TEST(HierarchicalGeometryTest, MaxShardAndSubCoverEveryTile) {
  const Geometry g{10'000'019, 4, 8, 16};
  for (int t = 0; t < g.nTiles; t++) {
    EXPECT_LE(g.shard(t, 3).count, g.maxShard(3));
    EXPECT_LE(g.sub(t, 3, 2).count, g.maxSub(3, 2));
  }
  EXPECT_EQ(g.maxShard(7), g.shard(g.nTiles - 1, 7).count);
}

TEST(HierarchicalGeometryTest, InterAlgoIsExchangeOnlyForTwoNodes) {
  EXPECT_EQ((Geometry{0, 2, 8, 1}.inter()), InterAlgo::Exchange);
  EXPECT_EQ((Geometry{0, 4, 8, 1}.inter()), InterAlgo::ReduceScatterAllGather);
  EXPECT_EQ((Geometry{0, 8, 8, 1}.inter()), InterAlgo::ReduceScatterAllGather);
}

TEST(HierarchicalGeometryTest, FeasibilityNeedsOneAlignedUnitPerSubShard) {
  EXPECT_TRUE(geometryFeasible(2 * 8 * kAlignElems, 2, 8));
  EXPECT_FALSE(geometryFeasible(2 * 8 * kAlignElems - 1, 2, 8));
}

TEST(HierarchicalGeometryTest, TileCountFollowsTileBytesAndCaps) {
  constexpr size_t kMiB = 1024 * 1024;
  // 64 MiB of floats at 16 MiB tiles.
  EXPECT_EQ(chooseTiles(16 * kMiB, 4, 2, 8, 16 * kMiB, 16), 4);
  // Capped by maxTiles.
  EXPECT_EQ(chooseTiles(256 * kMiB, 4, 2, 8, 1 * kMiB, 16), 16);
  // Smaller than one tile.
  EXPECT_EQ(chooseTiles(1024 * 1024, 4, 2, 8, 32 * kMiB, 16), 1);
  // Capped by how many minimum-size tiles fit: 3 units of 2*8*512.
  EXPECT_EQ(chooseTiles(3 * 2 * 8 * kAlignElems, 4, 2, 8, 1, 16), 3);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
