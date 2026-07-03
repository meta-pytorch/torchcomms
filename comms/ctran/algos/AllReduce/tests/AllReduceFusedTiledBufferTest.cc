// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <array>
#include <cstddef>

#include "comms/prims/core/TiledBuffer.cuh"

namespace {

struct TileView {
  std::size_t offsetBytes;
  std::size_t bytes;
  std::size_t strideBytes;
};

TileView
tileView(char* base, std::size_t totalBytes, int numTiles, int tileId) {
  comms::prims::TiledBuffer<char> tiles(base, totalBytes, numTiles);
  return TileView{
      .offsetBytes = static_cast<std::size_t>(tiles.tile_data(tileId) - base),
      .bytes = tiles.tile_bytes(tileId),
      .strideBytes = tiles.tile_elements,
  };
}

TEST(AllReduceFusedTiledBuffer, BlockTileSplitUsesNativeTiledBufferStride) {
  std::array<char, 1000> buffer{};
  const auto tile0 = tileView(buffer.data(), buffer.size(), 3, 0);
  const auto tile1 = tileView(buffer.data(), buffer.size(), 3, 1);
  const auto tile2 = tileView(buffer.data(), buffer.size(), 3, 2);

  EXPECT_EQ(tile0.strideBytes, 336);
  EXPECT_EQ(tile0.offsetBytes, 0);
  EXPECT_EQ(tile0.bytes, 336);
  EXPECT_EQ(tile1.offsetBytes, 336);
  EXPECT_EQ(tile1.bytes, 336);
  EXPECT_EQ(tile2.offsetBytes, 672);
  EXPECT_EQ(tile2.bytes, 328);
}

TEST(AllReduceFusedTiledBuffer, BlockTileSplitKeepsDegenerateCases) {
  std::array<char, 1040> buffer{};
  EXPECT_EQ(tileView(buffer.data(), 0, 8, 0).bytes, 0);
  EXPECT_EQ(tileView(buffer.data(), 1025, 1, 0).offsetBytes, 0);
  EXPECT_EQ(tileView(buffer.data(), 1025, 1, 0).bytes, 1025);
  EXPECT_EQ(tileView(buffer.data(), 1000, 3, 3).offsetBytes, 1008);
  EXPECT_EQ(tileView(buffer.data(), 1000, 3, 3).bytes, 0);
}

TEST(AllReduceFusedTiledBuffer, LaneSplitUsesNestedTiledBuffer) {
  std::array<char, 1000> buffer{};
  const auto blockTile = tileView(buffer.data(), buffer.size(), 3, 0);
  const auto lane0 =
      tileView(buffer.data() + blockTile.offsetBytes, blockTile.bytes, 2, 0);
  const auto lane1 =
      tileView(buffer.data() + blockTile.offsetBytes, blockTile.bytes, 2, 1);

  EXPECT_EQ(blockTile.bytes, 336);
  EXPECT_EQ(lane0.offsetBytes, 0);
  EXPECT_EQ(lane0.bytes, 176);
  EXPECT_EQ(lane1.offsetBytes, 176);
  EXPECT_EQ(lane1.bytes, 160);
  EXPECT_EQ(lane0.bytes + lane1.bytes, blockTile.bytes);
}

TEST(AllReduceFusedTiledBuffer, LaneSplitKeepsTailInSecondLane) {
  std::array<char, 1000> buffer{};
  const auto blockTail = tileView(buffer.data(), buffer.size(), 3, 2);
  const auto lane0 =
      tileView(buffer.data() + blockTail.offsetBytes, blockTail.bytes, 2, 0);
  const auto lane1 =
      tileView(buffer.data() + blockTail.offsetBytes, blockTail.bytes, 2, 1);

  EXPECT_EQ(blockTail.bytes, 328);
  EXPECT_EQ(lane0.bytes, 176);
  EXPECT_EQ(lane1.bytes, 152);
  EXPECT_EQ(lane0.bytes + lane1.bytes, blockTail.bytes);
}

TEST(AllReduceFusedTiledBuffer, CoversUnalignedNear64MiBPayload) {
  constexpr std::size_t kUnaligned64MiBBytes =
      64UL * 1024 * 1024 + sizeof(float);
  constexpr int kNumPartitions = 8;
  constexpr std::size_t kNativeTileAlignmentBytes = 16;

  comms::prims::TiledBuffer<char> tiles(
      nullptr, kUnaligned64MiBBytes, kNumPartitions);
  const std::size_t unalignedPartitionBytes =
      (kUnaligned64MiBBytes + kNumPartitions - 1) / kNumPartitions;

  EXPECT_NE(tiles.tile_elements, unalignedPartitionBytes);
  EXPECT_EQ(tiles.tile_elements % kNativeTileAlignmentBytes, 0);

  std::size_t coveredBytes = 0;
  for (int tileId = 0; tileId < kNumPartitions; ++tileId) {
    coveredBytes += tiles.tile_bytes(tileId);
  }
  EXPECT_EQ(coveredBytes, kUnaligned64MiBBytes);
}

} // namespace
