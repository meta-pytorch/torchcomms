// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <cstdint>

#include "comms/ctran/algos/AllReduce/AllReduceFusedTypes.h"

namespace {

using ctran::allreduce::common::compute_aligned_tile_parition_size;
using ctran::allreduce::common::kTileParitionAligmentBytes;

TEST(AllReduceFusedProtocolAlignment, BlockTileSplitAlignsMultiBlockStart) {
  EXPECT_EQ(compute_aligned_tile_parition_size(1024, 1, 4), 256);
  EXPECT_EQ(compute_aligned_tile_parition_size(1025, 1, 4), 384);
  EXPECT_EQ(compute_aligned_tile_parition_size(64, 1, 8), 128);
}

TEST(AllReduceFusedProtocolAlignment, BlockTileSplitKeepsDegenerateCases) {
  EXPECT_EQ(compute_aligned_tile_parition_size(0, 1, 8), 0);
  EXPECT_EQ(compute_aligned_tile_parition_size(1025, 1, 1), 1025);
  EXPECT_EQ(compute_aligned_tile_parition_size(1025, 1, 0), 1025);
  EXPECT_EQ(compute_aligned_tile_parition_size(1025, 0, 4), 1025);
}

TEST(AllReduceFusedProtocolAlignment, TreeLaneSplitAlignsFloatLaneStart) {
  constexpr size_t kFloatElemsPerProtocolStep =
      kTileParitionAligmentBytes / sizeof(float);

  for (size_t elems = 1; elems <= 160; ++elems) {
    const size_t lane0PartitionElems =
        compute_aligned_tile_parition_size(elems, sizeof(float), 2);
    const size_t lane0Elems =
        lane0PartitionElems < elems ? lane0PartitionElems : elems;
    EXPECT_LE(lane0Elems, elems);
    EXPECT_GT(lane0Elems, 0);
    if (lane0Elems < elems) {
      EXPECT_EQ(lane0Elems % kFloatElemsPerProtocolStep, 0)
          << "elems=" << elems;
    }
  }
}

TEST(AllReduceFusedProtocolAlignment, TreeLaneSplitAlignsHalfLaneStart) {
  constexpr size_t kHalfElemsPerProtocolStep =
      kTileParitionAligmentBytes / sizeof(uint16_t);

  for (size_t elems = 1; elems <= 192; ++elems) {
    const size_t lane0PartitionElems =
        compute_aligned_tile_parition_size(elems, sizeof(uint16_t), 2);
    const size_t lane0Elems =
        lane0PartitionElems < elems ? lane0PartitionElems : elems;
    EXPECT_LE(lane0Elems, elems);
    EXPECT_GT(lane0Elems, 0);
    if (lane0Elems < elems) {
      EXPECT_EQ(lane0Elems % kHalfElemsPerProtocolStep, 0) << "elems=" << elems;
    }
  }
}

TEST(AllReduceFusedProtocolAlignment, TreeLaneSplitStaysNearHalfTile) {
  EXPECT_EQ(compute_aligned_tile_parition_size(31, sizeof(float), 2), 32);
  EXPECT_EQ(compute_aligned_tile_parition_size(32, sizeof(float), 2), 32);
  EXPECT_EQ(compute_aligned_tile_parition_size(33, sizeof(float), 2), 32);
  EXPECT_EQ(compute_aligned_tile_parition_size(1024, sizeof(float), 2), 512);
  EXPECT_EQ(compute_aligned_tile_parition_size(1025, sizeof(float), 2), 544);
  EXPECT_EQ(compute_aligned_tile_parition_size(1056, sizeof(float), 2), 544);
  EXPECT_EQ(compute_aligned_tile_parition_size(1057, sizeof(float), 2), 544);
}

TEST(AllReduceFusedProtocolAlignment, AlignsElementSizesThatDoNotDivide128) {
  EXPECT_EQ(compute_aligned_tile_parition_size(0, sizeof(float), 2), 0);
  EXPECT_EQ(compute_aligned_tile_parition_size(9, 3, 2), 128);
  EXPECT_EQ(compute_aligned_tile_parition_size(129, 3, 2), 128);
  EXPECT_EQ(compute_aligned_tile_parition_size(257, 3, 2), 256);
}

TEST(AllReduceFusedProtocolAlignment, CoversUnalignedNear64MiBPayload) {
  constexpr size_t kUnaligned64MiBBytes = 64UL * 1024 * 1024 + sizeof(float);
  constexpr size_t kUnaligned64MiBElems = kUnaligned64MiBBytes / sizeof(float);
  constexpr size_t kNumPartitions = 8;
  constexpr size_t kFloatElemsPerProtocolStep =
      kTileParitionAligmentBytes / sizeof(float);

  const size_t unalignedPartitionElems =
      (kUnaligned64MiBElems + kNumPartitions - 1) / kNumPartitions;
  EXPECT_NE(unalignedPartitionElems % kFloatElemsPerProtocolStep, 0);
  EXPECT_EQ(
      compute_aligned_tile_parition_size(
          kUnaligned64MiBElems, sizeof(float), kNumPartitions) %
          kFloatElemsPerProtocolStep,
      0);
}

} // namespace
