// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstddef>

#include "comms/pipes/ll128/Ll128AutoTune.cuh"

namespace comms::pipes {

// =============================================================================
// ll128_auto_tune — unidirectional launch config
// =============================================================================

TEST(Ll128AutoTuneTest, ZeroBytes) {
  auto cfg = ll128_auto_tune(0);
  EXPECT_EQ(cfg.numBlocks, 0);
  EXPECT_EQ(cfg.numThreads, 0);
}

TEST(Ll128AutoTuneTest, BoundaryValues) {
  // At each threshold boundary, verify the block count transitions correctly.
  // All configs use 512 threads.

  // <= 2KB: 1 block
  auto at_2k = ll128_auto_tune(2048);
  EXPECT_EQ(at_2k.numBlocks, 1);
  EXPECT_EQ(at_2k.numThreads, 512);

  // 2KB + 1 crosses to 2 blocks
  auto at_2k1 = ll128_auto_tune(2049);
  EXPECT_EQ(at_2k1.numBlocks, 2);
  EXPECT_EQ(at_2k1.numThreads, 512);

  // <= 4KB: 2 blocks
  auto at_4k = ll128_auto_tune(4096);
  EXPECT_EQ(at_4k.numBlocks, 2);
  EXPECT_EQ(at_4k.numThreads, 512);

  // 4KB + 1 crosses to 4 blocks
  auto at_4k1 = ll128_auto_tune(4097);
  EXPECT_EQ(at_4k1.numBlocks, 4);
  EXPECT_EQ(at_4k1.numThreads, 512);

  // <= 8KB: 4 blocks
  auto at_8k = ll128_auto_tune(8192);
  EXPECT_EQ(at_8k.numBlocks, 4);
  EXPECT_EQ(at_8k.numThreads, 512);

  // <= 16KB: 8 blocks
  auto at_16k = ll128_auto_tune(16384);
  EXPECT_EQ(at_16k.numBlocks, 8);
  EXPECT_EQ(at_16k.numThreads, 512);

  // <= 32KB: 16 blocks
  auto at_32k = ll128_auto_tune(32768);
  EXPECT_EQ(at_32k.numBlocks, 16);
  EXPECT_EQ(at_32k.numThreads, 512);

  // <= 64KB: 32 blocks
  auto at_64k = ll128_auto_tune(65536);
  EXPECT_EQ(at_64k.numBlocks, 32);
  EXPECT_EQ(at_64k.numThreads, 512);

  // <= 128KB: 64 blocks
  auto at_128k = ll128_auto_tune(128 * 1024);
  EXPECT_EQ(at_128k.numBlocks, 64);
  EXPECT_EQ(at_128k.numThreads, 512);

  // <= 256KB: 128 blocks
  auto at_256k = ll128_auto_tune(256 * 1024);
  EXPECT_EQ(at_256k.numBlocks, 128);
  EXPECT_EQ(at_256k.numThreads, 512);

  // <= 512KB: 256 blocks
  auto at_512k = ll128_auto_tune(512 * 1024);
  EXPECT_EQ(at_512k.numBlocks, 256);
  EXPECT_EQ(at_512k.numThreads, 512);

  // <= 1MB: 512 blocks
  auto at_1m = ll128_auto_tune(1024 * 1024);
  EXPECT_EQ(at_1m.numBlocks, 512);
  EXPECT_EQ(at_1m.numThreads, 512);
}

TEST(Ll128AutoTuneTest, LargeMessage) {
  auto cfg = ll128_auto_tune(2 * 1024 * 1024);
  EXPECT_EQ(cfg.numBlocks, 1024);
  EXPECT_EQ(cfg.numThreads, 512);
}

TEST(Ll128AutoTuneTest, SmallMessage) {
  auto cfg = ll128_auto_tune(16);
  EXPECT_EQ(cfg.numBlocks, 1);
  EXPECT_EQ(cfg.numThreads, 512);
}

// =============================================================================
// ll128_auto_tune_bidirectional
// =============================================================================

TEST(Ll128AutoTuneTest, BidirectionalDoubles) {
  auto uni = ll128_auto_tune(4096);
  auto bidir = ll128_auto_tune_bidirectional(4096);
  EXPECT_EQ(bidir.numBlocks, uni.numBlocks * 2);
  EXPECT_EQ(bidir.numThreads, uni.numThreads);
}

TEST(Ll128AutoTuneTest, BidirectionalCap) {
  // At 1MB uni = 512 blocks, bidir would be 1024 (at the cap)
  auto bidir = ll128_auto_tune_bidirectional(1024 * 1024);
  EXPECT_EQ(bidir.numBlocks, 1024);

  // At 2MB uni = 1024 blocks, bidir should still cap at 1024
  auto bidir_2m = ll128_auto_tune_bidirectional(2 * 1024 * 1024);
  EXPECT_EQ(bidir_2m.numBlocks, 1024);
}

TEST(Ll128AutoTuneTest, BidirectionalMin) {
  // Zero bytes: 0 blocks, but bidir ensures at least 2
  auto bidir = ll128_auto_tune_bidirectional(0);
  // uni returns {0,0} for 0 bytes; 0*2 = 0, but min is 2
  EXPECT_GE(bidir.numBlocks, 2);
}

// =============================================================================
// ll128_auto_tune_alltoallv
// =============================================================================

TEST(Ll128AutoTuneTest, AllToAllvZero) {
  auto cfg = ll128_auto_tune_alltoallv(0, 8);
  EXPECT_EQ(cfg.numBlocks, 0);
  EXPECT_EQ(cfg.numThreads, 0);

  auto cfg2 = ll128_auto_tune_alltoallv(4096, 1);
  EXPECT_EQ(cfg2.numBlocks, 0);
  EXPECT_EQ(cfg2.numThreads, 0);
}

TEST(Ll128AutoTuneTest, AllToAllv8Rank) {
  // 4KB per peer: empirical = 16, min = 2*7 = 14 → 16
  auto cfg = ll128_auto_tune_alltoallv(4096, 8);
  EXPECT_EQ(cfg.numBlocks, 16);
  EXPECT_EQ(cfg.numThreads, 512);

  // 16KB per peer: empirical = 96, min = 14 → 96
  auto cfg2 = ll128_auto_tune_alltoallv(16384, 8);
  EXPECT_EQ(cfg2.numBlocks, 96);

  // 64KB per peer: empirical = 128, min = 14 → 128
  auto cfg3 = ll128_auto_tune_alltoallv(65536, 8);
  EXPECT_EQ(cfg3.numBlocks, 128);
}

TEST(Ll128AutoTuneTest, AllToAllvMinBlocks) {
  // nranks=4: min = 2 * (4-1) = 6
  auto cfg = ll128_auto_tune_alltoallv(16, 4);
  EXPECT_GE(cfg.numBlocks, 6);
}

} // namespace comms::pipes
