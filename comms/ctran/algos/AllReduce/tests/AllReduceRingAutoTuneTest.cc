// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include "comms/ctran/algos/AllReduce/AllReduceRingAutoTune.h"

using ctran::allreduce::ring::getAutoTunedNumBlocks;
using ctran::allreduce::ring::getAutoTunedPipeline;
using ctran::allreduce::ring::getAutoTunedThreadBlockSize;
using ctran::allreduce::ring::GpuArch;
using ctran::allreduce::ring::PipelineParams;

constexpr size_t KB = 1024ULL;
constexpr size_t MB = 1024ULL * 1024;
constexpr size_t GB = 1024ULL * 1024 * 1024;

// ============================================================================
// getAutoTunedNumBlocks — Default (GB200) arch
// ============================================================================

class AutoTuneNumBlocksTest : public ::testing::Test {
 protected:
  static constexpr int kUnlimited = 64;
};

TEST_F(AutoTuneNumBlocksTest, Default_EightRanks_BoundaryValues) {
  const int nRanks = 8;

  EXPECT_EQ(getAutoTunedNumBlocks(1 * KB, nRanks, kUnlimited), 1);
  EXPECT_EQ(getAutoTunedNumBlocks(8 * KB, nRanks, kUnlimited), 1);
  EXPECT_EQ(getAutoTunedNumBlocks(64 * KB, nRanks, kUnlimited), 1);
  EXPECT_EQ(getAutoTunedNumBlocks(256 * KB, nRanks, kUnlimited), 1);
  EXPECT_EQ(getAutoTunedNumBlocks(511 * KB, nRanks, kUnlimited), 1);
  EXPECT_EQ(getAutoTunedNumBlocks(512 * KB, nRanks, kUnlimited), 2);
  EXPECT_EQ(getAutoTunedNumBlocks(1 * MB, nRanks, kUnlimited), 2);
  EXPECT_EQ(getAutoTunedNumBlocks(2 * MB - 8, nRanks, kUnlimited), 2);
  EXPECT_EQ(getAutoTunedNumBlocks(2 * MB, nRanks, kUnlimited), 4);
  EXPECT_EQ(getAutoTunedNumBlocks(3 * MB, nRanks, kUnlimited), 4);
  EXPECT_EQ(getAutoTunedNumBlocks(4 * MB - 8, nRanks, kUnlimited), 4);
  EXPECT_EQ(getAutoTunedNumBlocks(4 * MB, nRanks, kUnlimited), 8);
  EXPECT_EQ(getAutoTunedNumBlocks(8 * MB, nRanks, kUnlimited), 8);
  EXPECT_EQ(getAutoTunedNumBlocks(64 * MB, nRanks, kUnlimited), 8);
  EXPECT_EQ(getAutoTunedNumBlocks(1 * GB, nRanks, kUnlimited), 8);
}

TEST_F(AutoTuneNumBlocksTest, Default_MaxOccupancyClamp) {
  const int nRanks = 8;

  EXPECT_EQ(getAutoTunedNumBlocks(1 * GB, nRanks, 4), 4);
  EXPECT_EQ(getAutoTunedNumBlocks(2 * MB, nRanks, 2), 2);
  EXPECT_EQ(getAutoTunedNumBlocks(1 * MB, nRanks, 1), 1);
  EXPECT_EQ(getAutoTunedNumBlocks(1 * GB, nRanks, 8), 8);
  EXPECT_EQ(getAutoTunedNumBlocks(1 * GB, nRanks, 16), 8);
}

TEST_F(AutoTuneNumBlocksTest, Default_TwoRanks) {
  const int nRanks = 2;

  EXPECT_EQ(getAutoTunedNumBlocks(64 * KB, nRanks, kUnlimited), 1);
  EXPECT_EQ(getAutoTunedNumBlocks(128 * KB, nRanks, kUnlimited), 2);
  EXPECT_EQ(getAutoTunedNumBlocks(512 * KB, nRanks, kUnlimited), 4);
  EXPECT_EQ(getAutoTunedNumBlocks(1 * MB, nRanks, kUnlimited), 8);
}

TEST_F(AutoTuneNumBlocksTest, Default_OneRank) {
  const int nRanks = 1;

  EXPECT_EQ(getAutoTunedNumBlocks(32 * KB, nRanks, kUnlimited), 1);
  EXPECT_EQ(getAutoTunedNumBlocks(64 * KB, nRanks, kUnlimited), 2);
  EXPECT_EQ(getAutoTunedNumBlocks(256 * KB, nRanks, kUnlimited), 4);
  EXPECT_EQ(getAutoTunedNumBlocks(512 * KB, nRanks, kUnlimited), 8);
}

// ============================================================================
// getAutoTunedNumBlocks — Hopper (H100) arch
// ============================================================================

TEST_F(AutoTuneNumBlocksTest, Hopper_EightRanks_BoundaryValues) {
  const int nRanks = 8;
  const auto arch = GpuArch::Hopper;

  EXPECT_EQ(getAutoTunedNumBlocks(1 * KB, nRanks, kUnlimited, arch), 1);
  EXPECT_EQ(getAutoTunedNumBlocks(128 * KB, nRanks, kUnlimited, arch), 1);
  EXPECT_EQ(getAutoTunedNumBlocks(255 * KB, nRanks, kUnlimited, arch), 1);
  EXPECT_EQ(getAutoTunedNumBlocks(256 * KB, nRanks, kUnlimited, arch), 2);
  EXPECT_EQ(getAutoTunedNumBlocks(512 * KB, nRanks, kUnlimited, arch), 2);
  EXPECT_EQ(getAutoTunedNumBlocks(4 * MB, nRanks, kUnlimited, arch), 2);
  EXPECT_EQ(getAutoTunedNumBlocks(8 * MB - 8, nRanks, kUnlimited, arch), 2);
  EXPECT_EQ(getAutoTunedNumBlocks(8 * MB, nRanks, kUnlimited, arch), 4);
  EXPECT_EQ(getAutoTunedNumBlocks(64 * MB, nRanks, kUnlimited, arch), 4);
  EXPECT_EQ(getAutoTunedNumBlocks(128 * MB - 8, nRanks, kUnlimited, arch), 4);
  EXPECT_EQ(getAutoTunedNumBlocks(128 * MB, nRanks, kUnlimited, arch), 8);
  EXPECT_EQ(getAutoTunedNumBlocks(1 * GB, nRanks, kUnlimited, arch), 8);
}

TEST_F(AutoTuneNumBlocksTest, Hopper_MaxOccupancyClamp) {
  const int nRanks = 8;
  const auto arch = GpuArch::Hopper;

  EXPECT_EQ(getAutoTunedNumBlocks(1 * GB, nRanks, 4, arch), 4);
  EXPECT_EQ(getAutoTunedNumBlocks(128 * MB, nRanks, 2, arch), 2);
  EXPECT_EQ(getAutoTunedNumBlocks(8 * MB, nRanks, 1, arch), 1);
}

// ============================================================================
// getAutoTunedThreadBlockSize
// ============================================================================

class AutoTuneThreadBlockSizeTest : public ::testing::Test {
 protected:
  static constexpr int kDefaultThreads = 512;
};

TEST_F(AutoTuneThreadBlockSizeTest, Default_ReturnsDefault) {
  const int nRanks = 8;

  EXPECT_EQ(
      getAutoTunedThreadBlockSize(1 * KB, nRanks, kDefaultThreads), 512);
  EXPECT_EQ(
      getAutoTunedThreadBlockSize(1 * MB, nRanks, kDefaultThreads), 512);
  EXPECT_EQ(
      getAutoTunedThreadBlockSize(1 * GB, nRanks, kDefaultThreads), 512);
  EXPECT_EQ(
      getAutoTunedThreadBlockSize(1 * GB, nRanks, 384), 384);
}

TEST_F(AutoTuneThreadBlockSizeTest, Hopper_EightRanks) {
  const int nRanks = 8;
  const auto arch = GpuArch::Hopper;

  EXPECT_EQ(
      getAutoTunedThreadBlockSize(1 * KB, nRanks, kDefaultThreads, arch), 384);
  EXPECT_EQ(
      getAutoTunedThreadBlockSize(256 * KB, nRanks, kDefaultThreads, arch),
      384);
  EXPECT_EQ(
      getAutoTunedThreadBlockSize(4 * MB, nRanks, kDefaultThreads, arch), 384);
  EXPECT_EQ(
      getAutoTunedThreadBlockSize(
          8 * MB - 8, nRanks, kDefaultThreads, arch),
      384);
  EXPECT_EQ(
      getAutoTunedThreadBlockSize(8 * MB, nRanks, kDefaultThreads, arch), 512);
  EXPECT_EQ(
      getAutoTunedThreadBlockSize(64 * MB, nRanks, kDefaultThreads, arch), 512);
  EXPECT_EQ(
      getAutoTunedThreadBlockSize(1 * GB, nRanks, kDefaultThreads, arch), 512);
}

// ============================================================================
// getAutoTunedPipeline
// ============================================================================

class AutoTunePipelineTest : public ::testing::Test {
 protected:
  static void expectPipeline(
      const PipelineParams& p,
      size_t expectedChunkSize,
      size_t expectedNumChunks) {
    EXPECT_EQ(p.chunkSize, expectedChunkSize)
        << "chunkSize mismatch: got " << p.chunkSize << " expected "
        << expectedChunkSize;
    EXPECT_EQ(p.numChunks, expectedNumChunks)
        << "numChunks mismatch: got " << p.numChunks << " expected "
        << expectedNumChunks;
  }
};

TEST_F(AutoTunePipelineTest, SmallMaxBDP_ChunksReduced) {
  const int nRanks = 8;

  expectPipeline(getAutoTunedPipeline(1 * MB, 256 * KB, nRanks), 256 * KB, 1);
  expectPipeline(getAutoTunedPipeline(1 * MB, 512 * KB, nRanks), 256 * KB, 2);
}

TEST_F(AutoTunePipelineTest, BDPInvariant) {
  const int nRanks = 8;
  const size_t maxBDPs[] = {
      256 * KB, 512 * KB, 1 * MB, 2 * MB, 4 * MB,
      8 * MB, 16 * MB, 32 * MB, 64 * MB, 128 * MB};

  size_t msg = 1 * KB;
  while (msg <= 64 * GB) {
    for (auto maxBDP : maxBDPs) {
      auto p = getAutoTunedPipeline(msg, maxBDP, nRanks);
      EXPECT_LE(p.chunkSize * p.numChunks, maxBDP)
          << "BDP violated: msg=" << msg << " maxBDP=" << maxBDP
          << " chunkSize=" << p.chunkSize << " numChunks=" << p.numChunks;
      EXPECT_GE(p.chunkSize, 256 * KB) << "chunkSize below minimum";
      EXPECT_GE(p.numChunks, 1u) << "numChunks below 1";
    }
    msg *= 2;
  }
}

// ============================================================================
// Combined golden tables: Default arch, 8 ranks, all pow2 sizes 1K-64G
// ============================================================================

struct AutoTuneExpected {
  size_t msgBytes;
  int blocks;
  int threads;
  size_t chunkSize;
  size_t numChunks;
};

TEST(AutoTuneCombinedDefault, MaxBDP16M_8Ranks) {
  const int nRanks = 8;
  const size_t maxBDP = 16 * MB;
  const int maxOcc = 64;
  const int defThreads = 512;

  // clang-format off
  const AutoTuneExpected cases[] = {
      {      1 * KB, 1, 512,   256 * KB, 8},
      {      2 * KB, 1, 512,   256 * KB, 8},
      {      4 * KB, 1, 512,   256 * KB, 8},
      {      8 * KB, 1, 512,   256 * KB, 8},
      {     16 * KB, 1, 512,   256 * KB, 8},
      {     32 * KB, 1, 512,   256 * KB, 8},
      {     64 * KB, 1, 512,   256 * KB, 8},
      {    128 * KB, 1, 512,   256 * KB, 8},
      {    256 * KB, 1, 512,   256 * KB, 8},
      {    512 * KB, 2, 512,   256 * KB, 8},
      {      1 * MB, 2, 512,   256 * KB, 8},
      {      2 * MB, 4, 512,   256 * KB, 8},
      {      4 * MB, 8, 512,   256 * KB, 8},
      {      8 * MB, 8, 512,   256 * KB, 8},
      {     16 * MB, 8, 512,   512 * KB, 8},
      {     32 * MB, 8, 512,     1 * MB, 8},
      {     64 * MB, 8, 512,     2 * MB, 8},
      {    128 * MB, 8, 512,     2 * MB, 8},
      {    256 * MB, 8, 512,     2 * MB, 8},
      {    512 * MB, 8, 512,     2 * MB, 8},
      {      1 * GB, 8, 512,     2 * MB, 8},
      {      2 * GB, 8, 512,     2 * MB, 8},
      {      4 * GB, 8, 512,     2 * MB, 8},
      {      8 * GB, 8, 512,     2 * MB, 8},
      {     16 * GB, 8, 512,     2 * MB, 8},
      {     32 * GB, 8, 512,     2 * MB, 8},
      {     64 * GB, 8, 512,     2 * MB, 8},
  };
  // clang-format on

  for (const auto& c : cases) {
    EXPECT_EQ(getAutoTunedNumBlocks(c.msgBytes, nRanks, maxOcc), c.blocks)
        << "blocks mismatch at msg=" << c.msgBytes;
    EXPECT_EQ(
        getAutoTunedThreadBlockSize(c.msgBytes, nRanks, defThreads), c.threads)
        << "threads mismatch at msg=" << c.msgBytes;
    auto p = getAutoTunedPipeline(c.msgBytes, maxBDP, nRanks);
    EXPECT_EQ(p.chunkSize, c.chunkSize)
        << "chunkSize mismatch at msg=" << c.msgBytes;
    EXPECT_EQ(p.numChunks, c.numChunks)
        << "numChunks mismatch at msg=" << c.msgBytes;
  }
}

TEST(AutoTuneCombinedDefault, MaxBDP32M_8Ranks) {
  const int nRanks = 8;
  const size_t maxBDP = 32 * MB;
  const int maxOcc = 64;
  const int defThreads = 512;

  // clang-format off
  const AutoTuneExpected cases[] = {
      {      1 * KB, 1, 512,   256 * KB, 8},
      {      2 * KB, 1, 512,   256 * KB, 8},
      {      4 * KB, 1, 512,   256 * KB, 8},
      {      8 * KB, 1, 512,   256 * KB, 8},
      {     16 * KB, 1, 512,   256 * KB, 8},
      {     32 * KB, 1, 512,   256 * KB, 8},
      {     64 * KB, 1, 512,   256 * KB, 8},
      {    128 * KB, 1, 512,   256 * KB, 8},
      {    256 * KB, 1, 512,   256 * KB, 8},
      {    512 * KB, 2, 512,   256 * KB, 8},
      {      1 * MB, 2, 512,   256 * KB, 8},
      {      2 * MB, 4, 512,   256 * KB, 8},
      {      4 * MB, 8, 512,   256 * KB, 8},
      {      8 * MB, 8, 512,   256 * KB, 8},
      {     16 * MB, 8, 512,   512 * KB, 8},
      {     32 * MB, 8, 512,     1 * MB, 8},
      {     64 * MB, 8, 512,     2 * MB, 8},
      {    128 * MB, 8, 512,     4 * MB, 8},
      {    256 * MB, 8, 512,     4 * MB, 8},
      {    512 * MB, 8, 512,     4 * MB, 8},
      {      1 * GB, 8, 512,     4 * MB, 8},
      {      2 * GB, 8, 512,     4 * MB, 8},
      {      4 * GB, 8, 512,     4 * MB, 8},
      {      8 * GB, 8, 512,     4 * MB, 8},
      {     16 * GB, 8, 512,     4 * MB, 8},
      {     32 * GB, 8, 512,     4 * MB, 8},
      {     64 * GB, 8, 512,     4 * MB, 8},
  };
  // clang-format on

  for (const auto& c : cases) {
    EXPECT_EQ(getAutoTunedNumBlocks(c.msgBytes, nRanks, maxOcc), c.blocks)
        << "blocks mismatch at msg=" << c.msgBytes;
    EXPECT_EQ(
        getAutoTunedThreadBlockSize(c.msgBytes, nRanks, defThreads), c.threads)
        << "threads mismatch at msg=" << c.msgBytes;
    auto p = getAutoTunedPipeline(c.msgBytes, maxBDP, nRanks);
    EXPECT_EQ(p.chunkSize, c.chunkSize)
        << "chunkSize mismatch at msg=" << c.msgBytes;
    EXPECT_EQ(p.numChunks, c.numChunks)
        << "numChunks mismatch at msg=" << c.msgBytes;
  }
}

TEST(AutoTuneCombinedDefault, MaxBDP64M_8Ranks) {
  const int nRanks = 8;
  const size_t maxBDP = 64 * MB;
  const int maxOcc = 64;
  const int defThreads = 512;

  // clang-format off
  const AutoTuneExpected cases[] = {
      {      1 * KB, 1, 512,   256 * KB, 8},
      {      2 * KB, 1, 512,   256 * KB, 8},
      {      4 * KB, 1, 512,   256 * KB, 8},
      {      8 * KB, 1, 512,   256 * KB, 8},
      {     16 * KB, 1, 512,   256 * KB, 8},
      {     32 * KB, 1, 512,   256 * KB, 8},
      {     64 * KB, 1, 512,   256 * KB, 8},
      {    128 * KB, 1, 512,   256 * KB, 8},
      {    256 * KB, 1, 512,   256 * KB, 8},
      {    512 * KB, 2, 512,   256 * KB, 8},
      {      1 * MB, 2, 512,   256 * KB, 8},
      {      2 * MB, 4, 512,   256 * KB, 8},
      {      4 * MB, 8, 512,   256 * KB, 8},
      {      8 * MB, 8, 512,   256 * KB, 8},
      {     16 * MB, 8, 512,   512 * KB, 8},
      {     32 * MB, 8, 512,     1 * MB, 8},
      {     64 * MB, 8, 512,     2 * MB, 8},
      {    128 * MB, 8, 512,     4 * MB, 8},
      {    256 * MB, 8, 512,     8 * MB, 8},
      {    512 * MB, 8, 512,     8 * MB, 8},
      {      1 * GB, 8, 512,     8 * MB, 8},
      {      2 * GB, 8, 512,     8 * MB, 8},
      {      4 * GB, 8, 512,     8 * MB, 8},
      {      8 * GB, 8, 512,     8 * MB, 8},
      {     16 * GB, 8, 512,     8 * MB, 8},
      {     32 * GB, 8, 512,     8 * MB, 8},
      {     64 * GB, 8, 512,     8 * MB, 8},
  };
  // clang-format on

  for (const auto& c : cases) {
    EXPECT_EQ(getAutoTunedNumBlocks(c.msgBytes, nRanks, maxOcc), c.blocks)
        << "blocks mismatch at msg=" << c.msgBytes;
    EXPECT_EQ(
        getAutoTunedThreadBlockSize(c.msgBytes, nRanks, defThreads), c.threads)
        << "threads mismatch at msg=" << c.msgBytes;
    auto p = getAutoTunedPipeline(c.msgBytes, maxBDP, nRanks);
    EXPECT_EQ(p.chunkSize, c.chunkSize)
        << "chunkSize mismatch at msg=" << c.msgBytes;
    EXPECT_EQ(p.numChunks, c.numChunks)
        << "numChunks mismatch at msg=" << c.msgBytes;
  }
}

TEST(AutoTuneCombinedDefault, MaxBDP128M_8Ranks) {
  const int nRanks = 8;
  const size_t maxBDP = 128 * MB;
  const int maxOcc = 64;
  const int defThreads = 512;

  // clang-format off
  const AutoTuneExpected cases[] = {
      {      1 * KB, 1, 512,   256 * KB, 8},
      {      2 * KB, 1, 512,   256 * KB, 8},
      {      4 * KB, 1, 512,   256 * KB, 8},
      {      8 * KB, 1, 512,   256 * KB, 8},
      {     16 * KB, 1, 512,   256 * KB, 8},
      {     32 * KB, 1, 512,   256 * KB, 8},
      {     64 * KB, 1, 512,   256 * KB, 8},
      {    128 * KB, 1, 512,   256 * KB, 8},
      {    256 * KB, 1, 512,   256 * KB, 8},
      {    512 * KB, 2, 512,   256 * KB, 8},
      {      1 * MB, 2, 512,   256 * KB, 8},
      {      2 * MB, 4, 512,   256 * KB, 8},
      {      4 * MB, 8, 512,   256 * KB, 8},
      {      8 * MB, 8, 512,   256 * KB, 8},
      {     16 * MB, 8, 512,   512 * KB, 8},
      {     32 * MB, 8, 512,     1 * MB, 8},
      {     64 * MB, 8, 512,     2 * MB, 8},
      {    128 * MB, 8, 512,     4 * MB, 8},
      {    256 * MB, 8, 512,     8 * MB, 8},
      {    512 * MB, 8, 512,    16 * MB, 8},
      {      1 * GB, 8, 512,    16 * MB, 8},
      {      2 * GB, 8, 512,    16 * MB, 8},
      {      4 * GB, 8, 512,    16 * MB, 8},
      {      8 * GB, 8, 512,    16 * MB, 8},
      {     16 * GB, 8, 512,    16 * MB, 8},
      {     32 * GB, 8, 512,    16 * MB, 8},
      {     64 * GB, 8, 512,    16 * MB, 8},
  };
  // clang-format on

  for (const auto& c : cases) {
    EXPECT_EQ(getAutoTunedNumBlocks(c.msgBytes, nRanks, maxOcc), c.blocks)
        << "blocks mismatch at msg=" << c.msgBytes;
    EXPECT_EQ(
        getAutoTunedThreadBlockSize(c.msgBytes, nRanks, defThreads), c.threads)
        << "threads mismatch at msg=" << c.msgBytes;
    auto p = getAutoTunedPipeline(c.msgBytes, maxBDP, nRanks);
    EXPECT_EQ(p.chunkSize, c.chunkSize)
        << "chunkSize mismatch at msg=" << c.msgBytes;
    EXPECT_EQ(p.numChunks, c.numChunks)
        << "numChunks mismatch at msg=" << c.msgBytes;
  }
}

// ============================================================================
// Combined golden tables: Hopper (H100) arch, 8 ranks, all pow2 sizes 1K-64G
// ============================================================================

TEST(AutoTuneCombinedHopper, MaxBDP16M_8Ranks) {
  const int nRanks = 8;
  const size_t maxBDP = 16 * MB;
  const int maxOcc = 64;
  const int defThreads = 512;
  const auto arch = GpuArch::Hopper;

  // clang-format off
  const AutoTuneExpected cases[] = {
      {      1 * KB, 1, 384,   256 * KB, 8},
      {      2 * KB, 1, 384,   256 * KB, 8},
      {      4 * KB, 1, 384,   256 * KB, 8},
      {      8 * KB, 1, 384,   256 * KB, 8},
      {     16 * KB, 1, 384,   256 * KB, 8},
      {     32 * KB, 1, 384,   256 * KB, 8},
      {     64 * KB, 1, 384,   256 * KB, 8},
      {    128 * KB, 1, 384,   256 * KB, 8},
      {    256 * KB, 2, 384,   256 * KB, 8},
      {    512 * KB, 2, 384,   256 * KB, 8},
      {      1 * MB, 2, 384,   256 * KB, 8},
      {      2 * MB, 2, 384,   256 * KB, 8},
      {      4 * MB, 2, 384,   256 * KB, 8},
      {      8 * MB, 4, 512,   256 * KB, 8},
      {     16 * MB, 4, 512,   512 * KB, 8},
      {     32 * MB, 4, 512,     1 * MB, 8},
      {     64 * MB, 4, 512,     2 * MB, 8},
      {    128 * MB, 8, 512,     2 * MB, 8},
      {    256 * MB, 8, 512,     2 * MB, 8},
      {    512 * MB, 8, 512,     2 * MB, 8},
      {      1 * GB, 8, 512,     2 * MB, 8},
      {      2 * GB, 8, 512,     2 * MB, 8},
      {      4 * GB, 8, 512,     2 * MB, 8},
      {      8 * GB, 8, 512,     2 * MB, 8},
      {     16 * GB, 8, 512,     2 * MB, 8},
      {     32 * GB, 8, 512,     2 * MB, 8},
      {     64 * GB, 8, 512,     2 * MB, 8},
  };
  // clang-format on

  for (const auto& c : cases) {
    EXPECT_EQ(
        getAutoTunedNumBlocks(c.msgBytes, nRanks, maxOcc, arch), c.blocks)
        << "blocks mismatch at msg=" << c.msgBytes;
    EXPECT_EQ(
        getAutoTunedThreadBlockSize(c.msgBytes, nRanks, defThreads, arch),
        c.threads)
        << "threads mismatch at msg=" << c.msgBytes;
    auto p = getAutoTunedPipeline(c.msgBytes, maxBDP, nRanks);
    EXPECT_EQ(p.chunkSize, c.chunkSize)
        << "chunkSize mismatch at msg=" << c.msgBytes;
    EXPECT_EQ(p.numChunks, c.numChunks)
        << "numChunks mismatch at msg=" << c.msgBytes;
  }
}

TEST(AutoTuneCombinedHopper, MaxBDP64M_8Ranks) {
  const int nRanks = 8;
  const size_t maxBDP = 64 * MB;
  const int maxOcc = 64;
  const int defThreads = 512;
  const auto arch = GpuArch::Hopper;

  // clang-format off
  const AutoTuneExpected cases[] = {
      {      1 * KB, 1, 384,   256 * KB, 8},
      {      2 * KB, 1, 384,   256 * KB, 8},
      {      4 * KB, 1, 384,   256 * KB, 8},
      {      8 * KB, 1, 384,   256 * KB, 8},
      {     16 * KB, 1, 384,   256 * KB, 8},
      {     32 * KB, 1, 384,   256 * KB, 8},
      {     64 * KB, 1, 384,   256 * KB, 8},
      {    128 * KB, 1, 384,   256 * KB, 8},
      {    256 * KB, 2, 384,   256 * KB, 8},
      {    512 * KB, 2, 384,   256 * KB, 8},
      {      1 * MB, 2, 384,   256 * KB, 8},
      {      2 * MB, 2, 384,   256 * KB, 8},
      {      4 * MB, 2, 384,   256 * KB, 8},
      {      8 * MB, 4, 512,   256 * KB, 8},
      {     16 * MB, 4, 512,   512 * KB, 8},
      {     32 * MB, 4, 512,     1 * MB, 8},
      {     64 * MB, 4, 512,     2 * MB, 8},
      {    128 * MB, 8, 512,     4 * MB, 8},
      {    256 * MB, 8, 512,     8 * MB, 8},
      {    512 * MB, 8, 512,     8 * MB, 8},
      {      1 * GB, 8, 512,     8 * MB, 8},
      {      2 * GB, 8, 512,     8 * MB, 8},
      {      4 * GB, 8, 512,     8 * MB, 8},
      {      8 * GB, 8, 512,     8 * MB, 8},
      {     16 * GB, 8, 512,     8 * MB, 8},
      {     32 * GB, 8, 512,     8 * MB, 8},
      {     64 * GB, 8, 512,     8 * MB, 8},
  };
  // clang-format on

  for (const auto& c : cases) {
    EXPECT_EQ(
        getAutoTunedNumBlocks(c.msgBytes, nRanks, maxOcc, arch), c.blocks)
        << "blocks mismatch at msg=" << c.msgBytes;
    EXPECT_EQ(
        getAutoTunedThreadBlockSize(c.msgBytes, nRanks, defThreads, arch),
        c.threads)
        << "threads mismatch at msg=" << c.msgBytes;
    auto p = getAutoTunedPipeline(c.msgBytes, maxBDP, nRanks);
    EXPECT_EQ(p.chunkSize, c.chunkSize)
        << "chunkSize mismatch at msg=" << c.msgBytes;
    EXPECT_EQ(p.numChunks, c.numChunks)
        << "numChunks mismatch at msg=" << c.msgBytes;
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
