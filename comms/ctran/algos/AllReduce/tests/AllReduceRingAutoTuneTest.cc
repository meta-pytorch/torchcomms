// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <algorithm>
#include <bit>
#include <set>
#include <vector>

#include <gtest/gtest.h>

#include "comms/ctran/algos/AllReduce/AllReduceRingAutoTune.h"
#include "comms/utils/cvars/nccl_cvars.h"

using ctran::allreduce::ring::getAutoTunedParams;
using ctran::allreduce::ring::GpuArch;

constexpr size_t KB = 1024ULL;
constexpr size_t MB = 1024ULL * 1024;
constexpr size_t GB = 1024ULL * 1024 * 1024;

// RAII guard for the maxBDP CVAR override. Restores to default on destruction.
class MaxBDPOverride {
 public:
  explicit MaxBDPOverride(size_t maxBDP) {
    NCCL_CTRAN_ALLREDUCE_RING_AUTO_TUNE_MAX_BDP = static_cast<int>(maxBDP);
  }
  ~MaxBDPOverride() {
    NCCL_CTRAN_ALLREDUCE_RING_AUTO_TUNE_MAX_BDP =
        NCCL_CTRAN_ALLREDUCE_RING_AUTO_TUNE_MAX_BDP_DEFAULTCVARVALUE;
  }
};

// RAII guard for TMPBUF_CHUNK_SIZE CVAR.
class ChunkSizeOverride {
 public:
  explicit ChunkSizeOverride(int v) {
    NCCL_CTRAN_ALLREDUCE_RING_TMPBUF_CHUNK_SIZE = v;
  }
  ~ChunkSizeOverride() {
    NCCL_CTRAN_ALLREDUCE_RING_TMPBUF_CHUNK_SIZE =
        NCCL_CTRAN_ALLREDUCE_RING_TMPBUF_CHUNK_SIZE_DEFAULTCVARVALUE;
  }
};

// RAII guard for TMPBUF_NUM_CHUNKS CVAR.
class NumChunksOverride {
 public:
  explicit NumChunksOverride(int v) {
    NCCL_CTRAN_ALLREDUCE_RING_TMPBUF_NUM_CHUNKS = v;
  }
  ~NumChunksOverride() {
    NCCL_CTRAN_ALLREDUCE_RING_TMPBUF_NUM_CHUNKS =
        NCCL_CTRAN_ALLREDUCE_RING_TMPBUF_NUM_CHUNKS_DEFAULTCVARVALUE;
  }
};

// RAII guard for MAX_NUM_THREAD_BLOCKS CVAR.
class NumBlocksOverride {
 public:
  explicit NumBlocksOverride(int v) {
    NCCL_CTRAN_ALLREDUCE_RING_MAX_NUM_THREAD_BLOCKS = v;
  }
  ~NumBlocksOverride() {
    NCCL_CTRAN_ALLREDUCE_RING_MAX_NUM_THREAD_BLOCKS =
        NCCL_CTRAN_ALLREDUCE_RING_MAX_NUM_THREAD_BLOCKS_DEFAULTCVARVALUE;
  }
};

// RAII guard for THREAD_BLOCK_SIZE CVAR.
class BlockSizeOverride {
 public:
  explicit BlockSizeOverride(int v) {
    NCCL_CTRAN_ALLREDUCE_RING_THREAD_BLOCK_SIZE = v;
  }
  ~BlockSizeOverride() {
    NCCL_CTRAN_ALLREDUCE_RING_THREAD_BLOCK_SIZE =
        NCCL_CTRAN_ALLREDUCE_RING_THREAD_BLOCK_SIZE_DEFAULTCVARVALUE;
  }
};

// ============================================================================
// getAutoTunedParams golden tables: different arch / BDP & nranks.
// ============================================================================

struct AutoTuneExpected {
  size_t msgBytes;
  int blocks;
  int threads;
  size_t chunkSize;
  size_t numChunks;
};

// ============================================================================
// getAutoTunedParams golden tables: Default arch, 8 ranks, pow2 sizes 1K-64G
// ============================================================================

template <size_t N>
void verifyAutoTune(
    const AutoTuneExpected (&cases)[N],
    int nRanks,
    int maxOccNumBlocks,
    int defThreads,
    GpuArch arch = GpuArch::Default) {
  for (const auto& c : cases) {
    auto at = getAutoTunedParams(
        c.msgBytes, nRanks, maxOccNumBlocks, defThreads, 1, arch);
    EXPECT_EQ(at.block.numBlocks, c.blocks)
        << "blocks mismatch at msg=" << c.msgBytes;
    EXPECT_EQ(at.block.blockSize, c.threads)
        << "threads mismatch at msg=" << c.msgBytes;
    EXPECT_EQ(at.pipeline.chunkSize, c.chunkSize)
        << "chunkSize mismatch at msg=" << c.msgBytes;
    EXPECT_EQ(at.pipeline.numChunks, c.numChunks)
        << "numChunks mismatch at msg=" << c.msgBytes;
  }
}

TEST(AutoTuneCombinedDefault, MaxBDP16M_8Ranks) {
  MaxBDPOverride o(16 * MB);
  const int nRanks = 8;
  const int maxOccNumBlocks = 64;
  const int defThreads = 512;

  // clang-format off
  const AutoTuneExpected cases[] = {
      {      1 * KB, 1, 512,         128, 8},
      {      2 * KB, 1, 512,         256, 8},
      {      4 * KB, 1, 512,         512, 8},
      {      8 * KB, 1, 512,      1 * KB, 8},
      {     16 * KB, 1, 512,      2 * KB, 8},
      {     32 * KB, 1, 512,      4 * KB, 8},
      {     64 * KB, 2, 512,      8 * KB, 8},
      {    128 * KB, 2, 512,     16 * KB, 8},
      {    256 * KB, 2, 512,     16 * KB, 16},
      {    512 * KB, 4, 512,     32 * KB, 16},
      {      1 * MB, 8, 512,     64 * KB, 16},
      {      2 * MB, 8, 512,    128 * KB, 16},
      {      4 * MB, 8, 512,    256 * KB, 16},
      {      8 * MB, 8, 512,    256 * KB, 32},
      {     16 * MB, 8, 512,    512 * KB, 32},
      {     32 * MB, 8, 512,    512 * KB, 32},
      {     64 * MB, 8, 512,    512 * KB, 32},
      {    128 * MB, 8, 512,      1 * MB, 16},
      {    256 * MB, 8, 512,      2 * MB, 8},
      {    512 * MB, 8, 512,      2 * MB, 8},
      {      1 * GB, 8, 512,      2 * MB, 8},
      {      2 * GB, 8, 512,      2 * MB, 8},
      {      4 * GB, 8, 512,      2 * MB, 8},
      {      8 * GB, 8, 512,      2 * MB, 8},
      {     16 * GB, 8, 512,      2 * MB, 8},
      {     32 * GB, 8, 512,      2 * MB, 8},
      {     64 * GB, 8, 512,      2 * MB, 8},
  };
  // clang-format on

  verifyAutoTune(cases, nRanks, maxOccNumBlocks, defThreads);
}

TEST(AutoTuneCombinedDefault, MaxBDP32M_8Ranks) {
  MaxBDPOverride o(32 * MB);
  const int nRanks = 8;
  const int maxOccNumBlocks = 64;
  const int defThreads = 512;

  // clang-format off
  const AutoTuneExpected cases[] = {
      {      1 * KB, 1, 512,         128, 8},
      {      2 * KB, 1, 512,         256, 8},
      {      4 * KB, 1, 512,         512, 8},
      {      8 * KB, 1, 512,      1 * KB, 8},
      {     16 * KB, 1, 512,      2 * KB, 8},
      {     32 * KB, 1, 512,      4 * KB, 8},
      {     64 * KB, 2, 512,      8 * KB, 8},
      {    128 * KB, 2, 512,     16 * KB, 8},
      {    256 * KB, 2, 512,     16 * KB, 16},
      {    512 * KB, 4, 512,     32 * KB, 16},
      {      1 * MB, 8, 512,     64 * KB, 16},
      {      2 * MB, 8, 512,    128 * KB, 16},
      {      4 * MB, 8, 512,    256 * KB, 16},
      {      8 * MB, 8, 512,    256 * KB, 32},
      {     16 * MB, 8, 512,    512 * KB, 32},
      {     32 * MB, 8, 512,      1 * MB, 32},
      {     64 * MB, 8, 512,      1 * MB, 32},
      {    128 * MB, 8, 512,      2 * MB, 16},
      {    256 * MB, 8, 512,      4 * MB, 8},
      {    512 * MB, 8, 512,      4 * MB, 8},
      {      1 * GB, 8, 512,      4 * MB, 8},
      {      2 * GB, 8, 512,      4 * MB, 8},
      {      4 * GB, 8, 512,      4 * MB, 8},
      {      8 * GB, 8, 512,      4 * MB, 8},
      {     16 * GB, 8, 512,      4 * MB, 8},
      {     32 * GB, 8, 512,      4 * MB, 8},
      {     64 * GB, 8, 512,      4 * MB, 8},
  };
  // clang-format on

  verifyAutoTune(cases, nRanks, maxOccNumBlocks, defThreads);
}

TEST(AutoTuneCombinedDefault, MaxBDP64M_8Ranks) {
  MaxBDPOverride o(64 * MB);
  const int nRanks = 8;
  const int maxOccNumBlocks = 64;
  const int defThreads = 512;

  // clang-format off
  const AutoTuneExpected cases[] = {
      {      1 * KB, 1, 512,         128, 8},
      {      2 * KB, 1, 512,         256, 8},
      {      4 * KB, 1, 512,         512, 8},
      {      8 * KB, 1, 512,      1 * KB, 8},
      {     16 * KB, 1, 512,      2 * KB, 8},
      {     32 * KB, 1, 512,      4 * KB, 8},
      {     64 * KB, 2, 512,      8 * KB, 8},
      {    128 * KB, 2, 512,     16 * KB, 8},
      {    256 * KB, 2, 512,     16 * KB, 16},
      {    512 * KB, 4, 512,     32 * KB, 16},
      {      1 * MB, 8, 512,     64 * KB, 16},
      {      2 * MB, 8, 512,    128 * KB, 16},
      {      4 * MB, 8, 512,    256 * KB, 16},
      {      8 * MB, 8, 512,    256 * KB, 32},
      {     16 * MB, 8, 512,    512 * KB, 32},
      {     32 * MB, 8, 512,      1 * MB, 32},
      {     64 * MB, 8, 512,      2 * MB, 32},
      {    128 * MB, 8, 512,      4 * MB, 16},
      {    256 * MB, 8, 512,      8 * MB, 8},
      {    512 * MB, 8, 512,      8 * MB, 8},
      {      1 * GB, 8, 512,      8 * MB, 8},
      {      2 * GB, 8, 512,      8 * MB, 8},
      {      4 * GB, 8, 512,      8 * MB, 8},
      {      8 * GB, 8, 512,      8 * MB, 8},
      {     16 * GB, 8, 512,      8 * MB, 8},
      {     32 * GB, 8, 512,      8 * MB, 8},
      {     64 * GB, 8, 512,      8 * MB, 8},
  };
  // clang-format on

  verifyAutoTune(cases, nRanks, maxOccNumBlocks, defThreads);
}

TEST(AutoTuneCombinedDefault, MaxBDP128M_8Ranks) {
  MaxBDPOverride o(128 * MB);
  const int nRanks = 8;
  const int maxOccNumBlocks = 64;
  const int defThreads = 512;

  // clang-format off
  const AutoTuneExpected cases[] = {
      {      1 * KB, 1, 512,         128, 8},
      {      2 * KB, 1, 512,         256, 8},
      {      4 * KB, 1, 512,         512, 8},
      {      8 * KB, 1, 512,      1 * KB, 8},
      {     16 * KB, 1, 512,      2 * KB, 8},
      {     32 * KB, 1, 512,      4 * KB, 8},
      {     64 * KB, 2, 512,      8 * KB, 8},
      {    128 * KB, 2, 512,     16 * KB, 8},
      {    256 * KB, 2, 512,     16 * KB, 16},
      {    512 * KB, 4, 512,     32 * KB, 16},
      {      1 * MB, 8, 512,     64 * KB, 16},
      {      2 * MB, 8, 512,    128 * KB, 16},
      {      4 * MB, 8, 512,    256 * KB, 16},
      {      8 * MB, 8, 512,    256 * KB, 32},
      {     16 * MB, 8, 512,    512 * KB, 32},
      {     32 * MB, 8, 512,      1 * MB, 32},
      {     64 * MB, 8, 512,      2 * MB, 32},
      {    128 * MB, 8, 512,      8 * MB, 16},
      {    256 * MB, 8, 512,     16 * MB, 8},
      {    512 * MB, 8, 512,     16 * MB, 8},
      {      1 * GB, 8, 512,     16 * MB, 8},
      {      2 * GB, 8, 512,     16 * MB, 8},
      {      4 * GB, 8, 512,     16 * MB, 8},
      {      8 * GB, 8, 512,     16 * MB, 8},
      {     16 * GB, 8, 512,     16 * MB, 8},
      {     32 * GB, 8, 512,     16 * MB, 8},
      {     64 * GB, 8, 512,     16 * MB, 8},
  };
  // clang-format on

  verifyAutoTune(cases, nRanks, maxOccNumBlocks, defThreads);
}

// ============================================================================
// getAutoTunedParams golden tables: Hopper (H100) arch, 8 ranks, pow2 1K-64G
// ============================================================================

TEST(AutoTuneCombinedHopper, MaxBDP16M_8Ranks) {
  MaxBDPOverride o(16 * MB);
  const int nRanks = 8;
  const int maxOccNumBlocks = 64;
  const int defThreads = 512;
  const auto arch = GpuArch::Hopper;

  // clang-format off
  const AutoTuneExpected cases[] = {
      {      1 * KB, 1, 384,         128, 8},
      {      2 * KB, 1, 384,         256, 8},
      {      4 * KB, 1, 384,         512, 8},
      {      8 * KB, 1, 384,      1 * KB, 8},
      {     16 * KB, 1, 384,      2 * KB, 8},
      {     32 * KB, 1, 384,      4 * KB, 8},
      {     64 * KB, 1, 384,      8 * KB, 8},
      {    128 * KB, 1, 512,     16 * KB, 8},
      {    256 * KB, 1, 512,     16 * KB, 16},
      {    512 * KB, 1, 512,     32 * KB, 16},
      {      1 * MB, 1, 512,     64 * KB, 16},
      {      2 * MB, 2, 512,    128 * KB, 16},
      {      4 * MB, 2, 512,    256 * KB, 16},
      {      8 * MB, 2, 512,    256 * KB, 32},
      {     16 * MB, 4, 512,    512 * KB, 32},
      {     32 * MB, 4, 512,      1 * MB, 16},
      {     64 * MB, 4, 512,      2 * MB, 8},
      {    128 * MB, 4, 512,      2 * MB, 8},
      {    256 * MB, 4, 512,      2 * MB, 8},
      {    512 * MB, 4, 512,      2 * MB, 8},
      {      1 * GB, 4, 512,      2 * MB, 8},
      {      2 * GB, 4, 512,      2 * MB, 8},
      {      4 * GB, 4, 512,      2 * MB, 8},
      {      8 * GB, 4, 512,      2 * MB, 8},
      {     16 * GB, 4, 512,      2 * MB, 8},
      {     32 * GB, 4, 512,      2 * MB, 8},
      {     64 * GB, 4, 512,      2 * MB, 8},
  };
  // clang-format on

  verifyAutoTune(cases, nRanks, maxOccNumBlocks, defThreads, arch);
}

TEST(AutoTuneCombinedHopper, MaxBDP32M_8Ranks) {
  MaxBDPOverride o(32 * MB);
  const int nRanks = 8;
  const int maxOccNumBlocks = 64;
  const int defThreads = 512;
  const auto arch = GpuArch::Hopper;

  // clang-format off
  const AutoTuneExpected cases[] = {
      {      1 * KB, 1, 384,         128, 8},
      {      2 * KB, 1, 384,         256, 8},
      {      4 * KB, 1, 384,         512, 8},
      {      8 * KB, 1, 384,      1 * KB, 8},
      {     16 * KB, 1, 384,      2 * KB, 8},
      {     32 * KB, 1, 384,      4 * KB, 8},
      {     64 * KB, 1, 384,      8 * KB, 8},
      {    128 * KB, 1, 512,     16 * KB, 8},
      {    256 * KB, 1, 512,     16 * KB, 16},
      {    512 * KB, 1, 512,     32 * KB, 16},
      {      1 * MB, 1, 512,     64 * KB, 16},
      {      2 * MB, 2, 512,    128 * KB, 16},
      {      4 * MB, 2, 512,    256 * KB, 16},
      {      8 * MB, 2, 512,    256 * KB, 32},
      {     16 * MB, 4, 512,    512 * KB, 32},
      {     32 * MB, 4, 512,      2 * MB, 16},
      {     64 * MB, 4, 512,      4 * MB, 8},
      {    128 * MB, 4, 512,      4 * MB, 8},
      {    256 * MB, 4, 512,      4 * MB, 8},
      {    512 * MB, 4, 512,      4 * MB, 8},
      {      1 * GB, 4, 512,      4 * MB, 8},
      {      2 * GB, 4, 512,      4 * MB, 8},
      {      4 * GB, 4, 512,      4 * MB, 8},
      {      8 * GB, 4, 512,      4 * MB, 8},
      {     16 * GB, 4, 512,      4 * MB, 8},
      {     32 * GB, 4, 512,      4 * MB, 8},
      {     64 * GB, 4, 512,      4 * MB, 8},
  };
  // clang-format on

  verifyAutoTune(cases, nRanks, maxOccNumBlocks, defThreads, arch);
}

TEST(AutoTuneCombinedHopper, MaxBDP64M_8Ranks) {
  MaxBDPOverride o(64 * MB);
  const int nRanks = 8;
  const int maxOccNumBlocks = 64;
  const int defThreads = 512;
  const auto arch = GpuArch::Hopper;

  // clang-format off
  const AutoTuneExpected cases[] = {
      {      1 * KB, 1, 384,         128, 8},
      {      2 * KB, 1, 384,         256, 8},
      {      4 * KB, 1, 384,         512, 8},
      {      8 * KB, 1, 384,      1 * KB, 8},
      {     16 * KB, 1, 384,      2 * KB, 8},
      {     32 * KB, 1, 384,      4 * KB, 8},
      {     64 * KB, 1, 384,      8 * KB, 8},
      {    128 * KB, 1, 512,     16 * KB, 8},
      {    256 * KB, 1, 512,     16 * KB, 16},
      {    512 * KB, 1, 512,     32 * KB, 16},
      {      1 * MB, 1, 512,     64 * KB, 16},
      {      2 * MB, 2, 512,    128 * KB, 16},
      {      4 * MB, 2, 512,    256 * KB, 16},
      {      8 * MB, 2, 512,    256 * KB, 32},
      {     16 * MB, 4, 512,    512 * KB, 32},
      {     32 * MB, 4, 512,      2 * MB, 16},
      {     64 * MB, 4, 512,      8 * MB, 8},
      {    128 * MB, 4, 512,      8 * MB, 8},
      {    256 * MB, 4, 512,      8 * MB, 8},
      {    512 * MB, 4, 512,      8 * MB, 8},
      {      1 * GB, 4, 512,      8 * MB, 8},
      {      2 * GB, 4, 512,      8 * MB, 8},
      {      4 * GB, 4, 512,      8 * MB, 8},
      {      8 * GB, 4, 512,      8 * MB, 8},
      {     16 * GB, 4, 512,      8 * MB, 8},
      {     32 * GB, 4, 512,      8 * MB, 8},
      {     64 * GB, 4, 512,      8 * MB, 8},
  };
  // clang-format on

  verifyAutoTune(cases, nRanks, maxOccNumBlocks, defThreads, arch);
}

TEST(AutoTuneCombinedHopper, MaxBDP128M_8Ranks) {
  MaxBDPOverride o(128 * MB);
  const int nRanks = 8;
  const int maxOccNumBlocks = 64;
  const int defThreads = 512;
  const auto arch = GpuArch::Hopper;

  // clang-format off
  const AutoTuneExpected cases[] = {
      {      1 * KB, 1, 384,         128, 8},
      {      2 * KB, 1, 384,         256, 8},
      {      4 * KB, 1, 384,         512, 8},
      {      8 * KB, 1, 384,      1 * KB, 8},
      {     16 * KB, 1, 384,      2 * KB, 8},
      {     32 * KB, 1, 384,      4 * KB, 8},
      {     64 * KB, 1, 384,      8 * KB, 8},
      {    128 * KB, 1, 512,     16 * KB, 8},
      {    256 * KB, 1, 512,     16 * KB, 16},
      {    512 * KB, 1, 512,     32 * KB, 16},
      {      1 * MB, 1, 512,     64 * KB, 16},
      {      2 * MB, 2, 512,    128 * KB, 16},
      {      4 * MB, 2, 512,    256 * KB, 16},
      {      8 * MB, 2, 512,    256 * KB, 32},
      {     16 * MB, 4, 512,    512 * KB, 32},
      {     32 * MB, 4, 512,      2 * MB, 16},
      {     64 * MB, 4, 512,      8 * MB, 8},
      {    128 * MB, 4, 512,     16 * MB, 8},
      {    256 * MB, 4, 512,     16 * MB, 8},
      {    512 * MB, 4, 512,     16 * MB, 8},
      {      1 * GB, 4, 512,     16 * MB, 8},
      {      2 * GB, 4, 512,     16 * MB, 8},
      {      4 * GB, 4, 512,     16 * MB, 8},
      {      8 * GB, 4, 512,     16 * MB, 8},
      {     16 * GB, 4, 512,     16 * MB, 8},
      {     32 * GB, 4, 512,     16 * MB, 8},
      {     64 * GB, 4, 512,     16 * MB, 8},
  };
  // clang-format on

  verifyAutoTune(cases, nRanks, maxOccNumBlocks, defThreads, arch);
}

// ============================================================================
// Rank-sweep tables: Default arch, arch-default BDP, ranks {8,16,32,64}
// ============================================================================

TEST(AutoTuneDefaultRankSweep, DefaultBDP_8Ranks) {
  const int nRanks = 8;
  const int maxOccNumBlocks = 64;
  const int defThreads = 512;

  // clang-format off
  const AutoTuneExpected cases[] = {
      {      1 * KB, 1, 512,         128, 8},
      {      2 * KB, 1, 512,         256, 8},
      {      4 * KB, 1, 512,         512, 8},
      {      8 * KB, 1, 512,      1 * KB, 8},
      {     16 * KB, 1, 512,      2 * KB, 8},
      {     32 * KB, 1, 512,      4 * KB, 8},
      {     64 * KB, 2, 512,      8 * KB, 8},
      {    128 * KB, 2, 512,     16 * KB, 8},
      {    256 * KB, 2, 512,     16 * KB, 16},
      {    512 * KB, 4, 512,     32 * KB, 16},
      {      1 * MB, 8, 512,     64 * KB, 16},
      {      2 * MB, 8, 512,    128 * KB, 16},
      {      4 * MB, 8, 512,    256 * KB, 16},
      {      8 * MB, 8, 512,    256 * KB, 32},
      {     16 * MB, 8, 512,    512 * KB, 32},
      {     32 * MB, 8, 512,      1 * MB, 32},
      {     64 * MB, 8, 512,      2 * MB, 32},
      {    128 * MB, 8, 512,      8 * MB, 16},
      {    256 * MB, 8, 512,     16 * MB, 8},
      {    512 * MB, 8, 512,     16 * MB, 8},
      {      1 * GB, 8, 512,     16 * MB, 8},
      {      2 * GB, 8, 512,     16 * MB, 8},
      {      4 * GB, 8, 512,     16 * MB, 8},
      {      8 * GB, 8, 512,     16 * MB, 8},
      {     16 * GB, 8, 512,     16 * MB, 8},
      {     32 * GB, 8, 512,     16 * MB, 8},
      {     64 * GB, 8, 512,     16 * MB, 8},
  };
  // clang-format on

  verifyAutoTune(cases, nRanks, maxOccNumBlocks, defThreads);
}

TEST(AutoTuneDefaultRankSweep, DefaultBDP_16Ranks) {
  const int nRanks = 16;
  const int maxOccNumBlocks = 64;
  const int defThreads = 512;

  // clang-format off
  const AutoTuneExpected cases[] = {
      {      1 * KB, 1, 512,          64, 16},
      {      2 * KB, 1, 512,         128, 16},
      {      4 * KB, 1, 512,         256, 16},
      {      8 * KB, 1, 512,         512, 16},
      {     16 * KB, 1, 512,      1 * KB, 16},
      {     32 * KB, 1, 512,      2 * KB, 16},
      {     64 * KB, 1, 512,      4 * KB, 16},
      {    128 * KB, 2, 512,      8 * KB, 16},
      {    256 * KB, 2, 512,     16 * KB, 16},
      {    512 * KB, 2, 512,     16 * KB, 32},
      {      1 * MB, 4, 512,     32 * KB, 32},
      {      2 * MB, 8, 512,     64 * KB, 32},
      {      4 * MB, 8, 512,    128 * KB, 32},
      {      8 * MB, 8, 512,    256 * KB, 32},
      {     16 * MB, 8, 512,    256 * KB, 64},
      {     32 * MB, 8, 512,    512 * KB, 64},
      {     64 * MB, 8, 512,      1 * MB, 64},
      {    128 * MB, 8, 512,      2 * MB, 64},
      {    256 * MB, 8, 512,      4 * MB, 32},
      {    512 * MB, 8, 512,      8 * MB, 16},
      {      1 * GB, 8, 512,      8 * MB, 16},
      {      2 * GB, 8, 512,      8 * MB, 16},
      {      4 * GB, 8, 512,      8 * MB, 16},
      {      8 * GB, 8, 512,      8 * MB, 16},
      {     16 * GB, 8, 512,      8 * MB, 16},
      {     32 * GB, 8, 512,      8 * MB, 16},
      {     64 * GB, 8, 512,      8 * MB, 16},
  };
  // clang-format on

  verifyAutoTune(cases, nRanks, maxOccNumBlocks, defThreads);
}

TEST(AutoTuneDefaultRankSweep, DefaultBDP_32Ranks) {
  const int nRanks = 32;
  const int maxOccNumBlocks = 64;
  const int defThreads = 512;

  // clang-format off
  const AutoTuneExpected cases[] = {
      {      1 * KB, 1, 512,          32, 32},
      {      2 * KB, 1, 512,          64, 32},
      {      4 * KB, 1, 512,         128, 32},
      {      8 * KB, 1, 512,         256, 32},
      {     16 * KB, 1, 512,         512, 32},
      {     32 * KB, 1, 512,      1 * KB, 32},
      {     64 * KB, 1, 512,      2 * KB, 32},
      {    128 * KB, 1, 512,      4 * KB, 32},
      {    256 * KB, 2, 512,      8 * KB, 32},
      {    512 * KB, 2, 512,     16 * KB, 32},
      {      1 * MB, 2, 512,     16 * KB, 64},
      {      2 * MB, 4, 512,     32 * KB, 64},
      {      4 * MB, 8, 512,     64 * KB, 64},
      {      8 * MB, 8, 512,    128 * KB, 64},
      {     16 * MB, 8, 512,    256 * KB, 64},
      {     32 * MB, 8, 512,    256 * KB, 128},
      {     64 * MB, 8, 512,    512 * KB, 128},
      {    128 * MB, 8, 512,      1 * MB, 128},
      {    256 * MB, 8, 512,      1 * MB, 128},
      {    512 * MB, 8, 512,      2 * MB, 64},
      {      1 * GB, 8, 512,      4 * MB, 32},
      {      2 * GB, 8, 512,      4 * MB, 32},
      {      4 * GB, 8, 512,      4 * MB, 32},
      {      8 * GB, 8, 512,      4 * MB, 32},
      {     16 * GB, 8, 512,      4 * MB, 32},
      {     32 * GB, 8, 512,      4 * MB, 32},
      {     64 * GB, 8, 512,      4 * MB, 32},
  };
  // clang-format on

  verifyAutoTune(cases, nRanks, maxOccNumBlocks, defThreads);
}

TEST(AutoTuneDefaultRankSweep, DefaultBDP_64Ranks) {
  const int nRanks = 64;
  const int maxOccNumBlocks = 64;
  const int defThreads = 512;

  // clang-format off
  const AutoTuneExpected cases[] = {
      {      1 * KB, 1, 512,          16, 64},
      {      2 * KB, 1, 512,          32, 64},
      {      4 * KB, 1, 512,          64, 64},
      {      8 * KB, 1, 512,         128, 64},
      {     16 * KB, 1, 512,         256, 64},
      {     32 * KB, 1, 512,         512, 64},
      {     64 * KB, 1, 512,      1 * KB, 64},
      {    128 * KB, 1, 512,      2 * KB, 64},
      {    256 * KB, 1, 512,      4 * KB, 64},
      {    512 * KB, 2, 512,      8 * KB, 64},
      {      1 * MB, 2, 512,     16 * KB, 64},
      {      2 * MB, 2, 512,     16 * KB, 128},
      {      4 * MB, 4, 512,     32 * KB, 128},
      {      8 * MB, 8, 512,     64 * KB, 128},
      {     16 * MB, 8, 512,    128 * KB, 128},
      {     32 * MB, 8, 512,    256 * KB, 128},
      {     64 * MB, 8, 512,    256 * KB, 256},
      {    128 * MB, 8, 512,    512 * KB, 256},
      {    256 * MB, 8, 512,    512 * KB, 256},
      {    512 * MB, 8, 512,    512 * KB, 256},
      {      1 * GB, 8, 512,      1 * MB, 128},
      {      2 * GB, 8, 512,      2 * MB, 64},
      {      4 * GB, 8, 512,      2 * MB, 64},
      {      8 * GB, 8, 512,      2 * MB, 64},
      {     16 * GB, 8, 512,      2 * MB, 64},
      {     32 * GB, 8, 512,      2 * MB, 64},
      {     64 * GB, 8, 512,      2 * MB, 64},
  };
  // clang-format on

  verifyAutoTune(cases, nRanks, maxOccNumBlocks, defThreads);
}

// ============================================================================
// Rank-sweep tables: Hopper arch, arch-default BDP, ranks {8,16,32,64}
// ============================================================================

TEST(AutoTuneHopperRankSweep, DefaultBDP_8Ranks) {
  const int nRanks = 8;
  const int maxOccNumBlocks = 64;
  const int defThreads = 512;
  const auto arch = GpuArch::Hopper;

  // clang-format off
  const AutoTuneExpected cases[] = {
      {      1 * KB, 1, 384,         128, 8},
      {      2 * KB, 1, 384,         256, 8},
      {      4 * KB, 1, 384,         512, 8},
      {      8 * KB, 1, 384,      1 * KB, 8},
      {     16 * KB, 1, 384,      2 * KB, 8},
      {     32 * KB, 1, 384,      4 * KB, 8},
      {     64 * KB, 1, 384,      8 * KB, 8},
      {    128 * KB, 1, 512,     16 * KB, 8},
      {    256 * KB, 1, 512,     16 * KB, 16},
      {    512 * KB, 1, 512,     32 * KB, 16},
      {      1 * MB, 1, 512,     64 * KB, 16},
      {      2 * MB, 2, 512,    128 * KB, 16},
      {      4 * MB, 2, 512,    256 * KB, 16},
      {      8 * MB, 2, 512,    256 * KB, 32},
      {     16 * MB, 4, 512,    512 * KB, 32},
      {     32 * MB, 4, 512,      2 * MB, 16},
      {     64 * MB, 4, 512,      4 * MB, 8},
      {    128 * MB, 4, 512,      4 * MB, 8},
      {    256 * MB, 4, 512,      4 * MB, 8},
      {    512 * MB, 4, 512,      4 * MB, 8},
      {      1 * GB, 4, 512,      4 * MB, 8},
      {      2 * GB, 4, 512,      4 * MB, 8},
      {      4 * GB, 4, 512,      4 * MB, 8},
      {      8 * GB, 4, 512,      4 * MB, 8},
      {     16 * GB, 4, 512,      4 * MB, 8},
      {     32 * GB, 4, 512,      4 * MB, 8},
      {     64 * GB, 4, 512,      4 * MB, 8},
  };
  // clang-format on

  verifyAutoTune(cases, nRanks, maxOccNumBlocks, defThreads, arch);
}

TEST(AutoTuneHopperRankSweep, DefaultBDP_16Ranks) {
  const int nRanks = 16;
  const int maxOccNumBlocks = 64;
  const int defThreads = 512;
  const auto arch = GpuArch::Hopper;

  // clang-format off
  const AutoTuneExpected cases[] = {
      {      1 * KB, 1, 384,          64, 16},
      {      2 * KB, 1, 384,         128, 16},
      {      4 * KB, 1, 384,         256, 16},
      {      8 * KB, 1, 384,         512, 16},
      {     16 * KB, 1, 384,      1 * KB, 16},
      {     32 * KB, 1, 384,      2 * KB, 16},
      {     64 * KB, 1, 384,      4 * KB, 16},
      {    128 * KB, 1, 384,      8 * KB, 16},
      {    256 * KB, 1, 512,     16 * KB, 16},
      {    512 * KB, 1, 512,     16 * KB, 32},
      {      1 * MB, 1, 512,     32 * KB, 32},
      {      2 * MB, 1, 512,     64 * KB, 32},
      {      4 * MB, 2, 512,    128 * KB, 32},
      {      8 * MB, 2, 512,    256 * KB, 32},
      {     16 * MB, 2, 512,    256 * KB, 64},
      {     32 * MB, 4, 512,    512 * KB, 64},
      {     64 * MB, 4, 512,      1 * MB, 32},
      {    128 * MB, 4, 512,      2 * MB, 16},
      {    256 * MB, 4, 512,      2 * MB, 16},
      {    512 * MB, 4, 512,      2 * MB, 16},
      {      1 * GB, 4, 512,      2 * MB, 16},
      {      2 * GB, 4, 512,      2 * MB, 16},
      {      4 * GB, 4, 512,      2 * MB, 16},
      {      8 * GB, 4, 512,      2 * MB, 16},
      {     16 * GB, 4, 512,      2 * MB, 16},
      {     32 * GB, 4, 512,      2 * MB, 16},
      {     64 * GB, 4, 512,      2 * MB, 16},
  };
  // clang-format on

  verifyAutoTune(cases, nRanks, maxOccNumBlocks, defThreads, arch);
}

TEST(AutoTuneHopperRankSweep, DefaultBDP_32Ranks) {
  const int nRanks = 32;
  const int maxOccNumBlocks = 64;
  const int defThreads = 512;
  const auto arch = GpuArch::Hopper;

  // clang-format off
  const AutoTuneExpected cases[] = {
      {      1 * KB, 1, 384,          32, 32},
      {      2 * KB, 1, 384,          64, 32},
      {      4 * KB, 1, 384,         128, 32},
      {      8 * KB, 1, 384,         256, 32},
      {     16 * KB, 1, 384,         512, 32},
      {     32 * KB, 1, 384,      1 * KB, 32},
      {     64 * KB, 1, 384,      2 * KB, 32},
      {    128 * KB, 1, 384,      4 * KB, 32},
      {    256 * KB, 1, 384,      8 * KB, 32},
      {    512 * KB, 1, 512,     16 * KB, 32},
      {      1 * MB, 1, 512,     16 * KB, 64},
      {      2 * MB, 1, 512,     32 * KB, 64},
      {      4 * MB, 1, 512,     64 * KB, 64},
      {      8 * MB, 2, 512,    128 * KB, 64},
      {     16 * MB, 2, 512,    256 * KB, 64},
      {     32 * MB, 2, 512,    256 * KB, 128},
      {     64 * MB, 2, 512,    256 * KB, 128},
      {    128 * MB, 4, 512,    512 * KB, 64},
      {    256 * MB, 4, 512,      1 * MB, 32},
      {    512 * MB, 4, 512,      1 * MB, 32},
      {      1 * GB, 4, 512,      1 * MB, 32},
      {      2 * GB, 4, 512,      1 * MB, 32},
      {      4 * GB, 4, 512,      1 * MB, 32},
      {      8 * GB, 4, 512,      1 * MB, 32},
      {     16 * GB, 4, 512,      1 * MB, 32},
      {     32 * GB, 4, 512,      1 * MB, 32},
      {     64 * GB, 4, 512,      1 * MB, 32},
  };
  // clang-format on

  verifyAutoTune(cases, nRanks, maxOccNumBlocks, defThreads, arch);
}

TEST(AutoTuneHopperRankSweep, DefaultBDP_64Ranks) {
  const int nRanks = 64;
  const int maxOccNumBlocks = 64;
  const int defThreads = 512;
  const auto arch = GpuArch::Hopper;

  // clang-format off
  const AutoTuneExpected cases[] = {
      {      1 * KB, 1, 384,          16, 64},
      {      2 * KB, 1, 384,          32, 64},
      {      4 * KB, 1, 384,          64, 64},
      {      8 * KB, 1, 384,         128, 64},
      {     16 * KB, 1, 384,         256, 64},
      {     32 * KB, 1, 384,         512, 64},
      {     64 * KB, 1, 384,      1 * KB, 64},
      {    128 * KB, 1, 384,      2 * KB, 64},
      {    256 * KB, 1, 384,      4 * KB, 64},
      {    512 * KB, 1, 384,      8 * KB, 64},
      {      1 * MB, 1, 512,     16 * KB, 64},
      {      2 * MB, 1, 512,     16 * KB, 128},
      {      4 * MB, 1, 512,     32 * KB, 128},
      {      8 * MB, 1, 512,     64 * KB, 128},
      {     16 * MB, 2, 512,    128 * KB, 128},
      {     32 * MB, 2, 512,    256 * KB, 128},
      {     64 * MB, 2, 512,    128 * KB, 256},
      {    128 * MB, 2, 512,    128 * KB, 256},
      {    256 * MB, 2, 512,    256 * KB, 128},
      {    512 * MB, 4, 512,    512 * KB, 64},
      {      1 * GB, 4, 512,    512 * KB, 64},
      {      2 * GB, 4, 512,    512 * KB, 64},
      {      4 * GB, 4, 512,    512 * KB, 64},
      {      8 * GB, 4, 512,    512 * KB, 64},
      {     16 * GB, 4, 512,    512 * KB, 64},
      {     32 * GB, 4, 512,    512 * KB, 64},
      {     64 * GB, 4, 512,    512 * KB, 64},
  };
  // clang-format on

  verifyAutoTune(cases, nRanks, maxOccNumBlocks, defThreads, arch);
}

// ============================================================================
// CVAR override tests for getAutoTunedParams
// ============================================================================

class AutoTuneCVAROverrideTest : public ::testing::Test {
 protected:
  static constexpr int kMaxOccNumBlocks = 64;
  static constexpr int kDefThreads = 512;
  static constexpr int kNRanks = 8;
  // Use a message size large enough that auto-tune produces non-trivial values.
  static constexpr size_t kMsg = 64 * MB;
};

// Chunk size CVAR alone overrides chunkSize, numChunks stays auto-tuned.
TEST_F(AutoTuneCVAROverrideTest, ChunkSizeOnly) {
  MaxBDPOverride bdp(128 * MB);
  ChunkSizeOverride cs(1 * MB);

  auto at = getAutoTunedParams(kMsg, kNRanks, kMaxOccNumBlocks, kDefThreads, 1);
  EXPECT_EQ(at.pipeline.chunkSize, 1 * MB);
  // numChunks is still auto-tuned (not overridden)
  EXPECT_GT(at.pipeline.numChunks, 0u);
}

// Num chunks CVAR alone overrides numChunks, chunkSize stays auto-tuned.
TEST_F(AutoTuneCVAROverrideTest, NumChunksOnly) {
  MaxBDPOverride bdp(128 * MB);
  NumChunksOverride nc(4);

  auto at = getAutoTunedParams(kMsg, kNRanks, kMaxOccNumBlocks, kDefThreads, 1);
  EXPECT_EQ(at.pipeline.numChunks, 4u);
  // chunkSize is still auto-tuned
  EXPECT_GT(at.pipeline.chunkSize, 0u);
}

// Both chunk CVARs set together.
TEST_F(AutoTuneCVAROverrideTest, ChunkSizeAndNumChunks) {
  MaxBDPOverride bdp(128 * MB);
  ChunkSizeOverride cs(2 * MB);
  NumChunksOverride nc(8);

  auto at = getAutoTunedParams(kMsg, kNRanks, kMaxOccNumBlocks, kDefThreads, 1);
  EXPECT_EQ(at.pipeline.chunkSize, 2 * MB);
  EXPECT_EQ(at.pipeline.numChunks, 8u);
}

// Block CVARs override auto-tuned block params.
// numBlocks CVAR acts as an upper-bound cap (does not inflate).
// blockSize CVAR is an unconditional override.
TEST_F(AutoTuneCVAROverrideTest, BlockOverrides) {
  MaxBDPOverride bdp(128 * MB);

  // Case 1: CVAR < auto-tuned numBlocks → caps down.
  // 64MB Default → auto-tune gives 8 blocks; CVAR caps to 3.
  {
    NumBlocksOverride nb(3);
    BlockSizeOverride bs(384);
    auto at =
        getAutoTunedParams(kMsg, kNRanks, kMaxOccNumBlocks, kDefThreads, 1);
    EXPECT_EQ(at.block.numBlocks, 3);
    EXPECT_EQ(at.block.blockSize, 384);
  }

  // Case 2: CVAR > auto-tuned numBlocks → auto-tuned value preserved.
  // Use a small chunkSize to get 1 auto-tuned block, set CVAR to 16.
  {
    ChunkSizeOverride cs(4 * KB); // Default tier <8K → 1 block
    NumBlocksOverride nb(16);
    auto at =
        getAutoTunedParams(kMsg, kNRanks, kMaxOccNumBlocks, kDefThreads, 1);
    EXPECT_EQ(at.block.numBlocks, 1); // not inflated to 16
  }

  // Case 3: CVAR > maxOccupancyNumBlocks → maxOccupancy still respected.
  // Large chunkSize auto-tunes to 8 blocks, maxOccNumBlocks=3 clamps to 3.
  // CVAR=5 is above maxOccNumBlocks but below unclamped auto-tune; must not
  // inflate past maxOccupancy.
  {
    constexpr int lowMaxOccNumBlocks = 3;
    NumBlocksOverride nb(5);
    auto at =
        getAutoTunedParams(kMsg, kNRanks, lowMaxOccNumBlocks, kDefThreads, 1);
    EXPECT_LE(at.block.numBlocks, lowMaxOccNumBlocks);
  }
}

// Chunk size override feeds into block params computation (Default arch).
// Default thresholds: <8K->1, 8K-32K->2, 32K-64K->4, >=64K->8
TEST_F(AutoTuneCVAROverrideTest, ChunkSizeAffectsBlockParams) {
  MaxBDPOverride bdp(128 * MB);

  struct Case {
    int chunkSize;
    int expectedBlocks;
  };
  // clang-format off
  const Case cases[] = {
      {  4 * KB, 1},
      {  8 * KB, 2},
      { 32 * KB, 4},
      { 64 * KB, 8},
      {  1 * MB, 8},
  };
  // clang-format on

  for (const auto& c : cases) {
    ChunkSizeOverride cs(c.chunkSize);
    auto at =
        getAutoTunedParams(kMsg, kNRanks, kMaxOccNumBlocks, kDefThreads, 1);
    EXPECT_EQ(at.block.numBlocks, c.expectedBlocks)
        << "chunkSize=" << c.chunkSize;
  }
}

// Chunk size override feeds into block params on Hopper arch.
// Hopper thresholds: <16K->{1,384}, 16K-128K->{1,512}, 128K-512K->{2,512},
// >=512K->{4,512}
TEST_F(AutoTuneCVAROverrideTest, ChunkSizeAffectsBlockParamsHopper) {
  MaxBDPOverride bdp(128 * MB);
  const auto arch = GpuArch::Hopper;

  struct Case {
    int chunkSize;
    int expectedBlocks;
    int expectedBlockSize;
  };
  // clang-format off
  const Case cases[] = {
      {   8 * KB, 1, 384},
      {  16 * KB, 1, 512},
      { 128 * KB, 2, 512},
      { 512 * KB, 4, 512},
  };
  // clang-format on

  for (const auto& c : cases) {
    ChunkSizeOverride cs(c.chunkSize);
    auto at = getAutoTunedParams(
        kMsg, kNRanks, kMaxOccNumBlocks, kDefThreads, 1, arch);
    EXPECT_EQ(at.block.numBlocks, c.expectedBlocks)
        << "chunkSize=" << c.chunkSize;
    EXPECT_EQ(at.block.blockSize, c.expectedBlockSize)
        << "chunkSize=" << c.chunkSize;
  }
}

// Block CVAR acts as an upper-bound cap, not an unconditional override.
// When auto-tune produces more blocks than the CVAR, it caps down.
// When auto-tune produces fewer blocks, the auto-tuned value is preserved.
TEST_F(AutoTuneCVAROverrideTest, BlockOverrideTakesPriorityOverChunkDerived) {
  MaxBDPOverride bdp(128 * MB);

  // Case 1: CVAR < auto-tuned → caps down.
  // chunkSize=1MB on Default → auto-tune gives 8 blocks; CVAR caps to 2.
  {
    ChunkSizeOverride cs(1 * MB);
    NumBlocksOverride nb(2);
    auto at =
        getAutoTunedParams(kMsg, kNRanks, kMaxOccNumBlocks, kDefThreads, 1);
    EXPECT_EQ(at.pipeline.chunkSize, 1 * MB);
    EXPECT_EQ(at.block.numBlocks, 2);
  }

  // Case 2: CVAR > auto-tuned → auto-tuned value preserved (cap is a no-op).
  // chunkSize=4KB on Default → auto-tune gives 1 block; CVAR=4 does not
  // inflate.
  {
    ChunkSizeOverride cs(4 * KB);
    NumBlocksOverride nb(4);
    auto at =
        getAutoTunedParams(kMsg, kNRanks, kMaxOccNumBlocks, kDefThreads, 1);
    EXPECT_EQ(at.pipeline.chunkSize, 4 * KB);
    EXPECT_EQ(at.block.numBlocks, 1);
  }
}

// All four CVARs set simultaneously.
TEST_F(AutoTuneCVAROverrideTest, AllFourOverrides) {
  MaxBDPOverride bdp(128 * MB);
  ChunkSizeOverride cs(512 * KB);
  NumChunksOverride nc(16);
  NumBlocksOverride nb(4);
  BlockSizeOverride bs(256);

  auto at = getAutoTunedParams(kMsg, kNRanks, kMaxOccNumBlocks, kDefThreads, 1);
  EXPECT_EQ(at.pipeline.chunkSize, 512 * KB);
  EXPECT_EQ(at.pipeline.numChunks, 16u);
  EXPECT_EQ(at.block.numBlocks, 4);
  EXPECT_EQ(at.block.blockSize, 256);
}

// ============================================================================
// AutoTuned Invariants
// ============================================================================

class AutoTuneInvariantTest : public ::testing::Test {
 protected:
  static constexpr int kMaxOccNumBlocks = 64;
  static constexpr int kDefThreads = 512;
  static constexpr int kNRanks = 8;
  // Use a message size large enough that auto-tune produces non-trivial values.
  static constexpr size_t kMsg = 64 * MB;
};

// Spot-check that small maxBDP values correctly reduce pipeline chunks.
TEST_F(AutoTuneInvariantTest, SmallMaxBDP_ChunksReduced) {
  {
    MaxBDPOverride o(256 * KB);
    auto at =
        getAutoTunedParams(1 * MB, kNRanks, kMaxOccNumBlocks, kDefThreads, 1);
    EXPECT_EQ(at.pipeline.chunkSize, 16 * KB);
    EXPECT_EQ(at.pipeline.numChunks, 16u);
  }
  {
    MaxBDPOverride o(512 * KB);
    auto at =
        getAutoTunedParams(1 * MB, kNRanks, kMaxOccNumBlocks, kDefThreads, 1);
    EXPECT_EQ(at.pipeline.chunkSize, 32 * KB);
    EXPECT_EQ(at.pipeline.numChunks, 16u);
  }
}

// maxOccupancyNumBlocks clamps block count; blockSize clamped by
// defaultThreads. Hopper tiers have explicit blockSize values (384, 512) that
// can exceed defaultThreads, exercising the std::min(blockSize, defaultThreads)
// path.
TEST_F(AutoTuneInvariantTest, MaxOccupancyClampWithBlockSize) {
  MaxBDPOverride bdp(128 * MB);
  const auto arch = GpuArch::Hopper;

  // Hopper tier: chunkSize < 16K -> {1 block, 384 threads}
  // With defaultThreads=256, blockSize should clamp to 256.
  // With maxOccupancyNumBlocks=1, numBlocks stays 1 (no clamp needed).
  {
    ChunkSizeOverride cs(8 * KB);
    // Verify unclamped tier values are larger (clamping is meaningful)
    auto unclamped = getAutoTunedParams(
        kMsg,
        kNRanks,
        /*maxOccupancyNumBlocks=*/kMaxOccNumBlocks,
        kDefThreads,
        1,
        arch);
    ASSERT_GE(unclamped.block.blockSize, 256);

    auto at = getAutoTunedParams(
        kMsg,
        kNRanks,
        /*maxOccupancyNumBlocks=*/1,
        /*defaultThreads=*/256,
        1,
        arch);
    EXPECT_EQ(at.block.numBlocks, 1);
    EXPECT_EQ(at.block.blockSize, 256); // clamped from 384
  }

  // Hopper tier: chunkSize >= 512K -> {4 blocks, 512 threads}
  // With defaultThreads=256, blockSize should clamp to 256.
  // With maxOccupancyNumBlocks=2, numBlocks should clamp to 2.
  {
    ChunkSizeOverride cs(1 * MB);
    // Verify unclamped tier values are larger (clamping is meaningful)
    auto unclamped = getAutoTunedParams(
        kMsg,
        kNRanks,
        /*maxOccupancyNumBlocks=*/kMaxOccNumBlocks,
        kDefThreads,
        1,
        arch);
    ASSERT_GE(unclamped.block.numBlocks, 2);
    ASSERT_GE(unclamped.block.blockSize, 256);

    auto at = getAutoTunedParams(
        kMsg,
        kNRanks,
        /*maxOccupancyNumBlocks=*/2,
        /*defaultThreads=*/256,
        1,
        arch);
    EXPECT_EQ(at.block.numBlocks, 2); // clamped from 4
    EXPECT_EQ(at.block.blockSize, 256); // clamped from 512
  }
}

// Build basePoints for probes.
// For each consecutive pair (p, pNext), generates 7 probe points:
//   p, p+1, mid-1, mid, mid+1, pNext-1, pNext
// where mid = (p + pNext) / 2.
template <typename T>
std::vector<T> buildProbes(const std::vector<T>& basePoints) {
  std::set<T> probes;
  for (size_t i = 0; i < basePoints.size(); ++i) {
    T p = basePoints[i];
    if (i + 1 < basePoints.size()) {
      T pNext = basePoints[i + 1];
      T mid = static_cast<T>((p + pNext) / 2);
      // clang-format off
      probes.insert(p);          // exact p
      probes.insert(p + 1);      // just above p
      probes.insert(mid - 1);     // just below midpoint
      probes.insert(mid);         // exact midpoint
      probes.insert(mid + 1);     // just above midpoint
      probes.insert(pNext - 1);  // just below pNext
      probes.insert(pNext);      // exact pNext
      // clang-format on
    } else {
      probes.insert(p);
    }
  }

  // Filter out zero/negative values that arise from offset arithmetic.
  std::vector<T> result;
  for (auto v : probes) {
    if (v > 0) {
      result.push_back(v);
    }
  }
  return result;
}

// Generate BDP base points.
// buildProbes then adds boundary probes around each consecutive pair.
std::vector<size_t> bdpBasePoints() {
  std::vector<size_t> base;
  for (size_t b = 256 * KB; b <= 128 * MB; b *= 2) {
    base.push_back(b);
  }
  return base;
}

// Generate rank base points.
// buildProbes then adds boundary probes around each consecutive pair.
std::vector<int> rankBasePoints() {
  std::vector<int> base;
  for (int r = 2; r <= 8; ++r) {
    base.push_back(r);
  }
  for (int p = 16; p <= 128; p *= 2) {
    base.push_back(p);
  }
  return base;
}

// Generate message-size base points.
// buildProbes then adds boundary probes around each consecutive pair.
std::vector<size_t> msgBytesBasePoints() {
  std::vector<size_t> base;
  for (size_t p = 1; p <= 64 * GB; p *= 2) {
    base.push_back(p);
  }
  return base;
}

// Mirror of roundToNearestPow2 from AllReduceRingAutoTune.cc for test
// verification. Rounds n to the nearest power of 2; ties round up.
size_t roundToNearestPow2(size_t n) {
  if (n <= 1) {
    return 1;
  }
  if (std::has_single_bit(n)) {
    return n;
  }
  int bits = std::countl_zero(n);
  size_t lo = size_t{1} << (sizeof(size_t) * 8 - 1 - bits);
  size_t hi = lo << 1;
  return (n - lo < hi - n) ? lo : hi;
}

// Verify non-pow2 msgBytes produces same tuning decision as nearest pow2.
void checkPow2Equivalence(
    size_t msgBytes,
    int nRanks,
    int maxOccNumBlocks,
    int defThreads,
    size_t maxBDP) {
  auto at =
      getAutoTunedParams(msgBytes, nRanks, maxOccNumBlocks, defThreads, 1);
  size_t pow2Msg = roundToNearestPow2(msgBytes);
  auto atPow2 =
      getAutoTunedParams(pow2Msg, nRanks, maxOccNumBlocks, defThreads, 1);
  EXPECT_EQ(at.pipeline.chunkSize, atPow2.pipeline.chunkSize)
      << "pow2 mismatch chunkSize: msgBytes=" << msgBytes << " pow2=" << pow2Msg
      << " ranks=" << nRanks << " maxBDP=" << maxBDP;
  EXPECT_EQ(at.pipeline.numChunks, atPow2.pipeline.numChunks)
      << "pow2 mismatch numChunks: msgBytes=" << msgBytes << " pow2=" << pow2Msg
      << " ranks=" << nRanks << " maxBDP=" << maxBDP;
  EXPECT_EQ(at.block.numBlocks, atPow2.block.numBlocks)
      << "pow2 mismatch numBlocks: msgBytes=" << msgBytes << " pow2=" << pow2Msg
      << " ranks=" << nRanks << " maxBDP=" << maxBDP;
  EXPECT_EQ(at.block.blockSize, atPow2.block.blockSize)
      << "pow2 mismatch blockSize: msgBytes=" << msgBytes << " pow2=" << pow2Msg
      << " ranks=" << nRanks << " maxBDP=" << maxBDP;
}

// Verify chunkSize * numChunks <= maxBDP and basic sanity.
void checkBDPBudget(
    size_t msgBytes,
    int nRanks,
    int maxOccNumBlocks,
    int defThreads,
    size_t maxBDP) {
  auto at =
      getAutoTunedParams(msgBytes, nRanks, maxOccNumBlocks, defThreads, 1);
  EXPECT_GT(at.pipeline.chunkSize, 0u)
      << "chunkSize=0: msgBytes=" << msgBytes << " ranks=" << nRanks
      << " maxBDP=" << maxBDP;
  EXPECT_GT(at.pipeline.numChunks, 0u)
      << "numChunks=0: msgBytes=" << msgBytes << " ranks=" << nRanks
      << " maxBDP=" << maxBDP;
  EXPECT_LE(at.pipeline.chunkSize * at.pipeline.numChunks, maxBDP)
      << "BDP violated: msgBytes=" << msgBytes << " ranks=" << nRanks
      << " maxBDP=" << maxBDP << " chunkSize=" << at.pipeline.chunkSize
      << " numChunks=" << at.pipeline.numChunks;
  EXPECT_GT(at.block.numBlocks, 0)
      << "numBlocks=0: msgBytes=" << msgBytes << " ranks=" << nRanks
      << " maxBDP=" << maxBDP;
  EXPECT_GT(at.block.blockSize, 0)
      << "blockSize=0: msgBytes=" << msgBytes << " ranks=" << nRanks
      << " maxBDP=" << maxBDP;
}

// Verify chunkSize is aligned to each typeSize in {1, 2, 4, 8},
// and 16-byte aligned for vectorized device access when large enough.
void checkChunkAlignment(
    size_t msgBytes,
    int nRanks,
    int maxOccNumBlocks,
    int defThreads,
    size_t maxBDP) {
  constexpr size_t typeSizes[] = {1, 2, 4, 8};
  for (auto typeSize : typeSizes) {
    if (msgBytes < typeSize) {
      continue;
    }
    auto at = getAutoTunedParams(
        msgBytes, nRanks, maxOccNumBlocks, defThreads, typeSize);
    EXPECT_EQ(at.pipeline.chunkSize % typeSize, 0u)
        << "Not aligned: msgBytes=" << msgBytes << " ranks=" << nRanks
        << " typeSize=" << typeSize << " maxBDP=" << maxBDP
        << " chunkSize=" << at.pipeline.chunkSize;
    EXPECT_GE(at.pipeline.chunkSize, typeSize)
        << "chunkSize < typeSize: msgBytes=" << msgBytes << " ranks=" << nRanks
        << " typeSize=" << typeSize << " maxBDP=" << maxBDP
        << " chunkSize=" << at.pipeline.chunkSize;
  }

  // 16B alignment for vectorized load/store, only meaningful when
  // chunkSize can be >= 16.
  if (msgBytes >= static_cast<size_t>(nRanks) * 16) {
    auto at =
        getAutoTunedParams(msgBytes, nRanks, maxOccNumBlocks, defThreads, 1);
    EXPECT_EQ(at.pipeline.chunkSize % 16, 0u)
        << "Not 16B aligned: msgBytes=" << msgBytes << " ranks=" << nRanks
        << " maxBDP=" << maxBDP << " chunkSize=" << at.pipeline.chunkSize;
  }
}

// Combined invariant checks over the full (maxBDP, nRanks, msgBytes) space.
// For each combination we verify:
//   1. Non-pow2 msgBytes produces same tuning as nearest pow2
//   2. chunkSize * numChunks <= maxBDP (BDP budget respected)
//   3. chunkSize aligned to typeSize {1,2,4,8} and 16B for vectorized access
TEST_F(AutoTuneInvariantTest, CombinedInvariants) {
  const auto rankProbes = buildProbes(rankBasePoints());
  const auto maxBDPs = bdpBasePoints();
  const auto msgBytesProbes = buildProbes(msgBytesBasePoints());

  for (auto maxBDP : maxBDPs) {
    MaxBDPOverride o(maxBDP);
    for (auto nRanks : rankProbes) {
      if (nRanks < 2) {
        continue;
      }
      for (auto msgBytes : msgBytesProbes) {
        checkPow2Equivalence(
            msgBytes, nRanks, kMaxOccNumBlocks, kDefThreads, maxBDP);
        checkBDPBudget(msgBytes, nRanks, kMaxOccNumBlocks, kDefThreads, maxBDP);
        checkChunkAlignment(
            msgBytes, nRanks, kMaxOccNumBlocks, kDefThreads, maxBDP);
      }
    }
  }
}
