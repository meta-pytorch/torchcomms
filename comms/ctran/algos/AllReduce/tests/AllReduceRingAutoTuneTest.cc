// Copyright (c) Meta Platforms, Inc. and affiliates.

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
    int maxOcc,
    int defThreads,
    GpuArch arch = GpuArch::Default) {
  for (const auto& c : cases) {
    auto at = getAutoTunedParams(c.msgBytes, nRanks, maxOcc, defThreads, arch);
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
  const int maxOcc = 64;
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

  verifyAutoTune(cases, nRanks, maxOcc, defThreads);
}

TEST(AutoTuneCombinedDefault, MaxBDP32M_8Ranks) {
  MaxBDPOverride o(32 * MB);
  const int nRanks = 8;
  const int maxOcc = 64;
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

  verifyAutoTune(cases, nRanks, maxOcc, defThreads);
}

TEST(AutoTuneCombinedDefault, MaxBDP64M_8Ranks) {
  MaxBDPOverride o(64 * MB);
  const int nRanks = 8;
  const int maxOcc = 64;
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

  verifyAutoTune(cases, nRanks, maxOcc, defThreads);
}

TEST(AutoTuneCombinedDefault, MaxBDP128M_8Ranks) {
  MaxBDPOverride o(128 * MB);
  const int nRanks = 8;
  const int maxOcc = 64;
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

  verifyAutoTune(cases, nRanks, maxOcc, defThreads);
}

// ============================================================================
// getAutoTunedParams golden tables: Hopper (H100) arch, 8 ranks, pow2 1K-64G
// ============================================================================

TEST(AutoTuneCombinedHopper, MaxBDP16M_8Ranks) {
  MaxBDPOverride o(16 * MB);
  const int nRanks = 8;
  const int maxOcc = 64;
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

  verifyAutoTune(cases, nRanks, maxOcc, defThreads, arch);
}

TEST(AutoTuneCombinedHopper, MaxBDP32M_8Ranks) {
  MaxBDPOverride o(32 * MB);
  const int nRanks = 8;
  const int maxOcc = 64;
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

  verifyAutoTune(cases, nRanks, maxOcc, defThreads, arch);
}

TEST(AutoTuneCombinedHopper, MaxBDP64M_8Ranks) {
  MaxBDPOverride o(64 * MB);
  const int nRanks = 8;
  const int maxOcc = 64;
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

  verifyAutoTune(cases, nRanks, maxOcc, defThreads, arch);
}

TEST(AutoTuneCombinedHopper, MaxBDP128M_8Ranks) {
  MaxBDPOverride o(128 * MB);
  const int nRanks = 8;
  const int maxOcc = 64;
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

  verifyAutoTune(cases, nRanks, maxOcc, defThreads, arch);
}

// ============================================================================
// Rank-sweep tables: Default arch, arch-default BDP, ranks {8,16,32,64}
// ============================================================================

TEST(AutoTuneDefaultRankSweep, DefaultBDP_8Ranks) {
  const int nRanks = 8;
  const int maxOcc = 64;
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

  verifyAutoTune(cases, nRanks, maxOcc, defThreads);
}

TEST(AutoTuneDefaultRankSweep, DefaultBDP_16Ranks) {
  const int nRanks = 16;
  const int maxOcc = 64;
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

  verifyAutoTune(cases, nRanks, maxOcc, defThreads);
}

TEST(AutoTuneDefaultRankSweep, DefaultBDP_32Ranks) {
  const int nRanks = 32;
  const int maxOcc = 64;
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

  verifyAutoTune(cases, nRanks, maxOcc, defThreads);
}

TEST(AutoTuneDefaultRankSweep, DefaultBDP_64Ranks) {
  const int nRanks = 64;
  const int maxOcc = 64;
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

  verifyAutoTune(cases, nRanks, maxOcc, defThreads);
}

// ============================================================================
// Rank-sweep tables: Hopper arch, arch-default BDP, ranks {8,16,32,64}
// ============================================================================

TEST(AutoTuneHopperRankSweep, DefaultBDP_8Ranks) {
  const int nRanks = 8;
  const int maxOcc = 64;
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

  verifyAutoTune(cases, nRanks, maxOcc, defThreads, arch);
}

TEST(AutoTuneHopperRankSweep, DefaultBDP_16Ranks) {
  const int nRanks = 16;
  const int maxOcc = 64;
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

  verifyAutoTune(cases, nRanks, maxOcc, defThreads, arch);
}

TEST(AutoTuneHopperRankSweep, DefaultBDP_32Ranks) {
  const int nRanks = 32;
  const int maxOcc = 64;
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

  verifyAutoTune(cases, nRanks, maxOcc, defThreads, arch);
}

TEST(AutoTuneHopperRankSweep, DefaultBDP_64Ranks) {
  const int nRanks = 64;
  const int maxOcc = 64;
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

  verifyAutoTune(cases, nRanks, maxOcc, defThreads, arch);
}

// ============================================================================
// CVAR override tests for getAutoTunedParams
// ============================================================================

class AutoTuneCVAROverrideTest : public ::testing::Test {
 protected:
  static constexpr int kMaxOcc = 64;
  static constexpr int kDefThreads = 512;
  static constexpr int kNRanks = 8;
  // Use a message size large enough that auto-tune produces non-trivial values.
  static constexpr size_t kMsg = 64 * MB;
};

// Chunk size CVAR alone overrides chunkSize, numChunks stays auto-tuned.
TEST_F(AutoTuneCVAROverrideTest, ChunkSizeOnly) {
  MaxBDPOverride bdp(128 * MB);
  ChunkSizeOverride cs(1 * MB);

  auto at = getAutoTunedParams(kMsg, kNRanks, kMaxOcc, kDefThreads);
  EXPECT_EQ(at.pipeline.chunkSize, 1 * MB);
  // numChunks is still auto-tuned (not overridden)
  EXPECT_GT(at.pipeline.numChunks, 0u);
}

// Num chunks CVAR alone overrides numChunks, chunkSize stays auto-tuned.
TEST_F(AutoTuneCVAROverrideTest, NumChunksOnly) {
  MaxBDPOverride bdp(128 * MB);
  NumChunksOverride nc(4);

  auto at = getAutoTunedParams(kMsg, kNRanks, kMaxOcc, kDefThreads);
  EXPECT_EQ(at.pipeline.numChunks, 4u);
  // chunkSize is still auto-tuned
  EXPECT_GT(at.pipeline.chunkSize, 0u);
}

// Both chunk CVARs set together.
TEST_F(AutoTuneCVAROverrideTest, ChunkSizeAndNumChunks) {
  MaxBDPOverride bdp(128 * MB);
  ChunkSizeOverride cs(2 * MB);
  NumChunksOverride nc(8);

  auto at = getAutoTunedParams(kMsg, kNRanks, kMaxOcc, kDefThreads);
  EXPECT_EQ(at.pipeline.chunkSize, 2 * MB);
  EXPECT_EQ(at.pipeline.numChunks, 8u);
}

// Block CVARs override auto-tuned block params.
TEST_F(AutoTuneCVAROverrideTest, BlockOverrides) {
  MaxBDPOverride bdp(128 * MB);
  NumBlocksOverride nb(3);
  BlockSizeOverride bs(384);

  auto at = getAutoTunedParams(kMsg, kNRanks, kMaxOcc, kDefThreads);
  EXPECT_EQ(at.block.numBlocks, 3);
  EXPECT_EQ(at.block.blockSize, 384);
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
    auto at = getAutoTunedParams(kMsg, kNRanks, kMaxOcc, kDefThreads);
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
    auto at = getAutoTunedParams(kMsg, kNRanks, kMaxOcc, kDefThreads, arch);
    EXPECT_EQ(at.block.numBlocks, c.expectedBlocks)
        << "chunkSize=" << c.chunkSize;
    EXPECT_EQ(at.block.blockSize, c.expectedBlockSize)
        << "chunkSize=" << c.chunkSize;
  }
}

// Block CVARs take priority over the block params derived from chunk override.
TEST_F(AutoTuneCVAROverrideTest, BlockOverrideTakesPriorityOverChunkDerived) {
  MaxBDPOverride bdp(128 * MB);
  ChunkSizeOverride cs(1 * MB); // would auto-tune to 8 blocks on Default
  NumBlocksOverride nb(2); // explicit override wins

  auto at = getAutoTunedParams(kMsg, kNRanks, kMaxOcc, kDefThreads);
  EXPECT_EQ(at.pipeline.chunkSize, 1 * MB);
  EXPECT_EQ(at.block.numBlocks, 2);
}

// All four CVARs set simultaneously.
TEST_F(AutoTuneCVAROverrideTest, AllFourOverrides) {
  MaxBDPOverride bdp(128 * MB);
  ChunkSizeOverride cs(512 * KB);
  NumChunksOverride nc(16);
  NumBlocksOverride nb(4);
  BlockSizeOverride bs(256);

  auto at = getAutoTunedParams(kMsg, kNRanks, kMaxOcc, kDefThreads);
  EXPECT_EQ(at.pipeline.chunkSize, 512 * KB);
  EXPECT_EQ(at.pipeline.numChunks, 16u);
  EXPECT_EQ(at.block.numBlocks, 4);
  EXPECT_EQ(at.block.blockSize, 256);
}

// Pipeline BDP invariant: chunkSize * numChunks <= maxBDP for all configs.
TEST_F(AutoTuneCVAROverrideTest, BDPInvariant) {
  const size_t maxBDPs[] = {
      256 * KB,
      512 * KB,
      1 * MB,
      2 * MB,
      4 * MB,
      8 * MB,
      16 * MB,
      32 * MB,
      64 * MB,
      128 * MB};

  size_t msg = 1 * KB;
  while (msg <= 64 * GB) {
    for (auto maxBDP : maxBDPs) {
      MaxBDPOverride o(maxBDP);
      auto at = getAutoTunedParams(msg, kNRanks, kMaxOcc, kDefThreads);
      EXPECT_LE(at.pipeline.chunkSize * at.pipeline.numChunks, maxBDP)
          << "BDP violated: msg=" << msg << " maxBDP=" << maxBDP
          << " chunkSize=" << at.pipeline.chunkSize
          << " numChunks=" << at.pipeline.numChunks;
    }
    msg *= 2;
  }
}

// Spot-check that small maxBDP values correctly reduce pipeline chunks.
TEST_F(AutoTuneCVAROverrideTest, SmallMaxBDP_ChunksReduced) {
  {
    MaxBDPOverride o(256 * KB);
    auto at = getAutoTunedParams(1 * MB, kNRanks, kMaxOcc, kDefThreads);
    EXPECT_EQ(at.pipeline.chunkSize, 16 * KB);
    EXPECT_EQ(at.pipeline.numChunks, 16u);
  }
  {
    MaxBDPOverride o(512 * KB);
    auto at = getAutoTunedParams(1 * MB, kNRanks, kMaxOcc, kDefThreads);
    EXPECT_EQ(at.pipeline.chunkSize, 32 * KB);
    EXPECT_EQ(at.pipeline.numChunks, 16u);
  }
}

// maxOccupancyBlocks clamps block count; blockSize clamped by defaultThreads.
// Hopper tiers have explicit blockSize values (384, 512) that can exceed
// defaultThreads, exercising the std::min(blockSize, defaultThreads) path.
TEST_F(AutoTuneCVAROverrideTest, MaxOccupancyClampWithBlockSize) {
  MaxBDPOverride bdp(128 * MB);
  const auto arch = GpuArch::Hopper;

  // Hopper tier: chunkSize < 16K -> {1 block, 384 threads}
  // With defaultThreads=256, blockSize should clamp to 256.
  // With maxOccupancyBlocks=1, numBlocks stays 1 (no clamp needed).
  {
    ChunkSizeOverride cs(8 * KB);
    // Verify unclamped tier values are larger (clamping is meaningful)
    auto unclamped = getAutoTunedParams(
        kMsg, kNRanks, /*maxOccupancyBlocks=*/kMaxOcc, kDefThreads, arch);
    ASSERT_GE(unclamped.block.blockSize, 256);

    auto at = getAutoTunedParams(
        kMsg, kNRanks, /*maxOccupancyBlocks=*/1, /*defaultThreads=*/256, arch);
    EXPECT_EQ(at.block.numBlocks, 1);
    EXPECT_EQ(at.block.blockSize, 256); // clamped from 384
  }

  // Hopper tier: chunkSize >= 512K -> {4 blocks, 512 threads}
  // With defaultThreads=256, blockSize should clamp to 256.
  // With maxOccupancyBlocks=2, numBlocks should clamp to 2.
  {
    ChunkSizeOverride cs(1 * MB);
    // Verify unclamped tier values are larger (clamping is meaningful)
    auto unclamped = getAutoTunedParams(
        kMsg, kNRanks, /*maxOccupancyBlocks=*/kMaxOcc, kDefThreads, arch);
    ASSERT_GE(unclamped.block.numBlocks, 2);
    ASSERT_GE(unclamped.block.blockSize, 256);

    auto at = getAutoTunedParams(
        kMsg, kNRanks, /*maxOccupancyBlocks=*/2, /*defaultThreads=*/256, arch);
    EXPECT_EQ(at.block.numBlocks, 2); // clamped from 4
    EXPECT_EQ(at.block.blockSize, 256); // clamped from 512
  }
}

// Non-pow2 message sizes: verifies that auto-tune decisions for non-pow2
// inputs match the nearest pow2 (roundToNearestPow2 snaps internally).
// Probes 7 points per boundary: pow2(n), pow2(n)+1, mid-1, mid, mid+1,
// pow2(n+1)-1, pow2(n+1).
TEST_F(AutoTuneCVAROverrideTest, NonPow2MessageSizes) {
  MaxBDPOverride bdp(128 * MB);

  // clang-format off
  const size_t pow2Sizes[] = {
      8 * KB,   // block tier boundary
      64 * KB,  // block tier boundary
      256 * KB, // pipeline chunking region
      1 * MB,   // pipeline depth transition
      16 * MB,  // large message region
      32 * MB,  // pipeline depth change
  };
  // clang-format on

  for (auto p2 : pow2Sizes) {
    const size_t p2next = p2 * 2;
    const size_t mid = (p2 + p2next) / 2;

    struct Probe {
      size_t msg;
      size_t expectedPow2;
    };
    // clang-format off
    const Probe probes[] = {
        {p2,         p2},      // exact pow2(n)
        {p2 + 1,     p2},      // just above, rounds down
        {mid - 1,    p2},      // just below midpoint, rounds down
        {mid,        p2next},  // exact midpoint, ties round up
        {mid + 1,    p2next},  // just above midpoint, rounds up
        {p2next - 1, p2next},  // just below pow2(n+1), rounds up
        {p2next,     p2next},  // exact pow2(n+1)
    };
    // clang-format on

    for (const auto& pr : probes) {
      auto actual = getAutoTunedParams(pr.msg, kNRanks, kMaxOcc, kDefThreads);
      auto expected =
          getAutoTunedParams(pr.expectedPow2, kNRanks, kMaxOcc, kDefThreads);
      EXPECT_EQ(actual.pipeline.chunkSize, expected.pipeline.chunkSize)
          << "chunkSize: msg=" << pr.msg
          << " expected pow2=" << pr.expectedPow2;
      EXPECT_EQ(actual.pipeline.numChunks, expected.pipeline.numChunks)
          << "numChunks: msg=" << pr.msg
          << " expected pow2=" << pr.expectedPow2;
      EXPECT_EQ(actual.block.numBlocks, expected.block.numBlocks)
          << "numBlocks: msg=" << pr.msg
          << " expected pow2=" << pr.expectedPow2;
      EXPECT_EQ(actual.block.blockSize, expected.block.blockSize)
          << "blockSize: msg=" << pr.msg
          << " expected pow2=" << pr.expectedPow2;
    }
  }
}
