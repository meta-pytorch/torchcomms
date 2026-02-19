// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/AllReduce/AllReduceRingAutoTune.h"

#include <algorithm>
#include <bit>
#include <cstddef>

#include <fmt/format.h>

#include "comms/ctran/algos/CtranAlgoConsts.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

namespace ctran::allreduce::ring {

namespace {

inline size_t getArchMaxBDP(GpuArch arch) {
  switch (arch) {
    case GpuArch::Hopper:
      return kHopperMaxBDP;
    default:
      return kDefaultMaxBDP;
  }
}

// Round n to the nearest power of 2. Ties (exact midpoint) round up.
// Returns 1 for n <= 1.
size_t roundToNearestPow2(size_t n) {
  if (n <= 1) {
    return 1;
  }
  if (std::has_single_bit(n)) {
    return n; // already pow2
  }
  // floor pow2: clear all but the highest set bit
  int bits = std::countl_zero(n);
  size_t lo = size_t{1} << (sizeof(size_t) * 8 - 1 - bits);
  size_t hi = lo << 1;
  return (n - lo < hi - n) ? lo : hi;
}

PipelineParams
getAutoTunedPipeline(size_t messageBytes, int nRanks, GpuArch arch) {
  size_t maxBDP = getArchMaxBDP(arch);
  if (NCCL_CTRAN_ALLREDUCE_RING_AUTO_TUNE_MAX_BDP > 0) {
    maxBDP = static_cast<size_t>(NCCL_CTRAN_ALLREDUCE_RING_AUTO_TUNE_MAX_BDP);
  }

  size_t partitionMessageBytes = roundToNearestPow2(messageBytes);
  while (partitionMessageBytes > maxBDP) {
    partitionMessageBytes /= 2;
  }

  static constexpr size_t kMinChunkSize = 256 * 1024; // 256KB
  static constexpr size_t kMaxChunkSize = 16ULL * 1024 * 1024; // 16MB

  static constexpr size_t k1M = 1ULL * 1024 * 1024;
  static constexpr size_t k4M = 4ULL * 1024 * 1024;
  const size_t perRankMessageBytes = roundToNearestPow2(messageBytes) / nRanks;
  // prioritize pipeline depth more when message sizes are larger
  // at medium message sizes, this helps to push smaller chunks out faster.
  // at larger message sizes, this also helps to smooth out the pipeline when
  // chunksize is already large enough.
  const size_t kPipelineDepth =
      (perRankMessageBytes < k1M) ? 2 : ((perRankMessageBytes < k4M) ? 4 : 2);

  // within partition
  size_t numChunks = kPipelineDepth * static_cast<size_t>(nRanks);
  size_t chunkSize = partitionMessageBytes / numChunks;
  chunkSize = std::clamp(chunkSize, kMinChunkSize, kMaxChunkSize);

  // Ensure we don't exceed allocated buffer, reduce chunkSize first, then
  // numChunks
  size_t bdp = chunkSize * numChunks;
  while (bdp > maxBDP && ((chunkSize / 2) >= kMinChunkSize)) {
    chunkSize /= 2;
    bdp = chunkSize * numChunks;
  }
  while (bdp > maxBDP && ((numChunks / 2) >= 1)) {
    numChunks /= 2;
    bdp = chunkSize * numChunks;
  }

  if (bdp > maxBDP) {
    static constexpr int kTenMinutes = 10 * 60 * 1000;
    CLOGF_EVERY_MS(
        WARN,
        kTenMinutes,
        fmt::format(
            "AutoTune error: nRanks {}, maxBDP {} < numChunks {} x {}B",
            nRanks,
            maxBDP,
            numChunks,
            chunkSize));
    return {kMaxChunkSize, static_cast<size_t>(nRanks)};
  }

  // if message size is too small, report smaller chunksize, same for numChunks
  chunkSize = std::min(partitionMessageBytes, chunkSize);
  numChunks =
      std::max((partitionMessageBytes + chunkSize - 1) / chunkSize, 1UL);

  return {chunkSize, numChunks};
}

BlockParams getAutoTunedBlockParams(
    size_t chunkSize,
    int maxOccupancyBlocks,
    int defaultThreads,
    GpuArch arch) {
  // Lookup table: {exclusive chunkSize upper bound, numBlocks, blockSize}.
  // blockSize == 0 means use defaultThreads (Default arch pass-through).
  struct Tier {
    size_t upTo;
    int numBlocks;
    int blockSize;
  };

  // clang-format off
  static constexpr size_t kMax = ~size_t{0};
  static constexpr Tier kDefaultTiers[] = {
    { 8ULL * 1024,   1, 0},
    {32ULL * 1024,   2, 0},
    {64ULL * 1024,   4, 0},
    {kMax,           8, 0},
  };
  static constexpr Tier kHopperTiers[] = {
    { 16ULL * 1024,  1, 384},
    {128ULL * 1024,  1, 512},
    {512ULL * 1024,  2, 512},
    {kMax,           4, 512},
  };
  // clang-format on

  const auto* tiers = kDefaultTiers;
  auto nTiers = sizeof(kDefaultTiers) / sizeof(kDefaultTiers[0]);
  if (arch == GpuArch::Hopper) {
    tiers = kHopperTiers;
    nTiers = sizeof(kHopperTiers) / sizeof(kHopperTiers[0]);
  }

  for (size_t i = 0; i < nTiers; ++i) {
    if (chunkSize < tiers[i].upTo) {
      return {
          std::min(tiers[i].numBlocks, maxOccupancyBlocks),
          tiers[i].blockSize ? tiers[i].blockSize : defaultThreads};
    }
  }
  // Unreachable: kMax sentinel guarantees a match
  return {};
}

} // namespace

AutoTuneParams getAutoTunedParams(
    size_t messageBytes,
    int nRanks,
    int maxOccupancyBlocks,
    int defaultThreads,
    GpuArch arch) {
  auto p = getAutoTunedPipeline(messageBytes, nRanks, arch);
  if (NCCL_CTRAN_ALLREDUCE_RING_TMPBUF_NUM_CHUNKS > 0) {
    p.numChunks = NCCL_CTRAN_ALLREDUCE_RING_TMPBUF_NUM_CHUNKS;
  }
  if (NCCL_CTRAN_ALLREDUCE_RING_TMPBUF_CHUNK_SIZE > 0) {
    p.chunkSize = NCCL_CTRAN_ALLREDUCE_RING_TMPBUF_CHUNK_SIZE;
  }

  auto bp = getAutoTunedBlockParams(
      p.chunkSize, maxOccupancyBlocks, defaultThreads, arch);
  if (NCCL_CTRAN_ALLREDUCE_RING_MAX_NUM_THREAD_BLOCKS > 0) {
    bp.numBlocks = NCCL_CTRAN_ALLREDUCE_RING_MAX_NUM_THREAD_BLOCKS;
  }
  if (NCCL_CTRAN_ALLREDUCE_RING_THREAD_BLOCK_SIZE > 0) {
    bp.blockSize = NCCL_CTRAN_ALLREDUCE_RING_THREAD_BLOCK_SIZE;
  }

  return {p, bp};
}

void logAutoTuneDecisions(
    int nRanks,
    int maxOccupancyBlocks,
    int defaultThreads,
    GpuArch arch) {
  static constexpr int kPow2MaxExponent = 25; // 32GB
  static constexpr size_t kKB = 1024ULL;
  for (int i = 0; i <= kPow2MaxExponent; i++) {
    const size_t sz = (1 << i) * kKB;
    const auto at = getAutoTunedParams(
        sz, nRanks, maxOccupancyBlocks, defaultThreads, arch);
    CLOGF(
        DBG,
        "AutoTune ranks {}, msg {}B: blocks {}, chunks {} x {}B",
        nRanks,
        sz,
        at.block.numBlocks,
        at.pipeline.numChunks,
        at.pipeline.chunkSize);

    if (i != kPow2MaxExponent) {
      const size_t szNext = (1 << (i + 1)) * kKB;
      const size_t mid = (sz + szNext) / 2;
      const auto mat = getAutoTunedParams(
          mid, nRanks, maxOccupancyBlocks, defaultThreads, arch);
      CLOGF(
          DBG,
          "AutoTune ranks {}, msg {}B: blocks {}, chunks {} x {}B",
          nRanks,
          szNext,
          mat.block.numBlocks,
          mat.pipeline.numChunks,
          mat.pipeline.chunkSize);
    }
  }
}

} // namespace ctran::allreduce::ring
