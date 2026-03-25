// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>
#include <optional>

#include "comms/ctran/backends/CtranCtrl.h"

namespace ctran::allreduce::ring {

enum class GpuArch {
  Default, // GB200 / Blackwell (Phase 2 tuning)
  Hopper, // H100 / SM 9.0
};

struct PipelineParams {
  size_t chunkSize;
  size_t numChunks;
};

struct BlockParams {
  int numBlocks;
  int blockSize;
};

struct AutoTuneParams {
  PipelineParams pipeline;
  BlockParams block;
};

// Combined auto-tune: pipeline chunking + block/thread selection.
//
// Pipeline stage: auto-tunes chunkSize and numChunks based on message size,
// nRanks, and maxBDP. Values satisfy chunkSize * numChunks <= maxBDP.
//
// Block stage: auto-tunes numBlocks and blockSize based on chunkSize and arch.
//
// CVAR overrides (applied after auto-tune, highest priority):
//   TMPBUF_CHUNK_SIZE, TMPBUF_NUM_CHUNKS override pipeline params.
//   MAX_NUM_THREAD_BLOCKS caps numBlocks (upper bound, does not inflate).
//   THREAD_BLOCK_SIZE overrides blockSize (throws if > max occupancy).
//   Chunk size override feeds into block params computation.
AutoTuneParams getAutoTunedParams(
    size_t messageBytes,
    int nRanks,
    int maxOccupancyNumBlocks,
    int maxOccupancyBlockSize,
    size_t typeSize,
    GpuArch arch = GpuArch::Default);

// Log of auto-tune decisions for pow2 message sizes from 1KB to 32GB.
void logAutoTuneDecisions(
    int nRanks,
    int maxOccupancyNumBlocks,
    int maxOccupancyBlockSize,
    size_t typeSize,
    GpuArch arch = GpuArch::Default);

// Resolve IB config: explicit ALGO cvar > autotune-derived (Blackwell+) >
// nullopt.
std::optional<CtranIbConfig>
resolveIbConfig(CtranIbConfig* explicitConfig, GpuArch arch, size_t chunkSize);

} // namespace ctran::allreduce::ring
