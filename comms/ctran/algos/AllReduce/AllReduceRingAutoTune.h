// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstddef>

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
//   MAX_NUM_THREAD_BLOCKS, THREAD_BLOCK_SIZE override block params.
//   Chunk size override feeds into block params computation.
AutoTuneParams getAutoTunedParams(
    size_t messageBytes,
    int nRanks,
    int maxOccupancyBlocks,
    int defaultThreads,
    GpuArch arch = GpuArch::Default);

// Log of auto-tune decisions for pow2 message sizes from 1KB to 32GB.
void logAutoTuneDecisions(
    int nRanks,
    int maxOccupancyBlocks,
    int defaultThreads,
    GpuArch arch = GpuArch::Default);

} // namespace ctran::allreduce::ring
