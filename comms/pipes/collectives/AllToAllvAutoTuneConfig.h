// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
#define AUTOTUNE_HOST_DEVICE __host__ __device__
#else
#define AUTOTUNE_HOST_DEVICE
#endif

namespace comms::pipes {

// ── Init-time configs (apply globally, not per-msg-size) ──

struct NvlInitConfig {
  std::size_t nvlChunkSize = 16 * 1024; // 16KB (from 1x8 sweep)
  std::size_t nvlDataBufferSize = 1 * 1024 * 1024; // 1MB (from 1x8 sweep)
  int pipelineDepth = 2; // (from 1x8 sweep)
};

struct IbgdaInitConfig {
  std::size_t stagingBufferSize = 256 * 1024; // 256KB (from 2x8 sweep)
  int pipelineDepth = 4; // (from 2x8 sweep)
  std::size_t ibgdaChunkSize = 64 * 1024; // 64KB (from 2x8 sweep)
};

// ── NVLink-only per-message-size config ──

struct NvlPerMsgConfig {
  std::size_t maxMsgSize; // upper bound for this config (inclusive)
  int numBlocks;
  int numThreads;
};

// ── IBGDA+NVLink hybrid per-message-size config ──

struct HybridPerMsgConfig {
  std::size_t maxMsgSize; // upper bound for this config (inclusive)
  int numBlocks;
  int numThreads;
  int nvlSendWarps; // 0 = auto
  int nvlRecvWarps; // 0 = auto
  int ibgdaSendWarps; // 0 = auto
  int ibgdaRecvWarps; // 0 = auto
  int selfWarps; // 0 = auto
};

// ── Default lookup tables (populated from sweep results) ──
//
// Sorted by maxMsgSize ascending. Lookup finds the first entry where
// maxMsgSize >= requested size (same pattern as Ll128AutoTune.cuh).
//
// NVL defaults are derived from AllToAllvBenchmark.cc empirical results:
//   - Small messages: fewer blocks sufficient
//   - 256KB+: 8-16 blocks matches NCCL channel count
//   - 512MB+: 32 blocks optimal
//
// Hybrid defaults use 1 block (IBGDA benchmark shows each block-doubling
// doubles latency); warp reserve 0 = auto-compute from peer counts.

// ── Maximum grid dimensions across all autotune lookup tables ──
//
// The physical launch grid must be at least this large so that the
// kernel-side autotune effective warp count is never capped by the
// launch parameters. These values are the max of {numBlocks, numThreads}
// across all entries in kDefaultNvlConfigs and kDefaultHybridConfigs.
constexpr int kMaxAutotuneBlocks = 64;
constexpr int kMaxAutotuneThreads = 512;

// ── Lookup functions ──
//
// Linear scan over the sorted table. Returns the first entry where
// maxMsgSize >= bytes. If bytes exceeds all entries, the last entry
// is returned (largest config is always a safe fallback).
//
// Tables are defined as local constexpr arrays inside each function so
// they are accessible from both host and device code. NVCC does not
// make namespace-scope static constexpr arrays available in __device__
// functions (--expt-relaxed-constexpr only covers constexpr functions).

AUTOTUNE_HOST_DEVICE inline NvlPerMsgConfig getNvlConfigForMsgSize(
    std::size_t bytes) {
  // clang-format off
  constexpr NvlPerMsgConfig kConfigs[] = {
      // From 1x8 H100 NVLink sweep (chunk=16KB pipe=2 buf=1MB)
      {1 * 1024,               8,  512},  // ≤1KB   1.48x vs NCCL
      {2 * 1024,               4,  512},  // ≤2KB   1.32x vs NCCL
      {4 * 1024,               4,  512},  // ≤4KB   1.38x vs NCCL
      {8 * 1024,               8,  128},  // ≤8KB   1.24x vs NCCL
      {16 * 1024,              8,  128},  // ≤16KB  1.10x vs NCCL
      {32 * 1024,             16,  128},  // ≤32KB  1.03x vs NCCL
      {64 * 1024,             16,  128},  // ≤64KB  1.03x vs NCCL
      {128 * 1024,            16,  256},  // ≤128KB 1.16x vs NCCL
      {256 * 1024,            64,  128},  // ≤256KB 1.20x vs NCCL
      {512 * 1024,            64,  256},  // ≤512KB 1.54x vs NCCL
      {1 * 1024 * 1024,       64,  512},  // ≤1MB   1.13x vs NCCL
      {2 * 1024 * 1024,       64,  512},  // ≤2MB   1.20x vs NCCL
      {4 * 1024 * 1024,       64,  512},  // ≤4MB   1.18x vs NCCL
      {8 * 1024 * 1024,       64,  512},  // ≤8MB   1.09x vs NCCL
      {16 * 1024 * 1024,      64,  512},  // ≤16MB  1.04x vs NCCL
      {32 * 1024 * 1024,      64,  512},  // ≤32MB  1.00x vs NCCL
      {64 * 1024 * 1024,      64,  512},  // ≤64MB  1.00x vs NCCL
      {128 * 1024 * 1024,     64,  512},  // ≤128MB 0.95x vs NCCL
      {256 * 1024 * 1024,     64,  512},  // ≤256MB 0.92x vs NCCL
  };
  // clang-format on
  constexpr std::size_t kNum = sizeof(kConfigs) / sizeof(kConfigs[0]);
  for (std::size_t i = 0; i < kNum; ++i) {
    if (kConfigs[i].maxMsgSize >= bytes) {
      return kConfigs[i];
    }
  }
  return kConfigs[kNum - 1];
}

AUTOTUNE_HOST_DEVICE inline HybridPerMsgConfig getHybridConfigForMsgSize(
    std::size_t bytes) {
  // clang-format off
  constexpr HybridPerMsgConfig kConfigs[] = {
      // From 2x8 H100 hybrid sweep (ibStagBuf=256KB ibPipe=4 ibChunk=64KB nvlChunk=2MB)
      {1 * 1024,              16, 256, 0, 0, 0, 0, 0},  // ≤1KB   1.20x vs NCCL
      {2 * 1024,              16, 128, 0, 0, 0, 0, 0},  // ≤2KB   1.60x vs NCCL
      {4 * 1024,               8, 256, 0, 0, 0, 0, 0},  // ≤4KB   1.51x vs NCCL
      {8 * 1024,               8, 256, 0, 0, 0, 0, 0},  // ≤8KB   1.48x vs NCCL
      {16 * 1024,              8, 256, 0, 0, 0, 0, 0},  // ≤16KB  1.46x vs NCCL
      {32 * 1024,             16, 128, 0, 0, 0, 0, 0},  // ≤32KB  1.46x vs NCCL
      {64 * 1024,             16, 128, 0, 0, 0, 0, 0},  // ≤64KB  1.54x vs NCCL
      {128 * 1024,            16, 128, 0, 0, 0, 0, 0},  // ≤128KB 1.47x vs NCCL
      {256 * 1024,            16, 128, 0, 0, 0, 0, 0},  // ≤256KB 1.25x vs NCCL
      {512 * 1024,            64, 128, 0, 0, 0, 0, 0},  // ≤512KB 1.36x vs NCCL
      {1 * 1024 * 1024,       16, 128, 0, 0, 0, 0, 0},  // ≤1MB   1.08x vs NCCL
      {2 * 1024 * 1024,       32, 128, 0, 0, 0, 0, 0},  // ≤2MB   1.12x vs NCCL
      {4 * 1024 * 1024,       16, 128, 0, 0, 0, 0, 0},  // ≤4MB   1.14x vs NCCL
      {8 * 1024 * 1024,       16, 128, 0, 0, 0, 0, 0},  // ≤8MB   1.12x vs NCCL
      {16 * 1024 * 1024,      16, 128, 0, 0, 0, 0, 0},  // ≤16MB  1.09x vs NCCL
      {32 * 1024 * 1024,      32, 128, 0, 0, 0, 0, 0},  // ≤32MB  1.09x vs NCCL
      {64 * 1024 * 1024,      32, 128, 0, 0, 0, 0, 0},  // ≤64MB  1.09x vs NCCL
      {128 * 1024 * 1024,     32, 128, 0, 0, 0, 0, 0},  // ≤128MB 1.09x vs NCCL
      {256 * 1024 * 1024,     32, 128, 0, 0, 0, 0, 0},  // ≤256MB 1.09x vs NCCL
  };
  // clang-format on
  constexpr std::size_t kNum = sizeof(kConfigs) / sizeof(kConfigs[0]);
  for (std::size_t i = 0; i < kNum; ++i) {
    if (kConfigs[i].maxMsgSize >= bytes) {
      return kConfigs[i];
    }
  }
  return kConfigs[kNum - 1];
}

// ── Top-level config holder ──
//
// Init-time configs with in-code defaults. CVAR overrides are applied
// in ctranInitializePipes() on top of these defaults.

struct AllToAllvAutoTuneConfig {
  NvlInitConfig nvlInit;
  IbgdaInitConfig ibgdaInit;
};

} // namespace comms::pipes
