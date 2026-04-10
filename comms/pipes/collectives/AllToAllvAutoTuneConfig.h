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
  std::size_t nvlChunkSize = 32 * 1024; // 32KB (from 1x8 exhaustive sweep)
  std::size_t nvlDataBufferSize =
      8 * 1024 * 1024; // 8MB (from 1x8 exhaustive sweep)
  int pipelineDepth = 4; // (from 1x8 exhaustive sweep)
};

struct IbgdaInitConfig {
  std::size_t stagingBufferSize =
      256 * 1024; // 256KB (from 2x8 exhaustive sweep)
  int pipelineDepth = 4; // (from 2x8 exhaustive sweep)
  std::size_t ibgdaChunkSize = 64 * 1024; // 64KB (from 2x8 exhaustive sweep)
};

// ── NVLink-only per-message-size config ──

struct NvlPerMsgConfig {
  std::size_t maxMsgSize;
  int numBlocks;
  int numThreads;
};

// ── IBGDA+NVLink hybrid per-message-size config ──

struct HybridPerMsgConfig {
  std::size_t maxMsgSize;
  int numBlocks;
  int numThreads;
  int nvlSendWarps; // 0 = auto
  int nvlRecvWarps; // 0 = auto
  int ibgdaSendWarps; // 0 = auto
  int ibgdaRecvWarps; // 0 = auto
  int selfWarps; // 0 = auto
};

// ── Maximum grid dimensions across all autotune lookup tables ──
//
// The physical launch grid must be at least this large so that the
// kernel-side autotune effective warp count is never capped by the
// launch parameters. These values are the max of {numBlocks, numThreads}
// across all entries in kDefaultNvlConfigs and kDefaultHybridConfigs.
constexpr int kMaxAutotuneBlocks = 64;
constexpr int kMaxAutotuneThreads = 256;

// ── Default lookup tables (populated from exhaustive sweep results) ──
//
// Sorted by maxMsgSize ascending. Lookup finds the first entry where
// maxMsgSize >= requested size.
//
// The array data is defined via macros so it can also be embedded as
// function-local constexpr arrays inside __host__ __device__ lookup
// functions (namespace-scope constexpr arrays are not visible in CUDA
// device code).

// clang-format off

// From 1x8 H100 NVLink exhaustive sweep (chunk=32KB pipe=4 buf=8MB)
// Format: {maxMsgSize, numBlocks, numThreads}
#define PIPES_NVL_CONFIG_ENTRIES                                 \
    {1 * 1024,               4,  128},  /* 1KB   1.39x */       \
    {2 * 1024,               4,  128},  /* 2KB   1.11x */       \
    {4 * 1024,               4,  128},  /* 4KB   1.34x */       \
    {8 * 1024,               4,  128},  /* 8KB   1.21x */       \
    {16 * 1024,             64,  128},  /* 16KB  1.34x */       \
    {32 * 1024,             32,  128},  /* 32KB  0.90x */       \
    {64 * 1024,             32,  128},  /* 64KB  0.88x */       \
    {128 * 1024,            16,  128},  /* 128KB 1.17x */       \
    {256 * 1024,            64,  128},  /* 256KB 1.11x */       \
    {512 * 1024,            64,  128},  /* 512KB 1.45x */       \
    {1 * 1024 * 1024,       64,  256},  /* 1MB   1.10x */       \
    {2 * 1024 * 1024,       64,  256},  /* 2MB   1.17x */       \
    {4 * 1024 * 1024,       64,  256},  /* 4MB   1.12x */       \
    {8 * 1024 * 1024,       64,  256},  /* 8MB   1.10x */       \
    {16 * 1024 * 1024,      64,  256},  /* 16MB  1.02x */       \
    {32 * 1024 * 1024,      64,  256},  /* 32MB  0.99x */       \
    {64 * 1024 * 1024,      64,  256},  /* 64MB  0.97x */       \
    {128 * 1024 * 1024,     64,  256},  /* 128MB 0.93x */       \
    {256 * 1024 * 1024,     64,  256}   /* 256MB 0.92x */

// From 2x8 H100 hybrid exhaustive sweep
// (ibStagBuf=256KB ibPipe=4 ibChunk=64KB nvlChunk=32KB nvlBuf=2MB nvlPipe=2)
// Format: {maxMsgSize, numBlocks, numThreads,
//          nvlSendWarps, nvlRecvWarps, ibgdaSendWarps, ibgdaRecvWarps, selfWarps}
#define PIPES_HYBRID_CONFIG_ENTRIES                                            \
    {1 * 1024,              16, 128,  0,  0,  0,  0, 0},  /* 1KB   1.50x */   \
    {2 * 1024,              16, 128,  0,  0,  0,  0, 0},  /* 2KB   1.40x */   \
    {4 * 1024,              64, 128,  0,  0,  0,  0, 0},  /* 4KB   1.40x */   \
    {8 * 1024,              32, 256,  0,  0,  0,  0, 0},  /* 8KB   1.43x */   \
    {16 * 1024,             64, 128,  0,  0,  0,  0, 0},  /* 16KB  1.41x */   \
    {32 * 1024,             64, 256,  0,  0,  0,  0, 0},  /* 32KB  1.40x */   \
    {64 * 1024,             16, 128,  0,  0,  0,  0, 0},  /* 64KB  1.48x */   \
    {128 * 1024,            16, 128,  0,  0,  0,  0, 0},  /* 128KB 1.41x */   \
    {256 * 1024,            16, 128,  0,  0,  0,  0, 0},  /* 256KB 1.15x */   \
    {512 * 1024,            64, 128,  0,  0,  0,  0, 0},  /* 512KB 1.26x */   \
    {1 * 1024 * 1024,       32, 128,  0,  0,  0,  0, 0},  /* 1MB   0.99x */   \
    {2 * 1024 * 1024,       32, 128,  0,  0,  0,  0, 0},  /* 2MB   1.05x */   \
    {4 * 1024 * 1024,       64, 128,  0,  0,  0,  0, 0},  /* 4MB   1.07x */   \
    {8 * 1024 * 1024,       64, 128,  0,  0,  0,  0, 0},  /* 8MB   1.04x */   \
    {16 * 1024 * 1024,      64, 128,  0,  0,  0,  0, 0},  /* 16MB  1.03x */   \
    {32 * 1024 * 1024,      64, 128,  0,  0,  0,  0, 0},  /* 32MB  1.04x */   \
    {64 * 1024 * 1024,      64, 128,  0,  0,  0,  0, 0},  /* 64MB  1.05x */   \
    {128 * 1024 * 1024,     64, 128,  0,  0,  0,  0, 0},  /* 128MB 1.05x */   \
    {256 * 1024 * 1024,     64, 128,  0,  0,  0,  0, 0}   /* 256MB 1.06x */

// Host-visible namespace-scope arrays (used by getMaxIbgdaChannelsPerPeer
// and other host-only code).
constexpr NvlPerMsgConfig kDefaultNvlConfigs[] = {
    PIPES_NVL_CONFIG_ENTRIES
};
constexpr std::size_t kNumNvlConfigs =
    sizeof(kDefaultNvlConfigs) / sizeof(kDefaultNvlConfigs[0]);

constexpr HybridPerMsgConfig kDefaultHybridConfigs[] = {
    PIPES_HYBRID_CONFIG_ENTRIES
};
constexpr std::size_t kNumHybridConfigs =
    sizeof(kDefaultHybridConfigs) / sizeof(kDefaultHybridConfigs[0]);

// Device-accessible copies of the config arrays. Namespace-scope constexpr
// arrays have no device linkage in CUDA, so we maintain separate __device__
// copies in device global memory (not on the per-thread stack).
#ifdef __CUDACC__
static __device__ constexpr NvlPerMsgConfig kDeviceNvlConfigs[] = {
    PIPES_NVL_CONFIG_ENTRIES
};
static __device__ constexpr HybridPerMsgConfig kDeviceHybridConfigs[] = {
    PIPES_HYBRID_CONFIG_ENTRIES
};
#endif
// clang-format on

// ── Lookup functions ──
//
// These are __host__ __device__ so the ctran unified kernel can call them.
// In device code we use the __device__ arrays above; on the host we use
// the constexpr namespace-scope arrays.

AUTOTUNE_HOST_DEVICE inline NvlPerMsgConfig getNvlConfigForMsgSize(
    std::size_t bytes) {
#ifdef __CUDA_ARCH__
  const auto* configs = kDeviceNvlConfigs;
#else
  const auto* configs = kDefaultNvlConfigs;
#endif
  for (std::size_t i = 0; i < kNumNvlConfigs; ++i) {
    if (configs[i].maxMsgSize >= bytes) {
      return configs[i];
    }
  }
  return configs[kNumNvlConfigs - 1];
}

AUTOTUNE_HOST_DEVICE inline HybridPerMsgConfig getHybridConfigForMsgSize(
    std::size_t bytes) {
#ifdef __CUDA_ARCH__
  const auto* configs = kDeviceHybridConfigs;
#else
  const auto* configs = kDefaultHybridConfigs;
#endif
  for (std::size_t i = 0; i < kNumHybridConfigs; ++i) {
    if (configs[i].maxMsgSize >= bytes) {
      return configs[i];
    }
  }
  return configs[kNumHybridConfigs - 1];
}

// ── Multi-channel derivation ──
//
// Scans the hybrid autotune table to derive the maximum number of IBGDA
// channels per peer. Used at setup time to size per-channel signal and
// counter buffers. Returns 1 (single channel) if no entry has explicit
// ibgdaSendWarps.

constexpr int getMaxIbgdaChannelsPerPeer(int numIbgdaPeers) {
  if (numIbgdaPeers <= 0) {
    return 1;
  }
  int maxCh = 1;
  for (std::size_t i = 0; i < kNumHybridConfigs; ++i) {
    if (kDefaultHybridConfigs[i].ibgdaSendWarps > 0) {
      int ch = kDefaultHybridConfigs[i].ibgdaSendWarps / numIbgdaPeers;
      if (ch > maxCh) {
        maxCh = ch;
      }
    }
  }
  return maxCh;
}

// ── Top-level config holder ──

struct AllToAllvAutoTuneConfig {
  NvlInitConfig nvlInit;
  IbgdaInitConfig ibgdaInit;
};

} // namespace comms::pipes
