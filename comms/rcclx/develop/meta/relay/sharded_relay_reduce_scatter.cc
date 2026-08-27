/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "sharded_relay_reduce_scatter.h"
#include "comm.h"
#include "sharded_relay_allreduce_kernels.h"
#include "sharded_relay_route.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <map>
#include <mutex>
#include <tuple>

// GPU memory access alignment in elements for chunk size rounding.
// Distinct from the CPU CACHE_LINE_SIZE (64 bytes) defined in comm.h
// which is used for struct padding.
static constexpr size_t CHUNK_ALIGN_ELEMENTS =
    rcclx::relay::kRelayChunkAlignElements;

// Infra below (ScratchBufferCache, rank-config builder, DISPATCH macros) is a
// deliberate copy of the file-local helpers in sharded_relay_allreduce.cc.
// They are file-local (static / anonymous namespace) there, so they cannot be
// linked across translation units; the reduce-scatter TU re-declares its own
// copies in an anonymous namespace to keep them internal and ODR-safe. The
// GPU kernels themselves are NOT duplicated — they are reused via
// sharded_relay_allreduce_kernels.h (dtype-generic, collective-agnostic).
namespace {

/**
 * Scratch Buffer Cache Singleton
 *
 * Amortizes cudaMalloc/cudaFree costs by caching and reusing scratch buffers.
 * Thread-safe, with one buffer per (device, stream, key). See the allreduce
 * copy for a full description; this is an independent cache scoped to
 * reduce-scatter.
 */
class ScratchBufferCache {
 public:
  static ScratchBufferCache& getInstance() {
    static ScratchBufferCache instance;
    return instance;
  }

  void* get(int key, size_t requiredBytes, cudaStream_t stream) {
    if (requiredBytes == 0) {
      return nullptr;
    }

    int device;
    cudaGetDevice(&device);

    std::lock_guard<std::mutex> lock(mutex_);

    // Keyed by (device, stream, key). The stream is part of the key because two
    // relay collectives can run concurrently on one device on different streams
    // (independent communicators do exactly this): sharing one staging buffer
    // between them corrupts both. It also makes the stream-ordered free below
    // safe -- an entry is only ever read or written by the stream that owns it.
    auto& entry = buffers_[std::make_tuple(
        device, static_cast<const void*>(stream), key)];

    if (entry.buffer == nullptr || entry.size < requiredBytes) {
      if (entry.buffer != nullptr) {
        cudaFreeAsync(entry.buffer, stream);
      }

      size_t allocSize = requiredBytes;
      if (allocSize >= 1024 * 1024) {
        allocSize =
            ((requiredBytes + 64 * 1024 * 1024 - 1) / (64 * 1024 * 1024)) *
            (64 * 1024 * 1024);
      }

      cudaError_t err = cudaMallocAsync(&entry.buffer, allocSize, stream);
      if (err != cudaSuccess) {
        entry.buffer = nullptr;
        entry.size = 0;
        return nullptr;
      }
      entry.size = allocSize;
    }

    return entry.buffer;
  }

  void clear(cudaStream_t stream = nullptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    // Each buffer is freed on the stream that owns it (from the key), not on
    // the caller's stream: a stream-ordered free on an unrelated stream would
    // not be ordered against the owner's pending work.
    (void)stream;
    for (auto& pair : buffers_) {
      if (pair.second.buffer != nullptr) {
        cudaFreeAsync(
            pair.second.buffer,
            static_cast<cudaStream_t>(
                const_cast<void*>(std::get<1>(pair.first))));
        pair.second.buffer = nullptr;
        pair.second.size = 0;
      }
    }
    buffers_.clear();
  }

  ScratchBufferCache(const ScratchBufferCache&) = delete;
  ScratchBufferCache& operator=(const ScratchBufferCache&) = delete;

 private:
  ScratchBufferCache() = default;
  ~ScratchBufferCache() = default;

  struct BufferEntry {
    void* buffer = nullptr;
    size_t size = 0;
  };

  std::mutex mutex_;
  // (device, stream, group) -> grow-only staging buffer.
  std::map<std::tuple<int, const void*, int>, BufferEntry> buffers_;
};

// Maximum number of helper ranks supported per group.
constexpr int SHARDED_RELAY_MAX_HELPERS = 8;

// Maximum number of active ranks per group. The recursive-halving relay
// schedule (round-r partner = myActiveIndex XOR round) requires nActiveRanks to
// be a power of two; supported values are 2 and 4 (on an 8-GPU node this leaves
// 6 or 4 helpers respectively).
constexpr int SHARDED_RELAY_MAX_ACTIVE = 8;

// Returns true if v is a power of two (v >= 1).
inline bool isPowerOfTwo(int v) {
  return v > 0 && (v & (v - 1)) == 0;
}

// Reverse the low `bits` bits of x. The recursive-halving reduce-scatter leaves
// active rank mi owning the segment at bit-reversed position bitReverse(mi);
// the reduce-scatter gather places block[j] at bit-reversed position so owner j
// ends up holding the reduced block[j].
inline int bitReverse(int x, int bits) {
  int r = 0;
  for (int b = 0; b < bits; b++) {
    r = (r << 1) | ((x >> b) & 1);
  }
  return r;
}

/**
 * Rank Configuration for Sharded Relay Reduce-Scatter
 *
 * Holds parsed active and helper rank information for a single group.
 * Supports a power-of-two number of active ranks per group (2 or 4).
 */
struct ShardedRelayRankConfig {
  int activeRanks[SHARDED_RELAY_MAX_ACTIVE]; // Active rank IDs (power of two)
  int nActiveRanks; // Number of active ranks (2 or 4)
  int helperRanks[SHARDED_RELAY_MAX_HELPERS]; // Helper rank IDs
  int numHelpers; // Number of helper ranks
  bool isActiveRank; // Is current rank active?
  int myActiveIndex; // Index in activeRanks array (-1 if helper)
  int myHelperIndex; // Index in helperRanks array (-1 if active)
};

/**
 * Build rank configuration from provided active ranks array.
 *
 * Requires a power-of-two active-rank count in [2, SHARDED_RELAY_MAX_ACTIVE];
 * the XOR round schedule of the A>2 recursive path depends on it.
 */
bool buildShardedRelayRankConfig(
    int nRanks,
    int rank,
    const int* activeRanksInput,
    int nActiveRanksInput,
    ShardedRelayRankConfig& config) {
  config.nActiveRanks = 0;
  config.numHelpers = 0;
  config.isActiveRank = false;
  config.myActiveIndex = -1;
  config.myHelperIndex = -1;

  // Validate input - require a power-of-two active-rank count in
  // [2, SHARDED_RELAY_MAX_ACTIVE]. The XOR round schedule depends on it.
  if (activeRanksInput == nullptr || nActiveRanksInput < 2 ||
      nActiveRanksInput > SHARDED_RELAY_MAX_ACTIVE ||
      !isPowerOfTwo(nActiveRanksInput)) {
    return false;
  }

  // Copy active ranks and validate
  for (int i = 0; i < nActiveRanksInput; i++) {
    int rankId = activeRanksInput[i];
    if (rankId >= 0 && rankId < nRanks) {
      config.activeRanks[config.nActiveRanks++] = rankId;
    }
  }

  // Validate: need exactly nActiveRanksInput valid active ranks
  if (config.nActiveRanks != nActiveRanksInput) {
    return false;
  }

  // Build list of helper ranks (all ranks NOT in activeRanks).
  for (int r = 0; r < nRanks; r++) {
    bool isActive = false;
    for (int a = 0; a < config.nActiveRanks; a++) {
      if (r == config.activeRanks[a]) {
        isActive = true;
        break;
      }
    }
    if (!isActive) {
      if (config.numHelpers >= SHARDED_RELAY_MAX_HELPERS) {
        return false;
      }
      config.helperRanks[config.numHelpers++] = r;
    }
  }

  // Validate: need at least 1 helper
  if (config.numHelpers < 1) {
    return false;
  }

  // Determine if this rank is active
  for (int a = 0; a < config.nActiveRanks; a++) {
    if (rank == config.activeRanks[a]) {
      config.isActiveRank = true;
      config.myActiveIndex = a;
      break;
    }
  }

  // For helpers, determine which chunk index this rank handles
  if (!config.isActiveRank) {
    for (int i = 0; i < config.numHelpers; i++) {
      if (config.helperRanks[i] == rank) {
        config.myHelperIndex = i;
        break;
      }
    }
  }
  return true;
}

} // namespace

// Host-side dispatch macros for the reused generic kernels. These mirror the
// file-local DISPATCH_* macros in sharded_relay_allreduce.cc (the kernels they
// launch are declared in sharded_relay_allreduce_kernels.h).

#define LAUNCH_INCREMENTAL_ADD_KERNEL(TYPE, output, input, count, stream) \
  launchIncrementalAddKernel<TYPE>(output, input, count, stream)

#define DISPATCH_INCREMENTAL_ADD(datatype, output, input, count, stream)       \
  do {                                                                         \
    switch (datatype) {                                                        \
      case ncclInt8:                                                           \
        LAUNCH_INCREMENTAL_ADD_KERNEL(int8_t, output, input, count, stream);   \
        break;                                                                 \
      case ncclUint8:                                                          \
        LAUNCH_INCREMENTAL_ADD_KERNEL(uint8_t, output, input, count, stream);  \
        break;                                                                 \
      case ncclInt32:                                                          \
        LAUNCH_INCREMENTAL_ADD_KERNEL(int32_t, output, input, count, stream);  \
        break;                                                                 \
      case ncclUint32:                                                         \
        LAUNCH_INCREMENTAL_ADD_KERNEL(uint32_t, output, input, count, stream); \
        break;                                                                 \
      case ncclInt64:                                                          \
        LAUNCH_INCREMENTAL_ADD_KERNEL(int64_t, output, input, count, stream);  \
        break;                                                                 \
      case ncclUint64:                                                         \
        LAUNCH_INCREMENTAL_ADD_KERNEL(uint64_t, output, input, count, stream); \
        break;                                                                 \
      case ncclFloat16:                                                        \
        LAUNCH_INCREMENTAL_ADD_KERNEL(__half, output, input, count, stream);   \
        break;                                                                 \
      case ncclFloat:                                                          \
        LAUNCH_INCREMENTAL_ADD_KERNEL(float, output, input, count, stream);    \
        break;                                                                 \
      case ncclDouble:                                                         \
        LAUNCH_INCREMENTAL_ADD_KERNEL(double, output, input, count, stream);   \
        break;                                                                 \
      case ncclBfloat16:                                                       \
        LAUNCH_INCREMENTAL_ADD_KERNEL(                                         \
            __nv_bfloat16, output, input, count, stream);                      \
        break;                                                                 \
      default:                                                                 \
        break;                                                                 \
    }                                                                          \
  } while (0)

#define LAUNCH_SCALE_KERNEL(TYPE, data, count, divisor, stream) \
  launchScaleKernel<TYPE>(data, count, divisor, stream)

#define DISPATCH_SCALE(datatype, data, count, divisor, stream)            \
  do {                                                                    \
    switch (datatype) {                                                   \
      case ncclInt8:                                                      \
        LAUNCH_SCALE_KERNEL(int8_t, data, count, divisor, stream);        \
        break;                                                            \
      case ncclUint8:                                                     \
        LAUNCH_SCALE_KERNEL(uint8_t, data, count, divisor, stream);       \
        break;                                                            \
      case ncclInt32:                                                     \
        LAUNCH_SCALE_KERNEL(int32_t, data, count, divisor, stream);       \
        break;                                                            \
      case ncclUint32:                                                    \
        LAUNCH_SCALE_KERNEL(uint32_t, data, count, divisor, stream);      \
        break;                                                            \
      case ncclInt64:                                                     \
        LAUNCH_SCALE_KERNEL(int64_t, data, count, divisor, stream);       \
        break;                                                            \
      case ncclUint64:                                                    \
        LAUNCH_SCALE_KERNEL(uint64_t, data, count, divisor, stream);      \
        break;                                                            \
      case ncclFloat16:                                                   \
        LAUNCH_SCALE_KERNEL(__half, data, count, divisor, stream);        \
        break;                                                            \
      case ncclFloat:                                                     \
        LAUNCH_SCALE_KERNEL(float, data, count, divisor, stream);         \
        break;                                                            \
      case ncclDouble:                                                    \
        LAUNCH_SCALE_KERNEL(double, data, count, divisor, stream);        \
        break;                                                            \
      case ncclBfloat16:                                                  \
        LAUNCH_SCALE_KERNEL(__nv_bfloat16, data, count, divisor, stream); \
        break;                                                            \
      default:                                                            \
        break;                                                            \
    }                                                                     \
  } while (0)

#define LAUNCH_INCREMENTAL_ADD_AND_SCALE_KERNEL( \
    TYPE, output, input, count, divisor, stream) \
  launchIncrementalAddAndScaleKernel<TYPE>(      \
      output, input, count, divisor, stream)

#define DISPATCH_INCREMENTAL_ADD_AND_SCALE(                        \
    datatype, output, input, count, divisor, stream)               \
  do {                                                             \
    switch (datatype) {                                            \
      case ncclInt8:                                               \
        LAUNCH_INCREMENTAL_ADD_AND_SCALE_KERNEL(                   \
            int8_t, output, input, count, divisor, stream);        \
        break;                                                     \
      case ncclUint8:                                              \
        LAUNCH_INCREMENTAL_ADD_AND_SCALE_KERNEL(                   \
            uint8_t, output, input, count, divisor, stream);       \
        break;                                                     \
      case ncclInt32:                                              \
        LAUNCH_INCREMENTAL_ADD_AND_SCALE_KERNEL(                   \
            int32_t, output, input, count, divisor, stream);       \
        break;                                                     \
      case ncclUint32:                                             \
        LAUNCH_INCREMENTAL_ADD_AND_SCALE_KERNEL(                   \
            uint32_t, output, input, count, divisor, stream);      \
        break;                                                     \
      case ncclInt64:                                              \
        LAUNCH_INCREMENTAL_ADD_AND_SCALE_KERNEL(                   \
            int64_t, output, input, count, divisor, stream);       \
        break;                                                     \
      case ncclUint64:                                             \
        LAUNCH_INCREMENTAL_ADD_AND_SCALE_KERNEL(                   \
            uint64_t, output, input, count, divisor, stream);      \
        break;                                                     \
      case ncclFloat16:                                            \
        LAUNCH_INCREMENTAL_ADD_AND_SCALE_KERNEL(                   \
            __half, output, input, count, divisor, stream);        \
        break;                                                     \
      case ncclFloat:                                              \
        LAUNCH_INCREMENTAL_ADD_AND_SCALE_KERNEL(                   \
            float, output, input, count, divisor, stream);         \
        break;                                                     \
      case ncclDouble:                                             \
        LAUNCH_INCREMENTAL_ADD_AND_SCALE_KERNEL(                   \
            double, output, input, count, divisor, stream);        \
        break;                                                     \
      case ncclBfloat16:                                           \
        LAUNCH_INCREMENTAL_ADD_AND_SCALE_KERNEL(                   \
            __nv_bfloat16, output, input, count, divisor, stream); \
        break;                                                     \
      default:                                                     \
        break;                                                     \
    }                                                              \
  } while (0)

#define LAUNCH_FUSED_REDUCE_KERNEL(                       \
    TYPE, output, inputA, inputB, count, divisor, stream) \
  launchFusedReduceKernel<TYPE>(output, inputA, inputB, count, divisor, stream)

#define DISPATCH_FUSED_REDUCE(                                              \
    datatype, output, inputA, inputB, count, divisor, stream)               \
  do {                                                                      \
    switch (datatype) {                                                     \
      case ncclInt8:                                                        \
        LAUNCH_FUSED_REDUCE_KERNEL(                                         \
            int8_t, output, inputA, inputB, count, divisor, stream);        \
        break;                                                              \
      case ncclUint8:                                                       \
        LAUNCH_FUSED_REDUCE_KERNEL(                                         \
            uint8_t, output, inputA, inputB, count, divisor, stream);       \
        break;                                                              \
      case ncclInt32:                                                       \
        LAUNCH_FUSED_REDUCE_KERNEL(                                         \
            int32_t, output, inputA, inputB, count, divisor, stream);       \
        break;                                                              \
      case ncclUint32:                                                      \
        LAUNCH_FUSED_REDUCE_KERNEL(                                         \
            uint32_t, output, inputA, inputB, count, divisor, stream);      \
        break;                                                              \
      case ncclInt64:                                                       \
        LAUNCH_FUSED_REDUCE_KERNEL(                                         \
            int64_t, output, inputA, inputB, count, divisor, stream);       \
        break;                                                              \
      case ncclUint64:                                                      \
        LAUNCH_FUSED_REDUCE_KERNEL(                                         \
            uint64_t, output, inputA, inputB, count, divisor, stream);      \
        break;                                                              \
      case ncclFloat16:                                                     \
        LAUNCH_FUSED_REDUCE_KERNEL(                                         \
            __half, output, inputA, inputB, count, divisor, stream);        \
        break;                                                              \
      case ncclFloat:                                                       \
        LAUNCH_FUSED_REDUCE_KERNEL(                                         \
            float, output, inputA, inputB, count, divisor, stream);         \
        break;                                                              \
      case ncclDouble:                                                      \
        LAUNCH_FUSED_REDUCE_KERNEL(                                         \
            double, output, inputA, inputB, count, divisor, stream);        \
        break;                                                              \
      case ncclBfloat16:                                                    \
        LAUNCH_FUSED_REDUCE_KERNEL(                                         \
            __nv_bfloat16, output, inputA, inputB, count, divisor, stream); \
        break;                                                              \
      default:                                                              \
        break;                                                              \
    }                                                                       \
  } while (0)

#define LAUNCH_MULTI_REDUCE_KERNEL(                           \
    TYPE, dst, contribs, numContribs, count, divisor, stream) \
  launchMultiReduceKernel<TYPE>(                              \
      dst, contribs, numContribs, count, divisor, stream)

// Fused multi-input reduce: dst = (dst + sum of `numContribs` contiguous
// contribution blocks) [/ divisor], in one launch. Replaces a loop of
// per-contribution incremental adds plus a trailing scale.
#define DISPATCH_MULTI_REDUCE(                                             \
    datatype, dst, contribs, numContribs, count, divisor, stream)          \
  do {                                                                     \
    switch (datatype) {                                                    \
      case ncclInt8:                                                       \
        LAUNCH_MULTI_REDUCE_KERNEL(                                        \
            int8_t, dst, contribs, numContribs, count, divisor, stream);   \
        break;                                                             \
      case ncclUint8:                                                      \
        LAUNCH_MULTI_REDUCE_KERNEL(                                        \
            uint8_t, dst, contribs, numContribs, count, divisor, stream);  \
        break;                                                             \
      case ncclInt32:                                                      \
        LAUNCH_MULTI_REDUCE_KERNEL(                                        \
            int32_t, dst, contribs, numContribs, count, divisor, stream);  \
        break;                                                             \
      case ncclUint32:                                                     \
        LAUNCH_MULTI_REDUCE_KERNEL(                                        \
            uint32_t, dst, contribs, numContribs, count, divisor, stream); \
        break;                                                             \
      case ncclInt64:                                                      \
        LAUNCH_MULTI_REDUCE_KERNEL(                                        \
            int64_t, dst, contribs, numContribs, count, divisor, stream);  \
        break;                                                             \
      case ncclUint64:                                                     \
        LAUNCH_MULTI_REDUCE_KERNEL(                                        \
            uint64_t, dst, contribs, numContribs, count, divisor, stream); \
        break;                                                             \
      case ncclFloat16:                                                    \
        LAUNCH_MULTI_REDUCE_KERNEL(                                        \
            __half, dst, contribs, numContribs, count, divisor, stream);   \
        break;                                                             \
      case ncclFloat:                                                      \
        LAUNCH_MULTI_REDUCE_KERNEL(                                        \
            float, dst, contribs, numContribs, count, divisor, stream);    \
        break;                                                             \
      case ncclDouble:                                                     \
        LAUNCH_MULTI_REDUCE_KERNEL(                                        \
            double, dst, contribs, numContribs, count, divisor, stream);   \
        break;                                                             \
      case ncclBfloat16:                                                   \
        LAUNCH_MULTI_REDUCE_KERNEL(                                        \
            __nv_bfloat16,                                                 \
            dst,                                                           \
            contribs,                                                      \
            numContribs,                                                   \
            count,                                                         \
            divisor,                                                       \
            stream);                                                       \
        break;                                                             \
      default:                                                             \
        break;                                                             \
    }                                                                      \
  } while (0)

#define LAUNCH_SEEDED_MULTI_REDUCE_KERNEL(                          \
    TYPE, dst, seed, contribs, numContribs, count, divisor, stream) \
  launchSeededMultiReduceKernel<TYPE>(                              \
      dst, seed, contribs, numContribs, count, divisor, stream)

#define DISPATCH_SEEDED_MULTI_REDUCE(                                          \
    datatype, dst, seed, contribs, numContribs, count, divisor, stream)        \
  do {                                                                         \
    switch (datatype) {                                                        \
      case ncclInt8:                                                           \
        LAUNCH_SEEDED_MULTI_REDUCE_KERNEL(                                     \
            int8_t, dst, seed, contribs, numContribs, count, divisor, stream); \
        break;                                                                 \
      case ncclUint8:                                                          \
        LAUNCH_SEEDED_MULTI_REDUCE_KERNEL(                                     \
            uint8_t,                                                           \
            dst,                                                               \
            seed,                                                              \
            contribs,                                                          \
            numContribs,                                                       \
            count,                                                             \
            divisor,                                                           \
            stream);                                                           \
        break;                                                                 \
      case ncclInt32:                                                          \
        LAUNCH_SEEDED_MULTI_REDUCE_KERNEL(                                     \
            int32_t,                                                           \
            dst,                                                               \
            seed,                                                              \
            contribs,                                                          \
            numContribs,                                                       \
            count,                                                             \
            divisor,                                                           \
            stream);                                                           \
        break;                                                                 \
      case ncclUint32:                                                         \
        LAUNCH_SEEDED_MULTI_REDUCE_KERNEL(                                     \
            uint32_t,                                                          \
            dst,                                                               \
            seed,                                                              \
            contribs,                                                          \
            numContribs,                                                       \
            count,                                                             \
            divisor,                                                           \
            stream);                                                           \
        break;                                                                 \
      case ncclInt64:                                                          \
        LAUNCH_SEEDED_MULTI_REDUCE_KERNEL(                                     \
            int64_t,                                                           \
            dst,                                                               \
            seed,                                                              \
            contribs,                                                          \
            numContribs,                                                       \
            count,                                                             \
            divisor,                                                           \
            stream);                                                           \
        break;                                                                 \
      case ncclUint64:                                                         \
        LAUNCH_SEEDED_MULTI_REDUCE_KERNEL(                                     \
            uint64_t,                                                          \
            dst,                                                               \
            seed,                                                              \
            contribs,                                                          \
            numContribs,                                                       \
            count,                                                             \
            divisor,                                                           \
            stream);                                                           \
        break;                                                                 \
      case ncclFloat16:                                                        \
        LAUNCH_SEEDED_MULTI_REDUCE_KERNEL(                                     \
            __half, dst, seed, contribs, numContribs, count, divisor, stream); \
        break;                                                                 \
      case ncclFloat:                                                          \
        LAUNCH_SEEDED_MULTI_REDUCE_KERNEL(                                     \
            float, dst, seed, contribs, numContribs, count, divisor, stream);  \
        break;                                                                 \
      case ncclDouble:                                                         \
        LAUNCH_SEEDED_MULTI_REDUCE_KERNEL(                                     \
            double, dst, seed, contribs, numContribs, count, divisor, stream); \
        break;                                                                 \
      case ncclBfloat16:                                                       \
        LAUNCH_SEEDED_MULTI_REDUCE_KERNEL(                                     \
            __nv_bfloat16,                                                     \
            dst,                                                               \
            seed,                                                              \
            contribs,                                                          \
            numContribs,                                                       \
            count,                                                             \
            divisor,                                                           \
            stream);                                                           \
        break;                                                                 \
      default:                                                                 \
        break;                                                                 \
    }                                                                          \
  } while (0)

namespace {

// The DISPATCH_* macros above instantiate reduce kernels for exactly these
// types and fall through silently (default: break) for anything else, so an
// unsupported datatype would return ncclSuccess having never reduced anything.
//
// This is deliberately a supported-set test rather than the
// `datatype < 0 || datatype >= ncclNumTypes` range test used by upstream
// ArgsCheck: ncclFloat8e4m3 and ncclFloat8e5m2 are valid NCCL types that
// ncclTypeSize() sizes at 1 byte, so they pass a range test but have no reduce
// kernel here. Keep this list in sync with the DISPATCH_* macros above.
bool isSupportedRelayDataType(ncclDataType_t datatype) {
  switch (datatype) {
    case ncclInt8:
    case ncclUint8:
    case ncclInt32:
    case ncclUint32:
    case ncclInt64:
    case ncclUint64:
    case ncclFloat16:
    case ncclFloat:
    case ncclDouble:
    case ncclBfloat16:
      return true;
    default:
      return false;
  }
}

} // namespace

/**
 * Two-active sharded relay reduce-scatter (original, performant path).
 *
 * Each group has exactly 2 active ranks; the logical collective is a 2-rank
 * reduce-scatter between them, accelerated by passthrough helpers that relay
 * sharded chunks of a single block (recvCounts[g] elements). This is the
 * production BM-FM path and is byte-for-byte unchanged from the original
 * implementation; the A>2 path lives in shardedRelayReduceScatterFlat.
 *
 * Per active rank (index myActiveIndex), with recvcount = recvCounts[g]:
 *   - sendBuff holds 2 × recvcount elements; block[i] = sendBuff[i*recvcount].
 *   - ownBlockOffset  = myActiveIndex    × recvcount (local contribution)
 *   - sendBlockOffset = otherActiveIndex × recvcount (shipped to other rank)
 *   - recvBuff[0..recvcount) = block[myActiveIndex](self) +
 *                              block[myActiveIndex](other).
 *
 * The relay relays the sendBlockOffset block chunk-by-chunk; the output block
 * (recvBuff) is seeded with the ownBlockOffset contribution then accumulates
 * the relayed/direct-exchanged chunks from the other active rank.
 *
 * In-place is detected when recvBuff == sendBuff + ownBlockOffset (the NCCL
 * reduce-scatter in-place convention). In that case recvBuff already holds the
 * local contribution (no seeding copy) and the direct chunk is reduced via a
 * scratch buffer to avoid overwriting the local data before it is read.
 */
static constexpr int kHelperScratchKeyBase = SHARDED_RELAY_MAX_GROUPS + 1;

static ncclResult_t shardedRelayReduceScatter2Active(
    const void* const* sendBuffs,
    void* const* recvBuffs,
    const size_t* recvCounts,
    ncclDataType_t datatype,
    int reductionDivisor,
    ncclComm_t comm,
    cudaStream_t stream,
    const ShardedRelayRankConfig* configs,
    int myActiveGroup,
    int numHelpers,
    int nGroups,
    size_t elementSize) {
  // ==========================================================================
  // SIZE-ADAPTIVE PURE-DIRECT FAST PATH (A==2)
  // ==========================================================================
  // At small sizes the 2-hop helper relay (phases 1-2 = two group boundaries
  // plus a helper HBM round trip) costs far more than the bandwidth it buys.
  // Instead the two active ranks exchange their full foreign block directly in
  // a single group (helpers idle) and reduce locally -- minimal latency, the
  // same shape as all-to-all. The size -> route mapping lives in
  // selectReduceScatterRoute() so the tests assert the same definition this
  // dispatch uses. This function is the A==2 path, so the selector is asked
  // about A==2 (its metric, 2 * recvCount * elemSize, is the bench's per-rank
  // input label here).
  if (rcclx::relay::selectReduceScatterRoute(
          2, numHelpers, nGroups, recvCounts, elementSize) ==
      rcclx::relay::ReduceScatterRoute::PureDirect) {
    void* pdScratch = nullptr;
    size_t pdRecvcount = 0;
    size_t pdOwnOff = 0;
    bool pdInPlace = false;
    if (myActiveGroup >= 0 && recvCounts[myActiveGroup] > 0) {
      const ShardedRelayRankConfig& cfg = configs[myActiveGroup];
      pdRecvcount = recvCounts[myActiveGroup];
      pdOwnOff = static_cast<size_t>(cfg.myActiveIndex) * pdRecvcount;
      const char* ownContrib =
          static_cast<const char*>(sendBuffs[myActiveGroup]) +
          pdOwnOff * elementSize;
      pdInPlace =
          (static_cast<const void*>(recvBuffs[myActiveGroup]) ==
           static_cast<const void*>(ownContrib));
      pdScratch = ScratchBufferCache::getInstance().get(
          SHARDED_RELAY_MAX_GROUPS, pdRecvcount * elementSize, stream);
      if (pdScratch == nullptr) {
        return ncclInternalError;
      }
    }

    NCCLCHECK(ncclGroupStart());
    for (int g = 0; g < nGroups; g++) {
      if (recvCounts[g] == 0) {
        continue;
      }
      const ShardedRelayRankConfig& cfg = configs[g];
      if (!cfg.isActiveRank) {
        continue; // helpers idle in pure-direct mode
      }
      size_t rc = recvCounts[g];
      int other = 1 - cfg.myActiveIndex;
      int partner = cfg.activeRanks[other];
      // Send my contribution to the partner's owned block; receive the
      // partner's contribution to my owned block into scratch.
      NCCLCHECK(ncclSend(
          static_cast<const char*>(sendBuffs[g]) +
              static_cast<size_t>(other) * rc * elementSize,
          rc,
          datatype,
          partner,
          comm,
          stream));
      NCCLCHECK(ncclRecv(
          static_cast<char*>(pdScratch), rc, datatype, partner, comm, stream));
    }
    NCCLCHECK(ncclGroupEnd());

    if (myActiveGroup >= 0 && pdRecvcount > 0) {
      void* out = recvBuffs[myActiveGroup];
      if (pdInPlace) {
        // recvBuff already aliases the local contribution; fold the partner's
        // exchanged block (scratch) in with a single fused add[/scale] kernel.
        if (reductionDivisor > 1) {
          DISPATCH_INCREMENTAL_ADD_AND_SCALE(
              datatype, out, pdScratch, pdRecvcount, reductionDivisor, stream);
        } else {
          DISPATCH_INCREMENTAL_ADD(
              datatype, out, pdScratch, pdRecvcount, stream);
        }
      } else {
        // Out-of-place: read the local contribution and the partner block in a
        // single fused kernel (out = (own + scratch)[/divisor]) instead of a
        // seeding memcpy followed by an add[/scale], removing one launch and
        // one HBM round trip from the small-message critical path.
        DISPATCH_FUSED_REDUCE(
            datatype,
            out,
            static_cast<const char*>(sendBuffs[myActiveGroup]) +
                pdOwnOff * elementSize,
            pdScratch,
            pdRecvcount,
            reductionDivisor,
            stream);
      }
    }
    return ncclSuccess;
  }

  // =========================================================================
  // CHUNK GEOMETRY: numHelpers relayed chunks + TWO direct chunks
  // =========================================================================
  // The active<->active link is idle while the relay scatter and forward run on
  // the cross links, so instead of a third comm group for a single direct
  // chunk, one direct chunk rides along with each relay group. With numChunks =
  // numHelpers + 2 every link carries exactly one chunk per direction per
  // group, making the critical path 2*recvCount/numChunks instead of the
  // 3*recvCount/(numHelpers+1) of a separate direct phase.
  //
  // Unlike allreduce, the helper CANNOT reduce here: its slot 0 holds a0's
  // contribution to a1's output and slot 1 holds a1's contribution to a0's
  // output — different outputs. Helpers stay pure passthrough and the active
  // rank reduces.
  const int numChunks = numHelpers + 2;

  size_t chunkSizes[SHARDED_RELAY_MAX_GROUPS];
  size_t relayTotals[SHARDED_RELAY_MAX_GROUPS]; // == direct chunk A's offset
  size_t dirASizes[SHARDED_RELAY_MAX_GROUPS];
  size_t dirBOffsets[SHARDED_RELAY_MAX_GROUPS];
  size_t dirBSizes[SHARDED_RELAY_MAX_GROUPS]; // absorbs the remainder

  for (int g = 0; g < nGroups; g++) {
    size_t count = recvCounts[g];

    // Zero-count groups are skipped by every loop below.
    if (count == 0) {
      chunkSizes[g] = 0;
      relayTotals[g] = 0;
      dirASizes[g] = 0;
      dirBOffsets[g] = 0;
      dirBSizes[g] = 0;
      continue;
    }

    size_t chunkSize = count / numChunks;
    chunkSize = (chunkSize / CHUNK_ALIGN_ELEMENTS) * CHUNK_ALIGN_ELEMENTS;
    chunkSizes[g] = chunkSize;
    relayTotals[g] = static_cast<size_t>(numHelpers) * chunkSize;
    if (chunkSize == 0) {
      dirASizes[g] = count / 2;
      dirBOffsets[g] = dirASizes[g];
    } else {
      dirASizes[g] = chunkSize;
      dirBOffsets[g] = relayTotals[g] + chunkSize;
    }
    dirBSizes[g] = count - dirBOffsets[g];
  }

  // =========================================================================
  // SCRATCH: the whole foreign contribution to my output block, contiguous
  // =========================================================================
  // One recvCount-element buffer laid out exactly like the output block: the
  // relayed chunks land at [0, relayTotal) and the two direct chunks fill
  // [relayTotal, recvCount). Because it mirrors the output layout, the entire
  // reduction collapses to ONE fused kernel launch at the end.
  void* foreignScratch = nullptr;
  size_t ownBlockOffset = 0;
  size_t sendBlockOffset = 0;
  bool isInPlace = false;
  if (myActiveGroup >= 0 && recvCounts[myActiveGroup] > 0) {
    const ShardedRelayRankConfig& cfg = configs[myActiveGroup];
    size_t recvcount = recvCounts[myActiveGroup];
    ownBlockOffset = static_cast<size_t>(cfg.myActiveIndex) * recvcount;
    sendBlockOffset = static_cast<size_t>(1 - cfg.myActiveIndex) * recvcount;

    // In-place when recvBuff aliases the local contribution block of sendBuff.
    const char* ownBlock = static_cast<const char*>(sendBuffs[myActiveGroup]) +
        ownBlockOffset * elementSize;
    isInPlace =
        (static_cast<const void*>(recvBuffs[myActiveGroup]) ==
         static_cast<const void*>(ownBlock));

    foreignScratch = ScratchBufferCache::getInstance().get(
        SHARDED_RELAY_MAX_GROUPS, recvcount * elementSize, stream);
    if (foreignScratch == nullptr) {
      return ncclInternalError;
    }
  }

  // Helper staging is kernel-owned scratch: callers pass a placeholder buffer
  // for groups where they are a helper. Each helper group holds nActiveRanks
  // chunks (one per active source) to receive and forward.
  void* helperScratch[SHARDED_RELAY_MAX_GROUPS] = {nullptr};
  for (int g = 0; g < nGroups; g++) {
    const ShardedRelayRankConfig& cfg = configs[g];
    if (!cfg.isActiveRank && recvCounts[g] > 0 && chunkSizes[g] > 0) {
      size_t needBytes =
          static_cast<size_t>(cfg.nActiveRanks) * chunkSizes[g] * elementSize;
      helperScratch[g] = ScratchBufferCache::getInstance().get(
          kHelperScratchKeyBase + g, needBytes, stream);
      if (helperScratch[g] == nullptr) {
        return ncclInternalError;
      }
    }
  }

  // =========================================================================
  // GROUP 1: relay scatter (active->helpers) || direct chunk A
  // (active<->active)
  // =========================================================================
  NCCLCHECK(ncclGroupStart());

  for (int g = 0; g < nGroups; g++) {
    if (recvCounts[g] == 0)
      continue;
    const ShardedRelayRankConfig& cfg = configs[g];
    size_t chunkSize = chunkSizes[g];

    if (cfg.isActiveRank) {
      // The block shipped to the other active rank.
      const char* sendBlock = static_cast<const char*>(sendBuffs[g]) +
          sendBlockOffset * elementSize;

      for (int h = 0; h < cfg.numHelpers && chunkSize > 0; h++) {
        NCCLCHECK(ncclSend(
            sendBlock + static_cast<size_t>(h) * chunkSize * elementSize,
            chunkSize,
            datatype,
            cfg.helperRanks[h],
            comm,
            stream));
      }

      // Direct chunk A over the otherwise-idle active<->active link.
      int partner = cfg.activeRanks[1 - cfg.myActiveIndex];
      if (dirASizes[g] > 0) {
        NCCLCHECK(ncclSend(
            sendBlock + relayTotals[g] * elementSize,
            dirASizes[g],
            datatype,
            partner,
            comm,
            stream));
        NCCLCHECK(ncclRecv(
            static_cast<char*>(foreignScratch) + relayTotals[g] * elementSize,
            dirASizes[g],
            datatype,
            partner,
            comm,
            stream));
      }
    } else if (chunkSize > 0) {
      // Helper: receive active rank a's chunk into slot a.
      char* helperBuf = static_cast<char*>(helperScratch[g]);
      for (int a = 0; a < cfg.nActiveRanks; a++) {
        NCCLCHECK(ncclRecv(
            helperBuf + static_cast<size_t>(a) * chunkSize * elementSize,
            chunkSize,
            datatype,
            cfg.activeRanks[a],
            comm,
            stream));
      }
    }
  }

  NCCLCHECK(ncclGroupEnd());

  // =========================================================================
  // GROUP 2: relay forward (helpers->active) || direct chunk B
  // (active<->active)
  // =========================================================================
  NCCLCHECK(ncclGroupStart());

  for (int g = 0; g < nGroups; g++) {
    if (recvCounts[g] == 0)
      continue;
    const ShardedRelayRankConfig& cfg = configs[g];
    size_t chunkSize = chunkSizes[g];

    if (cfg.isActiveRank) {
      char* scratch = static_cast<char*>(foreignScratch);

      for (int h = 0; h < cfg.numHelpers && chunkSize > 0; h++) {
        NCCLCHECK(ncclRecv(
            scratch + static_cast<size_t>(h) * chunkSize * elementSize,
            chunkSize,
            datatype,
            cfg.helperRanks[h],
            comm,
            stream));
      }

      // Direct chunk B, again over the idle active<->active link.
      int partner = cfg.activeRanks[1 - cfg.myActiveIndex];
      if (dirBSizes[g] > 0) {
        NCCLCHECK(ncclSend(
            static_cast<const char*>(sendBuffs[g]) +
                (sendBlockOffset + dirBOffsets[g]) * elementSize,
            dirBSizes[g],
            datatype,
            partner,
            comm,
            stream));
        NCCLCHECK(ncclRecv(
            scratch + dirBOffsets[g] * elementSize,
            dirBSizes[g],
            datatype,
            partner,
            comm,
            stream));
      }
    } else if (chunkSize > 0) {
      // Passthrough: slot a goes to the OTHER active rank, which owns it.
      const char* helperBuf = static_cast<const char*>(helperScratch[g]);
      for (int a = 0; a < cfg.nActiveRanks; a++) {
        NCCLCHECK(ncclSend(
            helperBuf + static_cast<size_t>(a) * chunkSize * elementSize,
            chunkSize,
            datatype,
            cfg.activeRanks[1 - a],
            comm,
            stream));
      }
    }
  }

  NCCLCHECK(ncclGroupEnd());

  // =========================================================================
  // REDUCE: one fused pass over the whole output block
  // =========================================================================
  // foreignScratch mirrors the output layout, so relayed and direct chunks are
  // reduced together in a single launch.
  if (myActiveGroup >= 0 && recvCounts[myActiveGroup] > 0) {
    size_t recvcount = recvCounts[myActiveGroup];
    void* out = recvBuffs[myActiveGroup];
    if (isInPlace) {
      if (reductionDivisor > 1) {
        DISPATCH_INCREMENTAL_ADD_AND_SCALE(
            datatype, out, foreignScratch, recvcount, reductionDivisor, stream);
      } else {
        DISPATCH_INCREMENTAL_ADD(
            datatype, out, foreignScratch, recvcount, stream);
      }
    } else {
      DISPATCH_FUSED_REDUCE(
          datatype,
          out,
          static_cast<const char*>(sendBuffs[myActiveGroup]) +
              ownBlockOffset * elementSize,
          foreignScratch,
          recvcount,
          reductionDivisor,
          stream);
    }
  }

  return ncclSuccess;
}

/**
 * Software-pipelined single-group 2-active sharded relay reduce-scatter.
 *
 * Same logical collective and the same passthrough helpers as
 * shardedRelayReduceScatter2Active, but for nGroups == 1 -- where the active
 * ranks and the helpers are disjoint sets -- the relay is tiled and pipelined
 * so both directions of every cross link stay busy. See relayA2PipelineTiles()
 * for why the two-group schedule cannot do that and what it costs.
 *
 * Helpers stay pure passthrough here, unlike allreduce: slot 0 is a0's
 * contribution to a1's output and slot 1 is a1's contribution to a0's output --
 * different outputs, so there is nothing to sum at the helper. The active rank
 * reduces once at the end, and foreignScratch still mirrors the output layout
 * so that stays a single fused launch no matter how many tiles the relay used.
 *
 * With T tiles and unit u = align(recvCount / ((H+1)*T + 1)), the block shipped
 * to the other active rank splits as:
 *   [0, H*T*u)         relay region; helper h owns [h*T*u, (h+1)*T*u), its tile
 *                      t at h*T*u + t*u
 *   [H*T*u, recvCount) direct region as T+1 chunks of u, the last absorbing the
 *                      /((H+1)*T + 1) remainder and the alignment loss
 *
 * Group k, for k in [0, T]: the active rank scatters tile k (k < T) to every
 * helper, receives tile k-1 (k > 0) of the partner's contribution from every
 * helper into the matching offset of foreignScratch, and exchanges direct chunk
 * k over the active<->active link; helper h receives tile k of each active's
 * chunk into ping-pong buffer k%2 and forwards buffer (k-1)%2 to the active
 * rank that owns it.
 */
static ncclResult_t shardedRelayReduceScatter2ActivePipelined(
    const void* const* sendBuffs,
    void* const* recvBuffs,
    const size_t* recvCounts,
    ncclDataType_t datatype,
    int reductionDivisor,
    ncclComm_t comm,
    cudaStream_t stream,
    const ShardedRelayRankConfig* configs,
    int myActiveGroup,
    int numHelpers,
    int nTiles,
    size_t elementSize) {
  const ShardedRelayRankConfig& cfg = configs[0];
  const size_t recvcount = recvCounts[0];
  const int H = numHelpers;
  const int T = nTiles;
  const size_t u = ((recvcount / (static_cast<size_t>(H + 1) * T + 1)) /
                    CHUNK_ALIGN_ELEMENTS) *
      CHUNK_ALIGN_ELEMENTS;
  if (u == 0) {
    return ncclInvalidArgument;
  }
  const size_t tileStride = static_cast<size_t>(T) * u;
  const size_t directBase = static_cast<size_t>(H) * tileStride;
  const size_t lastDirect = recvcount - directBase - tileStride;

  // Scratch mirroring the output block, so the whole reduction is one launch.
  void* foreignScratch = nullptr;
  size_t ownBlockOffset = 0;
  size_t sendBlockOffset = 0;
  bool isInPlace = false;
  if (myActiveGroup == 0) {
    ownBlockOffset = static_cast<size_t>(cfg.myActiveIndex) * recvcount;
    sendBlockOffset = static_cast<size_t>(1 - cfg.myActiveIndex) * recvcount;
    const char* ownBlock =
        static_cast<const char*>(sendBuffs[0]) + ownBlockOffset * elementSize;
    isInPlace =
        (static_cast<const void*>(recvBuffs[0]) ==
         static_cast<const void*>(ownBlock));
    foreignScratch = ScratchBufferCache::getInstance().get(
        SHARDED_RELAY_MAX_GROUPS, recvcount * elementSize, stream);
    if (foreignScratch == nullptr) {
      return ncclInternalError;
    }
  }

  // Helper staging: two ping-pong units per active source.
  char* hbuff = nullptr;
  if (!cfg.isActiveRank) {
    hbuff = static_cast<char*>(ScratchBufferCache::getInstance().get(
        kHelperScratchKeyBase,
        static_cast<size_t>(cfg.nActiveRanks) * 2 * u * elementSize,
        stream));
    if (hbuff == nullptr) {
      return ncclInternalError;
    }
  }
  auto helperSlot = [&](int a, int k) -> char* {
    return hbuff +
        (static_cast<size_t>(a) * 2 + static_cast<size_t>(k % 2)) * u *
        elementSize;
  };

  for (int k = 0; k <= T; k++) {
    NCCLCHECK(ncclGroupStart());
    if (cfg.isActiveRank) {
      const char* sendBlock = static_cast<const char*>(sendBuffs[0]) +
          sendBlockOffset * elementSize;
      char* scratch = static_cast<char*>(foreignScratch);
      const int partner = cfg.activeRanks[1 - cfg.myActiveIndex];
      const size_t directOffset = directBase + static_cast<size_t>(k) * u;
      const size_t directSize = (k < T) ? u : lastDirect;

      if (k < T) {
        for (int h = 0; h < H; h++) {
          NCCLCHECK(ncclSend(
              sendBlock +
                  (static_cast<size_t>(h) * tileStride +
                   static_cast<size_t>(k) * u) *
                      elementSize,
              u,
              datatype,
              cfg.helperRanks[h],
              comm,
              stream));
        }
      }
      NCCLCHECK(ncclSend(
          sendBlock + directOffset * elementSize,
          directSize,
          datatype,
          partner,
          comm,
          stream));
      if (k > 0) {
        for (int h = 0; h < H; h++) {
          NCCLCHECK(ncclRecv(
              scratch +
                  (static_cast<size_t>(h) * tileStride +
                   static_cast<size_t>(k - 1) * u) *
                      elementSize,
              u,
              datatype,
              cfg.helperRanks[h],
              comm,
              stream));
        }
      }
      NCCLCHECK(ncclRecv(
          scratch + directOffset * elementSize,
          directSize,
          datatype,
          partner,
          comm,
          stream));
    } else {
      if (k < T) {
        for (int a = 0; a < cfg.nActiveRanks; a++) {
          NCCLCHECK(ncclRecv(
              helperSlot(a, k), u, datatype, cfg.activeRanks[a], comm, stream));
        }
      }
      if (k > 0) {
        for (int a = 0; a < cfg.nActiveRanks; a++) {
          NCCLCHECK(ncclSend(
              helperSlot(a, k - 1),
              u,
              datatype,
              cfg.activeRanks[1 - a],
              comm,
              stream));
        }
      }
    }
    NCCLCHECK(ncclGroupEnd());
  }

  // One fused pass over the whole output block.
  if (myActiveGroup == 0) {
    void* out = recvBuffs[0];
    if (isInPlace) {
      if (reductionDivisor > 1) {
        DISPATCH_INCREMENTAL_ADD_AND_SCALE(
            datatype, out, foreignScratch, recvcount, reductionDivisor, stream);
      } else {
        DISPATCH_INCREMENTAL_ADD(
            datatype, out, foreignScratch, recvcount, stream);
      }
    } else {
      DISPATCH_FUSED_REDUCE(
          datatype,
          out,
          static_cast<const char*>(sendBuffs[0]) + ownBlockOffset * elementSize,
          foreignScratch,
          recvcount,
          reductionDivisor,
          stream);
    }
  }

  return ncclSuccess;
}

/**
 * Reduce-scatter for > 2 active ranks (two-group flat relay with
 * reduce-at-helper).
 *
 * Each active rank's sendBuff holds A blocks of recvCount; block[j] is this
 * rank's contribution to owner j's output. Owner j's output is the sum over all
 * A sources of block[j].
 *
 * Every block is split into a DIRECT region, exchanged 1-hop over the intra
 * active<->active links, and an OFFLOAD region, routed 2-hop through the
 * otherwise-idle helpers. A helper owns one position slice of every block: it
 * collects that slice from the A-1 non-owner sources, SUMS them, and sends the
 * single reduced slice on to the owner, which folds in its own contribution.
 * Reducing at the helper is what keeps the return hop cheap -- (A-1) chunks in,
 * one chunk out.
 *
 * Link accounting, per direction, in units of the per-(owner, helper) chunk cs.
 * A rank's helpers are the active ranks of another group, so its scatter and
 * its helper duty are egress on the same cross links:
 *
 *   group 1:  cross = (A-1)*cs (my A-1 foreign blocks' slice h)   intra =
 * (A-1)*cs group 2:  cross = cs       (one reduced slice per owner)      intra
 * = cs
 *
 * Balancing intra against cross in each group puts (A-1)*cs of the direct
 * region in group 1 and cs in group 2, so the direct region is A*cs and the
 * offload region is H*cs: cs = recvCount/(A+H), eight equal units on the 8-GPU
 * node. The critical path is (A-1)*cs + cs = A*cs = recvCount/2, against
 * recvCount for NCCL's intra-only reduce-scatter -- a 2x ceiling. (The previous
 * recursive-halving path measured ~0.96x.)
 *
 * Block layout [0, recvCount): direct region [0, A*cs) whose first (A-1)*cs go
 * out in group 1 and whose last cs goes out in group 2, then the offload region
 * [A*cs, recvCount) as H slices of cs, slice h owned by helper h.
 *
 * Helper scratch = recvBuffs[g], holding one cs chunk per (owner, contributing
 * source) pair: A*(A-1)*cs elements. The reduction is done in place into each
 * owner's first slot.
 *
 * Below a size threshold the offload is disabled and this degenerates to a
 * single-group pure-direct all-to-all reduce-scatter with the helpers idle.
 */
static ncclResult_t shardedRelayReduceScatterFlat(
    const void* const* sendBuffs,
    void* const* recvBuffs,
    const size_t* recvCounts,
    ncclDataType_t datatype,
    int reductionDivisor,
    ncclComm_t comm,
    cudaStream_t stream,
    const ShardedRelayRankConfig* configs,
    int myActiveGroup,
    int numHelpers,
    int nActiveRanksPerGroup,
    int nGroups,
    size_t elementSize) {
  const int A = nActiveRanksPerGroup;
  const int H = numHelpers;

  // The size -> route mapping lives in selectReduceScatterRoute() so the tests
  // assert the same definition this dispatch uses: the offload's extra hop plus
  // the second group boundary only pay for themselves past ~48 MB; below that
  // the single-group pure-direct all-to-all reduce-scatter wins outright
  // (1.24-1.42x at <= 576 KB, vs 0.69x if the offload is forced on at 4.5 MB).
  const bool useOffload = rcclx::relay::selectReduceScatterRoute(
                              A, H, nGroups, recvCounts, elementSize) ==
      rcclx::relay::ReduceScatterRoute::FlatOffload;

  // Per-group geometry. cs = recvCount/(A+H) aligned down; the direct region
  // absorbs the remainder so directSz + H*cs == recvCount.
  size_t csArr[SHARDED_RELAY_MAX_GROUPS];
  size_t directArr[SHARDED_RELAY_MAX_GROUPS];
  size_t d1Arr[SHARDED_RELAY_MAX_GROUPS];
  for (int g = 0; g < nGroups; g++) {
    size_t rc = recvCounts[g];
    if (rc == 0) {
      csArr[g] = 0;
      directArr[g] = 0;
      d1Arr[g] = 0;
      continue;
    }
    size_t cs = useOffload ? (rc / static_cast<size_t>(A + H)) : 0;
    cs = (cs / CHUNK_ALIGN_ELEMENTS) * CHUNK_ALIGN_ELEMENTS;
    csArr[g] = cs;
    directArr[g] = rc - static_cast<size_t>(H) * cs;
    d1Arr[g] = (cs > 0) ? (directArr[g] - cs) : directArr[g];
  }

  // Active-rank scratch. dScratch holds the A-1 peer contributions to my direct
  // region, one contiguous directSz block each, so the whole direct region
  // reduces in a single multi-input pass. oScratch mirrors my output's offload
  // region and receives the H helper-reduced slices straight into place.
  void* dScratch = nullptr;
  void* oScratch = nullptr;
  bool isInPlace = false;
  if (myActiveGroup >= 0 && recvCounts[myActiveGroup] > 0) {
    const ShardedRelayRankConfig& cfg = configs[myActiveGroup];
    const char* ownBlock = static_cast<const char*>(sendBuffs[myActiveGroup]) +
        static_cast<size_t>(cfg.myActiveIndex) * recvCounts[myActiveGroup] *
            elementSize;
    isInPlace =
        (static_cast<const void*>(recvBuffs[myActiveGroup]) ==
         static_cast<const void*>(ownBlock));

    dScratch = ScratchBufferCache::getInstance().get(
        SHARDED_RELAY_MAX_GROUPS,
        static_cast<size_t>(A - 1) * directArr[myActiveGroup] * elementSize,
        stream);
    if (dScratch == nullptr) {
      return ncclInternalError;
    }
    if (csArr[myActiveGroup] > 0) {
      oScratch = ScratchBufferCache::getInstance().get(
          myActiveGroup,
          static_cast<size_t>(H) * csArr[myActiveGroup] * elementSize,
          stream);
      if (oScratch == nullptr) {
        return ncclInternalError;
      }
    }
  }

  // Helper staging is kernel-owned scratch: callers pass a placeholder buffer
  // for groups where they are a helper. Each helper group holds A*(A-1) offload
  // slices of cs (one per (owner, contributing source) pair).
  void* helperScratch[SHARDED_RELAY_MAX_GROUPS] = {nullptr};
  for (int g = 0; g < nGroups; g++) {
    const ShardedRelayRankConfig& cfg = configs[g];
    if (!cfg.isActiveRank && recvCounts[g] > 0 && csArr[g] > 0) {
      size_t needBytes =
          static_cast<size_t>(A) * (A - 1) * csArr[g] * elementSize;
      helperScratch[g] = ScratchBufferCache::getInstance().get(
          kHelperScratchKeyBase + g, needBytes, stream);
      if (helperScratch[g] == nullptr) {
        return ncclInternalError;
      }
    }
  }

  // =========================================================================
  // GROUP 1: direct part 1 (active<->active) || offload scatter
  // (active->helper)
  // =========================================================================
  NCCLCHECK(ncclGroupStart());
  for (int g = 0; g < nGroups; g++) {
    if (recvCounts[g] == 0)
      continue;
    const ShardedRelayRankConfig& cfg = configs[g];
    size_t rc = recvCounts[g];
    size_t cs = csArr[g];
    size_t directSz = directArr[g];
    size_t d1 = d1Arr[g];

    if (cfg.isActiveRank) {
      const char* sendbuff = static_cast<const char*>(sendBuffs[g]);
      int m = cfg.myActiveIndex;

      if (d1 > 0) {
        for (int k = 0; k < A; k++) {
          if (k == m)
            continue;
          NCCLCHECK(ncclSend(
              sendbuff + static_cast<size_t>(k) * rc * elementSize,
              d1,
              datatype,
              cfg.activeRanks[k],
              comm,
              stream));
        }
        for (int s = 0; s < A; s++) {
          if (s == m)
            continue;
          int p = (s < m) ? s : s - 1;
          NCCLCHECK(ncclRecv(
              static_cast<char*>(dScratch) +
                  static_cast<size_t>(p) * directSz * elementSize,
              d1,
              datatype,
              cfg.activeRanks[s],
              comm,
              stream));
        }
      }

      // Offload scatter: helper h gets slice h of each of my foreign blocks.
      for (int h = 0; h < cfg.numHelpers && cs > 0; h++) {
        for (int j = 0; j < A; j++) {
          if (j == m)
            continue;
          NCCLCHECK(ncclSend(
              sendbuff +
                  (static_cast<size_t>(j) * rc + directSz +
                   static_cast<size_t>(h) * cs) *
                      elementSize,
              cs,
              datatype,
              cfg.helperRanks[h],
              comm,
              stream));
        }
      }
    } else if (cs > 0) {
      // Helper: collect owner j's slice from each contributing source s != j.
      char* hbuff = static_cast<char*>(helperScratch[g]);
      for (int s = 0; s < cfg.nActiveRanks; s++) {
        for (int j = 0; j < A; j++) {
          if (j == s)
            continue;
          int t = (s < j) ? s : s - 1;
          size_t slot = static_cast<size_t>(j) * (A - 1) + t;
          NCCLCHECK(ncclRecv(
              hbuff + slot * cs * elementSize,
              cs,
              datatype,
              cfg.activeRanks[s],
              comm,
              stream));
        }
      }
    }
  }
  NCCLCHECK(ncclGroupEnd());

  bool anyOffload = false;
  for (int g = 0; g < nGroups; g++) {
    if (recvCounts[g] > 0 && csArr[g] > 0) {
      anyOffload = true;
      break;
    }
  }

  // =========================================================================
  // HELPER REDUCE: sum each owner's A-1 contributions in place
  // =========================================================================
  // No divisor here: the owner applies the AVG divisor once when it folds in
  // its own contribution.
  if (anyOffload) {
    for (int g = 0; g < nGroups; g++) {
      if (recvCounts[g] == 0 || csArr[g] == 0 || configs[g].isActiveRank)
        continue;
      char* hbuff = static_cast<char*>(helperScratch[g]);
      size_t cs = csArr[g];
      for (int j = 0; j < A; j++) {
        char* dst = hbuff + static_cast<size_t>(j) * (A - 1) * cs * elementSize;
        DISPATCH_MULTI_REDUCE(
            datatype, dst, dst + cs * elementSize, A - 2, cs, 1, stream);
      }
    }
  }

  // =========================================================================
  // GROUP 2: direct part 2 (active<->active) || reduced offload
  // (helper->active)
  // =========================================================================
  if (anyOffload) {
    NCCLCHECK(ncclGroupStart());
    for (int g = 0; g < nGroups; g++) {
      if (recvCounts[g] == 0 || csArr[g] == 0)
        continue;
      const ShardedRelayRankConfig& cfg = configs[g];
      size_t rc = recvCounts[g];
      size_t cs = csArr[g];
      size_t directSz = directArr[g];
      size_t d1 = d1Arr[g];
      size_t d2 = directSz - d1;

      if (cfg.isActiveRank) {
        const char* sendbuff = static_cast<const char*>(sendBuffs[g]);
        int m = cfg.myActiveIndex;

        if (d2 > 0) {
          for (int k = 0; k < A; k++) {
            if (k == m)
              continue;
            NCCLCHECK(ncclSend(
                sendbuff + (static_cast<size_t>(k) * rc + d1) * elementSize,
                d2,
                datatype,
                cfg.activeRanks[k],
                comm,
                stream));
          }
          for (int s = 0; s < A; s++) {
            if (s == m)
              continue;
            int p = (s < m) ? s : s - 1;
            NCCLCHECK(ncclRecv(
                static_cast<char*>(dScratch) +
                    (static_cast<size_t>(p) * directSz + d1) * elementSize,
                d2,
                datatype,
                cfg.activeRanks[s],
                comm,
                stream));
          }
        }

        // Reduced offload slices land contiguously, mirroring my output.
        for (int h = 0; h < cfg.numHelpers; h++) {
          NCCLCHECK(ncclRecv(
              static_cast<char*>(oScratch) +
                  static_cast<size_t>(h) * cs * elementSize,
              cs,
              datatype,
              cfg.helperRanks[h],
              comm,
              stream));
        }
      } else {
        // Helper: hand owner j its single reduced slice.
        const char* hbuff = static_cast<const char*>(helperScratch[g]);
        for (int j = 0; j < A; j++) {
          NCCLCHECK(ncclSend(
              hbuff + static_cast<size_t>(j) * (A - 1) * cs * elementSize,
              cs,
              datatype,
              cfg.activeRanks[j],
              comm,
              stream));
        }
      }
    }
    NCCLCHECK(ncclGroupEnd());
  }

  // =========================================================================
  // OWNER REDUCE: fold my own contribution into the direct and offload regions
  // =========================================================================
  if (myActiveGroup >= 0 && recvCounts[myActiveGroup] > 0) {
    const ShardedRelayRankConfig& cfg = configs[myActiveGroup];
    size_t rc = recvCounts[myActiveGroup];
    size_t cs = csArr[myActiveGroup];
    size_t directSz = directArr[myActiveGroup];
    char* out = static_cast<char*>(recvBuffs[myActiveGroup]);
    const char* ownBlock = static_cast<const char*>(sendBuffs[myActiveGroup]) +
        static_cast<size_t>(cfg.myActiveIndex) * rc * elementSize;

    // Direct region: own + the A-1 peer blocks, one fused multi-input pass.
    if (A == 4 && directSz > 0 && !isInPlace) {
      DISPATCH_SEEDED_MULTI_REDUCE(
          datatype,
          out,
          ownBlock,
          dScratch,
          A - 1,
          directSz,
          reductionDivisor,
          stream);
    } else {
      if (!isInPlace) {
        cudaMemcpyAsync(
            out,
            ownBlock,
            directSz * elementSize,
            cudaMemcpyDeviceToDevice,
            stream);
      }
      DISPATCH_MULTI_REDUCE(
          datatype, out, dScratch, A - 1, directSz, reductionDivisor, stream);
    }

    // Offload region: own + the H helper-reduced slices (each already a sum of
    // the A-1 other sources).
    if (cs > 0) {
      char* oOut = out + directSz * elementSize;
      size_t oCount = rc - directSz;
      if (isInPlace) {
        if (reductionDivisor > 1) {
          DISPATCH_INCREMENTAL_ADD_AND_SCALE(
              datatype, oOut, oScratch, oCount, reductionDivisor, stream);
        } else {
          DISPATCH_INCREMENTAL_ADD(datatype, oOut, oScratch, oCount, stream);
        }
      } else {
        DISPATCH_FUSED_REDUCE(
            datatype,
            oOut,
            ownBlock + directSz * elementSize,
            oScratch,
            oCount,
            reductionDivisor,
            stream);
      }
    }
  }

  return ncclSuccess;
}

/**
 * Fused Multi-Group Sharded Relay Reduce-Scatter.
 *
 * Executes multiple sharded relay reduce-scatters in one fused call,
 * phase-synced across all groups so XGMI links carry unidirectional traffic.
 * Helpers are pure passthrough; reductions happen on the active ranks. Each
 * rank is ACTIVE for exactly one group and a HELPER for the others.
 *
 * nActiveRanksPerGroup must be a power of two (2 or 4): A==2 uses the original
 * 2-active path; A>2 uses the bandwidth-optimal recursive path.
 */
HOT ncclResult_t ncclShardedRelayMultiGroupReduceScatterImpl(
    const void* const* sendBuffs,
    void* const* recvBuffs,
    const size_t* recvCounts,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    cudaStream_t stream,
    const int* const* allActiveRanks,
    int nActiveRanksPerGroup,
    int nGroups) {
  int nRanks, rank;
  NCCLCHECK(ncclCommCount(comm, &nRanks));
  NCCLCHECK(ncclCommUserRank(comm, &rank));

  // Validate every argument before touching recvCounts: the all-zero scan
  // below indexes recvCounts[0..nGroups), so a null pointer or an out-of-range
  // nGroups has to be rejected first. Bounds-checking nGroups up here also
  // means nGroups <= 0 reports ncclInvalidArgument rather than skipping the
  // scan entirely and returning ncclSuccess.
  if (nGroups < 1 || nGroups > SHARDED_RELAY_MAX_GROUPS) {
    return ncclInvalidArgument;
  }

  if (recvBuffs == nullptr || allActiveRanks == nullptr ||
      recvCounts == nullptr || sendBuffs == nullptr) {
    return ncclInvalidArgument;
  }

  // Require a power-of-two active-rank count (>= 2) for the XOR schedule.
  if (nActiveRanksPerGroup < 2 || !isPowerOfTwo(nActiveRanksPerGroup)) {
    return ncclInvalidArgument;
  }

  // Validate operation - only SUM and AVG are supported
  if (op != ncclSum && op != ncclAvg) {
    return ncclInvalidArgument;
  }

  if (!isSupportedRelayDataType(datatype)) {
    return ncclInvalidArgument;
  }

  // Check if all recvCounts are zero
  bool allZero = true;
  for (int g = 0; g < nGroups; g++) {
    if (recvCounts[g] != 0) {
      allZero = false;
      break;
    }
  }
  if (allZero) {
    return ncclSuccess;
  }

  size_t elementSize = ncclTypeSize(datatype);

  // Compute divisor for reduction: 1 for SUM, nActiveRanksPerGroup for AVG
  int reductionDivisor = (op == ncclAvg) ? nActiveRanksPerGroup : 1;

  // =========================================================================
  // BUILD RANK CONFIGURATION FOR ALL GROUPS
  // =========================================================================
  ShardedRelayRankConfig configs[SHARDED_RELAY_MAX_GROUPS];
  int myActiveGroup = -1; // Which group is this rank active for?

  for (int g = 0; g < nGroups; g++) {
    if (!buildShardedRelayRankConfig(
            nRanks,
            rank,
            allActiveRanks[g],
            nActiveRanksPerGroup,
            configs[g])) {
      return ncclInvalidArgument;
    }
    if (configs[g].isActiveRank) {
      myActiveGroup = g;
    }
  }

  // All groups should have the same number of helpers (same chunk structure)
  int numHelpers = configs[0].numHelpers;

  if (nActiveRanksPerGroup == 2) {
    // 2-active path is unchanged internally; feed it per-group contiguous
    // buffers. Helper groups use their scratch (recvBuffs[g]); the active
    // group uses its caller input/output buffers directly.
    const void* sendBuffs2[SHARDED_RELAY_MAX_GROUPS];
    void* recvBuffs2[SHARDED_RELAY_MAX_GROUPS];
    for (int g = 0; g < nGroups; g++) {
      sendBuffs2[g] = sendBuffs[g];
      recvBuffs2[g] = recvBuffs[g];
    }
    // A single-group relay call has the helpers to itself, so the scatter and
    // the forward run on opposite directions of each cross link and can be
    // software-pipelined into one duplex stream. relayA2PipelineTiles() returns
    // 1 whenever that does not apply, and the small-message pure-direct route
    // (owned by shardedRelayReduceScatter2Active) never pipelines.
    const int nTiles = (rcclx::relay::selectReduceScatterRoute(
                            2, numHelpers, nGroups, recvCounts, elementSize) ==
                        rcclx::relay::ReduceScatterRoute::A2Relay)
        ? rcclx::relay::relayA2PipelineTiles(
              2,
              numHelpers,
              nGroups,
              rcclx::relay::relayMaxCount(recvCounts, nGroups),
              elementSize)
        : 1;
    if (nTiles > 1) {
      return shardedRelayReduceScatter2ActivePipelined(
          sendBuffs2,
          recvBuffs2,
          recvCounts,
          datatype,
          reductionDivisor,
          comm,
          stream,
          configs,
          myActiveGroup,
          numHelpers,
          nTiles,
          elementSize);
    }
    return shardedRelayReduceScatter2Active(
        sendBuffs2,
        recvBuffs2,
        recvCounts,
        datatype,
        reductionDivisor,
        comm,
        stream,
        configs,
        myActiveGroup,
        numHelpers,
        nGroups,
        elementSize);
  }
  // A>2: two-group flat relay with reduce-at-helper woven with a direct
  // all-to-all reduce-scatter over the intra links.
  return shardedRelayReduceScatterFlat(
      sendBuffs,
      recvBuffs,
      recvCounts,
      datatype,
      reductionDivisor,
      comm,
      stream,
      configs,
      myActiveGroup,
      numHelpers,
      nActiveRanksPerGroup,
      nGroups,
      elementSize);
}
