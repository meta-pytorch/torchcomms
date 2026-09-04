/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "sharded_relay_allreduce.h"
#include "comm.h"
#include "sharded_relay_allreduce_kernels.h"
#include "sharded_relay_graph_scratch.h"
#include "sharded_relay_lp.h"
#include "sharded_relay_lp_arena.h"
#include "sharded_relay_lp_kernels.h"
#include "sharded_relay_oneshot.h"
#include "sharded_relay_route.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <map>
#include <mutex>
#include <tuple>

// GPU memory access alignment in elements for chunk size rounding.
// Distinct from the CPU CACHE_LINE_SIZE (64 bytes) defined in comm.h
// which is used for struct padding.
static constexpr size_t CHUNK_ALIGN_ELEMENTS = 128;

// The helpers below are duplicated, by design, across every sharded-relay TU
// (reduce-scatter, all-gather, all-to-all, hierarchical all-to-all). Each of
// those TUs keeps its copy in an anonymous namespace so the class types and
// their inline members have internal linkage and can never be merged across
// translation units; this TU does the same. The GPU kernels are NOT duplicated
// -- they are shared via sharded_relay_allreduce_kernels.h.
namespace {

/**
 * Scratch Buffer Cache Singleton
 *
 * Amortizes cudaMalloc/cudaFree costs by caching and reusing scratch buffers.
 * Thread-safe, with one buffer per (device, stream, key).
 *
 * Key features:
 * - Multiple buffers per device (keyed by stream and group, so concurrent
 *   collectives on different streams never share staging)
 * - Automatically grows buffer if larger size needed
 * - Never shrinks (to avoid repeated alloc/free for varying sizes)
 * - Thread-safe access with mutex protection
 */
class ScratchBufferCache {
 public:
  static ScratchBufferCache& getInstance() {
    static ScratchBufferCache instance;
    return instance;
  }

  /**
   * Get a scratch buffer with a specific key (for multi-group support).
   * Each key maintains its own buffer, allowing multiple independent scratch
   * buffers per device.
   *
   * The returned memory is UNINITIALISED: it is either a fresh pool allocation
   * or a recycled one still holding bytes from an earlier call. Callers must
   * fully overwrite every element they later read; each one does so today by
   * receiving into the whole buffer before reducing over it. Zeroing here
   * instead would add a full HBM pass per call, and would convert any future
   * coverage gap into a silently wrong sum (zero is the SUM identity) rather
   * than visible garbage.
   *
   * @param key Unique key to identify this scratch buffer (e.g., group index)
   * @param requiredBytes Minimum size in bytes needed
   * @param stream CUDA stream
   * @return Pointer to device memory of at least requiredBytes
   */
  void* get(int key, size_t requiredBytes, cudaStream_t stream) {
    if (requiredBytes == 0) {
      return nullptr;
    }

    // A capturing stream must not use this cache. hipMallocAsync would record
    // an allocation node whose address is only valid while the graph runs, and
    // a later growth would hipFreeAsync a pointer this graph has already baked
    // in. Captures get a graph-scoped buffer instead; see
    // sharded_relay_graph_scratch.h.
    //
    // Ahead of the lock on purpose: this is a HIP runtime call, and there is no
    // reason to hold a process-wide mutex across it while relay collectives run
    // concurrently on other streams. Measured either way it is in the noise, so
    // this is hygiene rather than a fix for anything.
    struct ncclCudaGraph graph;
    if (ncclCudaGetCapturingGraph(&graph, stream) != ncclSuccess) {
      return nullptr;
    }
    if (ncclCudaGraphValid(graph)) {
      return rcclx::relay::graphScratchGet(
          this, key, requiredBytes, stream, graph);
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
        // Use async free to avoid blocking - memory will be freed after
        // all preceding operations on this stream complete
        cudaFreeAsync(entry.buffer, stream);
      }

      size_t allocSize = requiredBytes;
      if (allocSize >= 1024 * 1024) {
        // Round up to next 64MB boundary for larger buffers
        allocSize =
            ((requiredBytes + 64 * 1024 * 1024 - 1) / (64 * 1024 * 1024)) *
            (64 * 1024 * 1024);
      }

      // Use async malloc to avoid blocking - this is critical for avoiding
      // deadlocks when different ranks reach this point at different times
      // while others are waiting in NCCL collectives
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

  /**
   * Clear all cached buffers. Call during shutdown or when memory pressure is
   * high.
   */
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

  // Prevent copying
  ScratchBufferCache(const ScratchBufferCache&) = delete;
  ScratchBufferCache& operator=(const ScratchBufferCache&) = delete;

 private:
  ScratchBufferCache() = default;

  ~ScratchBufferCache() {
    // Note: Don't call cudaFree in destructor during program exit,
    // as CUDA runtime may already be shut down.
    // Buffers will be automatically freed when the process exits.
  }

  struct BufferEntry {
    void* buffer = nullptr;
    size_t size = 0;
  };

  std::mutex mutex_;
  // (device, stream, group) -> grow-only staging buffer.
  std::map<std::tuple<int, const void*, int>, BufferEntry> buffers_;
};

} // namespace

/**
 * GPU kernel for incremental reduction: output[i] += input[i]
 * Used to add received chunks directly into the buffer.
 *
 * Kernel and launcher live in sharded_relay_allreduce_kernels.cu so the
 * `__global__` body is preserved when the surrounding TU is compiled with
 * `--offload-host-only` (see comments in def_build.bzl).
 */

#define LAUNCH_INCREMENTAL_ADD_KERNEL(TYPE, output, input, count, stream) \
  launchIncrementalAddKernel<TYPE>(output, input, count, stream)

/**
 * Helper macro to dispatch incremental add kernel by datatype
 */
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

/**
 * GPU kernel for scaling: output[i] = output[i] / divisor
 * Used to compute average after sum reduction (for ncclAvg operation).
 *
 * Kernel and launcher live in sharded_relay_allreduce_kernels.cu.
 */

#define LAUNCH_SCALE_KERNEL(TYPE, data, count, divisor, stream) \
  launchScaleKernel<TYPE>(data, count, divisor, stream)

/**
 * Helper macro to dispatch scale kernel by datatype
 */
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

/**
 * GPU kernel for fused incremental add + scale:
 *   output[i] = (output[i] + input[i]) / divisor
 *
 * Combines DISPATCH_INCREMENTAL_ADD + DISPATCH_SCALE into a single HBM pass
 * (read output, read input, write output once instead of twice).  Used by
 * the active rank to merge passthrough relay scratch into recvbuff while
 * applying the AVG divisor in one fused kernel.
 *
 * When divisor == 1, this collapses to a plain incremental add — but the
 * caller should prefer DISPATCH_INCREMENTAL_ADD in that case to avoid the
 * unnecessary divide.
 *
 * Kernel and launcher live in sharded_relay_allreduce_kernels.cu.
 */

#define LAUNCH_INCREMENTAL_ADD_AND_SCALE_KERNEL( \
    TYPE, output, input, count, divisor, stream) \
  launchIncrementalAddAndScaleKernel<TYPE>(      \
      output, input, count, divisor, stream)

/**
 * Helper macro to dispatch fused incremental-add + scale kernel by datatype
 */
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

/**
 * GPU kernel for fused reduction: output[i] = (a[i] + b[i]) / divisor
 * When divisor == 1, this is a simple sum: output[i] = a[i] + b[i]
 * When divisor == 2, this computes the average: output[i] = (a[i] + b[i]) / 2
 *
 * Used by helper ranks to combine data from both active ranks and compute
 * sum or average in a single kernel launch (avoiding separate add + scale).
 *
 * Kernel and launcher live in sharded_relay_allreduce_kernels.cu.
 */

#define LAUNCH_FUSED_REDUCE_KERNEL(                       \
    TYPE, output, inputA, inputB, count, divisor, stream) \
  launchFusedReduceKernel<TYPE>(output, inputA, inputB, count, divisor, stream)

/**
 * Helper macro to dispatch fused reduce kernel by datatype
 * divisor == 1 for SUM, divisor == 2 for AVG (with 2 active ranks)
 */
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

// =========================================================================
// LOW-PRECISION DISPATCH
// =========================================================================
// Only bf16 and fp32 appear here, because lpDtypeSupported() admits only those
// and the gate declines everything else back to full precision. The `default:`
// arm is therefore unreachable rather than a silent fallthrough; it aborts the
// call instead of returning ncclSuccess having quantized nothing, which is how
// a gate bug surfaces as an error rather than as wrong numbers.
#define DISPATCH_LP(datatype, CALL, ...)                                                                      \
  do {                                                                                                        \
    switch (datatype) {                                                                                       \
      case ncclFloat32:                                                                                       \
        CALL<float>(__VA_ARGS__);                                                                             \
        break;                                                                                                \
      case ncclBfloat16:                                                                                      \
        CALL<__nv_bfloat16>(__VA_ARGS__);                                                                     \
        break;                                                                                                \
      default:                                                                                                \
        WARN(                                                                                                 \
            "Sharded relay: low precision reached an unsupported datatype %d; the eligibility gate is wrong", \
            static_cast<int>(datatype));                                                                      \
        return ncclInternalError;                                                                             \
    }                                                                                                         \
  } while (0)

#define DISPATCH_LP_QUANTIZE(datatype, wireOut, in, count, stream) \
  DISPATCH_LP(datatype, launchLpQuantizeKernel, wireOut, in, count, stream)

#define DISPATCH_LP_DEQUANTIZE(datatype, out, wireIn, count, stream) \
  DISPATCH_LP(datatype, launchLpDequantizeKernel, out, wireIn, count, stream)

#define DISPATCH_LP_MULTI_REDUCE(                           \
    datatype, dst, wireContribs, n, count, divisor, stream) \
  DISPATCH_LP(                                              \
      datatype,                                             \
      launchLpMultiReduceKernel,                            \
      dst,                                                  \
      wireContribs,                                         \
      n,                                                    \
      count,                                                \
      divisor,                                              \
      stream)

#define DISPATCH_LP_SEEDED_MULTI_REDUCE(                          \
    datatype, dst, seed, wireContribs, n, count, divisor, stream) \
  DISPATCH_LP(                                                    \
      datatype,                                                   \
      launchLpSeededMultiReduceKernel,                            \
      dst,                                                        \
      seed,                                                       \
      wireContribs,                                               \
      n,                                                          \
      count,                                                      \
      divisor,                                                    \
      stream)

namespace {

// The DISPATCH_* macros above instantiate reduce kernels for exactly these
// types and fall through silently (default: break) for anything else, so an
// unsupported datatype would return ncclSuccess having never reduced anything.
//
// This is deliberately a supported-set test rather than the
// `datatype < 0 || datatype >= ncclNumTypes` range test used by upstream
// ArgsCheck: ncclFloat8e4m3 and ncclFloat8e5m2 are valid NCCL types that
// ncclTypeSize() sizes at 1 byte, so they pass a range test but have no reduce
// kernel here. It also keeps ncclTypeSize()'s int -1 for an unknown type out of
// the `size_t elementSize` below, where it would become SIZE_MAX rather than 0.
// Keep this list in sync with the DISPATCH_* macros above.
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

// Maximum number of helper ranks supported per group.
static constexpr int SHARDED_RELAY_MAX_HELPERS = 8;

// Maximum number of active ranks per group. The hypercube exchange schedule
// (round-r partner = myActiveIndex XOR round) requires nActiveRanks to be a
// power of two; supported values are 2 and 4 (on an 8-GPU node this leaves 6
// or 4 helpers respectively).
static constexpr int SHARDED_RELAY_MAX_ACTIVE = 8;

// Returns true if v is a power of two (v >= 1).
static inline bool isPowerOfTwo(int v) {
  return v > 0 && (v & (v - 1)) == 0;
}

/**
 * Rank Configuration for Sharded Relay AllReduce
 *
 * Holds parsed active and helper rank information for a single group.
 * Supports a power-of-two number of active ranks per group (2 or 4).
 */
namespace {

struct ShardedRelayRankConfig {
  int activeRanks[SHARDED_RELAY_MAX_ACTIVE]; // Active rank IDs (power of two)
  int nActiveRanks; // Number of active ranks (2 or 4)
  int helperRanks[SHARDED_RELAY_MAX_HELPERS]; // Helper rank IDs
  int numHelpers; // Number of helper ranks
  bool isActiveRank; // Is current rank active?
  int myActiveIndex; // Index in activeRanks array (-1 if helper)
  int myHelperIndex; // Index in helperRanks array (-1 if active)
};

} // namespace

/**
 * Build rank configuration from provided active ranks array.
 * NOTE: This implementation requires exactly 2 active ranks per group.
 *
 * @param nRanks Total number of ranks in the communicator
 * @param rank Current rank ID
 * @param activeRanksInput Array of active rank IDs from caller (must have
 * exactly 2)
 * @param nActiveRanksInput Number of active ranks (must be exactly 2)
 * @param config Output configuration struct
 * @return true if configuration is valid, false otherwise
 */
static bool buildShardedRelayRankConfig(
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
  // Bounded by SHARDED_RELAY_MAX_HELPERS to prevent stack buffer overflow
  // for communicators with more than (SHARDED_RELAY_MAX_HELPERS + 2) ranks.
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

/**
 * Two-active sharded relay allreduce.
 *
 * Two comm groups; helpers reduce the pair of chunks they receive and hand the
 * result straight back to both active ranks. Each of the two direct
 * active<->active chunks rides along with one of the relay groups, so no link
 * ever idles.
 */
// Distinct ScratchBufferCache key range for kernel-owned per-group helper
// staging, so it never collides with the active-rank scratch (keys
// 0..SHARDED_RELAY_MAX_GROUPS). Lets callers pass placeholder buffers for the
// groups where they are a helper.
static constexpr int kHelperScratchKeyBase = SHARDED_RELAY_MAX_GROUPS + 1;

#define DISPATCH_ONESHOT_PUSH_REDUCE(                 \
    datatype,                                         \
    handled,                                          \
    out,                                              \
    sendBuff,                                         \
    table,                                            \
    ranks,                                            \
    nActive,                                          \
    myRank,                                           \
    mySlot,                                           \
    rc,                                               \
    srcStride,                                        \
    ownOffset,                                        \
    slotBytes,                                        \
    seq,                                              \
    divisor,                                          \
    stream)                                           \
  do {                                                \
    (handled) = true;                                 \
    switch (datatype) {                               \
      case ncclInt32:                                 \
        launchOneShotPushReduceKernel<int32_t>(       \
            out,                                      \
            sendBuff,                                 \
            table,                                    \
            ranks,                                    \
            nActive,                                  \
            myRank,                                   \
            mySlot,                                   \
            rc,                                       \
            srcStride,                                \
            ownOffset,                                \
            slotBytes,                                \
            seq,                                      \
            divisor,                                  \
            stream);                                  \
        break;                                        \
      case ncclUint32:                                \
        launchOneShotPushReduceKernel<uint32_t>(      \
            out,                                      \
            sendBuff,                                 \
            table,                                    \
            ranks,                                    \
            nActive,                                  \
            myRank,                                   \
            mySlot,                                   \
            rc,                                       \
            srcStride,                                \
            ownOffset,                                \
            slotBytes,                                \
            seq,                                      \
            divisor,                                  \
            stream);                                  \
        break;                                        \
      case ncclInt64:                                 \
        launchOneShotPushReduceKernel<int64_t>(       \
            out,                                      \
            sendBuff,                                 \
            table,                                    \
            ranks,                                    \
            nActive,                                  \
            myRank,                                   \
            mySlot,                                   \
            rc,                                       \
            srcStride,                                \
            ownOffset,                                \
            slotBytes,                                \
            seq,                                      \
            divisor,                                  \
            stream);                                  \
        break;                                        \
      case ncclUint64:                                \
        launchOneShotPushReduceKernel<uint64_t>(      \
            out,                                      \
            sendBuff,                                 \
            table,                                    \
            ranks,                                    \
            nActive,                                  \
            myRank,                                   \
            mySlot,                                   \
            rc,                                       \
            srcStride,                                \
            ownOffset,                                \
            slotBytes,                                \
            seq,                                      \
            divisor,                                  \
            stream);                                  \
        break;                                        \
      case ncclFloat16:                               \
        launchOneShotPushReduceKernel<__half>(        \
            out,                                      \
            sendBuff,                                 \
            table,                                    \
            ranks,                                    \
            nActive,                                  \
            myRank,                                   \
            mySlot,                                   \
            rc,                                       \
            srcStride,                                \
            ownOffset,                                \
            slotBytes,                                \
            seq,                                      \
            divisor,                                  \
            stream);                                  \
        break;                                        \
      case ncclFloat:                                 \
        launchOneShotPushReduceKernel<float>(         \
            out,                                      \
            sendBuff,                                 \
            table,                                    \
            ranks,                                    \
            nActive,                                  \
            myRank,                                   \
            mySlot,                                   \
            rc,                                       \
            srcStride,                                \
            ownOffset,                                \
            slotBytes,                                \
            seq,                                      \
            divisor,                                  \
            stream);                                  \
        break;                                        \
      case ncclDouble:                                \
        launchOneShotPushReduceKernel<double>(        \
            out,                                      \
            sendBuff,                                 \
            table,                                    \
            ranks,                                    \
            nActive,                                  \
            myRank,                                   \
            mySlot,                                   \
            rc,                                       \
            srcStride,                                \
            ownOffset,                                \
            slotBytes,                                \
            seq,                                      \
            divisor,                                  \
            stream);                                  \
        break;                                        \
      case ncclBfloat16:                              \
        launchOneShotPushReduceKernel<__nv_bfloat16>( \
            out,                                      \
            sendBuff,                                 \
            table,                                    \
            ranks,                                    \
            nActive,                                  \
            myRank,                                   \
            mySlot,                                   \
            rc,                                       \
            srcStride,                                \
            ownOffset,                                \
            slotBytes,                                \
            seq,                                      \
            divisor,                                  \
            stream);                                  \
        break;                                        \
      default:                                        \
        (handled) = false;                            \
        break;                                        \
    }                                                 \
  } while (0)

// Try the one-shot IPC kernel for a single-group A-active allreduce, and report
// whether it ran.
//
// The pure-direct allreduce schedules -- the A==2 full exchange and the A>2 one
// added alongside it -- are both TWO launches: one group, then one reduce. That
// is one more than NCCL, and it is the same shape the reduce-scatter had before
// the one-shot path replaced it. The fix transfers directly: allreduce is the
// srcStride == 0 case of the same push-reduce kernel, because every peer is
// owed the whole buffer rather than a per-peer block, and every rank keeps the
// whole result rather than a shard.
//
// Bytes moved are unchanged -- (A-1)*count pushed per rank, exactly what the
// group form sent -- so this trades a launch for nothing.
//
// Every predicate is derived from sizes and the communicator only, so all ranks
// reach the same decision: a rank that ran the one-shot kernel while a peer
// took the ncclSend path would spin forever rather than merely run slower.
// oneShotAcquire() is COLLECTIVE on first use, so it is called before any
// branch on myActiveGroup and it agrees its own success across ranks.
static bool tryOneShotAllReduce(
    const void* const* sendBuffs,
    void* const* recvBuffs,
    const size_t* counts,
    ncclDataType_t datatype,
    int reductionDivisor,
    ncclComm_t comm,
    cudaStream_t stream,
    const ShardedRelayRankConfig* configs,
    int myActiveGroup,
    int nActiveRanksPerGroup,
    int nGroups,
    size_t elementSize) {
  const size_t maxCount = rcclx::relay::relayMaxCount(counts, nGroups);
  if (nGroups != 1 || maxCount == 0) {
    return false;
  }
  // Each of the A staging slots holds a whole contribution here, not a shard,
  // so the gate is the per-rank count directly.
  if (maxCount * elementSize > rcclx::relay::kRelayOneShotMaxBytes) {
    return false;
  }
  // Creating the region is not capturable: it does a bootstrap all-gather and a
  // synchronous hipMemset. Using one that already exists is fine, so under
  // capture take the path only if the region is already up.
  struct ncclCudaGraph graph;
  if (ncclCudaGetCapturingGraph(&graph, stream) != ncclSuccess) {
    return false;
  }
  if (ncclCudaGraphValid(graph) && !rcclx::relay::oneShotReady(comm)) {
    return false;
  }

  rcclx::relay::OneShotLaunch osl{};
  if (!rcclx::relay::oneShotAcquire(comm, &osl)) {
    return false;
  }

  // Helpers have nothing to do, but they DID have to reach the acquire above,
  // which is the whole reason it sits before this branch.
  if (myActiveGroup < 0 || counts[myActiveGroup] == 0) {
    return true;
  }

  const ShardedRelayRankConfig& cfg = configs[myActiveGroup];
  rcclx::relay::OneShotRanks ranks{};
  for (int a = 0; a < nActiveRanksPerGroup; a++) {
    ranks.r[a] = cfg.activeRanks[a];
  }
  bool handled = false;
  DISPATCH_ONESHOT_PUSH_REDUCE(
      datatype,
      handled,
      recvBuffs[myActiveGroup],
      sendBuffs[myActiveGroup],
      osl.table,
      ranks,
      nActiveRanksPerGroup,
      comm->rank,
      cfg.myActiveIndex,
      counts[myActiveGroup],
      /*srcStride=*/0,
      /*ownOffset=*/0,
      osl.slotBytes,
      osl.seq,
      reductionDivisor,
      stream);
  // An unsupported datatype must not silently produce nothing. Falling back is
  // safe only because the dtype is identical on every rank, so either all of
  // them fall back or none do -- and in pure-direct mode the helpers post
  // nothing anyway, so a helper that already returned true is equivalent.
  return handled;
}

namespace {

/**
 * The wire buffers one 2-active relay call needs, carved from the
 * communicator's low-precision arena.
 *
 * Every region is carved UNCONDITIONALLY and at the WORST CASE over all groups,
 * even though a rank is active for exactly one group and a helper for the rest.
 * That is deliberate on two counts. It makes the partition byte-identical on
 * every rank and on every call with the same geometry, which is what lets a
 * captured graph replay against the same addresses. And it makes the footprint
 * a function of the counts alone, so the capacity check below is
 * rank-independent -- a rank that sized its own roles would decline on a
 * different set of calls than its peers, and low precision has to be unanimous
 * or the two disagree on wire byte counts and the call hangs.
 */
struct A2LpPlan {
  char* sendShadow{nullptr}; // wire(count): the whole active send buffer
  char* relayRecv{
      nullptr}; // wire(relayTotal): mirrors recvBuff's relayed region
  char* directRecv{nullptr}; // wire(direct region)
  char* helper[SHARDED_RELAY_MAX_GROUPS]{};
  bool valid{false};
};

// The size-and-dtype half of the gate, in one place so the dispatcher cannot
// accidentally feed it a different size metric than the route selector uses.
rcclx::relay::LpGateInputs allReduceLpGate(
    ncclDataType_t datatype,
    const size_t* counts,
    int nGroups,
    int nActiveRanksPerGroup,
    size_t elementSize,
    bool relayRouteSelected) {
  rcclx::relay::LpGateInputs in;
  in.coll = rcclx::relay::LpCollective::AllReduce;
  in.datatype = datatype;
  in.counts = counts;
  in.nGroups = nGroups;
  in.nActiveRanksPerGroup = nActiveRanksPerGroup;
  // max(counts) * elementSize is exactly selectAllReduceRoute()'s metric, so
  // the low-precision threshold and the route threshold are directly
  // comparable.
  in.routeSizeBytes =
      rcclx::relay::relayMaxCount(counts, nGroups) * elementSize;
  in.relayRouteSelected = relayRouteSelected;
  return in;
}

size_t a2LpAlign(size_t bytes) {
  return ((bytes + rcclx::relay::LpArenaCarver::kAlign - 1) /
          rcclx::relay::LpArenaCarver::kAlign) *
      rcclx::relay::LpArenaCarver::kAlign;
}

// Bytes one call needs, derived only from the counts and the chunk geometry
// (which is itself derived only from the counts), so every rank computes the
// same number.
size_t a2LpRequiredBytes(
    const size_t* counts,
    const size_t* relayTotals,
    const size_t* chunkSizes,
    int nGroups,
    int nActiveRanks) {
  const size_t maxCount = rcclx::relay::relayMaxCount(counts, nGroups);
  const size_t maxRelay = rcclx::relay::relayMaxCount(relayTotals, nGroups);
  const size_t maxChunk = rcclx::relay::relayMaxCount(chunkSizes, nGroups);
  // directRecv is bounded by maxCount rather than max(count - relayTotal),
  // which costs a little arena and removes a per-group subtraction from a
  // number every rank has to agree on.
  size_t total = a2LpAlign(rcclx::relay::lpWireBytes(maxCount)) +
      a2LpAlign(rcclx::relay::lpWireBytes(maxRelay)) +
      a2LpAlign(rcclx::relay::lpWireBytes(maxCount));
  total += static_cast<size_t>(nGroups) *
      a2LpAlign(
               static_cast<size_t>(nActiveRanks) *
               rcclx::relay::lpWireBytes(maxChunk));
  return total;
}

A2LpPlan a2LpCarve(
    const rcclx::relay::LpArenaLease& lease,
    const size_t* counts,
    const size_t* relayTotals,
    const size_t* chunkSizes,
    int nGroups,
    int nActiveRanks) {
  const size_t maxCount = rcclx::relay::relayMaxCount(counts, nGroups);
  const size_t maxRelay = rcclx::relay::relayMaxCount(relayTotals, nGroups);
  const size_t maxChunk = rcclx::relay::relayMaxCount(chunkSizes, nGroups);

  A2LpPlan p{};
  rcclx::relay::LpArenaCarver carver(lease);
  p.sendShadow = carver.take(rcclx::relay::lpWireBytes(maxCount));
  p.relayRecv = carver.take(rcclx::relay::lpWireBytes(maxRelay));
  p.directRecv = carver.take(rcclx::relay::lpWireBytes(maxCount));
  for (int g = 0; g < nGroups; g++) {
    p.helper[g] = carver.take(
        static_cast<size_t>(nActiveRanks) *
        rcclx::relay::lpWireBytes(maxChunk));
  }
  p.valid = carver.ok();
  return p;
}

/**
 * Every region boundary this schedule sends or receives at is a whole number of
 * wire blocks.
 *
 * 128-aligned per-group counts (what lpEligible() checks) make relayTotal, dirA
 * and dirB aligned -- but ONLY when the aligned chunk size is non-zero. When
 * count / numChunks floors to zero the geometry falls back to splitting the
 * buffer in HALF, and count / 2 is a multiple of 64, not of 128. That is not a
 * tail-only error: dirB starts at count/2, so under low precision the boundary
 * lands mid-block and both directions of the exchange disagree about where the
 * second chunk begins.
 *
 * The degenerate branch is easy to reach in a FUSED call, where one small group
 * rides along with a large one: the gate only ever sees the largest group's
 * size, so counts={2097152, 384} passes on the big sibling while the 384
 * element group takes the halving branch. Its own size is a legal 128 multiple,
 * which is why the count check cannot catch this.
 *
 * Pure function of the counts and the chunk geometry (itself a pure function of
 * the counts), so every rank declines together.
 */
bool a2LpGeometryOk(
    const size_t* counts,
    const size_t* chunkSizes,
    int nGroups) {
  for (int g = 0; g < nGroups; g++) {
    if (counts[g] > 0 && chunkSizes[g] == 0) {
      return false;
    }
  }
  return true;
}

/**
 * Turn the caller's request into a decision, and carve the buffers if it holds.
 *
 * COLLECTIVE: lpArenaAcquire() runs a bootstrap unanimity vote on first use, so
 * every rank must reach this whenever the dispatcher's size-only gate said yes
 * -- including the helper ranks, which is why it is called before any role
 * branch. Every reason it can return false is derived from the counts or is
 * already agreed across the communicator, so all ranks decline together.
 */
bool a2LpPrepare(
    bool wantLp,
    ncclComm_t comm,
    cudaStream_t stream,
    const size_t* counts,
    const size_t* relayTotals,
    const size_t* chunkSizes,
    int nGroups,
    int nActiveRanks,
    A2LpPlan* out) {
  if (!wantLp) {
    return false;
  }

  if (!a2LpGeometryOk(counts, chunkSizes, nGroups)) {
    rcclx::relay::lpRecordDecline(rcclx::relay::LpDecline::Alignment);
    return false;
  }

  // Creating the arena is not capturable: it runs a bootstrap all-gather. Using
  // one that already exists is fine, so under capture take the path only if the
  // arena is already up. Same precedent as the one-shot region.
  struct ncclCudaGraph graph;
  if (ncclCudaGetCapturingGraph(&graph, stream) != ncclSuccess) {
    rcclx::relay::lpRecordDecline(rcclx::relay::LpDecline::GraphCapture);
    return false;
  }
  if (ncclCudaGraphValid(graph) && !rcclx::relay::lpArenaReady(comm)) {
    rcclx::relay::lpRecordDecline(rcclx::relay::LpDecline::GraphCapture);
    return false;
  }

  rcclx::relay::LpArenaLease lease{};
  if (!rcclx::relay::lpArenaAcquire(comm, &lease)) {
    rcclx::relay::lpRecordDecline(rcclx::relay::LpDecline::Arena);
    return false;
  }
  if (a2LpRequiredBytes(
          counts, relayTotals, chunkSizes, nGroups, nActiveRanks) >
      lease.bytes) {
    rcclx::relay::lpRecordDecline(rcclx::relay::LpDecline::Arena);
    return false;
  }

  *out =
      a2LpCarve(lease, counts, relayTotals, chunkSizes, nGroups, nActiveRanks);
  if (!out->valid) {
    rcclx::relay::lpRecordDecline(rcclx::relay::LpDecline::Arena);
    return false;
  }
  rcclx::relay::lpRecordEngage();
  return true;
}

} // namespace

static ncclResult_t shardedRelayAllReduce2Active(
    const void* const* sendBuffs,
    void* const* recvBuffs,
    const size_t* counts,
    ncclDataType_t datatype,
    int reductionDivisor,
    ncclComm_t comm,
    cudaStream_t stream,
    const ShardedRelayRankConfig* configs,
    int myActiveGroup,
    int numHelpers,
    int nGroups,
    size_t elementSize,
    bool wantLp) {
  // ==========================================================================
  // SIZE-ADAPTIVE PURE-DIRECT FAST PATH (A==2)
  // ==========================================================================
  // At small sizes the helper relay is dominated by launch and handshake
  // latency, not bandwidth. Fall back to a single full exchange between the two
  // active ranks (helpers idle) and reduce locally. A reduce-scatter +
  // all-gather pair would move the same count per link direction (count/2
  // twice) but needs TWO group boundaries, so the plain exchange is strictly
  // better in this regime. The size -> route mapping lives in
  // selectAllReduceRoute() so the tests assert the same definition this
  // dispatch uses. This function is the A==2 path, so the selector is asked
  // about A==2.
  // The pure-direct route is launch-bound, not bandwidth-bound: low precision
  // would only add launches there, so the gate already excludes it and this
  // branch never sees wantLp true. Asserted by construction rather than
  // re-checked -- see the gate in the dispatcher.
  if (rcclx::relay::selectAllReduceRoute(2, nGroups, counts, elementSize) ==
      rcclx::relay::AllReduceRoute::PureDirect) {
    // One kernel instead of the group-plus-reduce pair below. See
    // tryOneShotAllReduce().
    if (tryOneShotAllReduce(
            sendBuffs,
            recvBuffs,
            counts,
            datatype,
            reductionDivisor,
            comm,
            stream,
            configs,
            myActiveGroup,
            2,
            nGroups,
            elementSize)) {
      return ncclSuccess;
    }

    void* pdScratch = nullptr;
    if (myActiveGroup >= 0 && counts[myActiveGroup] > 0) {
      pdScratch = ScratchBufferCache::getInstance().get(
          SHARDED_RELAY_MAX_GROUPS,
          counts[myActiveGroup] * elementSize,
          stream);
      if (pdScratch == nullptr) {
        return ncclInternalError;
      }
    }

    NCCLCHECK(ncclGroupStart());
    for (int g = 0; g < nGroups; g++) {
      if (counts[g] == 0) {
        continue;
      }
      const ShardedRelayRankConfig& cfg = configs[g];
      if (!cfg.isActiveRank) {
        continue; // helpers idle
      }
      int partner = cfg.activeRanks[1 - cfg.myActiveIndex];
      NCCLCHECK(
          ncclSend(sendBuffs[g], counts[g], datatype, partner, comm, stream));
      NCCLCHECK(
          ncclRecv(pdScratch, counts[g], datatype, partner, comm, stream));
    }
    NCCLCHECK(ncclGroupEnd());

    if (myActiveGroup >= 0 && counts[myActiveGroup] > 0) {
      size_t count = counts[myActiveGroup];
      void* out = recvBuffs[myActiveGroup];
      if (sendBuffs[myActiveGroup] == out) {
        if (reductionDivisor > 1) {
          DISPATCH_INCREMENTAL_ADD_AND_SCALE(
              datatype, out, pdScratch, count, reductionDivisor, stream);
        } else {
          DISPATCH_INCREMENTAL_ADD(datatype, out, pdScratch, count, stream);
        }
      } else {
        DISPATCH_FUSED_REDUCE(
            datatype,
            out,
            sendBuffs[myActiveGroup],
            pdScratch,
            count,
            reductionDivisor,
            stream);
      }
    }
    return ncclSuccess;
  }

  // =========================================================================
  // CHUNK GEOMETRY: numHelpers relayed chunks + TWO direct chunks
  // =========================================================================
  // The active<->active link carries nothing while the relay scatter and
  // forward run on the cross links, so instead of a third comm group for a
  // single direct chunk, one direct chunk rides along with each relay group.
  // With numChunks = numHelpers + 2 every link then carries exactly one chunk
  // per direction per group, making the critical path 2*count/numChunks. A
  // separate direct phase costs 3*count/(numHelpers+1) — 1.7x more on an 8-GPU
  // node.
  const int numChunks = numHelpers + 2;

  size_t chunkSizes[SHARDED_RELAY_MAX_GROUPS];
  size_t relayTotals[SHARDED_RELAY_MAX_GROUPS]; // == direct chunk A's offset
  size_t dirASizes[SHARDED_RELAY_MAX_GROUPS];
  size_t dirBOffsets[SHARDED_RELAY_MAX_GROUPS];
  size_t dirBSizes[SHARDED_RELAY_MAX_GROUPS]; // absorbs the remainder

  for (int g = 0; g < nGroups; g++) {
    size_t count = counts[g];

    // Zero-count groups are skipped by every loop below, so their geometry is
    // never read.
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
  // LOW PRECISION: decide, acquire the arena, and quantize the send buffer
  // =========================================================================
  // Collective: every rank reaches a2LpPrepare() when the dispatcher's
  // size-only gate said yes, and every way it can decline is agreed across the
  // communicator, so all ranks run the same wire format or none do.
  A2LpPlan lpPlan{};
  const bool lp = a2LpPrepare(
      wantLp,
      comm,
      stream,
      counts,
      relayTotals,
      chunkSizes,
      nGroups,
      configs[0].nActiveRanks,
      &lpPlan);
  const rcclx::relay::RelayWire wire =
      rcclx::relay::lpWireFor(datatype, elementSize, lp);

  // One quantize per group over the ENTIRE boundary-crossing send region,
  // before the first ncclGroupStart. Every byte of the active buffer crosses a
  // rank boundary here -- the relayed chunks cover [0, relayTotal) and the two
  // direct chunks cover [relayTotal, count) -- so this is the whole buffer.
  // Hoisted rather than done per chunk: per-chunk would cost numChunks launches
  // instead of one for no benefit, since nothing between them reads the result.
  if (lp && myActiveGroup >= 0 && counts[myActiveGroup] > 0) {
    DISPATCH_LP_QUANTIZE(
        datatype,
        lpPlan.sendShadow,
        sendBuffs[myActiveGroup],
        counts[myActiveGroup],
        stream);
  }

  // Where a boundary-crossing send reads from: the quantized shadow under low
  // precision, the caller's buffer otherwise. Offsets are in ELEMENTS of the
  // caller's dtype either way, which is what keeps every call site below
  // unchanged in shape.
  auto sendFrom = [&](int g, size_t offsetElems) -> const char* {
    if (lp) {
      return lpPlan.sendShadow + wire.bytes(offsetElems);
    }
    return static_cast<const char*>(sendBuffs[g]) + offsetElems * elementSize;
  };

  // =========================================================================
  // SCRATCH: the two received direct chunks, contiguous (in-place only)
  // =========================================================================
  // Out-of-place receives the direct chunks straight into recvBuff and reduces
  // against sendBuff. In-place must stage them so the local contribution is
  // still readable when the fused reduce runs.
  void* directScratch = nullptr;
  bool isInPlace = false;
  size_t myDirOffset = 0; // offset of direct chunk A within the active buffers
  size_t myDirTotal = 0; // both direct chunks together

  if (myActiveGroup >= 0 && counts[myActiveGroup] > 0) {
    isInPlace = (sendBuffs[myActiveGroup] == recvBuffs[myActiveGroup]);
    myDirOffset = relayTotals[myActiveGroup];
    myDirTotal = counts[myActiveGroup] - myDirOffset;
    // Under low precision the arrivals are wire bytes, so they cannot land in
    // recvBuff even out-of-place: lpPlan.directRecv stages them in both cases
    // and the closing fold reads them from there.
    if (isInPlace && !lp) {
      directScratch = ScratchBufferCache::getInstance().get(
          myActiveGroup, myDirTotal * elementSize, stream);
      if (directScratch == nullptr) {
        return ncclInternalError;
      }
    }
  }

  // Destination for received direct chunk A / chunk B.
  auto directDst = [&](size_t offsetWithinDirect) -> char* {
    if (lp) {
      return lpPlan.directRecv + wire.bytes(offsetWithinDirect);
    }
    if (isInPlace) {
      return static_cast<char*>(directScratch) +
          offsetWithinDirect * elementSize;
    }
    return static_cast<char*>(recvBuffs[myActiveGroup]) +
        (myDirOffset + offsetWithinDirect) * elementSize;
  };

  // Helper staging is kernel-owned scratch: callers pass a placeholder buffer
  // for groups where they are a helper. Each helper group needs nActiveRanks
  // chunks (to receive both actives' chunks, reduce, and forward).
  void* helperScratch[SHARDED_RELAY_MAX_GROUPS] = {nullptr};
  for (int g = 0; g < nGroups; g++) {
    const ShardedRelayRankConfig& cfg = configs[g];
    if (!cfg.isActiveRank && counts[g] > 0 && chunkSizes[g] > 0) {
      if (lp) {
        // Already carved, and in wire bytes: the helper reduces and forwards
        // wire blocks without ever learning the caller's dtype.
        helperScratch[g] = lpPlan.helper[g];
        continue;
      }
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
    if (counts[g] == 0)
      continue;
    const ShardedRelayRankConfig& cfg = configs[g];
    size_t chunkSize = chunkSizes[g];

    if (cfg.isActiveRank) {
      // Scatter: chunk h goes to helper h.
      for (int h = 0; h < cfg.numHelpers && chunkSize > 0; h++) {
        NCCLCHECK(ncclSend(
            sendFrom(g, static_cast<size_t>(h) * chunkSize),
            wire.count(chunkSize),
            wire.dtype,
            cfg.helperRanks[h],
            comm,
            stream));
      }

      // Direct chunk A over the otherwise-idle active<->active link.
      int partner = cfg.activeRanks[1 - cfg.myActiveIndex];
      if (dirASizes[g] > 0) {
        NCCLCHECK(ncclSend(
            sendFrom(g, relayTotals[g]),
            wire.count(dirASizes[g]),
            wire.dtype,
            partner,
            comm,
            stream));
        NCCLCHECK(ncclRecv(
            directDst(0),
            wire.count(dirASizes[g]),
            wire.dtype,
            partner,
            comm,
            stream));
      }
    } else if (chunkSize > 0) {
      // Helper: receive active rank a's chunk into slot a.
      char* helperBuf = static_cast<char*>(helperScratch[g]);
      for (int a = 0; a < cfg.nActiveRanks; a++) {
        NCCLCHECK(ncclRecv(
            helperBuf + wire.bytes(static_cast<size_t>(a) * chunkSize),
            wire.count(chunkSize),
            wire.dtype,
            cfg.activeRanks[a],
            comm,
            stream));
      }
    }
  }

  NCCLCHECK(ncclGroupEnd());

  // =========================================================================
  // HELPER REDUCE: slot 0 = (slot 0 + slot 1) / divisor
  // =========================================================================
  // Both active ranks send the SAME logical chunk index, so their sum is the
  // final allreduced value for that chunk. Reducing here instead of forwarding
  // both slots keeps the link cost identical (the helper still sends one chunk
  // to each active rank) while removing the active rank's relay scratch and its
  // fused add+scale over numHelpers/numChunks of the whole buffer. The work is
  // also spread over every helper GPU rather than piled onto the two actives.
  for (int g = 0; g < nGroups; g++) {
    if (counts[g] == 0 || chunkSizes[g] == 0 || configs[g].isActiveRank)
      continue;
    char* helperBuf = static_cast<char*>(helperScratch[g]);
    size_t chunkSize = chunkSizes[g];
    if (lp) {
      // Wire in, wire out, summed in fp32. The divisor lands HERE and not on
      // the active rank because the reduced chunk below goes straight to its
      // final place -- there is no active-side reduce for this region to defer
      // it to. Exact: the divisor is a power of two. See
      // sharded_relay_lp_kernels.h.
      launchLpReduceRequantizeKernel(
          helperBuf,
          helperBuf,
          configs[g].nActiveRanks,
          chunkSize,
          reductionDivisor,
          stream);
      continue;
    }
    DISPATCH_FUSED_REDUCE(
        datatype,
        helperBuf,
        helperBuf,
        helperBuf + chunkSize * elementSize,
        chunkSize,
        reductionDivisor,
        stream);
  }

  // =========================================================================
  // GROUP 2: reduced relay (helpers->active) || direct chunk B
  // (active<->active)
  // =========================================================================
  NCCLCHECK(ncclGroupStart());

  for (int g = 0; g < nGroups; g++) {
    if (counts[g] == 0)
      continue;
    const ShardedRelayRankConfig& cfg = configs[g];
    size_t chunkSize = chunkSizes[g];

    if (cfg.isActiveRank) {
      char* recvbuff = static_cast<char*>(recvBuffs[g]);

      // The helper's chunk is already reduced, so at full precision it lands
      // directly in its final place in recvBuff. Under low precision it arrives
      // as wire bytes, so it stages in lpPlan.relayRecv -- laid out to mirror
      // recvBuff's relayed region -- and one dequantize below writes the whole
      // region out at once.
      for (int h = 0; h < cfg.numHelpers && chunkSize > 0; h++) {
        const size_t offsetElems = static_cast<size_t>(h) * chunkSize;
        char* dst = lp ? (lpPlan.relayRecv + wire.bytes(offsetElems))
                       : (recvbuff + offsetElems * elementSize);
        NCCLCHECK(ncclRecv(
            dst,
            wire.count(chunkSize),
            wire.dtype,
            cfg.helperRanks[h],
            comm,
            stream));
      }

      // Direct chunk B, again over the idle active<->active link.
      int partner = cfg.activeRanks[1 - cfg.myActiveIndex];
      NCCLCHECK(ncclSend(
          sendFrom(g, dirBOffsets[g]),
          wire.count(dirBSizes[g]),
          wire.dtype,
          partner,
          comm,
          stream));
      NCCLCHECK(ncclRecv(
          directDst(dirASizes[g]),
          wire.count(dirBSizes[g]),
          wire.dtype,
          partner,
          comm,
          stream));
    } else if (chunkSize > 0) {
      // Helper: hand the reduced chunk to both active ranks.
      const char* helperBuf = static_cast<const char*>(helperScratch[g]);
      for (int a = 0; a < cfg.nActiveRanks; a++) {
        NCCLCHECK(ncclSend(
            helperBuf,
            wire.count(chunkSize),
            wire.dtype,
            cfg.activeRanks[a],
            comm,
            stream));
      }
    }
  }

  NCCLCHECK(ncclGroupEnd());

  // =========================================================================
  // LOW PRECISION: land the relayed region
  // =========================================================================
  // One launch for the whole [0, relayTotal) region, which is already final --
  // the helpers reduced it and applied the divisor. Nothing else touches it.
  if (lp && myActiveGroup >= 0 && relayTotals[myActiveGroup] > 0) {
    DISPATCH_LP_DEQUANTIZE(
        datatype,
        recvBuffs[myActiveGroup],
        lpPlan.relayRecv,
        relayTotals[myActiveGroup],
        stream);
  }

  // =========================================================================
  // DIRECT REDUCE: fold both received direct chunks in one fused pass
  // =========================================================================
  // Chunks A and B are adjacent in both the active buffers and the scratch, so
  // a single launch covers [relayTotal, count).
  if (myActiveGroup >= 0 && myDirTotal > 0) {
    char* dst = static_cast<char*>(recvBuffs[myActiveGroup]) +
        myDirOffset * elementSize;
    if (lp) {
      // One wire contribution (the partner's two direct chunks, contiguous)
      // folded against this rank's own, accumulated in fp32. In-place reads its
      // contribution from dst; out-of-place seeds from sendBuff.
      if (isInPlace) {
        DISPATCH_LP_MULTI_REDUCE(
            datatype,
            dst,
            lpPlan.directRecv,
            1,
            myDirTotal,
            reductionDivisor,
            stream);
      } else {
        DISPATCH_LP_SEEDED_MULTI_REDUCE(
            datatype,
            dst,
            static_cast<const char*>(sendBuffs[myActiveGroup]) +
                myDirOffset * elementSize,
            lpPlan.directRecv,
            1,
            myDirTotal,
            reductionDivisor,
            stream);
      }
    } else if (isInPlace) {
      if (reductionDivisor > 1) {
        DISPATCH_INCREMENTAL_ADD_AND_SCALE(
            datatype, dst, directScratch, myDirTotal, reductionDivisor, stream);
      } else {
        DISPATCH_INCREMENTAL_ADD(
            datatype, dst, directScratch, myDirTotal, stream);
      }
    } else {
      DISPATCH_FUSED_REDUCE(
          datatype,
          dst,
          static_cast<const char*>(sendBuffs[myActiveGroup]) +
              myDirOffset * elementSize,
          dst,
          myDirTotal,
          reductionDivisor,
          stream);
    }
  }

  return ncclSuccess;
}

/**
 * Software-pipelined single-group 2-active sharded relay allreduce.
 *
 * Same logical collective and the same reduce-at-helper as
 * shardedRelayAllReduce2Active, but for nGroups == 1 -- where the active ranks
 * and the helpers are disjoint sets -- the relay is tiled and pipelined so both
 * directions of every cross link stay busy. See relayPipelineTiles() for why
 * the two-group schedule cannot do that and what it costs.
 *
 * With T tiles and unit u = align(count / ((H+1)*T + 1)):
 *   [0, H*T*u)     relay region; helper h owns [h*T*u, (h+1)*T*u), its tile t
 * at h*T*u + t*u [H*T*u, count) direct region as T+1 chunks of u, the last
 * absorbing the
 *                  /((H+1)*T + 1) remainder and the alignment loss
 *
 * Group k, for k in [0, T]: the active rank scatters tile k (k < T) to every
 * helper, receives helper h's REDUCED tile k-1 (k > 0) straight into its final
 * place in recvBuff, and exchanges direct chunk k over the active<->active
 * link. Helper h receives tile k into ping-pong buffer k%2 and forwards the
 * already reduced buffer (k-1)%2 to both active ranks.
 *
 * The helper's reduce of tile k is issued between group k, which receives it,
 * and group k+1, which sends it -- so it is one launch per tile over u elements
 * instead of one over the whole chunk. Both active ranks send the same logical
 * tile index, so their sum is already the final allreduced value and the return
 * hop stays one chunk per active rank.
 *
 * Both in-place and out-of-place are supported. The relay region cannot alias
 * dangerously even in place: group k reads tile k from sendBuff while writing
 * tile k-1 into recvBuff, which are different offsets.
 */
static ncclResult_t shardedRelayAllReduce2ActivePipelined(
    const void* const* sendBuffs,
    void* const* recvBuffs,
    const size_t* counts,
    ncclDataType_t datatype,
    int reductionDivisor,
    ncclComm_t comm,
    cudaStream_t stream,
    const ShardedRelayRankConfig* configs,
    int myActiveGroup,
    int numHelpers,
    int nTiles,
    size_t elementSize,
    bool wantLp) {
  const ShardedRelayRankConfig& cfg = configs[0];
  const size_t count = counts[0];
  const int H = numHelpers;
  const int T = nTiles;
  const size_t u =
      ((count / (static_cast<size_t>(H + 1) * T + 1)) / CHUNK_ALIGN_ELEMENTS) *
      CHUNK_ALIGN_ELEMENTS;
  if (u == 0) {
    return ncclInvalidArgument;
  }
  const size_t tileStride = static_cast<size_t>(T) * u;
  const size_t directBase = static_cast<size_t>(H) * tileStride;
  const size_t directTotal = count - directBase;
  const size_t lastDirect = directTotal - tileStride;

  // =========================================================================
  // LOW PRECISION: decide, acquire the arena, and quantize the send buffer
  // =========================================================================
  // The plan's region set fits this schedule exactly once the geometry is
  // described in its terms: the relayed span is [0, directBase), and a helper
  // needs nActiveRanks * 2 ping-pong slots of u, which is nActiveRanks slots of
  // 2u. Both are 128-multiples, so wire.bytes() is additive across them and
  // wire.bytes(2u) == 2 * wire.bytes(u).
  const size_t lpRelayTotals[1] = {directBase};
  const size_t lpChunkSizes[1] = {2 * u};
  A2LpPlan lpPlan{};
  const bool lp = a2LpPrepare(
      wantLp,
      comm,
      stream,
      counts,
      lpRelayTotals,
      lpChunkSizes,
      /*nGroups=*/1,
      cfg.nActiveRanks,
      &lpPlan);
  const rcclx::relay::RelayWire wire =
      rcclx::relay::lpWireFor(datatype, elementSize, lp);

  // One quantize over the whole send buffer, before the first group of the
  // first stage. Hoisted out of the k-loop deliberately: per-tile would cost T
  // launches, and worse, in the in-place case tile k's fold can still be in
  // flight while tile k+1's send source is being read.
  if (lp && cfg.isActiveRank && count > 0) {
    DISPATCH_LP_QUANTIZE(
        datatype, lpPlan.sendShadow, sendBuffs[0], count, stream);
  }

  auto sendFrom = [&](size_t offsetElems) -> const char* {
    if (lp) {
      return lpPlan.sendShadow + wire.bytes(offsetElems);
    }
    return static_cast<const char*>(sendBuffs[0]) + offsetElems * elementSize;
  };

  // In-place must stage the received direct chunks so the local contribution is
  // still readable when the fused reduce runs; out-of-place lands them straight
  // in recvBuff and reduces against sendBuff.
  void* directScratch = nullptr;
  const bool isInPlace = (myActiveGroup == 0) && (sendBuffs[0] == recvBuffs[0]);
  if (myActiveGroup == 0 && isInPlace && !lp) {
    directScratch = ScratchBufferCache::getInstance().get(
        0, directTotal * elementSize, stream);
    if (directScratch == nullptr) {
      return ncclInternalError;
    }
  }
  auto directDst = [&](size_t offsetWithinDirect) -> char* {
    if (lp) {
      // Wire bytes cannot land in recvBuff even out-of-place.
      return lpPlan.directRecv + wire.bytes(offsetWithinDirect);
    }
    if (isInPlace) {
      return static_cast<char*>(directScratch) +
          offsetWithinDirect * elementSize;
    }
    return static_cast<char*>(recvBuffs[0]) +
        (directBase + offsetWithinDirect) * elementSize;
  };

  // Helper staging: two ping-pong slot pairs, one pair per pipeline stage in
  // flight. Small enough that a tile is reduced and forwarded out of cache.
  char* hbuff = nullptr;
  if (!cfg.isActiveRank) {
    if (lp) {
      hbuff = lpPlan.helper[0];
    } else {
      hbuff = static_cast<char*>(ScratchBufferCache::getInstance().get(
          kHelperScratchKeyBase,
          static_cast<size_t>(cfg.nActiveRanks) * 2 * u * elementSize,
          stream));
      if (hbuff == nullptr) {
        return ncclInternalError;
      }
    }
  }
  // Slot for active source a of the buffer that stage k uses.
  //
  // STAGE-MAJOR: the slot index is (k%2)*nActiveRanks + a, so the sources of
  // one stage are ADJACENT. Source-major -- (a*2 + k%2) -- put them two slots
  // apart, which is equivalent while the reduce takes two explicit pointers,
  // but not once it takes a base plus a contribution stride. Indices still
  // cover 0..2A-1 exactly once, and the allocation is the same size, so this is
  // behaviour-neutral.
  auto helperSlot = [&](int a, int k) -> char* {
    return hbuff +
        wire.bytes(
            (static_cast<size_t>(k % 2) *
                 static_cast<size_t>(cfg.nActiveRanks) +
             static_cast<size_t>(a)) *
            u);
  };

  for (int k = 0; k <= T; k++) {
    NCCLCHECK(ncclGroupStart());
    if (cfg.isActiveRank) {
      const char* sendbuff = static_cast<const char*>(sendBuffs[0]);
      char* recvbuff = static_cast<char*>(recvBuffs[0]);
      const int partner = cfg.activeRanks[1 - cfg.myActiveIndex];
      const size_t directOffset = static_cast<size_t>(k) * u;
      const size_t directSize = (k < T) ? u : lastDirect;

      if (k < T) {
        for (int h = 0; h < H; h++) {
          NCCLCHECK(ncclSend(
              sendFrom(
                  static_cast<size_t>(h) * tileStride +
                  static_cast<size_t>(k) * u),
              wire.count(u),
              wire.dtype,
              cfg.helperRanks[h],
              comm,
              stream));
        }
      }
      NCCLCHECK(ncclSend(
          sendFrom(directBase + directOffset),
          wire.count(directSize),
          wire.dtype,
          partner,
          comm,
          stream));
      if (k > 0) {
        // Already reduced by the helper, so this is its final value. At full
        // precision it lands where it belongs in recvBuff; under low precision
        // it stages in lpPlan.relayRecv, laid out to mirror recvBuff's relayed
        // span, and one dequantize after the loop writes [0, directBase) out at
        // once.
        for (int h = 0; h < H; h++) {
          const size_t offsetElems = static_cast<size_t>(h) * tileStride +
              static_cast<size_t>(k - 1) * u;
          char* dst = lp ? (lpPlan.relayRecv + wire.bytes(offsetElems))
                         : (recvbuff + offsetElems * elementSize);
          NCCLCHECK(ncclRecv(
              dst,
              wire.count(u),
              wire.dtype,
              cfg.helperRanks[h],
              comm,
              stream));
        }
      }
      NCCLCHECK(ncclRecv(
          directDst(directOffset),
          wire.count(directSize),
          wire.dtype,
          partner,
          comm,
          stream));
    } else {
      if (k < T) {
        for (int a = 0; a < cfg.nActiveRanks; a++) {
          NCCLCHECK(ncclRecv(
              helperSlot(a, k),
              wire.count(u),
              wire.dtype,
              cfg.activeRanks[a],
              comm,
              stream));
        }
      }
      if (k > 0) {
        for (int a = 0; a < cfg.nActiveRanks; a++) {
          NCCLCHECK(ncclSend(
              helperSlot(0, k - 1),
              wire.count(u),
              wire.dtype,
              cfg.activeRanks[a],
              comm,
              stream));
        }
      }
    }
    NCCLCHECK(ncclGroupEnd());

    // Reduce the tile this group received, before the next group forwards it.
    if (!cfg.isActiveRank && k < T) {
      if (lp) {
        // The stage's sources are adjacent slots, which is what the stage-major
        // layout above is for: this kernel reads nActiveRanks contributions
        // from one base at wire.bytes(u) stride.
        launchLpReduceRequantizeKernel(
            helperSlot(0, k),
            helperSlot(0, k),
            cfg.nActiveRanks,
            u,
            reductionDivisor,
            stream);
      } else {
        DISPATCH_FUSED_REDUCE(
            datatype,
            helperSlot(0, k),
            helperSlot(0, k),
            helperSlot(1, k),
            u,
            reductionDivisor,
            stream);
      }
    }
  }

  // Land the relayed span. One launch for [0, directBase), which the helpers
  // already reduced and already divided.
  if (lp && cfg.isActiveRank && directBase > 0) {
    DISPATCH_LP_DEQUANTIZE(
        datatype, recvBuffs[0], lpPlan.relayRecv, directBase, stream);
  }

  // Fold both received direct chunks in one fused pass; they are adjacent in
  // recvBuff and in the scratch.
  if (myActiveGroup == 0 && directTotal > 0) {
    char* dst = static_cast<char*>(recvBuffs[0]) + directBase * elementSize;
    if (lp) {
      if (isInPlace) {
        DISPATCH_LP_MULTI_REDUCE(
            datatype,
            dst,
            lpPlan.directRecv,
            1,
            directTotal,
            reductionDivisor,
            stream);
      } else {
        DISPATCH_LP_SEEDED_MULTI_REDUCE(
            datatype,
            dst,
            static_cast<const char*>(sendBuffs[0]) + directBase * elementSize,
            lpPlan.directRecv,
            1,
            directTotal,
            reductionDivisor,
            stream);
      }
    } else if (isInPlace) {
      if (reductionDivisor > 1) {
        DISPATCH_INCREMENTAL_ADD_AND_SCALE(
            datatype,
            dst,
            directScratch,
            directTotal,
            reductionDivisor,
            stream);
      } else {
        DISPATCH_INCREMENTAL_ADD(
            datatype, dst, directScratch, directTotal, stream);
      }
    } else {
      DISPATCH_FUSED_REDUCE(
          datatype,
          dst,
          static_cast<const char*>(sendBuffs[0]) + directBase * elementSize,
          dst,
          directTotal,
          reductionDivisor,
          stream);
    }
  }

  return ncclSuccess;
}

/**
 * AllReduce for > 2 active ranks (flat helper-reduce-AND-broadcast). Combines
 * the two patterns that individually beat NCCL at 4-active: the
 * reduce-AT-helper of the flat reduce-scatter and the broadcast-AT-helper of
 * the flat all-gather.
 *
 * Each rank's `count` is split into a DIRECT region pD (allreduced among the A
 * active ranks over the 1-hop intra links via a direct all-to-all
 * reduce-scatter + all-gather) and an OFFLOAD region pO (allreduced 2-hop
 * through the otherwise-idle helpers: every helper SUMS all A active ranks'
 * chunk and BROADCASTS the result back to all A). Both regions run concurrently
 * in two ncclGroups -- G1 scatter and G2 gather/broadcast -- with the
 * reductions in between. With f = pO/count = 0.5 each XGMI link carries
 * ~0.25*count (intra: 0.5*(1-f)*count for the direct RS+AG; cross:
 * 2*f*count/H for the offload scatter+broadcast), vs ~0.5*count/link for a
 * 4-rank NCCL allreduce that uses only the intra links. Replaces the
 * recursive-halving path.
 *
 * Scratch: dScratch holds the A-1 received direct shards. The helper's scratch
 * is recvBuffs[g] = rawBuf [0, A*oChunk) (one chunk per active source); the
 * fused helper reduce sums all A chunks in place into rawBuf[0], which is then
 * broadcast back to the active ranks. recvBuff is operated on in
 * place after being seeded from sendBuff (out-of-place) or aliasing it
 * (in-place). Requires count % A == 0 and assumes A == numHelpers (the 8-GPU
 * 4-active topology); the AVG divisor is applied once over the whole count.
 */
static ncclResult_t shardedRelayAllReduceFlat(
    const void* const* sendBuffs,
    void* const* recvBuffs,
    const size_t* counts,
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

  // Offload (helper reduce+broadcast) share. Per comm group the intra links
  // carry pD/A and the cross links carry pO/H, and with A == H on the 8-GPU
  // 4-active topology the two-group critical path is 2*max(pD, pO)/A. That is
  // minimized when the direct and offload regions are EQUAL, giving count/4 per
  // link against NCCL's count/2 over the intra links alone — a 2x ceiling. (The
  // previous 780 skewed everything onto the cross links for a 1.28x ceiling,
  // which matches the ~1.03x that was measured.)
  //
  // At small sizes the 2-hop offload only adds helper-hop latency, so disable
  // it there and run a pure-direct RS+AG among the A active ranks (helpers
  // idle). The size -> route mapping and the resulting offload share live in
  // selectAllReduceRoute()/allReduceOffloadPermille() so the tests assert the
  // same definition this dispatch uses.
  const size_t kOffPermille = rcclx::relay::allReduceOffloadPermille(
      rcclx::relay::selectAllReduceRoute(A, nGroups, counts, elementSize));

  for (int g = 0; g < nGroups; g++) {
    if (counts[g] != 0 && (counts[g] % static_cast<size_t>(A)) != 0) {
      return ncclInvalidArgument;
    }
  }

  // ==========================================================================
  // SINGLE-GROUP SMALL-MESSAGE FULL-EXCHANGE FAST PATH (A > 2)
  // ==========================================================================
  // Below the crossover this schedule's cost is entirely launch latency, and
  // the reduce-scatter + all-gather form it would otherwise run is the most
  // expensive shape in the relay set at FOUR launches on the critical path: the
  // sendbuff -> recvbuff staging copy, the reduce-scatter group, the shard
  // reduce, and the all-gather group. A plain full exchange needs TWO -- one
  // group in which every active rank ships its whole buffer to the other A-1,
  // then one fused reduce over all A contributions. It moves (A-1)*count per
  // link instead of 2*(A-1)*count/A, but below the crossover the bytes are not
  // what is being paid for. This is the same trade the A==2 path already makes
  // (see shardedRelayAllReduce2Active), generalized to A > 2.
  //
  // No separate size gate: the route selector already bounds the pure-direct
  // regime, so "offload disabled" IS "below the crossover". Restricted to
  // nGroups == 1 because a fused call has every rank active in one group and a
  // helper in the others, so its links carry several groups' traffic at once
  // and the bytes this trades away are not free there.
  if (kOffPermille == 0 && nGroups == 1) {
    // One kernel instead of the group-plus-reduce pair below. See
    // tryOneShotAllReduce().
    if (tryOneShotAllReduce(
            sendBuffs,
            recvBuffs,
            counts,
            datatype,
            reductionDivisor,
            comm,
            stream,
            configs,
            myActiveGroup,
            A,
            nGroups,
            elementSize)) {
      return ncclSuccess;
    }

    const size_t count = counts[0];
    void* xScratch = nullptr;
    if (myActiveGroup >= 0 && count > 0) {
      xScratch = ScratchBufferCache::getInstance().get(
          SHARDED_RELAY_MAX_GROUPS,
          static_cast<size_t>(A - 1) * count * elementSize,
          stream);
      if (xScratch == nullptr) {
        return ncclInternalError;
      }
    }

    if (count > 0) {
      NCCLCHECK(ncclGroupStart());
      const ShardedRelayRankConfig& cfg = configs[0];
      if (cfg.isActiveRank) {
        const int m = cfg.myActiveIndex;
        for (int k = 0; k < A; k++) {
          if (k == m) {
            continue;
          }
          NCCLCHECK(ncclSend(
              sendBuffs[0], count, datatype, cfg.activeRanks[k], comm, stream));
        }
        for (int s = 0; s < A; s++) {
          if (s == m) {
            continue;
          }
          const int p = (s < m) ? s : s - 1;
          NCCLCHECK(ncclRecv(
              static_cast<char*>(xScratch) +
                  static_cast<size_t>(p) * count * elementSize,
              count,
              datatype,
              cfg.activeRanks[s],
              comm,
              stream));
        }
      }
      NCCLCHECK(ncclGroupEnd());
    }

    if (myActiveGroup >= 0 && count > 0) {
      void* out = recvBuffs[0];
      if (out == sendBuffs[0]) {
        // In-place: recvbuff already holds this rank's contribution, so it is
        // both the seed and the destination.
        DISPATCH_MULTI_REDUCE(
            datatype, out, xScratch, A - 1, count, reductionDivisor, stream);
      } else {
        DISPATCH_SEEDED_MULTI_REDUCE(
            datatype,
            out,
            sendBuffs[0],
            xScratch,
            A - 1,
            count,
            reductionDivisor,
            stream);
      }
    }
    return ncclSuccess;
  }

  // Per-group geometry: pO (offload) aligned to H*CHUNK_ALIGN; pD = count - pO
  // is then divisible by A (== H). dShard = pD/A, oChunk = pO/H.
  size_t pDArr[SHARDED_RELAY_MAX_GROUPS];
  size_t pOArr[SHARDED_RELAY_MAX_GROUPS];
  for (int g = 0; g < nGroups; g++) {
    size_t count = counts[g];
    if (count == 0) {
      pDArr[g] = 0;
      pOArr[g] = 0;
      continue;
    }
    size_t alignH = static_cast<size_t>(H) * CHUNK_ALIGN_ELEMENTS;
    size_t pO = (count * kOffPermille) / 1000;
    pO = (pO / alignH) * alignH;
    if (pO > count) {
      pO = (count / alignH) * alignH;
    }
    pOArr[g] = pO;
    pDArr[g] = count - pO;
  }

  // Out-of-place: seed recvBuff = sendBuff once, then operate in place.
  if (myActiveGroup >= 0 && counts[myActiveGroup] > 0 &&
      sendBuffs[myActiveGroup] != recvBuffs[myActiveGroup]) {
    cudaMemcpyAsync(
        recvBuffs[myActiveGroup],
        sendBuffs[myActiveGroup],
        counts[myActiveGroup] * elementSize,
        cudaMemcpyDeviceToDevice,
        stream);
  }

  // Direct reduce-scatter scratch: the A-1 received direct shards.
  void* dScratch = nullptr;
  if (myActiveGroup >= 0 && counts[myActiveGroup] > 0 &&
      pDArr[myActiveGroup] > 0) {
    size_t dShard = pDArr[myActiveGroup] / A;
    size_t dBytes = static_cast<size_t>(A - 1) * dShard * elementSize;
    dScratch = ScratchBufferCache::getInstance().get(
        SHARDED_RELAY_MAX_GROUPS, dBytes, stream);
    if (dScratch == nullptr) {
      return ncclInternalError;
    }
  }

  // Helper staging is kernel-owned scratch: callers pass a placeholder buffer
  // for groups where they are a helper. Each helper group holds A offload
  // chunks (rawBuf[0..A)) to receive, reduce, and broadcast.
  void* helperScratch[SHARDED_RELAY_MAX_GROUPS] = {nullptr};
  for (int g = 0; g < nGroups; g++) {
    const ShardedRelayRankConfig& cfg = configs[g];
    size_t oChunkG = (pOArr[g] > 0) ? pOArr[g] / H : 0;
    if (!cfg.isActiveRank && counts[g] > 0 && oChunkG > 0) {
      size_t needBytes =
          static_cast<size_t>(cfg.nActiveRanks) * oChunkG * elementSize;
      helperScratch[g] = ScratchBufferCache::getInstance().get(
          kHelperScratchKeyBase + g, needBytes, stream);
      if (helperScratch[g] == nullptr) {
        return ncclInternalError;
      }
    }
  }

  // ===== Group 1: direct RS (active<->active) + offload scatter
  // (active->helper). ========================================================
  NCCLCHECK(ncclGroupStart());
  for (int g = 0; g < nGroups; g++) {
    if (counts[g] == 0)
      continue;
    const ShardedRelayRankConfig& cfg = configs[g];
    size_t pD = pDArr[g];
    size_t pO = pOArr[g];
    size_t dShard = (pD > 0) ? pD / A : 0;
    size_t oChunk = (pO > 0) ? pO / H : 0;
    if (cfg.isActiveRank) {
      char* recvbuff = static_cast<char*>(recvBuffs[g]);
      int m = cfg.myActiveIndex;
      // Direct RS: send my shard k to owner k; recv source s's shard m into its
      // dScratch slot p.
      if (dShard > 0) {
        for (int k = 0; k < A; k++) {
          if (k == m)
            continue;
          NCCLCHECK(ncclSend(
              recvbuff + static_cast<size_t>(k) * dShard * elementSize,
              dShard,
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
                  static_cast<size_t>(p) * dShard * elementSize,
              dShard,
              datatype,
              cfg.activeRanks[s],
              comm,
              stream));
        }
      }
      // Offload scatter: send my offload chunk h to helper h.
      if (oChunk > 0) {
        for (int h = 0; h < cfg.numHelpers; h++) {
          NCCLCHECK(ncclSend(
              recvbuff + (pD + static_cast<size_t>(h) * oChunk) * elementSize,
              oChunk,
              datatype,
              cfg.helperRanks[h],
              comm,
              stream));
        }
      }
    } else if (oChunk > 0) {
      // Helper: recv this helper's offload chunk from each active a into
      // rawBuf[a].
      char* rawBuf = static_cast<char*>(helperScratch[g]);
      for (int a = 0; a < cfg.nActiveRanks; a++) {
        NCCLCHECK(ncclRecv(
            rawBuf + static_cast<size_t>(a) * oChunk * elementSize,
            oChunk,
            datatype,
            cfg.activeRanks[a],
            comm,
            stream));
      }
    }
  }
  NCCLCHECK(ncclGroupEnd());

  // ===== Reduce: direct (active sums its shard) + offload (helper sums all A).
  for (int g = 0; g < nGroups; g++) {
    if (counts[g] == 0)
      continue;
    const ShardedRelayRankConfig& cfg = configs[g];
    size_t pD = pDArr[g];
    size_t pO = pOArr[g];
    size_t dShard = (pD > 0) ? pD / A : 0;
    size_t oChunk = (pO > 0) ? pO / H : 0;
    if (cfg.isActiveRank) {
      if (dShard > 0 && g == myActiveGroup) {
        char* recvbuff = static_cast<char*>(recvBuffs[g]);
        int m = cfg.myActiveIndex;
        char* dst = recvbuff + static_cast<size_t>(m) * dShard * elementSize;
        // Fused single-pass reduce: dst = (dst + A-1 direct contribs) / divisor
        // (AVG folded in), before the all-gather broadcasts the owned shard.
        DISPATCH_MULTI_REDUCE(
            datatype, dst, dScratch, A - 1, dShard, reductionDivisor, stream);
      }
    } else if (oChunk > 0) {
      // Helper: fused single-pass reduce of the A received chunks into
      // rawBuf[0] in-place (rawBuf[0] = (sum_a rawBuf[a]) / divisor), one
      // launch instead of memcpy + A-1 adds + scale. The broadcast below reads
      // rawBuf[0].
      char* rawBuf = static_cast<char*>(helperScratch[g]);
      DISPATCH_MULTI_REDUCE(
          datatype,
          rawBuf,
          rawBuf + oChunk * elementSize,
          cfg.nActiveRanks - 1,
          oChunk,
          reductionDivisor,
          stream);
    }
  }

  // ===== Group 2: direct AG (active<->active) + offload broadcast
  // (helper->active). ========================================================
  NCCLCHECK(ncclGroupStart());
  for (int g = 0; g < nGroups; g++) {
    if (counts[g] == 0)
      continue;
    const ShardedRelayRankConfig& cfg = configs[g];
    size_t pD = pDArr[g];
    size_t pO = pOArr[g];
    size_t dShard = (pD > 0) ? pD / A : 0;
    size_t oChunk = (pO > 0) ? pO / H : 0;
    if (cfg.isActiveRank) {
      char* recvbuff = static_cast<char*>(recvBuffs[g]);
      int m = cfg.myActiveIndex;
      // Direct AG: send my reduced shard m to every other active; recv their
      // reduced shard k into recvBuff[k].
      if (dShard > 0) {
        for (int k = 0; k < A; k++) {
          if (k == m)
            continue;
          NCCLCHECK(ncclSend(
              recvbuff + static_cast<size_t>(m) * dShard * elementSize,
              dShard,
              datatype,
              cfg.activeRanks[k],
              comm,
              stream));
          NCCLCHECK(ncclRecv(
              recvbuff + static_cast<size_t>(k) * dShard * elementSize,
              dShard,
              datatype,
              cfg.activeRanks[k],
              comm,
              stream));
        }
      }
      // Offload broadcast: recv reduced chunk h from helper h into recvBuff.
      if (oChunk > 0) {
        for (int h = 0; h < cfg.numHelpers; h++) {
          NCCLCHECK(ncclRecv(
              recvbuff + (pD + static_cast<size_t>(h) * oChunk) * elementSize,
              oChunk,
              datatype,
              cfg.helperRanks[h],
              comm,
              stream));
        }
      }
    } else if (oChunk > 0) {
      // Helper: broadcast the reduced chunk (now in rawBuf[0]) to all A active
      // ranks.
      char* rawBuf = static_cast<char*>(helperScratch[g]);
      for (int a = 0; a < cfg.nActiveRanks; a++) {
        NCCLCHECK(ncclSend(
            rawBuf, oChunk, datatype, cfg.activeRanks[a], comm, stream));
      }
    }
  }
  NCCLCHECK(ncclGroupEnd());

  // The AVG divisor was already folded into the per-shard direct reduce and the
  // per-chunk helper reduce above, so no separate whole-count scale is needed.
  return ncclSuccess;
}

/**
 * Software-pipelined single-group A>2 flat allreduce.
 *
 * Same two regions as shardedRelayAllReduceFlat -- a direct reduce-scatter plus
 * all-gather over the intra links, and an offload region the helpers reduce and
 * broadcast -- but for nGroups == 1, where the active ranks and the helpers are
 * disjoint, both are tiled and pipelined. See relayAllReducePipelineTiles()
 * for why this schedule needs its own depth selector and why it only pays from
 * depth 4 up.
 *
 * Two dependencies are pipelined here, not one:
 *   offload   scatter tile k in group k; the helper sums the A chunks; the
 *             reduced tile is broadcast back in group k+1. Because the helper
 *             takes one chunk per active rank and returns one per active rank,
 *             the cross link carries the same in each direction -- this is the
 *             4-active shape that is NOT throttled by a heavy direction.
 *   direct    reduce-scatter tile k in group k, the owner reduces its shard's
 *             tile k, and the all-gather of that tile goes out in group k+1.
 *
 * With y = align(count / ((A + 2H)*T)): the offload region is H*2*T*y (tile
 * 2y), the direct region is the rest with per-owner shard dShard = pD/A (tile
 * y, the last absorbing the remainder), and each of the T+1 groups carries 2y
 * on the busiest link, so the cost is 2*(T+1)*y -- at A = H = 4, count/4 for
 * the two-group schedule falling to 5*count/24 at depth 4 and 3*count/16 at
 * depth 8.
 *
 * The divisor is A + 2H rather than a fixed 12 because the layout consumes
 * 2*H*T units for the offload region and A*T for the direct region's shards. A
 * = 4 with H = 5..8 (a 9-to-12-rank comm) and A = 8 with H = 8 (a 16-rank one)
 * also reach this path, and a fixed 12 would make pO exceed count so pD = count
 * - pO underflows, handing a wild dShard to the reduce kernels and to ncclSend.
 *
 * dScratch is laid out TILE-major, not shard-major as in the flat path: the
 * owner's per-tile reduce has to fold the A-1 peer contributions for one tile
 * while the next group is already in flight, and the fused multi-input reduce
 * requires those to be contiguous with stride tileSz.
 *
 * Both in-place and out-of-place are supported; out-of-place seeds recvBuff
 * from sendBuff once and then operates in place, as the flat path does.
 */
static ncclResult_t shardedRelayAllReduceFlatPipelined(
    const void* const* sendBuffs,
    void* const* recvBuffs,
    const size_t* counts,
    ncclDataType_t datatype,
    int reductionDivisor,
    ncclComm_t comm,
    cudaStream_t stream,
    const ShardedRelayRankConfig* configs,
    int myActiveGroup,
    int numHelpers,
    int nActiveRanksPerGroup,
    int nTiles,
    size_t elementSize) {
  const ShardedRelayRankConfig& cfg = configs[0];
  const size_t count = counts[0];
  const int A = nActiveRanksPerGroup;
  const int H = numHelpers;
  const int T = nTiles;
  if ((count % static_cast<size_t>(A)) != 0) {
    return ncclInvalidArgument;
  }
  const size_t y =
      ((count /
        (static_cast<size_t>(
             rcclx::relay::relayAllReducePipelineUnitsPerTile(A, H)) *
         T)) /
       CHUNK_ALIGN_ELEMENTS) *
      CHUNK_ALIGN_ELEMENTS;
  if (y == 0) {
    return ncclInvalidArgument;
  }
  const size_t oTile = 2 * y;
  // The offload moves oTile = 2*y per link per group while the direct exchange
  // moves y, and RCCL budgets channels per operation -- so a cross link
  // carrying its bytes as one op gets half the channels of an intra link
  // carrying the same bytes as two. Splitting the offload into y-sized pieces
  // makes every operation in the group the same size. Gated on size for the
  // same reason as the all-gather mirror: measured 1.98x -> 2.05x at 1 GB
  // and 1.88x -> 1.97x at 512 MB, but ~1% worse at 135-144 MB where the extra
  // ops do not amortize.
  const int oN =
      (count * elementSize >= rcclx::relay::kRelayUniformDirectOpMinBytes) ? 2
                                                                           : 1;
  const size_t oPiece = oTile / static_cast<size_t>(oN);
  const size_t oChunk = static_cast<size_t>(T) * oTile;
  const size_t pO = static_cast<size_t>(H) * oChunk;
  const size_t pD = count - pO;
  const size_t dShard = pD / static_cast<size_t>(A);
  auto tileOffset = [&](int t) -> size_t { return static_cast<size_t>(t) * y; };
  auto tileSize = [&](int t) -> size_t {
    return (t < T - 1) ? y : (dShard - tileOffset(T - 1));
  };
  // Peer p's contribution to tile t, tile-major so one tile's A-1 inputs are
  // contiguous with stride tileSize(t).
  auto dSlot = [&](int t, int p) -> size_t {
    return static_cast<size_t>(A - 1) * tileOffset(t) +
        static_cast<size_t>(p) * tileSize(t);
  };

  void* dScratch = nullptr;
  if (myActiveGroup == 0) {
    if (sendBuffs[0] != recvBuffs[0]) {
      cudaMemcpyAsync(
          recvBuffs[0],
          sendBuffs[0],
          count * elementSize,
          cudaMemcpyDeviceToDevice,
          stream);
    }
    dScratch = ScratchBufferCache::getInstance().get(
        SHARDED_RELAY_MAX_GROUPS,
        static_cast<size_t>(A - 1) * dShard * elementSize,
        stream);
    if (dScratch == nullptr) {
      return ncclInternalError;
    }
  }

  // Helper staging: A chunks per ping-pong buffer, buffer-major so the A inputs
  // of one stage are contiguous with stride oTile for the fused reduce.
  char* hbuff = nullptr;
  if (!cfg.isActiveRank) {
    hbuff = static_cast<char*>(ScratchBufferCache::getInstance().get(
        kHelperScratchKeyBase,
        2 * static_cast<size_t>(A) * oTile * elementSize,
        stream));
    if (hbuff == nullptr) {
      return ncclInternalError;
    }
  }
  auto helperSlot = [&](int a, int k) -> char* {
    return hbuff +
        ((static_cast<size_t>(k % 2) * A + static_cast<size_t>(a)) * oTile) *
        elementSize;
  };

  for (int k = 0; k <= T; k++) {
    NCCLCHECK(ncclGroupStart());
    if (cfg.isActiveRank) {
      char* recvbuff = static_cast<char*>(recvBuffs[0]);
      const int m = cfg.myActiveIndex;

      // Sends: reduce-scatter tile k, then all-gather tile k-1. The receive
      // loops below use the same order, which is what keeps each peer's matched
      // pair in step.
      if (k < T) {
        for (int j = 0; j < A; j++) {
          if (j == m) {
            continue;
          }
          NCCLCHECK(ncclSend(
              recvbuff +
                  (static_cast<size_t>(j) * dShard + tileOffset(k)) *
                      elementSize,
              tileSize(k),
              datatype,
              cfg.activeRanks[j],
              comm,
              stream));
        }
      }
      if (k > 0) {
        for (int j = 0; j < A; j++) {
          if (j == m) {
            continue;
          }
          NCCLCHECK(ncclSend(
              recvbuff +
                  (static_cast<size_t>(m) * dShard + tileOffset(k - 1)) *
                      elementSize,
              tileSize(k - 1),
              datatype,
              cfg.activeRanks[j],
              comm,
              stream));
        }
      }
      if (k < T) {
        // Offload scatter, split into y-sized pieces so every operation in the
        // group is the same size as the direct tiles. oTile is exactly 2*y, so
        // the split is exact. See kRelayUniformDirectOpMinBytes for why this
        // matters: RCCL budgets channels per operation, so a link carrying its
        // bytes as one large op gets half the channels of a link carrying the
        // same bytes as two.
        for (int h = 0; h < H; h++) {
          for (int i = 0; i < oN; i++) {
            NCCLCHECK(ncclSend(
                recvbuff +
                    (pD + static_cast<size_t>(h) * oChunk +
                     static_cast<size_t>(k) * oTile +
                     static_cast<size_t>(i) * oPiece) *
                        elementSize,
                oPiece,
                datatype,
                cfg.helperRanks[h],
                comm,
                stream));
          }
        }
      }

      if (k < T) {
        for (int s = 0; s < A; s++) {
          if (s == m) {
            continue;
          }
          const int p = (s < m) ? s : s - 1;
          NCCLCHECK(ncclRecv(
              static_cast<char*>(dScratch) + dSlot(k, p) * elementSize,
              tileSize(k),
              datatype,
              cfg.activeRanks[s],
              comm,
              stream));
        }
      }
      if (k > 0) {
        for (int j = 0; j < A; j++) {
          if (j == m) {
            continue;
          }
          NCCLCHECK(ncclRecv(
              recvbuff +
                  (static_cast<size_t>(j) * dShard + tileOffset(k - 1)) *
                      elementSize,
              tileSize(k - 1),
              datatype,
              cfg.activeRanks[j],
              comm,
              stream));
        }
        // Already reduced by the helper, so this is the final value. Split to
        // match the scatter above.
        for (int h = 0; h < H; h++) {
          for (int i = 0; i < oN; i++) {
            NCCLCHECK(ncclRecv(
                recvbuff +
                    (pD + static_cast<size_t>(h) * oChunk +
                     static_cast<size_t>(k - 1) * oTile +
                     static_cast<size_t>(i) * oPiece) *
                        elementSize,
                oPiece,
                datatype,
                cfg.helperRanks[h],
                comm,
                stream));
          }
        }
      }
    } else {
      if (k < T) {
        for (int a = 0; a < A; a++) {
          for (int i = 0; i < oN; i++) {
            NCCLCHECK(ncclRecv(
                helperSlot(a, k) +
                    static_cast<size_t>(i) * oPiece * elementSize,
                oPiece,
                datatype,
                cfg.activeRanks[a],
                comm,
                stream));
          }
        }
      }
      if (k > 0) {
        for (int a = 0; a < A; a++) {
          for (int i = 0; i < oN; i++) {
            NCCLCHECK(ncclSend(
                helperSlot(0, k - 1) +
                    static_cast<size_t>(i) * oPiece * elementSize,
                oPiece,
                datatype,
                cfg.activeRanks[a],
                comm,
                stream));
          }
        }
      }
    }
    NCCLCHECK(ncclGroupEnd());

    if (k < T) {
      if (cfg.isActiveRank) {
        // Fold this tile of my own shard, before the next group gathers it.
        char* dst = static_cast<char*>(recvBuffs[0]) +
            (static_cast<size_t>(cfg.myActiveIndex) * dShard + tileOffset(k)) *
                elementSize;
        DISPATCH_MULTI_REDUCE(
            datatype,
            dst,
            static_cast<char*>(dScratch) + dSlot(k, 0) * elementSize,
            A - 1,
            tileSize(k),
            reductionDivisor,
            stream);
      } else {
        // Sum all A chunks into slot 0, which the next group broadcasts.
        DISPATCH_MULTI_REDUCE(
            datatype,
            helperSlot(0, k),
            helperSlot(1, k),
            A - 1,
            oTile,
            reductionDivisor,
            stream);
      }
    }
  }

  return ncclSuccess;
}

/**
 * Fused Multi-Group Sharded Relay AllReduce.
 *
 * Executes multiple sharded relay allreduces in one fused call, phase-synced
 * across all groups so XGMI links carry unidirectional traffic. Helpers are
 * pure passthrough; reductions happen on the active ranks. Each rank is ACTIVE
 * for exactly one group and a HELPER for the others.
 *
 * nActiveRanksPerGroup must be a power of two (2 or 4): A==2 uses the original
 * 2-active path; A>2 uses the flat helper-reduce-and-broadcast path.
 */
HOT ncclResult_t ncclShardedRelayMultiGroupAllReduceImpl(
    const void* const* sendBuffs,
    void* const* recvBuffs,
    const size_t* counts,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm_t comm,
    cudaStream_t stream,
    const int* const* allActiveRanks,
    int nActiveRanksPerGroup,
    int nGroups,
    int lowPrecision) {
  int nRanks, rank;
  NCCLCHECK(ncclCommCount(comm, &nRanks));
  NCCLCHECK(ncclCommUserRank(comm, &rank));

  // Validate every argument before touching counts: the all-zero scan below
  // indexes counts[0..nGroups), so a null pointer or an out-of-range nGroups
  // has to be rejected first. Bounds-checking nGroups up here also means
  // nGroups <= 0 reports ncclInvalidArgument rather than skipping the scan
  // entirely and returning ncclSuccess.
  if (nGroups < 1 || nGroups > SHARDED_RELAY_MAX_GROUPS) {
    return ncclInvalidArgument;
  }
  if (recvBuffs == nullptr || allActiveRanks == nullptr || counts == nullptr ||
      sendBuffs == nullptr) {
    return ncclInvalidArgument;
  }

  // Require a power-of-two active-rank count (>= 2) for the XOR schedule.
  if (nActiveRanksPerGroup < 2 || !isPowerOfTwo(nActiveRanksPerGroup)) {
    return ncclInvalidArgument;
  }

  if (op != ncclSum && op != ncclAvg) {
    return ncclInvalidArgument;
  }

  if (!isSupportedRelayDataType(datatype)) {
    return ncclInvalidArgument;
  }

  bool allZero = true;
  for (int g = 0; g < nGroups; g++) {
    if (counts[g] != 0) {
      allZero = false;
      break;
    }
  }
  if (allZero) {
    return ncclSuccess;
  }

  size_t elementSize = ncclTypeSize(datatype);
  int reductionDivisor = (op == ncclAvg) ? nActiveRanksPerGroup : 1;

  ShardedRelayRankConfig configs[SHARDED_RELAY_MAX_GROUPS];
  int myActiveGroup = -1;
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
  int numHelpers = configs[0].numHelpers;

  // Build per-group buffer arrays for the unchanged kernels. Helper groups use
  // their scratch (recvBuffs[g]); the active group uses the caller's contiguous
  // input/output buffers directly. Allreduce may be in-place (sendBuffs[g]
  // aliases recvBuffs[g]) or out-of-place, keyed off
  // sendBuffs[g]==recvBuffs[g].
  const void* sendBuffs2[SHARDED_RELAY_MAX_GROUPS];
  void* recvBuffs2[SHARDED_RELAY_MAX_GROUPS];
  for (int g = 0; g < nGroups; g++) {
    sendBuffs2[g] = sendBuffs[g];
    recvBuffs2[g] = recvBuffs[g];
  }

  ncclResult_t r;
  if (nActiveRanksPerGroup == 2) {
    // A single-group relay call has the helpers to itself, so the scatter and
    // the reduced forward run on opposite directions of each cross link and can
    // be software-pipelined into one duplex stream. relayPipelineTiles()
    // returns 1 whenever that does not apply, and the small-message pure-direct
    // route (owned by shardedRelayAllReduce2Active) never pipelines.
    const rcclx::relay::AllReduceRoute route =
        rcclx::relay::selectAllReduceRoute(2, nGroups, counts, elementSize);
    const int nTiles = (route == rcclx::relay::AllReduceRoute::A2Relay)
        ? rcclx::relay::relayPipelineTiles(
              nGroups,
              rcclx::relay::relayShapeA2(numHelpers),
              rcclx::relay::relayMaxCount(counts, nGroups),
              elementSize)
        : 1;

    // The caller's request narrowed by the size-only gate, so every rank
    // reaches the same answer without communicating. Both 2-active schedules
    // carry the wire format; the flat schedules still decline, which is safe
    // precisely because this decision is identical on every rank -- a per-rank
    // disagreement would be a hang, not a slowdown.
    bool wantLp = false;
    if (lowPrecision != 0) {
      wantLp = rcclx::relay::lpEligible(allReduceLpGate(
          datatype,
          counts,
          nGroups,
          nActiveRanksPerGroup,
          elementSize,
          route == rcclx::relay::AllReduceRoute::A2Relay));
    }

    r = (nTiles > 1) ? shardedRelayAllReduce2ActivePipelined(
                           sendBuffs2,
                           recvBuffs2,
                           counts,
                           datatype,
                           reductionDivisor,
                           comm,
                           stream,
                           configs,
                           myActiveGroup,
                           numHelpers,
                           nTiles,
                           elementSize,
                           wantLp)
                     : shardedRelayAllReduce2Active(
                           sendBuffs2,
                           recvBuffs2,
                           counts,
                           datatype,
                           reductionDivisor,
                           comm,
                           stream,
                           configs,
                           myActiveGroup,
                           numHelpers,
                           nGroups,
                           elementSize,
                           wantLp);
  } else {
    // A>2: flat helper-reduce-and-broadcast (direct RS+AG over intra woven with
    // offload reduce+broadcast through the helpers). Low precision is not
    // carried here yet, so a request declines -- identically on every rank,
    // since the condition is just the width.
    if (lowPrecision != 0) {
      rcclx::relay::lpRecordDecline(rcclx::relay::LpDecline::Route);
    }
    // A single-group call can pipeline both dependencies so each cross link
    // runs duplex; the selector returns 1 below the crossover, where the
    // two-group schedule is better.
    const int nTiles = rcclx::relay::relayAllReducePipelineTiles(
        nGroups,
        nActiveRanksPerGroup,
        numHelpers,
        rcclx::relay::relayMaxCount(counts, nGroups),
        elementSize);
    if (nTiles > 1) {
      return shardedRelayAllReduceFlatPipelined(
          sendBuffs2,
          recvBuffs2,
          counts,
          datatype,
          reductionDivisor,
          comm,
          stream,
          configs,
          myActiveGroup,
          numHelpers,
          nActiveRanksPerGroup,
          nTiles,
          elementSize);
    }
    r = shardedRelayAllReduceFlat(
        sendBuffs2,
        recvBuffs2,
        counts,
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

  return r;
}
