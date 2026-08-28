/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "sharded_relay_allreduce_kernels.h"

/**
 * GPU kernel for incremental reduction: output[i] += input[i]
 * Used to add received chunks directly into the buffer.
 */
template <typename T>
__global__ void incrementalAddKernel(T* output, const T* input, size_t count) {
  size_t threadId = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t totalThreads = static_cast<size_t>(blockDim.x) * gridDim.x;

  for (size_t elemIdx = threadId; elemIdx < count; elemIdx += totalThreads) {
    output[elemIdx] += input[elemIdx];
  }
}

template <typename T>
void launchIncrementalAddKernel(
    void* output,
    const void* input,
    size_t count,
    cudaStream_t stream) {
  const int blockSize = 256;
  int gridSize = (count + blockSize - 1) / blockSize;
  incrementalAddKernel<T><<<gridSize, blockSize, 0, stream>>>(
      static_cast<T*>(output), static_cast<const T*>(input), count);
}

/**
 * GPU kernel for scaling: output[i] = output[i] / divisor
 * Used to compute average after sum reduction (for ncclAvg operation).
 */
template <typename T>
__global__ void scaleKernel(T* data, size_t count, int divisor) {
  size_t threadId = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t totalThreads = static_cast<size_t>(blockDim.x) * gridDim.x;

  for (size_t elemIdx = threadId; elemIdx < count; elemIdx += totalThreads) {
    data[elemIdx] = data[elemIdx] / static_cast<T>(divisor);
  }
}

template <typename T>
void launchScaleKernel(
    void* data,
    size_t count,
    int divisor,
    cudaStream_t stream) {
  const int blockSize = 256;
  int gridSize = (count + blockSize - 1) / blockSize;
  scaleKernel<T><<<gridSize, blockSize, 0, stream>>>(
      static_cast<T*>(data), count, divisor);
}

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
 */
template <typename T>
__global__ void incrementalAddAndScaleKernel(
    T* output,
    const T* input,
    size_t count,
    int divisor) {
  size_t threadId = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t totalThreads = static_cast<size_t>(blockDim.x) * gridDim.x;

  for (size_t elemIdx = threadId; elemIdx < count; elemIdx += totalThreads) {
    T sum = output[elemIdx] + input[elemIdx];
    if (divisor > 1) {
      output[elemIdx] = sum / static_cast<T>(divisor);
    } else {
      output[elemIdx] = sum;
    }
  }
}

template <typename T>
void launchIncrementalAddAndScaleKernel(
    void* output,
    const void* input,
    size_t count,
    int divisor,
    cudaStream_t stream) {
  const int blockSize = 256;
  int gridSize = (count + blockSize - 1) / blockSize;
  incrementalAddAndScaleKernel<T><<<gridSize, blockSize, 0, stream>>>(
      static_cast<T*>(output), static_cast<const T*>(input), count, divisor);
}

/**
 * GPU kernel for fused reduction: output[i] = (a[i] + b[i]) / divisor
 * When divisor == 1, this is a simple sum: output[i] = a[i] + b[i]
 * When divisor == 2, this computes the average: output[i] = (a[i] + b[i]) / 2
 *
 * Used by helper ranks to combine data from both active ranks and compute
 * sum or average in a single kernel launch (avoiding separate add + scale).
 */
template <typename T>
__global__ void fusedReduceKernel(
    T* output,
    const T* inputA,
    const T* inputB,
    size_t count,
    int divisor) {
  size_t threadId = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t totalThreads = static_cast<size_t>(blockDim.x) * gridDim.x;

  for (size_t elemIdx = threadId; elemIdx < count; elemIdx += totalThreads) {
    T sum = inputA[elemIdx] + inputB[elemIdx];
    if (divisor > 1) {
      output[elemIdx] = sum / static_cast<T>(divisor);
    } else {
      output[elemIdx] = sum;
    }
  }
}

template <typename T>
void launchFusedReduceKernel(
    void* output,
    const void* inputA,
    const void* inputB,
    size_t count,
    int divisor,
    cudaStream_t stream) {
  const int blockSize = 256;
  int gridSize = (count + blockSize - 1) / blockSize;
  fusedReduceKernel<T><<<gridSize, blockSize, 0, stream>>>(
      static_cast<T*>(output),
      static_cast<const T*>(inputA),
      static_cast<const T*>(inputB),
      count,
      divisor);
}

/*
 * Fused multi-input reduce: dst[i] = (dst[i] + sum_p contribs[p*count + i]),
 * optionally divided by `divisor`, in a single pass / single launch.
 *
 * `contribs` is a contiguous array of `numContribs` blocks, each `count`
 * elements (block p starts at contribs + p*count).  This replaces a loop of
 * `numContribs` separate incremental-add launches (each read-modify-writing
 * dst) plus a trailing scale — reading dst once and writing it once instead of
 * numContribs+1 times, which cuts reduce-path HBM traffic.
 */
template <typename T>
__global__ void multiReduceKernel(
    T* dst,
    const T* contribs,
    int numContribs,
    size_t count,
    int divisor) {
  size_t threadId = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t totalThreads = static_cast<size_t>(blockDim.x) * gridDim.x;

  for (size_t elemIdx = threadId; elemIdx < count; elemIdx += totalThreads) {
    T acc = dst[elemIdx];
    for (int p = 0; p < numContribs; p++) {
      acc += contribs[static_cast<size_t>(p) * count + elemIdx];
    }
    if (divisor > 1) {
      dst[elemIdx] = acc / static_cast<T>(divisor);
    } else {
      dst[elemIdx] = acc;
    }
  }
}

template <typename T>
void launchMultiReduceKernel(
    void* dst,
    const void* contribs,
    int numContribs,
    size_t count,
    int divisor,
    cudaStream_t stream) {
  const int blockSize = 256;
  int gridSize = (count + blockSize - 1) / blockSize;
  multiReduceKernel<T><<<gridSize, blockSize, 0, stream>>>(
      static_cast<T*>(dst),
      static_cast<const T*>(contribs),
      numContribs,
      count,
      divisor);
}

/*
 * Seeded multi-input reduce. Each scalar result follows the exact sequence:
 * seed[i], then contribs[0][i] through contribs[numContribs - 1][i], then one
 * optional divide. The loop controls prevent reassociation and vectorization.
 */
template <typename T>
__global__ void seededMultiReduceKernel(
    T* dst,
    const T* seed,
    const T* contribs,
    int numContribs,
    size_t count,
    int divisor) {
#pragma clang fp reassociate(off)
  size_t threadId = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t totalThreads = static_cast<size_t>(blockDim.x) * gridDim.x;

#pragma clang loop vectorize(disable) interleave(disable)
  for (size_t elemIdx = threadId; elemIdx < count; elemIdx += totalThreads) {
    T acc = seed[elemIdx];
#pragma clang loop unroll(disable) vectorize(disable) interleave(disable)
    for (int p = 0; p < numContribs; p++) {
      acc = acc + contribs[static_cast<size_t>(p) * count + elemIdx];
    }
    if (divisor > 1) {
      acc = acc / static_cast<T>(divisor);
    }
    dst[elemIdx] = acc;
  }
}

template <typename T>
void launchSeededMultiReduceKernel(
    void* dst,
    const void* seed,
    const void* contribs,
    int numContribs,
    size_t count,
    int divisor,
    cudaStream_t stream) {
  const int blockSize = 256;
  int gridSize = (count + blockSize - 1) / blockSize;
  seededMultiReduceKernel<T><<<gridSize, blockSize, 0, stream>>>(
      static_cast<T*>(dst),
      static_cast<const T*>(seed),
      static_cast<const T*>(contribs),
      numContribs,
      count,
      divisor);
}

// Explicit template instantiations for every dtype used by the DISPATCH_*
// macros in sharded_relay_allreduce.cc.  Without these, the symbols would
// not exist with external linkage in the device object, and the host TU
// (which only sees `extern template` declarations via the header) would
// fail to link the launchers — leaving the underlying `__global__`
// kernels' host stubs unresolved at runtime.

/*
 * ONE-SHOT 2-RANK REDUCE-SCATTER
 *
 * A single kernel that both moves the data and reduces it, so it replaces an
 * ncclGroup plus a reduce launch. See sharded_relay_oneshot.h for why the group
 * itself, not the reduce, is what has to go.
 *
 * Ordering: the data stores must be visible to the peer BEFORE it observes the
 * flag, so thread 0 issues __threadfence_system() and then a release store; the
 * reader spins with an acquire load. The epoch comparison is wraparound-safe,
 * and flags are never reset -- the epoch only advances, so a stale flag from a
 * previous call always compares as not-yet-arrived for the current one.
 */
__device__ __forceinline__ bool oneShotEpochReached(
    uint32_t got,
    uint32_t want) {
  return (got - want) <= (uint32_t(-1) >> 1);
}

template <typename T>
__global__ void oneShotPushReduceKernel(
    T* __restrict__ out,
    const T* __restrict__ sendBuff,
    rcclx::relay::OneShotPeerTable table,
    rcclx::relay::OneShotRanks ranks,
    int nActive,
    int myRank,
    int mySlot,
    size_t rc,
    size_t srcStride,
    size_t ownOffset,
    size_t slotBytes,
    uint32_t epoch,
    int divisor) {
  // 16 bytes is the widest access, and moving the push as 16-byte units rather
  // than element-wise is what makes this competitive: measured, the kernel is
  // copy-throughput bound, not handshake bound (1 block is 6x slower than 64 at
  // 576 KB, while the extra 63 flag round trips cost nothing).
  constexpr int kEvec = static_cast<int>(16 / sizeof(T));
  struct alignas(16) Vec {
    T e[kEvec];
  };

  // My own staging slots: slot s receives source s's contribution to my block.
  const T* staged[rcclx::relay::kOneShotMaxRanks];
  for (int s = 0; s < nActive; s++) {
    staged[s] = reinterpret_cast<const T*>(
        table.staging[myRank] + static_cast<size_t>(s) * slotBytes);
  }
  const T* own = sendBuff + ownOffset;

  // Staging slots are hipMalloc'd and slot-aligned, so they are always 16-byte
  // aligned; the caller's buffers are not guaranteed to be, and a per-source
  // offset of rc elements is only aligned when rc*sizeof(T) is a multiple
  // of 16. Check rather than assume -- a misaligned call still works, just
  // element-wise.
  //
  // This is a LOCAL property: the two ranks own independent allocations, so one
  // can be aligned while the other is not. It therefore must NOT feed the block
  // partition below. Blocks handshake pairwise on a single block index, so if
  // the ranks partitioned differently, block b's flag would not cover the range
  // block b goes on to reduce -- it would read staging that the peer pushed
  // from some other block, whose flag it never waited on. vecOk selects only
  // HOW a block moves its own fixed range, never WHICH range it gets.
  const uintptr_t bits = reinterpret_cast<uintptr_t>(sendBuff) |
      reinterpret_cast<uintptr_t>(out) |
      reinterpret_cast<uintptr_t>(table.staging[myRank]);
  const bool vecOk = ((bits & 15u) == 0) &&
      ((srcStride * sizeof(T)) % 16u == 0) &&
      ((ownOffset * sizeof(T)) % 16u == 0) && ((rc * sizeof(T)) % 16u == 0);

  // Per-block element range, derived only from rc, kEvec and gridDim -- all
  // rank-agreed (gridDim comes from rc, kOneShotThreads and kOneShotMaxBlocks)
  // -- so both ranks assign the same [begin, end) to the same block index
  // whatever their buffer alignment. Ranges are whole 16-byte vectors so that
  // begin stays 16-byte aligned for a rank that can vectorize.
  const size_t nvec = rc / kEvec;
  const size_t vchunk = (nvec + gridDim.x - 1) / gridDim.x;
  size_t vb = vchunk * blockIdx.x;
  vb = (vb < nvec) ? vb : nvec;
  size_t ve = vb + vchunk;
  ve = (ve < nvec) ? ve : nvec;
  const size_t begin = vb * kEvec;
  // The last block also takes the ragged tail rc % kEvec.
  const size_t end = (blockIdx.x == gridDim.x - 1) ? rc : (ve * kEvec);
  // End of the 16-byte bulk this block moves vectorized; collapsing it to
  // begin sends the whole range through the element-wise loops instead.
  const size_t vecEnd = vecOk ? (ve * kEvec) : begin;

  // Step 1: push each peer's block into that peer's staging slot mySlot.
  for (int j = 0; j < nActive; j++) {
    if (j == mySlot) {
      continue;
    }
    T* dst = reinterpret_cast<T*>(
        table.staging[ranks.r[j]] + static_cast<size_t>(mySlot) * slotBytes);
    const T* src = sendBuff + static_cast<size_t>(j) * srcStride;
    if (vecOk && vecEnd > begin) {
      const Vec* s4 = reinterpret_cast<const Vec*>(src + begin);
      Vec* d4 = reinterpret_cast<Vec*>(dst + begin);
      const size_t nv = (vecEnd - begin) / kEvec;
      for (size_t v = threadIdx.x; v < nv; v += blockDim.x) {
        d4[v] = s4[v];
      }
    }
    for (size_t i = vecEnd + threadIdx.x; i < end; i += blockDim.x) {
      dst[i] = src[i];
    }
  }

  // Step 2: publish the stores, then raise every peer's flag for this block.
  // The release store must not be reordered before the data, hence the
  // system-scope fence; the reader pairs it with an acquire load.
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();
    for (int j = 0; j < nActive; j++) {
      if (j == mySlot) {
        continue;
      }
      __atomic_store_n(
          &table.flags[ranks.r[j]]
                      [mySlot * rcclx::relay::kOneShotMaxBlocks + blockIdx.x],
          epoch,
          __ATOMIC_RELEASE);
    }
  }

  // Step 3: wait for every source's matching block. Flags are never cleared:
  // the epoch only advances, so a value left by an earlier call always compares
  // as not-yet-arrived for this one.
  if (threadIdx.x == 0) {
    for (int s = 0; s < nActive; s++) {
      if (s == mySlot) {
        continue;
      }
      uint32_t* mine =
          &table
               .flags[myRank][s * rcclx::relay::kOneShotMaxBlocks + blockIdx.x];
      while (!oneShotEpochReached(
          __atomic_load_n(mine, __ATOMIC_ACQUIRE), epoch)) {
      }
    }
  }
  __syncthreads();

  // Step 4: reduce my own contribution with what every source staged. In-place
  // is safe: out aliases own at the same element index.
  if (vecOk && vecEnd > begin) {
    const Vec* o4 = reinterpret_cast<const Vec*>(own + begin);
    Vec* r4 = reinterpret_cast<Vec*>(out + begin);
    const size_t nv = (vecEnd - begin) / kEvec;
    for (size_t v = threadIdx.x; v < nv; v += blockDim.x) {
      Vec a = o4[v];
      for (int s = 0; s < nActive; s++) {
        if (s == mySlot) {
          continue;
        }
        const Vec b = reinterpret_cast<const Vec*>(staged[s] + begin)[v];
#pragma unroll
        for (int k = 0; k < kEvec; k++) {
          a.e[k] = a.e[k] + b.e[k];
        }
      }
      if (divisor > 1) {
#pragma unroll
        for (int k = 0; k < kEvec; k++) {
          a.e[k] = a.e[k] / static_cast<T>(divisor);
        }
      }
      r4[v] = a;
    }
  }
  for (size_t i = vecEnd + threadIdx.x; i < end; i += blockDim.x) {
    T acc = own[i];
    for (int s = 0; s < nActive; s++) {
      if (s == mySlot) {
        continue;
      }
      acc = acc + staged[s][i];
    }
    if (divisor > 1) {
      acc = acc / static_cast<T>(divisor);
    }
    out[i] = acc;
  }
}

template <typename T>
void launchOneShotPushReduceKernel(
    void* out,
    const void* sendBuff,
    const rcclx::relay::OneShotPeerTable& table,
    const rcclx::relay::OneShotRanks& ranks,
    int nActive,
    int myRank,
    int mySlot,
    size_t rc,
    size_t srcStride,
    size_t ownOffset,
    size_t slotBytes,
    uint32_t epoch,
    int divisor,
    cudaStream_t stream) {
  const int blockSize = rcclx::relay::kOneShotThreads;
  int gridSize = static_cast<int>((rc + blockSize - 1) / blockSize);
  if (gridSize < 1) {
    gridSize = 1;
  }
  if (gridSize > rcclx::relay::kOneShotMaxBlocks) {
    gridSize = rcclx::relay::kOneShotMaxBlocks;
  }
  oneShotPushReduceKernel<T><<<gridSize, blockSize, 0, stream>>>(
      static_cast<T*>(out),
      static_cast<const T*>(sendBuff),
      table,
      ranks,
      nActive,
      myRank,
      mySlot,
      rc,
      srcStride,
      ownOffset,
      slotBytes,
      epoch,
      divisor);
}

#define RCCLX_INSTANTIATE_RELAY_KERNELS(T)                                 \
  template void launchIncrementalAddKernel<T>(                             \
      void* output, const void* input, size_t count, cudaStream_t stream); \
  template void launchScaleKernel<T>(                                      \
      void* data, size_t count, int divisor, cudaStream_t stream);         \
  template void launchIncrementalAddAndScaleKernel<T>(                     \
      void* output,                                                        \
      const void* input,                                                   \
      size_t count,                                                        \
      int divisor,                                                         \
      cudaStream_t stream);                                                \
  template void launchFusedReduceKernel<T>(                                \
      void* output,                                                        \
      const void* inputA,                                                  \
      const void* inputB,                                                  \
      size_t count,                                                        \
      int divisor,                                                         \
      cudaStream_t stream);                                                \
  template void launchMultiReduceKernel<T>(                                \
      void* dst,                                                           \
      const void* contribs,                                                \
      int numContribs,                                                     \
      size_t count,                                                        \
      int divisor,                                                         \
      cudaStream_t stream);                                                \
  template void launchSeededMultiReduceKernel<T>(                          \
      void* dst,                                                           \
      const void* seed,                                                    \
      const void* contribs,                                                \
      int numContribs,                                                     \
      size_t count,                                                        \
      int divisor,                                                         \
      cudaStream_t stream);                                                \
  template void launchOneShotPushReduceKernel<T>(                          \
      void* out,                                                           \
      const void* sendBuff,                                                \
      const rcclx::relay::OneShotPeerTable& table,                         \
      const rcclx::relay::OneShotRanks& ranks,                             \
      int nActive,                                                         \
      int myRank,                                                          \
      int mySlot,                                                          \
      size_t rc,                                                           \
      size_t srcStride,                                                    \
      size_t ownOffset,                                                    \
      size_t slotBytes,                                                    \
      uint32_t epoch,                                                      \
      int divisor,                                                         \
      cudaStream_t stream);

RCCLX_INSTANTIATE_RELAY_KERNELS(int8_t)
RCCLX_INSTANTIATE_RELAY_KERNELS(uint8_t)
RCCLX_INSTANTIATE_RELAY_KERNELS(int32_t)
RCCLX_INSTANTIATE_RELAY_KERNELS(uint32_t)
RCCLX_INSTANTIATE_RELAY_KERNELS(int64_t)
RCCLX_INSTANTIATE_RELAY_KERNELS(uint64_t)
RCCLX_INSTANTIATE_RELAY_KERNELS(__half)
RCCLX_INSTANTIATE_RELAY_KERNELS(float)
RCCLX_INSTANTIATE_RELAY_KERNELS(double)
RCCLX_INSTANTIATE_RELAY_KERNELS(__nv_bfloat16)

#undef RCCLX_INSTANTIATE_RELAY_KERNELS
