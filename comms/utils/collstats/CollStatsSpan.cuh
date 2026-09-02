// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <cstdint>

#include "comms/utils/collstats/CollStatsDeviceBlock.h"
#include "comms/utils/collstats/CollStatsHistogram.h"
#include "comms/utils/collstats/CollStatsTypes.h"

// Device-side collective span: the two touch points a collective kernel adds to
// time itself. Entry records each block's start as a running minimum; the exit
// finalizer (the last block to arrive) computes the duration and records one
// observation. No block waits on another — the only barrier is the collective's
// own block-wide __syncthreads(), which the caller already issues.
//
// Gated to Hopper and above; below it, and when instrumentation is disabled (a
// null block), every entry point compiles to a no-op.
//
// Not implemented on AMD, and a green AMD build is not evidence otherwise:
// hipcc never defines __CUDA_ARCH__ (it defines __HIP_DEVICE_COMPILE__), so the
// gate below is false for every AMD arch and the whole file compiles away.
// Not yet rather than not possible -- %globaltimer is PTX-only, and AMD's
// nanosecond reference (__builtin_amdgcn_s_memrealtime) needs its own frequency
// handling before end-minus-start means the same thing on both vendors.

namespace meta::comms::collstats {

// %globaltimer in nanoseconds. It is already an ns reference clock, so a
// single-GPU end-minus-start needs no calibration.
__device__ __forceinline__ uint64_t collStatsGlobaltimerNs() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
#else
  return 0;
#endif
}

// Blocks in the launch grid. The span's arrival barrier counts one entry per
// block, elected on threadIdx.{x,y,z} == 0, so it counts every block however
// the grid is shaped. Comparing against gridDim.x alone would finalize a 2-D or
// 3-D launch on its first gridDim.x arrivals, timing a span that is still
// running.
__device__ __forceinline__ unsigned int collStatsGridBlocks() {
  return gridDim.x * gridDim.y * gridDim.z;
}

// Entry: the elected thread of each block folds its start time into the slot's
// running minimum. Call from all threads; the election is internal.
//
// An out-of-range slot is dropped rather than indexed. The host path
// (collStatsEnqueuePreReset) makes the same check, and the device is the only
// side where getting it wrong writes past the span allocation instead of
// merely losing an observation.
__device__ __forceinline__ void collStatsSpanEntry(
    CollStatsDeviceBlock* block,
    uint32_t slot) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (block == nullptr) {
    return;
  }
  // The slot bound is tested inside the election, so only the elected thread
  // loads numSlots rather than every thread in the block.
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 &&
      slot < block->numSlots) {
    atomicMin(
        reinterpret_cast<unsigned long long*>(&block->span[slot].start),
        static_cast<unsigned long long>(collStatsGlobaltimerNs()));
  }
#else
  (void)block;
  (void)slot;
#endif
}

// Exit finalizer taking a value slot resolved by the host key registry before
// launch, so the device performs no key lookup at all.
//
// The finalizer emits only; it does not reset the slot. Slot cleanup is owned
// by the stream-ordered pre-sequence reset (collStatsEnqueuePreReset), which
// runs before the next collective's first entry and cleans the slot even when a
// prior sequence aborted without finalizing.
//
// Election is internal and on all three thread indices, matching entry. The
// caller must still issue the block-wide __syncthreads() first. Callers elect
// too, but the common 1-D idiom (`if (threadIdx.x == 0)`) admits
// blockDim.y * blockDim.z threads per block in a 2-D or 3-D launch, and the
// extra arrivals mistime the span rather than losing it: the equality below
// still fires exactly once, but on the gridDim'th of gridDim * blockDim.y *
// blockDim.z increments -- while most blocks are still running -- so it reads
// `end` early and records a plausible, far-too-short duration. That is the
// failure this whole file is built to avoid, so the election is not left to
// the caller.
__device__ __forceinline__ void collStatsSpanFinalizeElectedById(
    CollStatsDeviceBlock* block,
    uint32_t slot,
    uint32_t keyId,
    uint64_t logicalBytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (block == nullptr || threadIdx.x != 0 || threadIdx.y != 0 ||
      threadIdx.z != 0) {
    return;
  }
  if (slot >= block->numSlots) {
    return;
  }
  CollStatSpanScratch* s = &block->span[slot];
  const uint64_t end = collStatsGlobaltimerNs();
  // Release: publish this block's entry atomicMin before its arrival becomes
  // visible. Acquire (below): the arrival count only orders the start
  // timestamps if the read of them cannot be hoisted above the count.
  __threadfence();
  const unsigned int prev = atomicAdd(&s->arrived, 1u);
  if (prev + 1u == collStatsGridBlocks()) {
    __threadfence();
    const uint64_t start =
        *reinterpret_cast<volatile const unsigned long long*>(&s->start);
    // Drop rather than record a duration that is not one. If no entry ever
    // landed the sentinel survives, and end - UINT64_MAX wraps to end + 1 -- a
    // sub-microsecond value that lands in the underflow bucket and reads as a
    // real, very fast collective rather than as missing data. A start left over
    // from an un-pre-reset predecessor fails the same way in the other
    // direction, pinning durMaxNs.
    if (start == kSpanStartInit || end < start) {
      return;
    }
    collStatsRecordById(block, keyId, end - start, logicalBytes);
  }
#else
  (void)block;
  (void)slot;
  (void)keyId;
  (void)logicalBytes;
#endif
}

// Derive the collective's logical message size from its launch args and
// finalize the span in one call, so each instrumented kernel repeats neither.
// `Args` is any collective's kernel-args type exposing `count` and
// `collStatsKeyId`; the element size comes from the payload type `T`. Call
// AFTER the block-wide __syncthreads(); the election is internal.
template <typename T, typename Args>
__device__ __forceinline__ void collStatsSpanFinalizeElectedColl(
    CollStatsDeviceBlock* block,
    uint32_t slot,
    const Args& args) {
  const uint64_t logicalBytes = static_cast<uint64_t>(args.count) * sizeof(T);
  collStatsSpanFinalizeElectedById(
      block, slot, args.collStatsKeyId, logicalBytes);
}

} // namespace meta::comms::collstats
