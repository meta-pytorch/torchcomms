/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <hip/hip_bf16.h>
#include <hip/hip_bfloat16.h>
#include <cstddef>
#include <cstdint>

#include "nccl.h"

/*
 * Host-callable launchers for the low-precision relay kernels.
 *
 * These are DEFINED in `sharded_relay_lp_kernels.cu`, which is compiled as a
 * monolithic (non-RDC) HIP translation unit so the host stub for the
 * `<<<...>>>` launch and the matching `__global__` body live in the same TU.
 * Every instantiation the host TUs may reference is forward-declared here, so a
 * host TU -- built with `--offload-host-only` -- never tries to instantiate one
 * itself. Same arrangement as sharded_relay_allreduce_kernels.h; see its
 * comment.
 *
 * ONLY bf16 AND fp32 ARE INSTANTIATED, because those are the only dtypes
 * lpDtypeSupported() admits. A dispatch that reaches this header for any other
 * dtype is a bug in the gate, and the missing instantiation makes it a link
 * error rather than silent wrong output.
 *
 * WIRE FORMAT, restated because these signatures depend on it: a wire buffer is
 * a sequence of 132-byte blocks, each 128 fp8e4m3 payload bytes followed by one
 * fp32 scale. `count` is always in ELEMENTS of the caller's dtype and must be a
 * multiple of 128 -- lpEligible()'s alignment gate is what guarantees that.
 * Where a launcher takes several wire contributions they are contiguous, block
 * p starting at `wireContribs + p * lpWireBytes(count)`, mirroring how the
 * full-precision multi-reduce launchers already lay out their contributions.
 *
 * WHERE THE DIVISOR GOES. Two kernels can apply it, and which one does depends
 * on whether the region has an active-side closing reduce at all. Regions the
 * ACTIVE rank folds (a direct exchange, whose contributions it reduces itself)
 * get it in `launchLpMultiReduceKernel`. Regions a HELPER reduces do not have
 * an active-side reduce -- the helper's already-reduced chunk lands directly in
 * its final place in the caller's receive buffer, so the only active-side step
 * is a dequantize -- and those get it in `launchLpReduceRequantizeKernel`. A
 * helper returning a pure sum would drop the ncclAvg divisor outright.
 *
 * `launchLpDequantizeKernel` deliberately has no divisor, so it stays a pure
 * format conversion: all-gather and all-to-all have no divisor at all and would
 * otherwise carry a dead parameter.
 */

/**
 * Quantize `count` elements into the wire format.
 *
 * One launch per group covering the ENTIRE boundary-crossing send region,
 * issued before the first ncclGroupStart(). Deliberately hoisted rather than
 * done per-tile: per-tile would cost T launches instead of 1, and would create
 * an ordering hazard in the in-place pipelined allreduce where tile t's fold
 * can be in flight while tile t+1's send source is still being read.
 */
template <typename T>
void launchLpQuantizeKernel(
    void* wireOut,
    const void* in,
    size_t count,
    cudaStream_t stream);

/**
 * Dequantize `count` elements from the wire format into the caller's dtype.
 *
 * Used by all-gather and all-to-all, whose receives land straight into recvBuff
 * today: under low precision the FOREIGN slots arrive in wire form and one
 * launch per group writes them out at the full-precision slot offsets. It must
 * cover foreign slots only and skip the folded diagonal slot, which is already
 * final and already full precision.
 */
template <typename T>
void launchLpDequantizeKernel(
    void* out,
    const void* wireIn,
    size_t count,
    cudaStream_t stream);

/**
 * Wire in, wire out: sum `numContribs` wire contributions in fp32, apply
 * `divisor`, and requantize, picking a fresh per-block absmax.
 *
 * The helper's reduce under low precision, replacing what DISPATCH_FUSED_REDUCE
 * / DISPATCH_MULTI_REDUCE does at full precision. Not templated on the caller's
 * dtype, because neither side of it is in the caller's dtype -- which is
 * exactly why the helper stays ignorant of what it is carrying.
 *
 * Applying the divisor here costs nothing in accuracy. It is always
 * `nActiveRanks`, which the dispatchers require to be a power of two, so
 * scaling before the requantize moves the block absmax by the same exact power
 * of two and leaves every fp8 code unchanged: dividing early and dividing late
 * are bit-identical. See the file comment for why it cannot be deferred to the
 * active rank instead.
 */
void launchLpReduceRequantizeKernel(
    void* wireOut,
    const void* wireContribs,
    int numContribs,
    size_t count,
    int divisor,
    cudaStream_t stream);

/**
 * Mixed reduce: dst (full precision) plus `numContribs` wire contributions,
 * accumulated in fp32, written back in the caller's dtype with the divisor
 * applied once.
 *
 * The active rank's closing fold under low precision. fp32 accumulation is the
 * point -- it is what keeps a reduction of e4m3 values from overflowing or
 * losing the sum's low bits.
 */
template <typename T>
void launchLpMultiReduceKernel(
    void* dst,
    const void* wireContribs,
    int numContribs,
    size_t count,
    int divisor,
    cudaStream_t stream);

/**
 * As above, but seeded from a separate full-precision buffer instead of reading
 * dst: result[i] = (seed[i] + sum_p wire_p[i]) / divisor, in that order.
 */
template <typename T>
void launchLpSeededMultiReduceKernel(
    void* dst,
    const void* seed,
    const void* wireContribs,
    int numContribs,
    size_t count,
    int divisor,
    cudaStream_t stream);

// Suppress instantiation in the host TU; the actual instantiations live in
// sharded_relay_lp_kernels.cu.
#define RCCLX_DECLARE_RELAY_LP_KERNEL_INSTANTIATIONS(T)                  \
  extern template void launchLpQuantizeKernel<T>(                        \
      void* wireOut, const void* in, size_t count, cudaStream_t stream); \
  extern template void launchLpDequantizeKernel<T>(                      \
      void* out, const void* wireIn, size_t count, cudaStream_t stream); \
  extern template void launchLpMultiReduceKernel<T>(                     \
      void* dst,                                                         \
      const void* wireContribs,                                          \
      int numContribs,                                                   \
      size_t count,                                                      \
      int divisor,                                                       \
      cudaStream_t stream);                                              \
  extern template void launchLpSeededMultiReduceKernel<T>(               \
      void* dst,                                                         \
      const void* seed,                                                  \
      const void* wireContribs,                                          \
      int numContribs,                                                   \
      size_t count,                                                      \
      int divisor,                                                       \
      cudaStream_t stream);

RCCLX_DECLARE_RELAY_LP_KERNEL_INSTANTIATIONS(float)
RCCLX_DECLARE_RELAY_LP_KERNEL_INSTANTIATIONS(__nv_bfloat16)

#undef RCCLX_DECLARE_RELAY_LP_KERNEL_INSTANTIATIONS
