/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Device bodies for the low-precision relay kernels. Separate from the header
// for the same reason sharded_relay_allreduce_kernels.cu is: this TU is
// compiled as a monolithic non-RDC HIP library WITHOUT `--offload-host-only`,
// so the
// `__global__` bodies survive into the archive, and the explicit instantiations
// at the bottom give the host TUs' `<<<...>>>` stubs something to bind to.

#include "meta/relay/sharded_relay_lp_kernels.h"

#include <hip/hip_runtime.h>

#include "meta/relay/sharded_relay_lp.h"
#include "rccl_float8.h"

namespace {

using rcclx::relay::kLpBlockBytes;
using rcclx::relay::kLpBlockElems;
using rcclx::relay::kLpInvNormalizeMax;
using rcclx::relay::lpWireBytes;

static_assert(
    sizeof(rccl_float8) == 1,
    "the wire format stores one fp8 code per byte");

// ONE WAVEFRONT OWNS ONE WIRE BLOCK, with each lane holding kLpElemsPerLane
// payload elements.
//
// This replaced a layout of one thread per element (128 threads spanning two
// wavefronts per block) whose absmax was a shared-memory BROADCAST loop: every
// thread read all 128 of its block's absolute values and took the max itself.
// The argument for it was that LDS broadcast is one instruction per wavefront
// and it needs one barrier instead of a tree's seven -- barriers being what a
// 256-byte-per-block kernel cannot afford.
//
// The barrier half of that was right and the conclusion did not follow.
// Reducing WITHIN a wavefront needs neither LDS nor barriers: __shfl_xor is a
// register permute. The old loop paid a 128-long SERIAL dependent fmaxf chain
// per thread, with an LDS read feeding every link, to produce one output byte
// -- a latency chain no amount of occupancy hides. Six shuffle steps replace
// it, and because a wavefront is implicitly in lockstep both __syncthreads()
// calls and the whole
// __shared__ array disappear with it. That also lets a wave whose block is out
// of range simply leave, instead of every thread staying resident to service a
// barrier.
//
// Sizing the block to the wavefront is what makes the reduction purely
// intra-wave; two wavefronts per block would put a cross-wave combine (and so
// LDS and a barrier) back into the critical path for one extra step of
// reduction.
constexpr int kLpLanesPerBlock = 64; // CDNA wavefront
constexpr int kLpThreadsPerCta = 256;
constexpr int kLpBlocksPerCta = kLpThreadsPerCta / kLpLanesPerBlock;
constexpr int kLpElemsPerLane = kLpBlockElems / kLpLanesPerBlock;

static_assert(
    kLpBlockElems % kLpLanesPerBlock == 0,
    "a wire block must divide evenly among the lanes of one wavefront");
static_assert(
    kLpThreadsPerCta % kLpLanesPerBlock == 0,
    "a workgroup must be a whole number of wavefronts");

// The intra-wavefront reduction below is only correct on a 64-lane wavefront:
// it xor-reduces over exactly kLpLanesPerBlock lanes and takes the whole
// block's absmax to live in that one wavefront. Fail at compile time rather
// than compute a per-half-block scale that every rank would still agree on and
// that would still decode -- silently, and slightly wrong.
#if defined(__AMDGCN_WAVEFRONT_SIZE) && __AMDGCN_WAVEFRONT_SIZE != 64
#error "sharded relay low precision kernels assume a 64-lane wavefront"
#endif

// Enough workgroups to fill the device several times over without making the
// grid-stride loop pointless. The kernels are memory bound, so the exact value
// is not delicate.
constexpr int kLpMaxCtas = 8192;

int ctaCount(size_t nBlocks) {
  const size_t ctas = (nBlocks + kLpBlocksPerCta - 1) / kLpBlocksPerCta;
  return static_cast<int>(
      ctas < kLpMaxCtas ? (ctas == 0 ? 1 : ctas) : kLpMaxCtas);
}

int elementCtaCount(size_t count) {
  const size_t ctas = (count + kLpThreadsPerCta - 1) / kLpThreadsPerCta;
  return static_cast<int>(
      ctas < kLpMaxCtas ? (ctas == 0 ? 1 : ctas) : kLpMaxCtas);
}

// Widening and narrowing in one place, so a dtype whose conversion needs an
// intrinsic rather than a cast is a two-line change here.
template <typename T>
__device__ __forceinline__ float lpToFloat(T v) {
  return static_cast<float>(v);
}

template <typename T>
__device__ __forceinline__ T lpFromFloat(float v) {
  return static_cast<T>(v);
}

// fp8e4m3 code <-> value, through rccl_float8 so the flavour question
// (__hip_fp8_e4m3_fnuz on gfx942, OCP __hip_fp8_e4m3 elsewhere) is answered in
// exactly one place in the tree. Encode and decode always run on the same
// device, so they always agree; the BYTES are arch-local, which is fine for an
// intra-node homogeneous relay and documented as latent otherwise.
__device__ __forceinline__ uint8_t lpEncodeByte(float v) {
  const rccl_float8 q(v);
  uint8_t byte;
  __builtin_memcpy(&byte, &q, 1);
  return byte;
}

__device__ __forceinline__ float lpDecodeByte(uint8_t byte) {
  rccl_float8 q{};
  __builtin_memcpy(&q, &byte, 1);
  return static_cast<float>(q);
}

// Where a block's inline scale lives. kLpBlockBytes is a multiple of 4 and
// every LP buffer offset is a whole number of blocks, so this is always 4-byte
// aligned.
__device__ __forceinline__ float* scaleSlot(uint8_t* block) {
  return reinterpret_cast<float*>(block + kLpBlockElems);
}

__device__ __forceinline__ float scaleOf(const uint8_t* block) {
  return *reinterpret_cast<const float*>(block + kLpBlockElems);
}

/*
 * Scale for a block whose absolute maximum is absMax.
 *
 * absMax * 2^-7 is EXACT (a power-of-two multiply), and the decode is
 * code * scale, so a block whose elements are all equal round-trips
 * bit-exactly: v / (v * 2^-7) has the exact quotient 128, fp8e4m3 represents
 * 128 exactly, and 128 * (v * 2^-7) is v again. Normalizing to the format's own
 * maximum (240 or 448) instead would make the scale a rounded fp32 value and
 * throw that property away for no precision gain. See sharded_relay_lp.h.
 *
 * An all-zero block gets scale 0 and codes 0, which decodes back to 0. A block
 * holding a NaN or an inf gets a non-finite scale and decodes non-finite; see
 * lpAbsMaxFold().
 */
__device__ __forceinline__ float lpScaleFor(float absMax) {
  return absMax * kLpInvNormalizeMax;
}

__device__ __forceinline__ uint8_t lpEncodeWithScale(float v, float scale) {
  return lpEncodeByte(scale > 0.0f ? v / scale : 0.0f);
}

/*
 * One step of the absmax fold, NaN-STICKY.
 *
 * Deliberately not fmaxf: fmaxf returns the NON-NaN operand, so an absmax
 * folded with it reports the max of a block's FINITE elements and reports 0 for
 * a block that is entirely NaN. Zero is the all-zero block's absmax, so a
 * diverged block would be written as scale 0 with 128 zero codes and decode
 * back to clean 0.0 -- silently erasing the divergence, and stopping a
 * post-collective isfinite() check from firing on exactly the runs that need
 * it.
 *
 * Sticky instead: once a NaN is in the accumulator it stays, because `v > acc`
 * is false for a NaN acc. absMax then being non-finite makes lpScaleFor()
 * non-finite, which makes every element of that block decode non-finite -- so
 * the block is destroyed either way, but visibly. An inf absmax reaches the
 * same place without help: the scale is inf, so every finite element encodes to
 * 0 and decodes to 0 * inf == NaN.
 *
 * Still commutative and associative over the non-negative inputs the callers
 * pass, so the xor order below does not change the result and every lane
 * computes the identical value, bit for bit.
 */
__device__ __forceinline__ float lpAbsMaxFold(float acc, float v) {
  return (v > acc || __builtin_isnan(v)) ? v : acc;
}

/*
 * Absolute maximum across the one wavefront that owns a wire block, left in
 * EVERY lane.
 *
 * log2(64) = 6 xor-shuffle steps, no LDS and no barrier. Every lane ends with
 * the full result, which is what the caller wants: all of them need the scale
 * to encode with, so a reduce-then-broadcast would be a wasted step.
 *
 * The fold is lpAbsMaxFold rather than fmaxf so a NaN survives the reduction;
 * it is associative over the non-negative inputs callers pass (fabsf), so the
 * xor order does not change the result -- every lane computes the identical
 * value, bit for bit, which is what keeps the scale a property of the block
 * rather than of the lane that happened to compute it.
 */
__device__ __forceinline__ float waveAbsMax(float absValue) {
  for (int mask = kLpLanesPerBlock / 2; mask > 0; mask >>= 1) {
    absValue =
        lpAbsMaxFold(absValue, __shfl_xor(absValue, mask, kLpLanesPerBlock));
  }
  return absValue;
}

} // namespace

// ---------------------------------------------------------------------------
// Quantize
// ---------------------------------------------------------------------------

template <typename T>
__global__ void lpQuantizeKernel(const T* in, uint8_t* wire, size_t nBlocks) {
  const int wave = static_cast<int>(threadIdx.x) / kLpLanesPerBlock;
  const int lane = static_cast<int>(threadIdx.x) % kLpLanesPerBlock;
  const size_t ctaStride =
      static_cast<size_t>(gridDim.x) * static_cast<size_t>(kLpBlocksPerCta);

  for (size_t base = static_cast<size_t>(blockIdx.x) * kLpBlocksPerCta;
       base < nBlocks;
       base += ctaStride) {
    const size_t b = base + static_cast<size_t>(wave);
    // No barrier in this loop, so a wavefront with no block to do just leaves.
    if (b >= nBlocks) {
      continue;
    }
    const T* src = in + b * kLpBlockElems;

    // Lane l takes elements l, l + 64, ... so each of the kLpElemsPerLane
    // accesses is one contiguous 64-lane run.
    float v[kLpElemsPerLane];
    float absMax = 0.0f;
    for (int i = 0; i < kLpElemsPerLane; i++) {
      v[i] = lpToFloat<T>(src[lane + i * kLpLanesPerBlock]);
      absMax = lpAbsMaxFold(absMax, fabsf(v[i]));
    }

    const float scale = lpScaleFor(waveAbsMax(absMax));
    uint8_t* block = wire + b * kLpBlockBytes;
    for (int i = 0; i < kLpElemsPerLane; i++) {
      block[lane + i * kLpLanesPerBlock] = lpEncodeWithScale(v[i], scale);
    }
    if (lane == 0) {
      *scaleSlot(block) = scale;
    }
  }
}

template <typename T>
void launchLpQuantizeKernel(
    void* wireOut,
    const void* in,
    size_t count,
    cudaStream_t stream) {
  const size_t nBlocks = count / kLpBlockElems;
  if (nBlocks == 0) {
    return;
  }
  lpQuantizeKernel<T><<<ctaCount(nBlocks), kLpThreadsPerCta, 0, stream>>>(
      static_cast<const T*>(in), static_cast<uint8_t*>(wireOut), nBlocks);
}

// ---------------------------------------------------------------------------
// Dequantize
// ---------------------------------------------------------------------------

template <typename T>
__global__ void lpDequantizeKernel(T* out, const uint8_t* wire, size_t count) {
  const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

  for (size_t e = tid; e < count; e += stride) {
    const uint8_t* block = wire + (e / kLpBlockElems) * kLpBlockBytes;
    // Broadcast read: 128 consecutive threads share one scale.
    const float scale = scaleOf(block);
    out[e] = lpFromFloat<T>(lpDecodeByte(block[e % kLpBlockElems]) * scale);
  }
}

template <typename T>
void launchLpDequantizeKernel(
    void* out,
    const void* wireIn,
    size_t count,
    cudaStream_t stream) {
  if (count == 0) {
    return;
  }
  lpDequantizeKernel<T>
      <<<elementCtaCount(count), kLpThreadsPerCta, 0, stream>>>(
          static_cast<T*>(out), static_cast<const uint8_t*>(wireIn), count);
}

// ---------------------------------------------------------------------------
// Helper-side reduce and requantize (wire in, wire out)
// ---------------------------------------------------------------------------

__global__ void lpReduceRequantizeKernel(
    uint8_t* wireOut,
    const uint8_t* wireContribs,
    int numContribs,
    size_t nBlocks,
    size_t contribStride,
    int divisor) {
  const int wave = static_cast<int>(threadIdx.x) / kLpLanesPerBlock;
  const int lane = static_cast<int>(threadIdx.x) % kLpLanesPerBlock;
  const size_t ctaStride =
      static_cast<size_t>(gridDim.x) * static_cast<size_t>(kLpBlocksPerCta);

  for (size_t base = static_cast<size_t>(blockIdx.x) * kLpBlocksPerCta;
       base < nBlocks;
       base += ctaStride) {
    const size_t b = base + static_cast<size_t>(wave);
    if (b >= nBlocks) {
      continue;
    }
    const size_t blockOffset = b * kLpBlockBytes;

    // fp32 accumulation, which is the whole reason the sum does not have to be
    // range-limited the way an fp8 accumulation would.
    float acc[kLpElemsPerLane] = {};
    for (int p = 0; p < numContribs; p++) {
      const uint8_t* block =
          wireContribs + static_cast<size_t>(p) * contribStride + blockOffset;
      const float contribScale = scaleOf(block);
      for (int i = 0; i < kLpElemsPerLane; i++) {
        acc[i] +=
            lpDecodeByte(block[lane + i * kLpLanesPerBlock]) * contribScale;
      }
    }

    float absMax = 0.0f;
    for (int i = 0; i < kLpElemsPerLane; i++) {
      // Exact: the divisor is a power of two, so this rescales the block's
      // absmax by the same factor and leaves every code below unchanged.
      if (divisor > 1) {
        acc[i] /= static_cast<float>(divisor);
      }
      absMax = lpAbsMaxFold(absMax, fabsf(acc[i]));
    }

    const float scale = lpScaleFor(waveAbsMax(absMax));
    uint8_t* block = wireOut + b * kLpBlockBytes;
    for (int i = 0; i < kLpElemsPerLane; i++) {
      block[lane + i * kLpLanesPerBlock] = lpEncodeWithScale(acc[i], scale);
    }
    if (lane == 0) {
      *scaleSlot(block) = scale;
    }
  }
}

void launchLpReduceRequantizeKernel(
    void* wireOut,
    const void* wireContribs,
    int numContribs,
    size_t count,
    int divisor,
    cudaStream_t stream) {
  const size_t nBlocks = count / kLpBlockElems;
  if (nBlocks == 0 || numContribs <= 0) {
    return;
  }
  lpReduceRequantizeKernel<<<ctaCount(nBlocks), kLpThreadsPerCta, 0, stream>>>(
      static_cast<uint8_t*>(wireOut),
      static_cast<const uint8_t*>(wireContribs),
      numContribs,
      nBlocks,
      lpWireBytes(count),
      divisor);
}

// ---------------------------------------------------------------------------
// Active-rank closing reduce (wire contributions in, caller's dtype out)
// ---------------------------------------------------------------------------

template <typename T>
__global__ void lpMultiReduceKernel(
    T* dst,
    const uint8_t* wireContribs,
    int numContribs,
    size_t count,
    size_t contribStride,
    int divisor) {
  const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

  for (size_t e = tid; e < count; e += stride) {
    const size_t blockOffset = (e / kLpBlockElems) * kLpBlockBytes;
    const size_t lane = e % kLpBlockElems;
    float acc = lpToFloat<T>(dst[e]);
    for (int p = 0; p < numContribs; p++) {
      const uint8_t* block =
          wireContribs + static_cast<size_t>(p) * contribStride + blockOffset;
      acc += lpDecodeByte(block[lane]) * scaleOf(block);
    }
    if (divisor > 1) {
      acc /= static_cast<float>(divisor);
    }
    dst[e] = lpFromFloat<T>(acc);
  }
}

template <typename T>
void launchLpMultiReduceKernel(
    void* dst,
    const void* wireContribs,
    int numContribs,
    size_t count,
    int divisor,
    cudaStream_t stream) {
  if (count == 0) {
    return;
  }
  lpMultiReduceKernel<T>
      <<<elementCtaCount(count), kLpThreadsPerCta, 0, stream>>>(
          static_cast<T*>(dst),
          static_cast<const uint8_t*>(wireContribs),
          numContribs,
          count,
          lpWireBytes(count),
          divisor);
}

template <typename T>
__global__ void lpSeededMultiReduceKernel(
    T* dst,
    const T* seed,
    const uint8_t* wireContribs,
    int numContribs,
    size_t count,
    size_t contribStride,
    int divisor) {
#pragma clang fp reassociate(off)
  const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

#pragma clang loop vectorize(disable) interleave(disable)
  for (size_t e = tid; e < count; e += stride) {
    const size_t blockOffset = (e / kLpBlockElems) * kLpBlockBytes;
    const size_t lane = e % kLpBlockElems;
    float acc = lpToFloat<T>(seed[e]);
#pragma clang loop unroll(disable) vectorize(disable) interleave(disable)
    for (int p = 0; p < numContribs; p++) {
      const uint8_t* block =
          wireContribs + static_cast<size_t>(p) * contribStride + blockOffset;
      acc = acc + lpDecodeByte(block[lane]) * scaleOf(block);
    }
    if (divisor > 1) {
      acc = acc / static_cast<float>(divisor);
    }
    dst[e] = lpFromFloat<T>(acc);
  }
}

template <typename T>
void launchLpSeededMultiReduceKernel(
    void* dst,
    const void* seed,
    const void* wireContribs,
    int numContribs,
    size_t count,
    int divisor,
    cudaStream_t stream) {
  if (count == 0) {
    return;
  }
  lpSeededMultiReduceKernel<T>
      <<<elementCtaCount(count), kLpThreadsPerCta, 0, stream>>>(
          static_cast<T*>(dst),
          static_cast<const T*>(seed),
          static_cast<const uint8_t*>(wireContribs),
          numContribs,
          count,
          lpWireBytes(count),
          divisor);
}

// Explicit instantiations. bf16 and fp32 only -- see the header.
#define RCCLX_INSTANTIATE_RELAY_LP_KERNELS(T)                            \
  template void launchLpQuantizeKernel<T>(                               \
      void* wireOut, const void* in, size_t count, cudaStream_t stream); \
  template void launchLpDequantizeKernel<T>(                             \
      void* out, const void* wireIn, size_t count, cudaStream_t stream); \
  template void launchLpMultiReduceKernel<T>(                            \
      void* dst,                                                         \
      const void* wireContribs,                                          \
      int numContribs,                                                   \
      size_t count,                                                      \
      int divisor,                                                       \
      cudaStream_t stream);                                              \
  template void launchLpSeededMultiReduceKernel<T>(                      \
      void* dst,                                                         \
      const void* seed,                                                  \
      const void* wireContribs,                                          \
      int numContribs,                                                   \
      size_t count,                                                      \
      int divisor,                                                       \
      cudaStream_t stream);

RCCLX_INSTANTIATE_RELAY_LP_KERNELS(float)
RCCLX_INSTANTIATE_RELAY_LP_KERNELS(__nv_bfloat16)

#undef RCCLX_INSTANTIATE_RELAY_LP_KERNELS
