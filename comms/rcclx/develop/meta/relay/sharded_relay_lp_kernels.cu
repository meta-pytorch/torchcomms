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
// EACH LANE OWNS FOUR CONTIGUOUS ELEMENTS, which is what makes the memory
// access wide. The previous layout gave a lane elements l, l+64, ... so that
// one wavefront covered exactly one block and the absmax was a full-wave
// reduction. Clean reduction, poor memory access: strided single-element access
// compiles to one load and one STORE PER ELEMENT -- global_load_ushort and
// global_store_byte for bf16 -- because neither a lane's reads nor its fp8
// writes are adjacent.
//
// Four contiguous elements instead: one global_load_dwordx2 for four bf16 and
// ONE global_store_dword for the four fp8 bytes. Same work, a QUARTER of the
// memory instructions (8 -> 2 per lane), confirmed in the emitted ISA.
//
// This is also why the stall it fixes was bf16-SPECIFIC. At a given message
// size bf16 has twice the elements of fp32, so it paid twice the per-element
// loads and stores; fp32, with half the elements for the same bytes, was much
// less exposed. Single-group allreduce A=2 measured 0.75x-0.92x in bf16
// above 31.5 MB against 1.14x-1.76x in fp32 at the SAME byte sizes, which is
// what pointed here.
//
// The CONVERSION was never the problem and is deliberately left alone:
// static_cast<float> on a bf16 plus fp8 construction already compiles to
// v_cvt_pk_fp8_f32, the packed hardware fp32->fp8 op, so hand-written asm for
// it would buy nothing.
constexpr int kLpLanesPerWave = 64; // CDNA wavefront
constexpr int kLpElemsPerLane = 4;
constexpr int kLpThreadsPerCta = 256;

// A block is owned by kLpBlockElems / kLpElemsPerLane = 32 lanes, HALF a
// wavefront, so a wavefront carries two blocks and a workgroup eight.
constexpr int kLpLanesPerBlock = kLpBlockElems / kLpElemsPerLane;
constexpr int kLpBlocksPerCta = kLpThreadsPerCta / kLpLanesPerBlock;

static_assert(
    kLpBlockElems % kLpElemsPerLane == 0,
    "a wire block must divide evenly into per-lane runs");
static_assert(
    kLpElemsPerLane == 4,
    "the single-dword fp8 store writes exactly four bytes per lane");
static_assert(
    kLpLanesPerWave % kLpLanesPerBlock == 0,
    "a wavefront must hold a whole number of blocks, so the absmax reduction "
    "stays inside one wavefront and needs no barrier");
static_assert(
    kLpThreadsPerCta % kLpLanesPerBlock == 0,
    "a workgroup must be a whole number of blocks");

// The reduction below xor-reduces over a BLOCK's lanes, and the block-per-wave
// arithmetic above assumes a 64-lane wavefront. Fail at compile time rather
// than compute a scale over the wrong lane set, which every rank would still
// agree on and which would still decode -- silently, and slightly wrong.
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
// Four fp8 codes written as ONE dword. A block is 132 bytes and every block
// offset is 4-block aligned, so byte offset 4*lane inside a block is always
// 4-byte aligned and this store is never misaligned.
__device__ __forceinline__ void lpStoreFourCodes(
    uint8_t* dst,
    const uint8_t (&codes)[4]) {
  uint32_t word;
  __builtin_memcpy(&word, codes, 4);
  __builtin_memcpy(dst, &word, 4);
}

__device__ __forceinline__ void lpLoadFourCodes(
    const uint8_t* src,
    uint8_t (&codes)[4]) {
  uint32_t word;
  __builtin_memcpy(&word, src, 4);
  __builtin_memcpy(codes, &word, 4);
}

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
 * An all-zero block gets scale 0 and codes 0, which decodes back to 0.
 */
__device__ __forceinline__ float lpScaleFor(float absMax) {
  return absMax * kLpInvNormalizeMax;
}

__device__ __forceinline__ uint8_t lpEncodeWithScale(float v, float scale) {
  return lpEncodeByte(scale > 0.0f ? v / scale : 0.0f);
}

/*
 * Absolute maximum across the one wavefront that owns a wire block, left in
 * EVERY lane.
 *
 * log2(64) = 6 xor-shuffle steps, no LDS and no barrier. Every lane ends with
 * the full result, which is what the caller wants: all of them need the scale
 * to encode with, so a reduce-then-broadcast would be a wasted step.
 *
 * fmaxf is associative and the inputs are already non-negative (callers pass
 * fabsf), so the xor order does not change the result -- every lane computes
 * the identical value, bit for bit, which is what keeps the scale a property of
 * the block rather than of the lane that happened to compute it.
 */
__device__ __forceinline__ float blockAbsMax(float absValue) {
  // log2(32) = 5 steps over the lanes of ONE block. The xor width is the
  // block's lane count rather than the wavefront's, so the two blocks sharing a
  // wavefront reduce independently and neither sees the other's values.
  for (int mask = kLpLanesPerBlock / 2; mask > 0; mask >>= 1) {
    absValue = fmaxf(absValue, __shfl_xor(absValue, mask, kLpLanesPerBlock));
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
    // Lane l owns the contiguous run [4l, 4l+4), so this is one wide load.
    const T* src = in + b * kLpBlockElems + lane * kLpElemsPerLane;

    float v[kLpElemsPerLane];
    float absMax = 0.0f;
    for (int i = 0; i < kLpElemsPerLane; i++) {
      v[i] = lpToFloat<T>(src[i]);
      absMax = fmaxf(absMax, fabsf(v[i]));
    }

    const float scale = lpScaleFor(blockAbsMax(absMax));
    uint8_t* block = wire + b * kLpBlockBytes;
    uint8_t codes[kLpElemsPerLane];
    for (int i = 0; i < kLpElemsPerLane; i++) {
      codes[i] = lpEncodeWithScale(v[i], scale);
    }
    lpStoreFourCodes(block + lane * kLpElemsPerLane, codes);
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

  // Four contiguous elements per thread, matching the quantize side: one dword
  // load of four fp8 codes, one scale load shared by all four, and one wide
  // store (dwordx2 for bf16, dwordx4 for fp32) instead of four of each.
  //
  // A run of four never straddles a block boundary, because kLpBlockElems is
  // 128 and 128 % 4 == 0. That is what lets the scale be loaded once per run
  // rather than once per element, and it is why the quad index maps to a block
  // with a plain shift rather than a per-element divide.
  const size_t quads = count / kLpElemsPerLane;
  for (size_t q = tid; q < quads; q += stride) {
    const size_t e = q * kLpElemsPerLane;
    const uint8_t* block = wire + (e / kLpBlockElems) * kLpBlockBytes;
    const float scale = scaleOf(block);

    uint8_t codes[kLpElemsPerLane];
    lpLoadFourCodes(block + (e % kLpBlockElems), codes);

    T vals[kLpElemsPerLane];
    for (int i = 0; i < kLpElemsPerLane; i++) {
      vals[i] = lpFromFloat<T>(lpDecodeByte(codes[i]) * scale);
    }
    __builtin_memcpy(out + e, vals, sizeof(vals));
  }

  // Tail, for a count that is not a whole number of quads. The low-precision
  // gate requires every per-group count to be a multiple of 128, and every
  // region boundary is 512-element aligned, so in practice this never runs --
  // but the kernel is called with per-REGION counts and the last region absorbs
  // a remainder, so it does not get to assume that.
  const size_t tailStart = quads * kLpElemsPerLane;
  for (size_t e = tailStart + tid; e < count; e += stride) {
    const uint8_t* block = wire + (e / kLpBlockElems) * kLpBlockBytes;
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
      // One dword load per contribution instead of four byte loads.
      uint8_t codes[kLpElemsPerLane];
      lpLoadFourCodes(block + lane * kLpElemsPerLane, codes);
      for (int i = 0; i < kLpElemsPerLane; i++) {
        acc[i] += lpDecodeByte(codes[i]) * contribScale;
      }
    }

    float absMax = 0.0f;
    for (int i = 0; i < kLpElemsPerLane; i++) {
      // Exact: the divisor is a power of two, so this rescales the block's
      // absmax by the same factor and leaves every code below unchanged.
      if (divisor > 1) {
        acc[i] /= static_cast<float>(divisor);
      }
      absMax = fmaxf(absMax, fabsf(acc[i]));
    }

    const float scale = lpScaleFor(blockAbsMax(absMax));
    uint8_t* block = wireOut + b * kLpBlockBytes;
    uint8_t codes[kLpElemsPerLane];
    for (int i = 0; i < kLpElemsPerLane; i++) {
      codes[i] = lpEncodeWithScale(acc[i], scale);
    }
    lpStoreFourCodes(block + lane * kLpElemsPerLane, codes);
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
