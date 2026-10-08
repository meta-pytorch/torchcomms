/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Combine: push locally-owned expert outputs back to the origin, and
 * accumulate them there in strict top-k order in fp32.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_EP_COMBINE_H_
#define RCCL_EP_COMBINE_H_

#include "device/ep_common.h"

namespace rccl_ep {

// Combine, non-hybrid intranode. The sum along num_topk runs in STRICT index
// order in float32, which is why the staging region is per (slot, topk) rather
// than pre-reduced per rank: with A owning slots {1,3} and B {0,2} the order is
// B0, A1, B2, A3, interleaved across ranks. Pre-reducing reorders that, and
// float addition is not associative, so bit-exactness fails.

// Sum `nrow` rows of `src`, named by `rows` and applied in the order given,
// into the single row `dst`, seeded by up to two optional bias rows. All four
// pointers address a ROW, not a tensor -- the caller has already applied its
// own token stride.
//
// Both combine paths reduce exactly this way and under the same contract:
// accumulate in fp32, round once at the end, and apply the terms as bias0,
// bias1, then rows in ascending k. That order is the bit-exactness guarantee,
// so it is defined here once rather than in each path.
//
// Vectorised at dwordx4: scalar 2-byte loads are an 8x loss of load width, and
// the split changes only which lane owns which element, never the order of
// additions applied to any single element.
__device__ __forceinline__ void reduce_rows_bf16(bf16_t* dst,
                                                 const bf16_t* __restrict__ src,
                                                 const int32_t* rows, int nrow,
                                                 const bf16_t* __restrict__ bias0,
                                                 const bf16_t* __restrict__ bias1,
                                                 int hidden, int lane) {
  constexpr int kPerVec = 8; // 8 bf16 = 16 B
  const int nvec = hidden / kPerVec;
  for (int v = lane; v < nvec; v += kWarpSize) {
    float acc[kPerVec];
    const int h0 = v * kPerVec;
#pragma unroll
    for (int u = 0; u < kPerVec; ++u) acc[u] = 0.f;
    if (bias0) {
      const uint4 b = *reinterpret_cast<const uint4*>(bias0 + h0);
      const bf16_t* bb = reinterpret_cast<const bf16_t*>(&b);
#pragma unroll
      for (int u = 0; u < kPerVec; ++u) acc[u] += __bfloat162float(bb[u]);
    }
    if (bias1) {
      const uint4 b = *reinterpret_cast<const uint4*>(bias1 + h0);
      const bf16_t* bb = reinterpret_cast<const bf16_t*>(&b);
#pragma unroll
      for (int u = 0; u < kPerVec; ++u) acc[u] += __bfloat162float(bb[u]);
    }
    for (int j = 0; j < nrow; ++j) { // ascending k, as strict order requires
      const uint4 y = *reinterpret_cast<const uint4*>(src + (size_t)rows[j] * hidden + h0);
      const bf16_t* yy = reinterpret_cast<const bf16_t*>(&y);
#pragma unroll
      for (int u = 0; u < kPerVec; ++u) acc[u] += __bfloat162float(yy[u]);
    }
    uint4 o;
    bf16_t* oo = reinterpret_cast<bf16_t*>(&o);
#pragma unroll
    for (int u = 0; u < kPerVec; ++u) oo[u] = __float2bfloat16(acc[u]);
    *reinterpret_cast<uint4*>(dst + h0) = o;
  }
  // scalar tail when hidden is not a multiple of 8
  for (int h = nvec * kPerVec + lane; h < hidden; h += kWarpSize) {
    float acc = 0.f;
    if (bias0) acc += __bfloat162float(bias0[h]);
    if (bias1) acc += __bfloat162float(bias1[h]);
    for (int j = 0; j < nrow; ++j) acc += __bfloat162float(src[(size_t)rows[j] * hidden + h]);
    dst[h] = __float2bfloat16(acc);
  }
}

// acc + y * s with the product rounded separately (no fma contraction). On HIP,
// __fadd_rn/__fmul_rn are plain operators that clang's default fp-contract fuses.
__device__ __forceinline__ float mul_add_unfused(float acc, float y, float s) {
#pragma clang fp contract(off)
  return acc + y * s;
}

// dst = bf16(sum_j src[row_j] * s_j): fp32, rows in the given order, one rounding.
// rowk packs row << 4 | k, so sorting by row keeps each row's top-k slot. sc null
// means s_j = 1, bit-identical to a plain sum; otherwise s_j = 0.f + sc[k_j] (maps
// -0.0 to +0.0). fma selects a fused multiply-add or two separate roundings.
__device__ __forceinline__ void reduce_rows_rk_bf16(bf16_t* dst,
                                                    const bf16_t* __restrict__ src,
                                                    const int32_t* rowk, int nrow,
                                                    const float* __restrict__ sc, int fma,
                                                    int hidden, int lane) {
  constexpr int kPerVec = 8;
  const int nvec = hidden / kPerVec;
  for (int v = lane; v < nvec; v += kWarpSize) {
    float acc[kPerVec];
    const int h0 = v * kPerVec;
#pragma unroll
    for (int u = 0; u < kPerVec; ++u) acc[u] = 0.f;
    for (int j = 0; j < nrow; ++j) {
      const uint4 y = *reinterpret_cast<const uint4*>(src + (size_t)(rowk[j] >> 4) * hidden + h0);
      const bf16_t* yy = reinterpret_cast<const bf16_t*>(&y);
      const float s = sc ? 0.f + sc[rowk[j] & 15] : 1.f;
#pragma unroll
      for (int u = 0; u < kPerVec; ++u)
        acc[u] = fma ? __builtin_fmaf(__bfloat162float(yy[u]), s, acc[u])
                     : mul_add_unfused(acc[u], __bfloat162float(yy[u]), s);
    }
    uint4 o;
    bf16_t* oo = reinterpret_cast<bf16_t*>(&o);
#pragma unroll
    for (int u = 0; u < kPerVec; ++u) oo[u] = __float2bfloat16(acc[u]);
    *reinterpret_cast<uint4*>(dst + h0) = o;
  }
  for (int h = nvec * kPerVec + lane; h < hidden; h += kWarpSize) {
    float acc = 0.f;
    for (int j = 0; j < nrow; ++j) {
      const float y = __bfloat162float(src[(size_t)(rowk[j] >> 4) * hidden + h]);
      const float s = sc ? 0.f + sc[rowk[j] & 15] : 1.f;
      acc = fma ? __builtin_fmaf(y, s, acc) : mul_add_unfused(acc, y, s);
    }
    dst[h] = __float2bfloat16(acc);
  }
}


// reduce_rows_rk_bf16 for at most 8 rows, same order and bits. Rows and scales sit in
// fixed-size register arrays, so every row load of a vector is issued before the first
// multiply-add. s[j] is already 0.f + weight, or 1.f unscaled.
__device__ __forceinline__ void reduce_rows8_scaled_bf16(bf16_t* dst,
                                                         const bf16_t* __restrict__ src,
                                                         const int32_t (&rows)[8],
                                                         const float (&s)[8], int nrow, int fma,
                                                         int hidden, int lane) {
  constexpr int kPerVec = 8;
  const int nvec = hidden / kPerVec;
#pragma unroll 1
  for (int v = lane; v < nvec; v += kWarpSize) {
    const int h0 = v * kPerVec;
    uint4 y[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      if (j < nrow) y[j] = *reinterpret_cast<const uint4*>(src + (size_t)rows[j] * hidden + h0);
    }
    float acc[kPerVec];
#pragma unroll
    for (int u = 0; u < kPerVec; ++u) acc[u] = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      if (j < nrow) {
        const bf16_t* yy = reinterpret_cast<const bf16_t*>(&y[j]);
#pragma unroll
        for (int u = 0; u < kPerVec; ++u)
          acc[u] = fma ? __builtin_fmaf(__bfloat162float(yy[u]), s[j], acc[u])
                       : mul_add_unfused(acc[u], __bfloat162float(yy[u]), s[j]);
      }
    }
    uint4 o;
    bf16_t* oo = reinterpret_cast<bf16_t*>(&o);
#pragma unroll
    for (int u = 0; u < kPerVec; ++u) oo[u] = __float2bfloat16(acc[u]);
    *reinterpret_cast<uint4*>(dst + h0) = o;
  }
  for (int h = nvec * kPerVec + lane; h < hidden; h += kWarpSize) {
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      if (j < nrow) {
        const float y = __bfloat162float(src[(size_t)rows[j] * hidden + h]);
        acc = fma ? __builtin_fmaf(y, s[j], acc) : mul_add_unfused(acc, y, s[j]);
      }
    }
    dst[h] = __float2bfloat16(acc);
  }
}

// reduce_rows_bf16 for at most 8 rows, in the same addition order (bias0, bias1, then
// rows), so bit-identical. Row bases stay in registers, so every row load of a vector
// is issued before the first add rather than one at a time.
__device__ __forceinline__ void reduce_rows8_bf16(bf16_t* dst,
                                                  const bf16_t* __restrict__ src,
                                                  const int32_t (&rows)[8], int nrow,
                                                  const bf16_t* __restrict__ bias0,
                                                  const bf16_t* __restrict__ bias1,
                                                  int hidden, int lane) {
  constexpr int kPerVec = 8; // 8 bf16 = 16 B
  const int nvec = hidden / kPerVec;
  // One vector per lane per trip. Unrolling this loop would double the live row loads
  // and push a 1024-thread block past its 128-VGPR budget into spills.
#pragma unroll 1
  for (int v = lane; v < nvec; v += kWarpSize) {
    const int h0 = v * kPerVec;
    uint4 y[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      if (j < nrow) y[j] = *reinterpret_cast<const uint4*>(src + (size_t)rows[j] * hidden + h0);
    }
    float acc[kPerVec];
#pragma unroll
    for (int u = 0; u < kPerVec; ++u) acc[u] = 0.f;
    if (bias0) {
      const uint4 b = *reinterpret_cast<const uint4*>(bias0 + h0);
      const bf16_t* bb = reinterpret_cast<const bf16_t*>(&b);
#pragma unroll
      for (int u = 0; u < kPerVec; ++u) acc[u] += __bfloat162float(bb[u]);
    }
    if (bias1) {
      const uint4 b = *reinterpret_cast<const uint4*>(bias1 + h0);
      const bf16_t* bb = reinterpret_cast<const bf16_t*>(&b);
#pragma unroll
      for (int u = 0; u < kPerVec; ++u) acc[u] += __bfloat162float(bb[u]);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) { // ascending k, as strict order requires
      if (j < nrow) {
        const bf16_t* yy = reinterpret_cast<const bf16_t*>(&y[j]);
#pragma unroll
        for (int u = 0; u < kPerVec; ++u) acc[u] += __bfloat162float(yy[u]);
      }
    }
    uint4 o;
    bf16_t* oo = reinterpret_cast<bf16_t*>(&o);
#pragma unroll
    for (int u = 0; u < kPerVec; ++u) oo[u] = __float2bfloat16(acc[u]);
    *reinterpret_cast<uint4*>(dst + h0) = o;
  }
  // scalar tail when hidden is not a multiple of 8
  for (int h = nvec * kPerVec + lane; h < hidden; h += kWarpSize) {
    float acc = 0.f;
    if (bias0) acc += __bfloat162float(bias0[h]);
    if (bias1) acc += __bfloat162float(bias1[h]);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      if (j < nrow) acc += __bfloat162float(src[(size_t)rows[j] * hidden + h]);
    }
    dst[h] = __float2bfloat16(acc);
  }
}

// combine_impl's reduce for a token with more than 8 rows on this rank. Out of line so
// its runtime-indexed arrays do not count against the fast path's register budget.
__device__ __noinline__ void reduce_token_rows_any(const bf16_t* __restrict__ in_y,
                                                   const int32_t* __restrict__ recv_topk,
                                                   const int32_t* __restrict__ row_map,
                                                   const int64_t* __restrict__ expert_rows,
                                                   const float* __restrict__ scales, int fma_mode,
                                                   int i, int K, int num_recv, bf16_t* dst,
                                                   int hidden, int lane) {
  // Row bases hoisted out of the element loop, packed as row << 4 | k so the
  // expert_rows sort also carries each row's top-k slot (its scale).
  int32_t rowk[kMaxTopk];
  int nrow = 0;
  for (int k = 0; k < K; ++k) {
    const int e = recv_topk[(size_t)i * K + k];
    if (e < 0) continue;
    const int r = expert_rows ? (int)expert_rows[(size_t)e * num_recv + i]
                              : row_map[(size_t)i * K + k];
    if (r < 0) continue;
    const int rk = (r << 4) | k;  // rows are distinct, so this sorts by row
    int j = nrow++;
    if (expert_rows)  // insertion sort, at most kMaxTopk rows
      for (; j > 0 && rowk[j - 1] > rk; --j) rowk[j] = rowk[j - 1];
    rowk[j] = rk;
  }
  // No bias on this side; the origin seeds its own. The store goes to a
  // PEER, where narrow stores cost more than against local HBM, which is
  // the other reason the reduce keeps them at dwordx4.
  reduce_rows_rk_bf16(dst, in_y, rowk, nrow, scales ? scales + (size_t)i * K : nullptr,
                      scales ? fma_mode : 1, hidden, lane);
}

// Expert side: push locally-owned values back to the origin, preserving slot k
// so the origin accumulates in order. Input layout is selected by `grouped` and
// `row_map`: grouped=1 is one already-reduced row per token pushed to k_last;
// grouped=0 is [nrecv, num_topk, hidden] at row i*K+k, or the expanded layout
// addressed through row_map. `expert_rows` (grouped only, instead of row_map)
// indexes a grouped-by-expert input; see the branch below.
__global__ void combine_impl(EpConfig cfg,
                             const bf16_t* __restrict__ in_y,        // [nrecv, hidden] or [rows, hidden]
                             const float* __restrict__ in_w,         // weights or nullptr
                             const int32_t* __restrict__ row_map,    // [nrecv, num_topk] or nullptr
                             const int64_t* __restrict__ expert_rows, // [experts_per_rank, nrecv] or nullptr
                             const float* __restrict__ scales,       // [nrecv, num_topk] or nullptr
                             int fma_mode,
                             const int32_t* __restrict__ recv_topk,  // [nrecv, num_topk] local ids, -1 outside
                             const int32_t* __restrict__ src_idx,    // [nrecv]
                             int num_recv, int grouped, WindowView self, WindowView* __restrict__ peer_views) {
  // Grid mirrors dispatch: block x owns one SOURCE rank, so the peer view is
  // loaded once per CTA and every write goes to a single peer. Parallelism is
  // gridDim.y * nwarps and only gridDim.y follows the budget; see kEpWaves.
  const int src_rank = blockIdx.x;
  if (src_rank >= cfg.num_ranks) return;   // block-uniform, so the barrier below is safe
  const int warp = get_warp_idx(), nwarps = get_num_warps(), lane = get_lane_idx();
  const int K = cfg.num_topk;

  // counts() lives in the peer window, so each term of this prefix sum is a
  // global load. Every thread needs the same two scalars, and the block is
  // kEpWaves wide, so computing it once and broadcasting through LDS trades
  // blockDim x src_rank loads for one barrier.
  __shared__ int s_begin, s_end;
  if (threadIdx.x == 0) {
    int b = 0;
    for (int r = 0; r < src_rank; ++r) b += self.counts()[r];
    s_begin = b;
    s_end = b + self.counts()[src_rank];
  }
  __syncthreads();
  const int begin = s_begin, end = s_end;

  const WindowView w = peer_views[src_rank];
  const int stride = nwarps * gridDim.y;

  for (int i = begin + blockIdx.y * nwarps + warp; i < end && i < num_recv; i += stride) {
    const int g = src_idx[i];
    const int src_tok = g % cfg.num_max_tokens_per_rank;
    const size_t slot = EpWindowLayout::slot(cfg, cfg.rank, src_tok);

    if (grouped && (row_map != nullptr || expert_rows != nullptr)) {
      // Grouped reduction over an EXPANDED input: form the per-rank partial
      // here, summing this rank's owned rows into one row. The origin folds that
      // row at the last top-k position this rank owns. fp32 with a single
      // rounding at the end; bf16 accumulation would not match.
      //
      // Summation order: ascending k for row_map; ascending row for expert_rows,
      // which in an expert-major layout is ascending expert id. Lane k resolves
      // slot k, and its rank among the valid lanes is its position in that order.
      int e = -1, r = -1;
      float sl = 1.f;
      if (lane < K) {
        e = recv_topk[(size_t)i * K + lane];
        if (e >= 0) r = expert_rows ? (int)expert_rows[(size_t)e * num_recv + i]
                                    : row_map[(size_t)i * K + lane];
        if (scales) sl = 0.f + scales[(size_t)i * K + lane];
      }
      const bool valid = r >= 0;  // r >= 0 implies lane < K and e >= 0
      const unsigned long long vm = __ballot(valid);
      const int nrow = __popcll(vm);
      // The origin reads this rank's row whenever it owns any slot, so write it
      // (as zeros) even when no slot resolved to a row.
      const bool owns = __ballot(e >= 0) != 0;
      if (owns && nrow <= 8) {
        int pos;
        if (expert_rows) {  // rows are distinct: position = valid lanes with a lower row
          pos = 0;
          for (int j = 0; j < K; ++j) {
            const int rj = __shfl(r, j, kWarpSize);
            pos += (((vm >> j) & 1) && rj < r) ? 1 : 0;
          }
        } else {
          pos = __popcll(vm & __lanemask_lt());
        }
        int32_t rows8[8];
        float s8[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          // src is wave-uniform (from a ballot), so readlane leaves the row and scale
          // in scalar registers, off the vector budget the reduce needs for its loads.
          const unsigned long long mj = __ballot(valid && pos == j);
          const int src = mj ? __builtin_ctzll(mj) : 0;
          rows8[j] = __builtin_amdgcn_readlane(r, src);
          s8[j] = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(sl), src));
        }
        reduce_rows8_scaled_bf16(w.y() + slot * cfg.hidden, in_y, rows8, s8, nrow,
                                 scales ? fma_mode : 1, cfg.hidden, lane);
      } else if (nrow > 8) {
        reduce_token_rows_any(in_y, recv_topk, row_map, expert_rows, scales, fma_mode, i, K,
                              num_recv, w.y() + slot * cfg.hidden, cfg.hidden, lane);
      }
      // With expert_rows, in_w is [rows, num_topk] in that layout, nonzero only at
      // each row's own slot; `0.f +` maps -0.0 to +0.0, as summing the rows would.
      if (in_w && valid)
        w.cw()[slot * K + lane] = expert_rows ? 0.f + in_w[(size_t)r * K + lane] : in_w[r];
    } else if (grouped) {
      // Grouped mode sends ONE row per token, so index by slot rather than by
      // (slot, k). The expanded layout is K times larger, and scattering single
      // rows across it is far slower than a dense write.
      // Whether this rank owns any slot of the token (lane k tests slot k).
      const bool owns = __ballot(lane < K && recv_topk[(size_t)i * K + lane] >= 0) != 0;
      if (owns) wave_copy_bf16(w.y() + slot * cfg.hidden, in_y + (size_t)i * cfg.hidden, cfg.hidden);
      // The origin needs its own top-k weights back; every rank that received
      // this token holds an identical copy of the full row, so whichever
      // arrives last wins and the result is the same either way.
      if (in_w) {
        for (int k = lane; k < K; k += kWarpSize) w.cw()[slot * K + k] = in_w[(size_t)i * K + k];
      }
    } else {
      for (int k = 0; k < K; ++k) {
        if (recv_topk[(size_t)i * K + k] < 0) continue;  // not ours
        const int row = row_map ? row_map[(size_t)i * K + k] : (int)(i * K + k);
        if (row < 0) continue;
        wave_copy_bf16(w.y() + (slot * K + k) * cfg.hidden, in_y + (size_t)row * cfg.hidden, cfg.hidden);
        if (in_w && lane == 0) w.cw()[slot * K + k] = in_w[row];
      }
    }
  }
  // No completion flag: the caller barriers between send and receive, and every
  // slot the receiver reads is one some rank is guaranteed to have written --
  // a token only reaches a rank that owns one of its experts.
}

// Origin-side reduce for a token with more than 8 contributing rows. Out of line so
// its runtime-indexed arrays do not count against the fast path's register budget.
__device__ __noinline__ void reduce_token_any_rows(EpConfig cfg, WindowView self,
                                                   const int32_t* __restrict__ topk_idx,
                                                   int t, int grouped,
                                                   const bf16_t* __restrict__ bias0,
                                                   const bf16_t* __restrict__ bias1,
                                                   bf16_t* __restrict__ out, int lane) {
  const int epr = cfg.experts_per_rank();
  const int K = cfg.num_topk;
  const size_t tbase = (size_t)t * cfg.hidden;
  // Decide once per token which slots carry a value, rather than re-deriving
  // it for every element of hidden.
  bool take[kMaxTopk];
  for (int k = 0; k < K; ++k) {
    const int e = topk_idx[(size_t)t * K + k];
    // Upper bound as well as lower: owner = e / epr reaches num_ranks for an
    // out-of-range id, and slot() would then index a whole rank-region past
    // the end of the window.
    take[k] = (e >= 0 && e < cfg.num_experts);
    if (take[k] && grouped) {
      const int owner = e / epr;
      for (int k2 = k + 1; k2 < K; ++k2) {
        const int e2 = topk_idx[(size_t)t * K + k2];
        if (e2 >= 0 && e2 / epr == owner) {
          take[k] = false;
          break;
        }
      }
    }
  }

  // Row base for each contributing slot, hoisted out of the element loop.
  // Recomputing owner and slot per element cost a topk_idx load and an
  // integer divide for every one of hidden x K accesses. Packed rather than
  // indexed by k, so the reduction carries no per-element branch; ascending k
  // is preserved, which is the part the bit-exactness contract cares about.
  int32_t row[kMaxTopk];
  int nrow = 0;
  for (int k = 0; k < K; ++k) {
    if (!take[k]) continue;
    const int owner = topk_idx[(size_t)t * K + k] / epr;
    const size_t slot = EpWindowLayout::slot(cfg, owner, t);
    // Must mirror the sender's layout: one row per slot when grouped, one
    // per (slot, k) when reading the expanded layout.
    row[nrow++] = (int32_t)(grouped ? slot : (slot * K + k));
  }

  reduce_rows_bf16(out + tbase, self.y(), row, nrow,
                   bias0 ? bias0 + tbase : nullptr,
                   bias1 ? bias1 + tbase : nullptr, cfg.hidden, lane);
}

// Origin side: accumulate across topk slots in strict order, in float32.
// bias0/bias1 are the accumulator's initial value and must seed it BEFORE the
// topk sum so they share the rounding sequence.
// `grouped` mirrors the sender: only the owner's last slot is populated.
__global__ void combine_reduce_epilogue_impl(EpConfig cfg, WindowView self,
                                             const int32_t* __restrict__ topk_idx, // [ntok, num_topk] GLOBAL expert ids
                                             int num_tokens,
                                             const bf16_t* __restrict__ bias0, // [ntok, hidden] or nullptr
                                             const bf16_t* __restrict__ bias1, // [ntok, hidden] or nullptr
                                             int grouped,
                                             bf16_t* __restrict__ out, // [ntok, hidden]
                                             float* __restrict__ out_w) { // [ntok, num_topk] or nullptr
  const int warp = get_warp_idx(), nwarps = get_num_warps(), lane = get_lane_idx();
  const int stride = nwarps * gridDim.x;
  const int epr = cfg.experts_per_rank();
  const int K = cfg.num_topk;

  // One wave per token, so gridDim.x * nwarps waves chase num_tokens, and this
  // loop is sensitive in BOTH directions: too few waves and each serialises
  // many tokens, too many and most retire without work while the block still
  // pays for them. Launched at kEpWaves, which is sized for the many-token
  // case; python/rccl_ep_capi.hip records what that costs when tokens are few.
  for (int t = blockIdx.x * nwarps + warp; t < num_tokens; t += stride) {
    const size_t tbase = (size_t)t * cfg.hidden;

    // Lane k decides slot k by the same rules as reduce_token_any_rows: the expert id
    // is in range and, when grouped, no LATER slot has the same owner (the sender
    // pushed that owner's row to its last slot). The ballot keeps ascending k.
    const int e = lane < K ? topk_idx[(size_t)t * K + lane] : -1;
    const int own = e >= 0 ? e / epr : -1;
    bool take_k = lane < K && e >= 0 && e < cfg.num_experts;
    if (grouped) {
      for (int d = 1; d < K; ++d) {
        const int later = __shfl(own, lane + d, kWarpSize);
        if (lane + d < K && later == own) take_k = false;
      }
    }
    const size_t my_slot = take_k ? EpWindowLayout::slot(cfg, own, t) : 0;
    const int32_t my_row = (int32_t)(grouped ? my_slot : (my_slot * K + lane));
    unsigned long long pending = __ballot(take_k);
    const int nrow_fast = __popcll(pending);
    if (nrow_fast <= 8) {
      int32_t rows8[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        rows8[j] = __shfl(my_row, pending ? __builtin_ctzll(pending) : 0, kWarpSize);
        pending &= pending - 1;
      }
      reduce_rows8_bf16(out + tbase, self.y(), rows8, nrow_fast,
                        bias0 ? bias0 + tbase : nullptr,
                        bias1 ? bias1 + tbase : nullptr, cfg.hidden, lane);
    } else {
      reduce_token_any_rows(cfg, self, topk_idx, t, grouped, bias0, bias1, out, lane);
    }

    if (out_w) {
      for (int k = lane; k < K; k += kWarpSize) {
        const int e = topk_idx[(size_t)t * K + k];
        // Any owner of this token has the full weight row; -1 slots were never
        // sent anywhere, so their weights are written as zero.
        const int owner = (e >= 0) ? (e / epr) : -1;
        out_w[(size_t)t * K + k] = (owner >= 0) ? self.cw()[EpWindowLayout::slot(cfg, owner, t) * K + k] : 0.f;
      }
    }
  }
}

} // namespace rccl_ep

#endif // RCCL_EP_COMBINE_H_
