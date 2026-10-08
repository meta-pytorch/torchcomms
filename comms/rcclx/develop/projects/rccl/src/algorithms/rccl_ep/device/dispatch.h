/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Dispatch: routing scan, the push of token payload into peer windows, and the
 * epilogue that compacts the staged per-source regions into the contiguous
 * output.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_EP_DISPATCH_H_
#define RCCL_EP_DISPATCH_H_

#include "device/ep_common.h"

namespace rccl_ep {

// Phase 1: per (token, dst rank), whether the token is sent and at which slot.
// Slot order must follow token order so the receiver's concatenation stays in
// token order. One CTA per destination; each wave counts the sends in its
// contiguous token range, an LDS scan gives each wave its first slot, and a
// second pass assigns slots in token order. All top-k ids are loaded before any
// is tested, so the loads overlap.
constexpr int kPlanMaxWaves = 32;

__global__ void k_ep_plan(EpConfig cfg,
                          const int32_t* __restrict__ topk_idx,  // [ntok, num_topk]
                          int num_tokens,
                          int32_t* __restrict__ slot_of,         // [num_ranks, ntok] -1 = not sent
                          int32_t* __restrict__ send_list,       // [num_ranks, ntok] slot -> token
                          int32_t* __restrict__ send_counts) {   // [num_ranks]
  const int dst = blockIdx.x;
  if (dst >= cfg.num_ranks) return;

  const int lo = cfg.expert_begin(dst), hi = cfg.expert_end(dst);
  const int K = cfg.num_topk;
  const int warp = get_warp_idx(), nwarps = get_num_warps(), lane = get_lane_idx();
  // Whole 64-token rounds per wave, so each round's scan covers full lanes.
  const int per_wave =
      ((num_tokens + nwarps - 1) / nwarps + kWarpSize - 1) / kWarpSize * kWarpSize;
  const int t_begin = min(num_tokens, warp * per_wave);
  const int t_end = min(num_tokens, t_begin + per_wave);

  auto sends = [&](int t) -> int {
    int send = 0;
    if (t < t_end) {
      int e[kMaxTopk];
#pragma unroll
      for (int k = 0; k < kMaxTopk; ++k) e[k] = k < K ? topk_idx[(size_t)t * K + k] : -1;
#pragma unroll
      for (int k = 0; k < kMaxTopk; ++k) send |= (e[k] >= lo && e[k] < hi) ? 1 : 0;
    }
    return send;
  };

  // Pass 1: this wave's send count.
  int wave_total = 0;
  for (int base = t_begin; base < t_end; base += kWarpSize) wave_total += reduce_add<int>(sends(base + lane));

  // Exclusive scan over the waves, in wave order = token order.
  __shared__ int wave_base[kPlanMaxWaves];
  if (lane == 0) wave_base[warp] = wave_total;
  __syncthreads();
  if (threadIdx.x == 0) {
    int acc = 0;
    for (int w = 0; w < nwarps; ++w) {
      const int v = wave_base[w];
      wave_base[w] = acc;
      acc += v;
    }
    send_counts[dst] = acc;
  }
  __syncthreads();

  // Pass 2: slots, in token order within the wave, from the wave's base.
  int running = wave_base[warp];
  for (int base = t_begin; base < t_end; base += kWarpSize) {
    const int t = base + lane;
    const int send = sends(t);
    const int excl = warp_exclusive_sum<int>(send);
    if (t < t_end) {
      const int slot = send ? (running + excl) : -1;
      slot_of[(size_t)dst * num_tokens + t] = slot;
      // The inverse map, so the send kernel iterates only the tokens it will
      // actually move. With topk 6 over 256 experts only ~55% of tokens go to
      // any given peer, so scanning all of them wastes ~45% of wave iterations.
      if (send) send_list[(size_t)dst * num_tokens + slot] = t;
    }
    running += reduce_add<int>(send);
  }
}

// k_ep_plan fused with the uncached dispatch's count exchange, which replaces barrier A.
// Each source publishes [send total, hist[epr], pairs_to[R], num_tokens] into dst's
// notify region under a monotonic sequence number; the grid's last CTA then waits for
// every source's publication to this rank and folds them (notify_reduce_wave). A
// source's plan starts only after its earlier work, including reads of its window, has
// finished, so seeing every publication gives barrier A's write-after-read guarantee.
//
// Grid is (num_ranks, chunks), one token per lane. Slots and ranks follow token order
// across chunks through `agg`: a CTA waits only for earlier chunks of its row, which
// were dispatched first, so the wait always ends.
//
// Payload and flag are stored by the SAME wave with a system-scope fence between them:
// vmcnt is per wave, so __syncthreads does not publish another wave's stores. Flags
// only increase and receivers never write them, so a stale cached line cannot
// overwrite a newer remote store.
//
// rank_out (optional): for each (token, k) with an expert on dst, the number of earlier
// tokens that chose that expert, i.e. its row within the (source, expert) group of
// k_ep_grouped_epilogue's layout.
constexpr int kPlanChunkWaves = 8;
constexpr int kPlanChunk = kPlanChunkWaves * kWarpSize;  // tokens per CTA
// agg row per (dst, chunk): [flag, send, hist[epr], pairs_to[R]]
__host__ __device__ constexpr int kPlanAggStride(int epr, int R) { return 2 + epr + R; }

// TopkT is the top-k id type. `topk_i32` (optional) receives the int32 copy the rest of
// the path reads, written by the dst 0 row, which loads every token.
template <typename TopkT>
__global__ void __launch_bounds__(kPlanChunk)
k_ep_plan_notify(EpConfig cfg,
                 const TopkT* __restrict__ topk_idx,    // [ntok, num_topk]
                 int32_t* __restrict__ topk_i32,        // [ntok, num_topk] or nullptr
                 int num_tokens,
                 int32_t* __restrict__ slot_of,         // [num_ranks, ntok] -1 = not sent
                 int32_t* __restrict__ send_list,       // [num_ranks, ntok] slot -> token
                 int32_t* __restrict__ send_counts,     // [num_ranks]
                 WindowView* __restrict__ peer_views, WindowView self, uint32_t seq,
                 int32_t* __restrict__ rank_out,        // [ntok, num_topk] or nullptr
                 int32_t* __restrict__ agg,             // [num_ranks, chunks, stride]
                 NotifyReduceOut nro) {
  const int dst = blockIdx.x, c = blockIdx.y, nchunks = gridDim.y;
  if (dst >= cfg.num_ranks) return;

  const int lo = cfg.expert_begin(dst), hi = cfg.expert_end(dst);
  const int K = cfg.num_topk, epr = cfg.experts_per_rank(), R = cfg.num_ranks;
  const int warp = get_warp_idx(), lane = get_lane_idx();
  const int t = c * kPlanChunk + warp * kWarpSize + lane;
  const bool live = t < num_tokens;
  const int S = kPlanAggStride(epr, R);
  int32_t* my_agg = agg + ((size_t)dst * nchunks + c) * S;

  __shared__ int hist[kMaxLocalExperts];      // this chunk's pairs per local expert
  __shared__ int base_le[kMaxLocalExperts];   // earlier chunks' pairs per local expert
  __shared__ int wave_base[kPlanChunkWaves];
  __shared__ int wave_cnt[kPlanChunkWaves * kMaxLocalExperts];  // rank_out only
  __shared__ int pairs_all[kMaxRanks];
  __shared__ int s_chunk_send, s_base_send;
  for (int j = threadIdx.x; j < epr; j += blockDim.x) hist[j] = 0;
  for (int j = threadIdx.x; j < kMaxRanks; j += blockDim.x) pairs_all[j] = 0;
  if (rank_out)
    for (int j = threadIdx.x; j < kPlanChunkWaves * epr; j += blockDim.x) wave_cnt[j] = 0;
  __syncthreads();

  // The token's top-k ids, loaded once and kept for both passes.
  int e[kMaxTopk];
#pragma unroll
  for (int k = 0; k < kMaxTopk; ++k) e[k] = (k < K && live) ? (int)topk_idx[(size_t)t * K + k] : -1;
  if (topk_i32 != nullptr && dst == 0 && live) {
#pragma unroll
    for (int k = 0; k < kMaxTopk; ++k)
      if (k < K) topk_i32[(size_t)t * K + k] = e[k];
  }

  // Pass 1: send flag, pairs per destination rank, the per-expert histogram.
  int send = 0;
  int pc[kMaxRanks] = {};
#pragma unroll
  for (int k = 0; k < kMaxTopk; ++k) {
    if (e[k] >= 0) {
      const int r = e[k] / epr;
#pragma unroll
      for (int rr = 0; rr < kMaxRanks; ++rr) pc[rr] += (r == rr) ? 1 : 0;
    }
    const bool mine = e[k] >= lo && e[k] < hi;
    send |= mine ? 1 : 0;
    if (mine) {
      atomicAdd(&hist[e[k] - lo], 1);
      if (rank_out) atomicAdd(&wave_cnt[warp * epr + e[k] - lo], 1);
    }
  }
  const int wave_total = reduce_add<int>(send);
  if (lane == 0) wave_base[warp] = wave_total;
#pragma unroll
  for (int rr = 0; rr < kMaxRanks; ++rr) {
    if (rr >= R) break;  // uniform
    const int v = reduce_add<int>(pc[rr]);
    if (lane == 0 && v) atomicAdd(&pairs_all[rr], v);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int acc = 0;
    for (int w = 0; w < kPlanChunkWaves; ++w) {
      const int v = wave_base[w];
      wave_base[w] = acc;
      acc += v;
    }
    s_chunk_send = acc;
  }
  if (rank_out) {  // per-expert counts of the earlier waves: each wave's rank base
    for (int j = threadIdx.x; j < epr; j += blockDim.x) {
      int acc = 0;
      for (int w = 0; w < kPlanChunkWaves; ++w) {
        const int v = wave_cnt[w * epr + j];
        wave_cnt[w * epr + j] = acc;
        acc += v;
      }
    }
  }
  __syncthreads();

  // Publish this chunk's totals, then take the earlier chunks' (agent scope: same GPU).
  if (threadIdx.x == 0) my_agg[1] = s_chunk_send;
  for (int j = threadIdx.x; j < epr; j += blockDim.x) my_agg[2 + j] = hist[j];
  for (int r = threadIdx.x; r < R; r += blockDim.x) my_agg[2 + epr + r] = pairs_all[r];
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    __hip_atomic_store(&my_agg[0], (int32_t)seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    for (int j = 0; j < c; ++j) {
      const int32_t* f = agg + ((size_t)dst * nchunks + j) * S;
      while (__hip_atomic_load(f, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) != (int32_t)seq)
        __builtin_amdgcn_s_sleep(1);
    }
  }
  __syncthreads();
  // Earlier chunks' totals. Agent-scope loads, so they see the flags' writers.
  auto aload = [](const int32_t* p) {
    return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  };
  const int32_t* row = agg + (size_t)dst * nchunks * S;
  if (threadIdx.x == 0) {
    int b = 0;
    for (int j = 0; j < c; ++j) b += aload(row + (size_t)j * S + 1);
    s_base_send = b;
  }
  for (int le = threadIdx.x; le < epr; le += blockDim.x) {
    int b = 0;
    for (int j = 0; j < c; ++j) b += aload(row + (size_t)j * S + 2 + le);
    base_le[le] = b;
  }
  __syncthreads();

  // Pass 2: slots, in token order, from the chunk's and the wave's base.
  const int excl = warp_exclusive_sum<int>(send);
  if (live) {
    const int slot = send ? (s_base_send + wave_base[warp] + excl) : -1;
    slot_of[(size_t)dst * num_tokens + t] = slot;
    // The inverse map; see k_ep_plan.
    if (send) send_list[(size_t)dst * num_tokens + slot] = t;
  }
  if (rank_out) {  // block-uniform; every lane joins the ballots below
    for (int le = 0; le < epr; ++le) {
      int kk = -1;  // a token's experts are distinct, so at most one k matches
#pragma unroll
      for (int k = 0; k < kMaxTopk; ++k) kk = (e[k] == lo + le) ? k : kk;
      const unsigned long long m = __ballot(kk >= 0);
      if (m == 0) continue;  // wave-uniform
      if (kk >= 0)
        rank_out[(size_t)t * K + kk] =
            base_le[le] + wave_cnt[warp * epr + le] + __popcll(m & __lanemask_lt());
    }
  }

  // The row's last chunk holds every total: publish to dst.
  if (c != nchunks - 1 || warp != 0) return;
  const int total_send = s_base_send + s_chunk_send;
  if (lane == 0) send_counts[dst] = total_send;
  const WindowView w = peer_views[dst];
  int32_t* pay = w.notify() + (size_t)cfg.rank * (2 + epr + R);
  if (lane == 0) pay[0] = total_send;
  for (int le = lane; le < epr; le += kWarpSize) pay[1 + le] = base_le[le] + hist[le];
  for (int r = lane; r < R; r += kWarpSize) {
    int p = pairs_all[r];
    for (int j = 0; j < c; ++j) p += aload(row + (size_t)j * S + 2 + epr + r);
    pay[1 + epr + r] = p;
  }
  if (lane == 0) pay[1 + epr + R] = num_tokens;
  __threadfence_system();
  __builtin_amdgcn_wave_barrier();
  if (lane == 0)
    __hip_atomic_store(&w.flags()[cfg.rank], seq, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  // CTA (R-1, nchunks-1) is dispatched last, so every other CTA of this grid is already
  // running or done when it waits here; lane r waits for source r.
  if (dst != R - 1) return;
  if (lane < R)
    while ((int32_t)(ld_acquire_sys<uint32_t>(&self.flags()[lane]) - seq) < 0) __builtin_amdgcn_s_sleep(1);
  __builtin_amdgcn_wave_barrier();
  notify_reduce_wave(cfg, self, nro);
}

// Phase 2: push payload and metadata into each destination's window. Grid is
// (num_ranks, ctas_per_rank), one wave per token. Only gridDim.y follows the
// caller's budget, so the block must be wide; see kEpWaves. with_meta: 0 pushes
// the payload only (cached replay), 1 adds the metadata, 2 also encodes each owned
// top-k slot as rank << 8 | local expert for k_ep_grouped_epilogue.
__global__ void dispatch_impl(EpConfig cfg,
                              const bf16_t* __restrict__ x,          // [ntok, hidden]
                              const int32_t* __restrict__ topk_idx,  // [ntok, num_topk]
                              const float* __restrict__ topk_w,      // [ntok, num_topk]
                              int num_tokens, const int32_t* __restrict__ send_list,
                              const int32_t* __restrict__ send_counts,
                              WindowView* __restrict__ peer_views,   // [num_ranks]
                              int with_meta,
                              const int32_t* __restrict__ rank) {    // [ntok, num_topk], mode 2
  const int dst = blockIdx.x;
  if (dst >= cfg.num_ranks) return;

  const WindowView w = peer_views[dst];
  const int lo = cfg.expert_begin(dst), hi = cfg.expert_end(dst);
  const int warp = get_warp_idx();
  const int nwarps = get_num_warps();
  const int lane = get_lane_idx();

  // One wave per token, over the DENSE list of tokens bound for this peer;
  // see the note in k_ep_plan for why the list is prebuilt.
  const int n = send_counts[dst];
  const int stride = nwarps * gridDim.y;
  const int start = blockIdx.y * nwarps + warp;

  // Prefetch the next token's index and this token's metadata (lane k holds slot k;
  // num_topk <= kMaxTopk < kWarpSize) so neither load sits between two copies.
  const int K = cfg.num_topk;
  int t = start < n ? send_list[(size_t)dst * num_tokens + start] : 0;
  for (int i = start; i < n; i += stride) {
    const int t_next = i + stride < n ? send_list[(size_t)dst * num_tokens + i + stride] : 0;
    const size_t slot = EpWindowLayout::slot(cfg, cfg.rank, i);
    int e = -1, rk = 0;
    float tw = 0.f;
    if (with_meta && lane < K) {
      e = topk_idx[(size_t)t * K + lane];
      tw = topk_w[(size_t)t * K + lane];
      if (with_meta == 2) rk = rank[(size_t)t * K + lane];
    }

    // payload
    wave_copy_bf16(w.x() + slot * cfg.hidden, x + (size_t)t * cfg.hidden, cfg.hidden);

    // metadata: mask topk_idx to the destination's expert range before sending
    if (with_meta && lane < K) {
      const bool mine = e >= lo && e < hi;
      w.topk_idx()[slot * K + lane] = !mine ? -1 : (with_meta == 2 ? (rk << 8) | (e - lo) : e);
      w.topk_w()[slot * K + lane] = tw;
    }
    if (with_meta && lane == 0) w.src_idx()[slot] = cfg.rank * cfg.num_max_tokens_per_rank + t;
    t = t_next;
  }

  // one wave per destination publishes the count, release-ordered so the
  // payload above is visible before the count that advertises it
  __syncthreads();
  if (warp == 0 && lane == 0) {
    st_release_sys<int32_t>(&w.counts()[cfg.rank], send_counts[dst]);
  }
}

__global__ void dispatch_impl(EpConfig cfg, const fp8_t* __restrict__ x,
                              const float* __restrict__ sf,          // [ntok, hidden_sf]
                              const int32_t* __restrict__ topk_idx, const float* __restrict__ topk_w, int num_tokens,
                              const int32_t* __restrict__ send_list, const int32_t* __restrict__ send_counts,
                              WindowView* __restrict__ peer_views) {
  const int dst = blockIdx.x;
  if (dst >= cfg.num_ranks) return;
  const WindowView w = peer_views[dst];
  const int lo = cfg.expert_begin(dst), hi = cfg.expert_end(dst);
  const int warp = get_warp_idx(), nwarps = get_num_warps(), lane = get_lane_idx();
  const int nsf = cfg.hidden_sf();
  const int n = send_counts[dst];

  // As in the bf16 kernel: the next index and this token's metadata load ahead of the copy.
  const int K = cfg.num_topk, stride = nwarps * gridDim.y, start = blockIdx.y * nwarps + warp;
  int t = start < n ? send_list[(size_t)dst * num_tokens + start] : 0;
  for (int i = start; i < n; i += stride) {
    const int t_next = i + stride < n ? send_list[(size_t)dst * num_tokens + i + stride] : 0;
    const size_t slot = EpWindowLayout::slot(cfg, cfg.rank, i);
    int e = -1;
    float tw = 0.f;
    if (lane < K) {
      e = topk_idx[(size_t)t * K + lane];
      tw = topk_w[(size_t)t * K + lane];
    }

    wave_copy_bytes((uint8_t*)(w.x_fp8() + slot * cfg.hidden), (const uint8_t*)(x + (size_t)t * cfg.hidden),
                    cfg.hidden);
    for (int j = lane; j < nsf; j += kWarpSize) w.sf()[slot * nsf + j] = sf[(size_t)t * nsf + j];
    if (lane < K) {
      w.topk_idx()[slot * K + lane] = (e >= lo && e < hi) ? e : -1;
      w.topk_w()[slot * K + lane] = tw;
    }
    if (lane == 0) w.src_idx()[slot] = cfg.rank * cfg.num_max_tokens_per_rank + t;
    t = t_next;
  }
  __syncthreads();
  if (warp == 0 && lane == 0) {
    st_release_sys<int32_t>(&w.counts()[cfg.rank], send_counts[dst]);
  }
}

// Phase 3: compact the staged regions into the contiguous output. Source-rank
// order, token order preserved within a region -- the required output layout.
// A pure local copy, the stage most sensitive to block width; see kEpWaves.
//
// Per-local-expert (token, slot) counts, accumulated by the copy epilogue in an LDS
// histogram per CTA and flushed with one global atomic per non-empty expert (integer
// adds, so order-independent). The caller zeroes `counts` first; null skips counting.
constexpr int kEpiMaxExperts = 256;

struct EpilogueCounts {
  int32_t* counts;  // [experts_per_rank] or nullptr
  int epr;
  bool lds;
  __device__ void begin(int32_t* c, int n, int* hist) {
    counts = c;
    epr = n;
    lds = c != nullptr && n <= kEpiMaxExperts;
    if (lds) for (int j = threadIdx.x; j < epr; j += blockDim.x) hist[j] = 0;
    __syncthreads();
  }
  __device__ void add(int local_expert, int* hist) const {
    if (counts == nullptr || local_expert < 0 || local_expert >= epr) return;
    if (lds) atomicAdd(&hist[local_expert], 1);
    else atomicAdd(&counts[local_expert], 1);
  }
  __device__ void end(const int* hist) const {
    __syncthreads();
    if (lds) {
      for (int j = threadIdx.x; j < epr; j += blockDim.x)
        if (hist[j]) atomicAdd(&counts[j], hist[j]);
    }
  }
};

__global__ void dispatch_copy_epilogue_impl(EpConfig cfg, WindowView self,
                                            const int32_t* __restrict__ recv_offsets,  // [num_ranks] prefix
                                            bf16_t* __restrict__ out_x, int32_t* __restrict__ out_topk_idx,
                                            float* __restrict__ out_topk_w, int32_t* __restrict__ out_src_idx,
                                            int32_t* __restrict__ expert_counts,       // [epr] or nullptr
                                            int offsets_inclusive) {  // 1: a handle's inclusive prefix
  const int src = blockIdx.x;
  if (src >= cfg.num_ranks) return;  // block-uniform, so the barriers in `cnt` are safe

  const int n = self.counts()[src];
  const int dst_base = offsets_inclusive ? (src ? recv_offsets[src - 1] : 0) : recv_offsets[src];
  const int warp = get_warp_idx(), nwarps = get_num_warps(), lane = get_lane_idx();
  const int my_lo = cfg.expert_begin(cfg.rank);
  __shared__ int hist[kEpiMaxExperts];
  EpilogueCounts cnt;
  cnt.begin(expert_counts, cfg.experts_per_rank(), hist);

  for (int i = blockIdx.y * nwarps + warp; i < n; i += nwarps * gridDim.y) {
    const size_t s = EpWindowLayout::slot(cfg, src, i);
    const size_t d = dst_base + i;

    wave_copy_local_bf16<true, 1>(out_x + d * cfg.hidden, self.x() + s * cfg.hidden, cfg.hidden);
    if (out_topk_idx == nullptr) continue;  // payload-only: grid-uniform

    for (int k = lane; k < cfg.num_topk; k += kWarpSize) {
      const int e = self.topk_idx()[s * cfg.num_topk + k];
      // rebase into local expert space; anything outside becomes -1
      const int le = (e >= 0) ? (e - my_lo) : -1;
      out_topk_idx[d * cfg.num_topk + k] = le;
      out_topk_w[d * cfg.num_topk + k] = self.topk_w()[s * cfg.num_topk + k];
      cnt.add(le, hist);
    }
    if (lane == 0) out_src_idx[d] = self.src_idx()[s];
  }
  cnt.end(hist);
}

__global__ void dispatch_copy_epilogue_impl(EpConfig cfg, WindowView self, const int32_t* __restrict__ recv_offsets,
                                            fp8_t* __restrict__ out_x, float* __restrict__ out_sf,
                                            int32_t* __restrict__ out_topk_idx, float* __restrict__ out_topk_w,
                                            int32_t* __restrict__ out_src_idx,
                                            int32_t* __restrict__ expert_counts) {     // [epr] or nullptr
  const int src = blockIdx.x;
  if (src >= cfg.num_ranks) return;
  const int n = self.counts()[src];
  const int dst_base = recv_offsets[src];
  const int warp = get_warp_idx(), nwarps = get_num_warps(), lane = get_lane_idx();
  const int my_lo = cfg.expert_begin(cfg.rank), nsf = cfg.hidden_sf();
  __shared__ int hist[kEpiMaxExperts];
  EpilogueCounts cnt;
  cnt.begin(expert_counts, cfg.experts_per_rank(), hist);

  for (int i = blockIdx.y * nwarps + warp; i < n; i += nwarps * gridDim.y) {
    const size_t s = EpWindowLayout::slot(cfg, src, i);
    const size_t d = dst_base + i;
    wave_copy_local_bytes<true, 1>((uint8_t*)(out_x + d * cfg.hidden), (const uint8_t*)(self.x_fp8() + s * cfg.hidden), cfg.hidden);
    for (int j = lane; j < nsf; j += kWarpSize) out_sf[d * nsf + j] = self.sf()[s * nsf + j];
    for (int k = lane; k < cfg.num_topk; k += kWarpSize) {
      const int e = self.topk_idx()[s * cfg.num_topk + k];
      const int le = (e >= 0) ? (e - my_lo) : -1;
      out_topk_idx[d * cfg.num_topk + k] = le;
      out_topk_w[d * cfg.num_topk + k] = self.topk_w()[s * cfg.num_topk + k];
      cnt.add(le, hist);
    }
    if (lane == 0) out_src_idx[d] = self.src_idx()[s];
  }
  cnt.end(hist);
}

// Phase 3, grouped: compact the staged regions straight into a grouped-by-expert
// layout (expert-major, then receive order). The row of (src, token, local expert le)
// is ebase[le] + srcpref[src][le] + the token's rank among src's tokens that chose le,
// which the sender encoded in the top-k word (dispatch_impl with_meta=2). Pure copies,
// so the output matches permuting the plain dispatch's output. bf16 only.
__global__ void k_ep_grouped_epilogue(EpConfig cfg, WindowView self,
                                      const int32_t* __restrict__ psum_rank,  // [R] inclusive
                                      const int32_t* __restrict__ srcpref,    // [R, epr]
                                      const int32_t* __restrict__ ebase,      // [epr]
                                      int num_recv,
                                      bf16_t* __restrict__ out_rows,          // [rows, hidden]
                                      int64_t* __restrict__ out_row_ids,      // [rows, K]
                                      float* __restrict__ out_row_w,          // [rows, K]
                                      int64_t* __restrict__ out_map,          // [epr, num_recv]
                                      int32_t* __restrict__ out_ids32,        // [num_recv, K]
                                      int64_t* __restrict__ out_ids64,        // [num_recv, K]
                                      float* __restrict__ out_w,              // [num_recv, K]
                                      int32_t* __restrict__ out_src) {        // [num_recv]
  const int src = blockIdx.x;
  if (src >= cfg.num_ranks) return;
  const int K = cfg.num_topk, epr = cfg.experts_per_rank();
  const int start = src ? psum_rank[src - 1] : 0;
  const int n = psum_rank[src] - start;
  const int warp = get_warp_idx(), nwarps = get_num_warps(), lane = get_lane_idx();
  const int nvec = cfg.hidden / 8;  // 16 B vectors of bf16; ep_create checks hidden % 8

  for (int i = blockIdx.y * nwarps + warp; i < n; i += nwarps * gridDim.y) {
    const size_t s = EpWindowLayout::slot(cfg, src, i);
    const size_t d = (size_t)start + i;
    int le = -1, row = -1;
    float w = 0.f;
    if (lane < K) {
      const int enc = self.topk_idx()[s * K + lane];
      w = self.topk_w()[s * K + lane];
      if (enc >= 0) {
        le = enc & 0xFF;
        row = ebase[le] + srcpref[src * epr + le] + (enc >> 8);
      }
      out_ids32[d * K + lane] = le;
      out_ids64[d * K + lane] = le;
      out_w[d * K + lane] = w;
    }
    if (lane == 0) out_src[d] = self.src_idx()[s];

    // Compact the token's valid (row, local expert) pairs into scalar registers, so the
    // loops below run over nrow rows rather than every top-k slot. Rows are distinct
    // and each is written once, so their order does not matter.
    const unsigned long long vm = __ballot(row >= 0);
    const int nrow = __popcll(vm);
    int rows[kMaxTopk], les[kMaxTopk];
    {
      unsigned long long m = vm;
#pragma unroll
      for (int j = 0; j < kMaxTopk; ++j) {
        const int ln = m ? __builtin_ctzll(m) : 0;
        rows[j] = __builtin_amdgcn_readlane(row, ln);
        les[j] = __builtin_amdgcn_readlane(le, ln);
        m &= m - 1;
      }
    }
    for (int e = lane; e < epr; e += kWarpSize) {
      int r = -1;
#pragma unroll
      for (int j = 0; j < kMaxTopk; ++j) r = (j < nrow && les[j] == e) ? rows[j] : r;
      out_map[(size_t)e * num_recv + d] = r;
    }
    if (lane < K) {
#pragma unroll
      for (int j = 0; j < kMaxTopk; ++j) {
        if (j >= nrow) break;  // uniform
        out_row_ids[(size_t)rows[j] * K + lane] = le;
        out_row_w[(size_t)rows[j] * K + lane] = w;
      }
    }
    // Nontemporal both ways, four loads in flight per lane (see wave_copy_local_bytes).
    const ep_v4i* sv = reinterpret_cast<const ep_v4i*>(self.x() + s * cfg.hidden);
    int v = lane;
    for (; v + 3 * kWarpSize < nvec; v += 4 * kWarpSize) {
      ep_v4i a[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) a[u] = __builtin_nontemporal_load(sv + v + u * kWarpSize);
#pragma unroll
      for (int j = 0; j < kMaxTopk; ++j) {
        if (j >= nrow) break;  // uniform
        ep_v4i* o = reinterpret_cast<ep_v4i*>(out_rows + (size_t)rows[j] * cfg.hidden);
#pragma unroll
        for (int u = 0; u < 4; ++u) __builtin_nontemporal_store(a[u], o + v + u * kWarpSize);
      }
    }
    for (; v < nvec; v += kWarpSize) {
      const ep_v4i val = __builtin_nontemporal_load(sv + v);
#pragma unroll
      for (int j = 0; j < kMaxTopk; ++j) {
        if (j >= nrow) break;  // uniform
        __builtin_nontemporal_store(val, reinterpret_cast<ep_v4i*>(out_rows + (size_t)rows[j] * cfg.hidden) + v);
      }
    }
  }
}

}  // namespace rccl_ep

#endif  // RCCL_EP_DISPATCH_H_
