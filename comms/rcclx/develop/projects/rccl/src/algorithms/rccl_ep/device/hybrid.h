/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Hybrid internode dispatch and combine, in two levels. A token crosses
 * nodes once per destination node, to the rail peer (same local index) there, which
 * forwards it over xGMI to each local rank owning one of its experts. Combine reverses
 * this: the forwarder folds one node partial per token and sends it back along the rail.
 *
 * The sender assigns each (token, rank) slot and carries it in the record,
 * so receiver windows match the flat path and its dispatch epilogues run unchanged.
 *
 * Remote-node buffers use the compact index np = n < D ? n : n - 1, D being this node.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_EP_HYBRID_H_
#define RCCL_EP_HYBRID_H_

#include "device/combine.h"
#include "device/dispatch.h"
#include "device/ep_common.h"

namespace rccl_ep {

// The hybrid shape, this rank's place in it, and one call's per-node counts. By value.
struct HybInfo {
  int SO, SU, D, u;           // nodes, ranks per node, this node, this rank's local index
  int nsend[kMaxNodes];       // records this rank sends node n (its tokens with an expert there)
  int nrecv[kMaxNodes];       // records its rail peer on node n sends this node
  int rec_bytes;              // record stride
  int off_sf, off_topk, off_w, off_src, off_slots;  // sections of a record
  int chunk, nchunks;         // this launch covers chunk `chunk` of every node's records
  int tok_split;              // combine: chunks are owner-token ranges (hyb_comb_range), not record ranges
  int taper;                  // dispatch records: the last chunk is half the others (hyb_rec_chunk)
};

__device__ __forceinline__ int hyb_np(const HybInfo& hi, int n) { return n < hi.D ? n : n - 1; }

// Chunk c of C over cnt records: [p0, p1). Both ends of the rail must split identically, as
// one side's nsend is the other's nrecv.
__host__ __device__ __forceinline__ void hyb_chunk(int cnt, int c, int C, int& p0, int& p1) {
  p0 = (int)((long long)cnt * c / C);
  p1 = (int)((long long)cnt * (c + 1) / C);
}

// A dispatch's chunk c of C over cnt records: hyb_chunk, or under hi.taper (RCCL_EP_GIN_STREAM)
// 2C - 1 equal parts with the last chunk taking one and the others two, so the forward of the
// last chunk to land, which nothing overlaps, is half as long. Both ends of the rail use it.
__host__ __device__ __forceinline__ void hyb_rec_chunk(const HybInfo& hi, int cnt, int c, int& p0, int& p1) {
  if (!hi.taper) {
    hyb_chunk(cnt, c, hi.nchunks, p0, p1);
    return;
  }
  const long long U = 2LL * hi.nchunks - 1;
  p0 = (int)(cnt * (2LL * c) / U);
  p1 = (int)(cnt * (c + 1 < hi.nchunks ? 2LL * (c + 1) : U) / U);
}

// Owner tokens [t0, t1) of chunk c of C, over num_max_tokens_per_rank T: 2C - 1 equal parts, the
// first chunk taking one and the others two, so the first range's push and fold, before which
// nothing is on the rail, are half as long.
__host__ __device__ __forceinline__ void hyb_tok_chunk(int T, int c, int C, int& t0, int& t1) {
  const long long U = 2LL * C - 1;
  t0 = (int)(T * (c == 0 ? 0 : 2LL * c - 1) / U);
  t1 = (int)(T * (2LL * c + 1) / U);
}

// The records of chunk hi.chunk that a combine folds and returns to node n: under hi.tok_split,
// those whose owner token (fwd_meta[np][p][0], ascending in p, as the records follow the owner's
// token order) lies in the chunk's token range, found by binary search; else hyb_chunk's record
// range. Token ranges let each local rank push a chunk's partials with no record numbering
// (combine_impl's tok_lo / tok_hi), so a chunk can fold and leave before the next is pushed.
__device__ __forceinline__ void hyb_comb_range(const HybInfo& hi, const int32_t* __restrict__ fwd_meta, int n,
                                               int T, int K, int& p0, int& p1) {
  if (!hi.tok_split) {
    hyb_chunk(hi.nrecv[n], hi.chunk, hi.nchunks, p0, p1);
    return;
  }
  int t0, t1;
  hyb_tok_chunk(T, hi.chunk, hi.nchunks, t0, t1);
  const int32_t* fm = fwd_meta + (size_t)hyb_np(hi, n) * T * (1 + K);
  int lo = 0, hi_ = hi.nrecv[n];
  while (lo < hi_) {  // first record with owner token >= t0
    const int mid = (lo + hi_) >> 1;
    if (fm[(size_t)mid * (1 + K)] < t0) lo = mid + 1; else hi_ = mid;
  }
  p0 = lo;
  hi_ = hi.nrecv[n];
  while (lo < hi_) {  // first record with owner token >= t1
    const int mid = (lo + hi_) >> 1;
    if (fm[(size_t)mid * (1 + K)] < t1) lo = mid + 1; else hi_ = mid;
  }
  p1 = lo;
}

// nrecv[n] for this rank out of the gathered count matrix (ep_hcounts: row q = [sendc (R) | node
// counts (SO) | ...], W ints): what its rail peer on node n sends this node.
__global__ void k_hyb_nrecv(const int32_t* __restrict__ xchg, int W, int R, int SU, int SO, int u, int D,
                            int32_t* __restrict__ nrecv) {
  const int n = threadIdx.x;
  if (n < SO) nrecv[n] = n == D ? 0 : xchg[(size_t)(n * SU + u) * W + R + D];
}

// sendc with every remote destination zeroed: the local push's counts.
__global__ void k_hyb_local_sendc(int num_ranks, int node_base, int node_size,
                                  const int32_t* __restrict__ sendc, int32_t* __restrict__ out) {
  const int r = threadIdx.x;
  if (r < num_ranks) out[r] = (r >= node_base && r < node_base + node_size) ? sendc[r] : 0;
}

// hist[e] += this rank's tokens with expert e in their top-k: what the receivers' copy
// epilogues will count, known before the payload moves. hist must be zeroed first.
__global__ void k_expert_hist(const int32_t* __restrict__ topk_idx, int n, int num_experts,
                              int32_t* __restrict__ hist) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    const int e = topk_idx[i];
    if (e >= 0 && e < num_experts) atomicAdd(hist + e, 1);
  }
}

// ep_hplan under RCCL_EP_HPLAN_FUSED=1: the rank plan (CTAs [0, R), rc) and the node plan (CTAs
// [R, R + SO), nc) in one launch, with this rank's row of the count exchange written as it goes:
// row = [sendc (R) | node counts (SO) | per-expert counts (E)]. Each rank CTA counts its
// destination's experts in LDS, and the destinations' expert ranges tile all E, so the row is
// what the two copies, the memset and k_expert_hist of ep_hcounts would leave.
__global__ void k_ep_hplan(EpConfig rc, EpConfig nc, const int32_t* __restrict__ topk_idx,
                           int num_tokens, int32_t* __restrict__ slot_of,
                           int32_t* __restrict__ send_list, int32_t* __restrict__ send_counts,
                           int32_t* __restrict__ node_pos, int32_t* __restrict__ node_list,
                           int32_t* __restrict__ node_cnt, int32_t* __restrict__ row) {
  const int b = blockIdx.x;
  if (b >= rc.num_ranks) {
    const int n = b - rc.num_ranks;
    if (n < nc.num_ranks)
      ep_plan_cta<true>(nc, n, topk_idx, num_tokens, node_pos, node_list, node_cnt,
                        row + rc.num_ranks, nullptr);
    return;
  }
  __shared__ int hist[kEpiMaxExperts];
  const int lo = rc.expert_begin(b), epr = rc.expert_end(b) - lo;
  for (int i = threadIdx.x; i < epr; i += blockDim.x) hist[i] = 0;
  __syncthreads();
  ep_plan_cta<true>(rc, b, topk_idx, num_tokens, slot_of, send_list, send_counts, row, hist);
  // ep_plan_cta's last __syncthreads follows every pass-1 count.
  int32_t* row_hist = row + rc.num_ranks + nc.num_ranks + lo;
  for (int i = threadIdx.x; i < epr; i += blockDim.x) row_hist[i] = hist[i];
}

// One record per (token, remote node holding one of its experts), in token order, into
// so_send[np]: [payload | sf (fp8) | topk ids | topk weights | src_token_global_idx | slot
// per local rank of that node, -1 if none]. with_meta 0 leaves the top-k sections unwritten.
// Covers chunks hi.chunk .. hi.chunk + span - 1.
template <bool kFp8>
__global__ void k_hyb_pack(EpConfig cfg, HybInfo hi, const void* __restrict__ x,
                           const float* __restrict__ sf, const int32_t* __restrict__ topk_idx,
                           const float* __restrict__ topk_w, int num_tokens,
                           const int32_t* __restrict__ node_list,  // [SO, num_tokens]
                           const int32_t* __restrict__ slot_of,    // [num_ranks, num_tokens]
                           uint8_t* __restrict__ so_send, int with_meta, int span) {
  const int n = blockIdx.x;
  if (n >= hi.SO || n == hi.D) return;
  const int np = hyb_np(hi, n);
  const int warp = get_warp_idx(), nwarps = get_num_warps(), lane = get_lane_idx();
  const int K = cfg.num_topk, T = cfg.num_max_tokens_per_rank, nsf = cfg.hidden_sf();
  const int row_bytes = kFp8 ? cfg.hidden : cfg.hidden * (int)sizeof(bf16_t);
  int p0, p1, d;
  hyb_rec_chunk(hi, hi.nsend[n], hi.chunk, p0, d);
  hyb_rec_chunk(hi, hi.nsend[n], hi.chunk + span - 1, d, p1);
  for (int p = p0 + blockIdx.y * nwarps + warp; p < p1; p += nwarps * gridDim.y) {
    const int t = node_list[(size_t)n * num_tokens + p];
    uint8_t* rec = so_send + ((size_t)np * T + p) * hi.rec_bytes;
    wave_copy_local_bytes<false, 1>(rec, (const uint8_t*)x + (size_t)t * row_bytes, row_bytes);
    if (kFp8)
      for (int j = lane; j < nsf; j += kWarpSize)
        reinterpret_cast<float*>(rec + hi.off_sf)[j] = sf[(size_t)t * nsf + j];
    if (with_meta && lane < K) {
      reinterpret_cast<int32_t*>(rec + hi.off_topk)[lane] = topk_idx[(size_t)t * K + lane];
      reinterpret_cast<float*>(rec + hi.off_w)[lane] = topk_w[(size_t)t * K + lane];
    }
    if (lane == 0) *reinterpret_cast<int32_t*>(rec + hi.off_src) = cfg.rank * T + t;
    if (lane < hi.SU)
      reinterpret_cast<int32_t*>(rec + hi.off_slots)[lane] =
          slot_of[(size_t)(n * hi.SU + lane) * num_tokens + t];
  }
}

// k_hyb_pack (bf16, with metadata) with each node's record count read from the node plan's
// node_cnt on the device, so it can run before the count exchange (RCCL_EP_HYB_PREFIX,
// RCCL_EP_GIN_STREAM). A copy rather than a shared body, so k_hyb_pack's code stays as it is.
// Covers chunks hi.chunk .. hi.chunk + span - 1.
__global__ void k_hyb_pack_dev(EpConfig cfg, HybInfo hi, const void* __restrict__ x,
                               const int32_t* __restrict__ topk_idx, const float* __restrict__ topk_w,
                               int num_tokens, const int32_t* __restrict__ node_list,
                               const int32_t* __restrict__ slot_of, const int32_t* __restrict__ node_cnt,
                               uint8_t* __restrict__ so_send, int span) {
  const int n = blockIdx.x;
  if (n >= hi.SO || n == hi.D) return;
  const int np = hyb_np(hi, n);
  const int warp = get_warp_idx(), nwarps = get_num_warps(), lane = get_lane_idx();
  const int K = cfg.num_topk, T = cfg.num_max_tokens_per_rank;
  const int row_bytes = cfg.hidden * (int)sizeof(bf16_t);
  int p0, p1, d;
  hyb_rec_chunk(hi, node_cnt[n], hi.chunk, p0, d);
  hyb_rec_chunk(hi, node_cnt[n], hi.chunk + span - 1, d, p1);
  for (int p = p0 + blockIdx.y * nwarps + warp; p < p1; p += nwarps * gridDim.y) {
    const int t = node_list[(size_t)n * num_tokens + p];
    uint8_t* rec = so_send + ((size_t)np * T + p) * hi.rec_bytes;
    wave_copy_local_bytes<false, 1>(rec, (const uint8_t*)x + (size_t)t * row_bytes, row_bytes);
    if (lane < K) {
      reinterpret_cast<int32_t*>(rec + hi.off_topk)[lane] = topk_idx[(size_t)t * K + lane];
      reinterpret_cast<float*>(rec + hi.off_w)[lane] = topk_w[(size_t)t * K + lane];
    }
    if (lane == 0) *reinterpret_cast<int32_t*>(rec + hi.off_src) = cfg.rank * T + t;
    if (lane < hi.SU)
      reinterpret_cast<int32_t*>(rec + hi.off_slots)[lane] =
          slot_of[(size_t)(n * hi.SU + lane) * num_tokens + t];
  }
}

// Forwarder: writes each record from the rail peer on node n into every local rank it names,
// at the sender's slot in source (n, u)'s region -- exactly what a direct push would write.
// with_meta 0 moves the payload only. fwd_meta (optional, [SO-1, T, 1+K]) keeps each record's
// owner token and top-k ids for the combine fold.
template <bool kFp8>
__global__ void k_hyb_forward(EpConfig cfg, HybInfo hi, const uint8_t* __restrict__ so_recv,
                              const WindowView* __restrict__ local_views,  // [SU], LSA
                              int with_meta, int32_t* __restrict__ fwd_meta) {
  const int n = blockIdx.x;
  if (n >= hi.SO || n == hi.D) return;
  const int np = hyb_np(hi, n);
  const int warp = get_warp_idx(), nwarps = get_num_warps(), lane = get_lane_idx();
  const int K = cfg.num_topk, T = cfg.num_max_tokens_per_rank, nsf = cfg.hidden_sf();
  const int src_rank = n * hi.SU + hi.u;
  int p0, p1;
  hyb_chunk(hi.nrecv[n], hi.chunk, hi.nchunks, p0, p1);
  for (int p = p0 + blockIdx.y * nwarps + warp; p < p1; p += nwarps * gridDim.y) {
    const uint8_t* rec = so_recv + ((size_t)np * T + p) * hi.rec_bytes;
    const int slot_v = lane < hi.SU ? reinterpret_cast<const int32_t*>(rec + hi.off_slots)[lane] : -1;
    int e = -1;
    float w = 0.f;
    if (with_meta && lane < K) {
      e = reinterpret_cast<const int32_t*>(rec + hi.off_topk)[lane];
      w = reinterpret_cast<const float*>(rec + hi.off_w)[lane];
    }
    const int src = with_meta ? *reinterpret_cast<const int32_t*>(rec + hi.off_src) : 0;
    for (unsigned long long m = __ballot(slot_v >= 0); m; m &= m - 1) {  // wave-uniform
      const int v = __builtin_ctzll(m);
      const size_t slot = EpWindowLayout::slot(cfg, src_rank, __shfl(slot_v, v, kWarpSize));
      const WindowView dv = local_views[v];
      if (kFp8) {
        wave_copy_bytes((uint8_t*)(dv.x_fp8() + slot * cfg.hidden), rec, cfg.hidden);
        for (int j = lane; j < nsf; j += kWarpSize)
          dv.sf()[slot * nsf + j] = reinterpret_cast<const float*>(rec + hi.off_sf)[j];
      } else {
        wave_copy_bf16(dv.x() + slot * cfg.hidden, reinterpret_cast<const bf16_t*>(rec), cfg.hidden);
      }
      if (with_meta) {
        const int dst = hi.D * hi.SU + v;
        if (lane < K) {
          dv.topk_idx()[slot * K + lane] = (e >= cfg.expert_begin(dst) && e < cfg.expert_end(dst)) ? e : -1;
          dv.topk_w()[slot * K + lane] = w;
        }
        if (lane == 0) dv.src_idx()[slot] = src;
      }
    }
    if (fwd_meta && with_meta) {
      int32_t* fm = fwd_meta + ((size_t)np * T + p) * (1 + K);
      if (lane == 0) fm[0] = src % T;
      if (lane < K) fm[1 + lane] = e;
    }
  }
}

// wave_copy_bytes with kUnroll 16-byte loads in flight per lane before their stores (plain
// stores: the destination is a peer's window).
template <int kUnroll>
__device__ __forceinline__ void wave_copy_peer_bytes(uint8_t* __restrict__ dst, const uint8_t* __restrict__ src,
                                                     int n) {
  const int lane = get_lane_idx();
  const int nvec = n / 16;
  const auto* s4 = reinterpret_cast<const uint4*>(src);
  auto* d4 = reinterpret_cast<uint4*>(dst);
  int i = lane;
  for (; i + (kUnroll - 1) * kWarpSize < nvec; i += kUnroll * kWarpSize) {
    uint4 v[kUnroll];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) v[u] = s4[i + u * kWarpSize];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) d4[i + u * kWarpSize] = v[u];
  }
  for (; i < nvec; i += kWarpSize) d4[i] = s4[i];
  for (int j = nvec * 16 + lane; j < n; j += kWarpSize) dst[j] = src[j];
}

// k_hyb_forward's writes, one wave per (record, local rank) pair instead of one per record, with
// four loads in flight per lane. RCCL_EP_GIN_STREAM forwards each chunk as it lands, and a chunk
// is a few hundred records: one wave per record left most waves idle and the rest copying their
// record to 3-4 ranks one latency-bound load at a time. Pair (p, 0) also writes fwd_meta.
template <bool kFp8>
__global__ void k_hyb_forward_pairs(EpConfig cfg, HybInfo hi, const uint8_t* __restrict__ so_recv,
                                    const WindowView* __restrict__ local_views,  // [SU], LSA
                                    int with_meta, int32_t* __restrict__ fwd_meta,
                                    const int32_t* __restrict__ dev_nrecv) {  // or null: hi.nrecv
  const int n = blockIdx.x;
  if (n >= hi.SO || n == hi.D) return;
  const int np = hyb_np(hi, n);
  const int warp = get_warp_idx(), nwarps = get_num_warps(), lane = get_lane_idx();
  const int K = cfg.num_topk, T = cfg.num_max_tokens_per_rank, nsf = cfg.hidden_sf(), SU = hi.SU;
  const int src_rank = n * SU + hi.u;
  int p0, p1;
  hyb_rec_chunk(hi, dev_nrecv ? dev_nrecv[n] : hi.nrecv[n], hi.chunk, p0, p1);
  for (int q = p0 * SU + blockIdx.y * nwarps + warp; q < p1 * SU; q += nwarps * gridDim.y) {
    const int p = q / SU, v = q - p * SU;
    const uint8_t* rec = so_recv + ((size_t)np * T + p) * hi.rec_bytes;
    const int slot_v = reinterpret_cast<const int32_t*>(rec + hi.off_slots)[v];  // wave-uniform
    const bool meta_row = fwd_meta && with_meta && v == 0;
    if (slot_v < 0 && !meta_row) continue;
    int e = -1;
    float w = 0.f;
    if (with_meta && lane < K) {
      e = reinterpret_cast<const int32_t*>(rec + hi.off_topk)[lane];
      w = reinterpret_cast<const float*>(rec + hi.off_w)[lane];
    }
    const int src = with_meta ? *reinterpret_cast<const int32_t*>(rec + hi.off_src) : 0;
    if (slot_v >= 0) {
      const size_t slot = EpWindowLayout::slot(cfg, src_rank, slot_v);
      const WindowView dv = local_views[v];
      if (kFp8) {
        wave_copy_peer_bytes<4>((uint8_t*)(dv.x_fp8() + slot * cfg.hidden), rec, cfg.hidden);
        for (int j = lane; j < nsf; j += kWarpSize)
          dv.sf()[slot * nsf + j] = reinterpret_cast<const float*>(rec + hi.off_sf)[j];
      } else {
        wave_copy_peer_bytes<4>((uint8_t*)(dv.x() + slot * cfg.hidden), rec, cfg.hidden * (int)sizeof(bf16_t));
      }
      if (with_meta) {
        const int dst = hi.D * SU + v;
        if (lane < K) {
          dv.topk_idx()[slot * K + lane] = (e >= cfg.expert_begin(dst) && e < cfg.expert_end(dst)) ? e : -1;
          dv.topk_w()[slot * K + lane] = w;
        }
        if (lane == 0) dv.src_idx()[slot] = src;
      }
    }
    if (meta_row) {
      int32_t* fm = fwd_meta + ((size_t)np * T + p) * (1 + K);
      if (lane == 0) fm[0] = src % T;
      if (lane < K) fm[1 + lane] = e;
    }
  }
}

// Lane k of a token's top-k: its expert's rank and node, and whether k is the LAST k with
// that rank / that node -- where the reference's grouped reduction leaves each group's sum.
struct HybSlot {
  int e, r, node;
  bool last_r, last_node;
};
__device__ __forceinline__ HybSlot hyb_slot(const EpConfig& cfg, int SU, int e, int lane) {
  const int K = cfg.num_topk, epr = cfg.experts_per_rank();
  HybSlot s;
  s.e = lane < K ? e : -1;
  const bool valid = s.e >= 0 && s.e < cfg.num_experts;
  s.r = valid ? s.e / epr : -1;
  s.node = valid ? s.r / SU : -1;
  s.last_r = valid;
  s.last_node = valid;
  for (int d = 1; d < K; ++d) {
    const int rr = __shfl(s.r, lane + d, kWarpSize);
    const int nn = __shfl(s.node, lane + d, kWarpSize);
    if (lane + d < K) {
      if (rr == s.r) s.last_r = false;
      if (nn == s.node) s.last_node = false;
    }
  }
  return s;
}

// Folds the rank partials staged for each forwarded record into its node partial comb_x[np][p]:
// fp32, ascending by each rank's last k, rounded once.
// Staging row is (np * SU + local rank) * T + owner token.
__global__ void k_hyb_fold(EpConfig cfg, HybInfo hi, const int32_t* __restrict__ fwd_meta,
                           const bf16_t* __restrict__ stage_x, const float* __restrict__ stage_w,
                           int with_w, bf16_t* __restrict__ comb_x, float* __restrict__ comb_w) {
  const int n = blockIdx.x;
  if (n >= hi.SO || n == hi.D) return;
  const int np = hyb_np(hi, n);
  const int warp = get_warp_idx(), nwarps = get_num_warps(), lane = get_lane_idx();
  const int K = cfg.num_topk, T = cfg.num_max_tokens_per_rank;
  int p0, p1;
  hyb_comb_range(hi, fwd_meta, n, T, K, p0, p1);
  for (int p = p0 + blockIdx.y * nwarps + warp; p < p1; p += nwarps * gridDim.y) {
    const int32_t* fm = fwd_meta + ((size_t)np * T + p) * (1 + K);
    const int t = fm[0];
    const HybSlot s = hyb_slot(cfg, hi.SU, lane < K ? fm[1 + lane] : -1, lane);
    const bool local = s.node == hi.D;
    const int r_loc = s.r - hi.D * hi.SU;
    const int32_t my_row = local ? (np * hi.SU + r_loc) * T + t : 0;
    unsigned long long take = __ballot(local && s.last_r);
    const int nrow = __popcll(take);
    int32_t rows8[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      rows8[j] = __shfl(my_row, take ? __builtin_ctzll(take) : 0, kWarpSize);
      take &= take - 1;
    }
    reduce_rows8_bf16(comb_x + ((size_t)np * T + p) * cfg.hidden, stage_x, rows8, nrow,
                      nullptr, nullptr, cfg.hidden, lane);
    if (with_w && lane < K)
      comb_w[((size_t)np * T + p) * K + lane] =
          local ? stage_w[((size_t)(np * hi.SU + r_loc) * T + t) * K + lane] : 0.f;
  }
}

// Owner-side reduce, one wave per token: biases, then the node partials in ascending order of
// each node's last k, fp32, rounded once; this node's partial is built here as the forwarders
// built theirs. The name contains 'combine_reduce_epilogue_impl' so that profilers matching that
// kernel name find it.
__global__ void hybrid_combine_reduce_epilogue_impl(EpConfig cfg, HybInfo hi, WindowView self,
                                                    const int32_t* __restrict__ topk_idx,  // [ntok, K] global
                                                    int num_tokens,
                                                    const int32_t* __restrict__ node_pos,  // [SO, ntok]
                                                    const bf16_t* __restrict__ land_x,
                                                    const float* __restrict__ land_w,
                                                    const bf16_t* __restrict__ bias0,
                                                    const bf16_t* __restrict__ bias1,
                                                    bf16_t* __restrict__ out, float* __restrict__ out_w,
                                                    int t_lo, int t_hi,  // tokens [t_lo, t_hi) only
                                                    int nsplit) {        // waves per token, splitting hidden
  const int warp = get_warp_idx(), nwarps = get_num_warps(), lane = get_lane_idx();
  const int K = cfg.num_topk, T = cfg.num_max_tokens_per_rank;
  constexpr int kPerVec = 8;
  const int nvec = cfg.hidden / kPerVec;
  const int t_end = t_hi < num_tokens ? t_hi : num_tokens;
  const int q = (blockIdx.x * nwarps + warp) % nsplit;  // this wave's part of each token's hidden
  const int v_lo = nvec * q / nsplit, v_hi = nvec * (q + 1) / nsplit;
  for (int t = t_lo + (blockIdx.x * nwarps + warp) / nsplit; t < t_end; t += nwarps * gridDim.x / nsplit) {
    const HybSlot s = hyb_slot(cfg, hi.SU, lane < K ? topk_idx[(size_t)t * K + lane] : -1, lane);
    const bool local = s.node == hi.D;
    // This node's rank rows, ascending last k.
    unsigned long long tl = __ballot(local && s.last_r);
    const int nl = __popcll(tl);
    const int32_t lrow = local ? (int32_t)EpWindowLayout::slot(cfg, s.r, t) : 0;
    int32_t lrows[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      lrows[j] = __shfl(lrow, tl ? __builtin_ctzll(tl) : 0, kWarpSize);
      tl &= tl - 1;
    }
    // The node partials, ascending last k: -1 marks this node's, else a landed row.
    unsigned long long tn = __ballot(s.last_node);
    const int nn = __popcll(tn);
    int32_t nrow_l = -1;
    if (s.last_node && !local)
      nrow_l = hyb_np(hi, s.node) * T + node_pos[(size_t)s.node * num_tokens + t];
    int32_t nrows[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      nrows[j] = __shfl(nrow_l, tn ? __builtin_ctzll(tn) : 0, kWarpSize);
      tn &= tn - 1;
    }
    const bf16_t* ybase = self.y();
    bf16_t* o = out + (size_t)t * cfg.hidden;
#pragma unroll 1
    for (int v = v_lo + lane; v < v_hi; v += kWarpSize) {
      const int h0 = v * kPerVec;
      float part[kPerVec];
#pragma unroll
      for (int u = 0; u < kPerVec; ++u) part[u] = 0.f;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (j < nl) {
          const uint4 y = *reinterpret_cast<const uint4*>(ybase + (size_t)lrows[j] * cfg.hidden + h0);
          const bf16_t* yy = reinterpret_cast<const bf16_t*>(&y);
#pragma unroll
          for (int u = 0; u < kPerVec; ++u) part[u] += __bfloat162float(yy[u]);
        }
      }
#pragma unroll
      for (int u = 0; u < kPerVec; ++u) part[u] = __bfloat162float(__float2bfloat16(part[u]));
      float acc[kPerVec];
#pragma unroll
      for (int u = 0; u < kPerVec; ++u) acc[u] = 0.f;
      if (bias0) {
        const uint4 b = *reinterpret_cast<const uint4*>(bias0 + (size_t)t * cfg.hidden + h0);
        const bf16_t* bb = reinterpret_cast<const bf16_t*>(&b);
#pragma unroll
        for (int u = 0; u < kPerVec; ++u) acc[u] += __bfloat162float(bb[u]);
      }
      if (bias1) {
        const uint4 b = *reinterpret_cast<const uint4*>(bias1 + (size_t)t * cfg.hidden + h0);
        const bf16_t* bb = reinterpret_cast<const bf16_t*>(&b);
#pragma unroll
        for (int u = 0; u < kPerVec; ++u) acc[u] += __bfloat162float(bb[u]);
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        if (j < nn) {
          if (nrows[j] < 0) {
#pragma unroll
            for (int u = 0; u < kPerVec; ++u) acc[u] += part[u];
          } else {
            const uint4 y = *reinterpret_cast<const uint4*>(land_x + (size_t)nrows[j] * cfg.hidden + h0);
            const bf16_t* yy = reinterpret_cast<const bf16_t*>(&y);
#pragma unroll
            for (int u = 0; u < kPerVec; ++u) acc[u] += __bfloat162float(yy[u]);
          }
        }
      }
      uint4 ov;
      bf16_t* oo = reinterpret_cast<bf16_t*>(&ov);
#pragma unroll
      for (int u = 0; u < kPerVec; ++u) oo[u] = __float2bfloat16(acc[u]);
      *reinterpret_cast<uint4*>(o + h0) = ov;
    }
    if (out_w && q == 0 && lane < K) {
      float w = 0.f;
      if (s.node == hi.D) {
        w = self.cw()[EpWindowLayout::slot(cfg, s.r, t) * K + lane];
      } else if (s.node >= 0) {
        const int pos = node_pos[(size_t)s.node * num_tokens + t];
        w = land_w[((size_t)hyb_np(hi, s.node) * T + pos) * K + lane];
      }
      out_w[(size_t)t * K + lane] = w;
    }
  }
}

}  // namespace rccl_ep

#endif  // RCCL_EP_HYBRID_H_
