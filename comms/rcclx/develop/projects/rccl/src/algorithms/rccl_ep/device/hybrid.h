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
};

__device__ __forceinline__ int hyb_np(const HybInfo& hi, int n) { return n < hi.D ? n : n - 1; }

// Chunk c of C over cnt records: [p0, p1). Both ends of the rail must split identically, as
// one side's nsend is the other's nrecv.
__host__ __device__ __forceinline__ void hyb_chunk(int cnt, int c, int C, int& p0, int& p1) {
  p0 = (int)((long long)cnt * c / C);
  p1 = (int)((long long)cnt * (c + 1) / C);
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

// One record per (token, remote node holding one of its experts), in token order, into
// so_send[np]: [payload | sf (fp8) | topk ids | topk weights | src_token_global_idx | slot
// per local rank of that node, -1 if none]. with_meta 0 leaves the top-k sections unwritten.
template <bool kFp8>
__global__ void k_hyb_pack(EpConfig cfg, HybInfo hi, const void* __restrict__ x,
                           const float* __restrict__ sf, const int32_t* __restrict__ topk_idx,
                           const float* __restrict__ topk_w, int num_tokens,
                           const int32_t* __restrict__ node_list,  // [SO, num_tokens]
                           const int32_t* __restrict__ slot_of,    // [num_ranks, num_tokens]
                           uint8_t* __restrict__ so_send, int with_meta) {
  const int n = blockIdx.x;
  if (n >= hi.SO || n == hi.D) return;
  const int np = hyb_np(hi, n);
  const int warp = get_warp_idx(), nwarps = get_num_warps(), lane = get_lane_idx();
  const int K = cfg.num_topk, T = cfg.num_max_tokens_per_rank, nsf = cfg.hidden_sf();
  const int row_bytes = kFp8 ? cfg.hidden : cfg.hidden * (int)sizeof(bf16_t);
  int p0, p1;
  hyb_chunk(hi.nsend[n], hi.chunk, hi.nchunks, p0, p1);
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
  hyb_chunk(hi.nrecv[n], hi.chunk, hi.nchunks, p0, p1);
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
                                                    bf16_t* __restrict__ out, float* __restrict__ out_w) {
  const int warp = get_warp_idx(), nwarps = get_num_warps(), lane = get_lane_idx();
  const int K = cfg.num_topk, T = cfg.num_max_tokens_per_rank;
  constexpr int kPerVec = 8;
  const int nvec = cfg.hidden / kPerVec;
  for (int t = blockIdx.x * nwarps + warp; t < num_tokens; t += nwarps * gridDim.x) {
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
    for (int v = lane; v < nvec; v += kWarpSize) {
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
    if (out_w && lane < K) {
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
