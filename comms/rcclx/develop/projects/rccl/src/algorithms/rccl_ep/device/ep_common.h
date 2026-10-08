/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Shared device types and primitives: the window view, the wave-wide copies,
 * and the cross-rank rendezvous.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_EP_COMMON_H_
#define RCCL_EP_COMMON_H_

#include <cstdint>

#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>
#include <nccl_device.h>

#include "device/hip_prims.h"
#include "include/ep_layout.h"

namespace rccl_ep {

using bf16_t = __hip_bfloat16;
// OCP e4m3 on gfx950. gfx942 uses the fnuz variant, which is a DIFFERENT wire
// format: the two are mutually exclusive per arch in HIP's FP8 API, so a
// mixed-arch communicator must be rejected at construction rather than
// silently exchanging incompatible bytes.
using fp8_t = __hip_fp8_e4m3;

// Byte view of a rank's symmetric window plus the offsets into it.
struct WindowView {
  uint8_t* base;
  EpWindowLayout l;

  __device__ __forceinline__ int32_t* counts() const {
    return (int32_t*)(base + l.off_counts);
  }
  __device__ __forceinline__ uint32_t* flags() const {
    return (uint32_t*)(base + l.off_flags);
  }
  __device__ __forceinline__ int32_t* notify() const {
    return (int32_t*)(base + l.off_notify);
  }
  __device__ __forceinline__ bf16_t* x() const {
    return (bf16_t*)(base + l.off_x);
  }
  __device__ __forceinline__ int32_t* topk_idx() const {
    return (int32_t*)(base + l.off_topk_idx);
  }
  __device__ __forceinline__ float* topk_w() const {
    return (float*)(base + l.off_topk_w);
  }
  __device__ __forceinline__ int32_t* src_idx() const {
    return (int32_t*)(base + l.off_src_idx);
  }
  __device__ __forceinline__ bf16_t* y() const {
    return (bf16_t*)(base + l.off_y);
  }
  __device__ __forceinline__ float* cw() const {
    return (float*)(base + l.off_cw);
  }
  __device__ __forceinline__ fp8_t* x_fp8() const {
    return (fp8_t*)(base + l.off_x);
  }
  __device__ __forceinline__ float* sf() const {
    return (float*)(base + l.off_sf);
  }
  __device__ __forceinline__ char* arch() const {
    return (char*)(base + l.off_arch);
  }
};

// Vectorised copy of `n` bf16 elements, one wave cooperating. Uses dwordx4
// (8 bf16) per lane, the widest store AMDGCN offers and the bulk-copy path for
// token payloads.
//
// Needs both pointers 16B aligned, which reduces to hidden being a multiple of
// 8; ep_create enforces it. The scalar loop below is a tail for a leftover
// element count, NOT a fallback for a misaligned base.
//
// Pushes into a peer's window use this copy: nontemporal stores are slower there.
__device__ __forceinline__ void wave_copy_bf16(bf16_t* __restrict__ dst, const bf16_t* __restrict__ src, int n) {
  const int lane = get_lane_idx();
  constexpr int kPerVec = 8;  // 8 * 2B = 16B = dwordx4
  const int nvec = n / kPerVec;

  const auto* s4 = reinterpret_cast<const uint4*>(src);
  auto* d4 = reinterpret_cast<uint4*>(dst);
  for (int i = lane; i < nvec; i += kWarpSize) d4[i] = s4[i];

  // scalar tail
  for (int i = nvec * kPerVec + lane; i < n; i += kWarpSize) dst[i] = src[i];
}

// wave_copy_bf16 / wave_copy_bytes for a LOCAL copy, with nontemporal stores since the
// output is not read again by the kernel. kNtLoad also streams the source past L2, for
// a row read exactly once. kUnroll issues that many loads per lane before the first
// store; otherwise the compiler emits load / wait / store per vector.
using ep_v4i = int __attribute__((ext_vector_type(4)));
template <bool kNtLoad>
__device__ __forceinline__ ep_v4i ld_v4(const ep_v4i* p) {
  if constexpr (kNtLoad) return __builtin_nontemporal_load(p);
  else return *p;
}
template <bool kNtLoad, int kUnroll>
__device__ __forceinline__ void wave_copy_local_bytes(uint8_t* __restrict__ dst,
                                                      const uint8_t* __restrict__ src, int n) {
  const int lane = get_lane_idx();
  const int nvec = n / 16;
  const auto* s4 = reinterpret_cast<const ep_v4i*>(src);
  auto* d4 = reinterpret_cast<ep_v4i*>(dst);
  int i = lane;
  if constexpr (kUnroll > 1) {
    for (; i + (kUnroll - 1) * kWarpSize < nvec; i += kUnroll * kWarpSize) {
      ep_v4i v[kUnroll];
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) v[u] = ld_v4<kNtLoad>(s4 + i + u * kWarpSize);
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) __builtin_nontemporal_store(v[u], d4 + i + u * kWarpSize);
    }
  }
  for (; i < nvec; i += kWarpSize) __builtin_nontemporal_store(ld_v4<kNtLoad>(s4 + i), d4 + i);
  for (int j = nvec * 16 + lane; j < n; j += kWarpSize) dst[j] = src[j];
}
template <bool kNtLoad, int kUnroll>
__device__ __forceinline__ void wave_copy_local_bf16(bf16_t* __restrict__ dst,
                                                     const bf16_t* __restrict__ src, int n) {
  // n % 8 == 0 (ep_create), so the byte copy has no tail
  wave_copy_local_bytes<kNtLoad, kUnroll>(reinterpret_cast<uint8_t*>(dst),
                                          reinterpret_cast<const uint8_t*>(src), n * 2);
}

// CTA count for k_ep_lsa_barrier. Must equal the ncclDevCommRequirements::
// lsaBarrierCount set in ep_configure and be used at every launch site; a
// mismatch deadlocks the rendezvous rather than failing to compile.
constexpr int kEpBarrierCtas = 8;

// Cross-rank rendezvous, device side. Replaces the host barrier. Barrier A, before a
// push, keeps it out of a peer's window until the peer has finished reading it;
// barrier B, after the push, makes the pushed rows visible before the epilogue.
__global__ void k_ep_lsa_barrier(ncclDevComm devComm) {
  ncclLsaBarrierSession<ncclCoopCta> bar{ncclCoopCta(), devComm, ncclTeamLsa(devComm), devComm.lsaBarrier,
                                         (uint32_t)blockIdx.x};
  bar.sync(ncclCoopCta(), cuda::memory_order_acq_rel);
}

// Exclusive prefix sum over the per-source counts, on device, so the receive
// offsets never round-trip through the host. Also zeroes `zero_counts` (nzero
// ints, if non-null) for the copy epilogue to accumulate into.
__global__ void k_ep_scan_counts(EpConfig cfg, WindowView self, int32_t* __restrict__ recv_offsets,
                                 int32_t* __restrict__ total_recv,
                                 int32_t* __restrict__ zero_counts, int nzero) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  if (zero_counts) {
    for (int i = 0; i < nzero; ++i) zero_counts[i] = 0;
  }
  int acc = 0;
  for (int r = 0; r < cfg.num_ranks; ++r) {
    recv_offsets[r] = acc;
    acc += ld_acquire_sys<int32_t>(&self.counts()[r]);
  }
  *total_recv = acc;
}

// Second half of the uncached dispatch's count exchange, run by one wave of
// k_ep_plan_notify's last CTA once every source has published: folds this rank's
// notify region into the receive offsets, total, per-local-expert counts and
// per-source prefix.
//
// host (optional, host-mapped) gets [total, counts[epr], recv_pairs[R], max_tokens]:
// the pairs each rank receives and the largest source token count, identical on every
// rank. host[0] is stored last, after a fence, so a host that sees it non-negative
// sees the rest. srcpref / ebase (optional, for k_ep_grouped_epilogue) are exclusive
// prefixes of the same histograms, over lower sources and over lower experts.
struct NotifyReduceOut {
  int32_t* recv_offsets;
  int32_t* total_recv;
  int32_t* expert_counts;
  int32_t* psum_rank;
  int32_t* host;     // or nullptr
  int32_t* srcpref;  // [R, epr] or nullptr
  int32_t* ebase;    // [epr] or nullptr
};

__device__ __forceinline__ void notify_reduce_wave(const EpConfig& cfg, const WindowView& self,
                                                   const NotifyReduceOut& o) {
  int32_t* __restrict__ recv_offsets = o.recv_offsets;
  int32_t* __restrict__ total_recv = o.total_recv;
  int32_t* __restrict__ expert_counts = o.expert_counts;
  int32_t* __restrict__ psum_rank = o.psum_rank;
  int32_t* __restrict__ host = o.host;
  int32_t* __restrict__ srcpref = o.srcpref;
  int32_t* __restrict__ ebase = o.ebase;
  const int lane = get_lane_idx();
  const int R = cfg.num_ranks, epr = cfg.experts_per_rank(), S = 2 + epr + R;
  const int32_t* nt = self.notify();
  // One system-scope acquire for the whole region: re-read the flags the caller waited
  // on, then fence, so every payload read below is an ordinary load.
  if (lane < R) (void)__hip_atomic_load(&self.flags()[lane], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  __atomic_thread_fence(__ATOMIC_ACQUIRE);
  for (int e = lane; e < epr; e += kWarpSize) {
    int v[kMaxRanks];
#pragma unroll
    for (int s = 0; s < kMaxRanks; ++s) v[s] = s < R ? nt[s * S + 1 + e] : 0;
    int c = 0;
#pragma unroll
    for (int s = 0; s < kMaxRanks; ++s) {
      if (s < R && srcpref) srcpref[s * epr + e] = c;
      c += v[s];
    }
    expert_counts[e] = c;
    if (host) __hip_atomic_store(&host[1 + e], c, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
  if (ebase) {  // lane e0 + lane wrote expert_counts[e0 + lane] above
    int carry = 0;
    for (int e0 = 0; e0 < epr; e0 += kWarpSize) {
      const int c = (e0 + lane < epr) ? expert_counts[e0 + lane] : 0;
      const int ex = warp_exclusive_sum<int>(c);
      if (e0 + lane < epr) ebase[e0 + lane] = carry + ex;
      carry += reduce_add<int>(c);
    }
  }
  if (host) {
    for (int r = lane; r < R; r += kWarpSize) {
      int v[kMaxRanks];
#pragma unroll
      for (int s = 0; s < kMaxRanks; ++s) v[s] = s < R ? nt[s * S + 1 + epr + r] : 0;
      int c = 0;
#pragma unroll
      for (int s = 0; s < kMaxRanks; ++s) c += v[s];
      __hip_atomic_store(&host[1 + epr + r], c, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
  }
  // Lane s holds source s's token count; a wave scan gives the offsets (R <= kMaxRanks
  // <= kWarpSize, so one pass).
  const int sent = lane < R ? nt[lane * S] : 0;
  const int ntok = lane < R ? nt[lane * S + 1 + epr + R] : 0;
  const int incl = warp_inclusive_sum<int>(sent);
  int max_tokens = ntok;
#pragma unroll
  for (int off = kWarpSize / 2; off > 0; off >>= 1) max_tokens = max(max_tokens, __shfl_xor(max_tokens, off, kWarpSize));
  if (lane < R) {
    recv_offsets[lane] = incl - sent;
    psum_rank[lane] = incl;
  }
  const int acc = __shfl(incl, R - 1, kWarpSize);
  if (lane == 0) {
    *total_recv = acc;
    if (host) __hip_atomic_store(&host[1 + epr + R], max_tokens, __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_SYSTEM);
  }
  // The fence is executed by the whole wave, so it orders every lane's host stores
  // above before lane 0's sentinel store below.
  __threadfence_system();
  __builtin_amdgcn_wave_barrier();
  if (host && lane == 0) __hip_atomic_store(&host[0], acc, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

// Vectorised copy of `n` BYTES, one wave cooperating: dwordx4 per lane plus a
// scalar tail. The fp8 payload path. `dst` and `src` must be 16B aligned, as
// wave_copy_bf16 above requires.
__device__ __forceinline__ void wave_copy_bytes(uint8_t* __restrict__ dst, const uint8_t* __restrict__ src, int n) {
  const int lane = get_lane_idx();
  const int nvec = n / 16;
  const auto* s4 = reinterpret_cast<const uint4*>(src);
  auto* d4 = reinterpret_cast<uint4*>(dst);
  for (int i = lane; i < nvec; i += kWarpSize) d4[i] = s4[i];
  for (int i = nvec * 16 + lane; i < n; i += kWarpSize) dst[i] = src[i];
}

}  // namespace rccl_ep

#endif  // RCCL_EP_COMMON_H_
