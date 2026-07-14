// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>
#include <type_traits>

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

#include "comms/prims/collectives/moe_ep/cpp/shared/kernels/Exception.cuh"
#include "comms/prims/collectives/moe_ep/cpp/shared/kernels/KernelConfigs.cuh"

// Helpers used by intranode dispatch / combine kernels.
//
// Pipes-specific notes:
//
//  - A `barrier_device<kNumRanks>(task_fifo_ptrs, head, rank)` star-pattern
//    atomic-add/sub barrier needs N peer-mapped FIFO arrays plus a rotating
//    `head` counter. We replace it everywhere with a thin `barrier_all_peers`
//    wrapper that calls into pipes' `MultiPeerNvlTransport::barrier_sync`
//    (one-line free function from D2). The kernel keeps the same call shape so
//    the body of `notify_dispatch` etc. doesn't have to change otherwise.

namespace comms::prims::moe_ep::kernels {

// ---------------------------------------------------------------------------
// Math / layout helpers (pure-compute, no comm).
// ---------------------------------------------------------------------------

template <typename DType>
__host__ __device__ __forceinline__ DType cell_div(DType a, DType b) {
  return (a + b - 1) / b;
}

template <typename DType>
__host__ __device__ __forceinline__ DType align(DType a, DType b) {
  return cell_div<DType>(a, b) * b;
}

__device__ __forceinline__ void get_channel_task_range(
    int num_tokens,
    int num_sms,
    int sm_id,
    int& token_start_idx,
    int& token_end_idx) {
  int num_tokens_per_sm = cell_div(num_tokens, num_sms);
  token_start_idx = min(num_tokens_per_sm * sm_id, num_tokens);
  token_end_idx = min(token_start_idx + num_tokens_per_sm, num_tokens);
}

// ---------------------------------------------------------------------------
// Memory-fence wrappers — bridge NVIDIA inline PTX vs HIP threadfence calls.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void memory_fence() {
#ifdef __HIP_PLATFORM_AMD__
  __threadfence_system();
#else
  asm volatile("fence.acq_rel.sys;" : : : "memory");
#endif
}

__device__ __forceinline__ void memory_fence_gpu() {
#ifdef __HIP_PLATFORM_AMD__
  __threadfence();
#else
  asm volatile("fence.acq_rel.gpu;" : : : "memory");
#endif
}

__device__ __forceinline__ void memory_fence_cta() {
#ifdef __HIP_PLATFORM_AMD__
  __threadfence_block();
#else
  asm volatile("fence.acq_rel.cta;" : : : "memory");
#endif
}

__device__ __forceinline__ void trap_kernel() {
#ifdef __HIP_PLATFORM_AMD__
  abort();
#else
  asm("trap;");
#endif
}

// ---------------------------------------------------------------------------
// Warp shuffle primitives — one wrapper for both NVIDIA and HIP.
// ---------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ T shfl_xor_sync_compat(T val, int lane_mask) {
#ifdef __HIP_PLATFORM_AMD__
  return __shfl_xor(val, lane_mask, kWarpSize);
#else
  return __shfl_xor_sync(kFullWarpMask, val, lane_mask, kWarpSize);
#endif
}

__device__ __forceinline__ int warp_reduce_sum(int value) {
  if constexpr (kWarpSize == 64) {
    value += shfl_xor_sync_compat<int>(value, 32);
  }
  value += shfl_xor_sync_compat<int>(value, 16);
  value += shfl_xor_sync_compat<int>(value, 8);
  value += shfl_xor_sync_compat<int>(value, 4);
  value += shfl_xor_sync_compat<int>(value, 2);
  value += shfl_xor_sync_compat<int>(value, 1);
  return value;
}

__device__ __forceinline__ int get_lane_id() {
#ifdef __HIP_PLATFORM_AMD__
  return threadIdx.x % kWarpSize;
#else
  int lane_id;
  asm("mov.s32 %0, %%laneid;" : "=r"(lane_id));
  return lane_id;
#endif
}

// ---------------------------------------------------------------------------
// Loads / stores with explicit memory ordering, used by the channel-state
// communication in dispatch / combine.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void st_relaxed_sys_global(int* ptr, int val) {
#ifdef __HIP_PLATFORM_AMD__
  __hip_atomic_store(ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
  asm volatile("st.relaxed.sys.global.s32 [%0], %1;"
               :
               : "l"(ptr), "r"(val)
               : "memory");
#endif
}

__device__ __forceinline__ void st_release_sys_global(int* ptr, int val) {
#ifdef __HIP_PLATFORM_AMD__
  __hip_atomic_store(ptr, val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
  asm volatile("st.release.sys.global.s32 [%0], %1;"
               :
               : "l"(ptr), "r"(val)
               : "memory");
#endif
}

__device__ __forceinline__ int ld_relaxed_sys_global(const int* ptr) {
#ifdef __HIP_PLATFORM_AMD__
  return __hip_atomic_load(ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#else
  int ret;
  asm volatile("ld.relaxed.sys.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
  return ret;
#endif
}

__device__ __forceinline__ int ld_acquire_sys_global(const int* ptr) {
#ifdef __HIP_PLATFORM_AMD__
  return __hip_atomic_load(ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
  int ret;
  asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
  return ret;
#endif
}

__device__ __forceinline__ int ld_volatile_global(const volatile int* ptr) {
#ifdef __HIP_PLATFORM_AMD__
  return *ptr;
#else
  int ret;
  asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
  return ret;
#endif
}

__device__ __forceinline__ int atomic_add_release_global(int* ptr, int value) {
#ifdef __HIP_PLATFORM_AMD__
  return __hip_atomic_fetch_add(
      ptr, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
  int ret;
  asm volatile("atom.add.release.gpu.global.s32 %0, [%1], %2;"
               : "=r"(ret)
               : "l"(ptr), "r"(value));
  return ret;
#endif
}

__device__ __forceinline__ long long wall_clock64_compat() {
#ifdef __HIP_PLATFORM_AMD__
  return static_cast<long long>(__builtin_amdgcn_s_memrealtime());
#else
  long long t;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(t));
  return t;
#endif
}

// ---------------------------------------------------------------------------
// Warp shuffle / sync wrappers (extended).
// ---------------------------------------------------------------------------

__device__ __forceinline__ void syncwarp() {
#ifdef __HIP_PLATFORM_AMD__
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
#else
  __syncwarp();
#endif
}

template <typename T>
__device__ __forceinline__ T shfl_sync_compat(T val, int src_lane) {
#ifdef __HIP_PLATFORM_AMD__
  return __shfl(val, src_lane, kWarpSize);
#else
  return __shfl_sync(kFullWarpMask, val, src_lane, kWarpSize);
#endif
}

// ---------------------------------------------------------------------------
// WARP_COPY unroll factor. AMD uses 2 vs NVIDIA's 4 because AMD's 64-lane
// warp already issues twice the data per stride; mismatched factors trigger
// AMD memory-pipe stalls / data races.
// ---------------------------------------------------------------------------
#ifdef __HIP_PLATFORM_AMD__
constexpr int kIntranodeUnrollFactor = 2;
#else
constexpr int kIntranodeUnrollFactor = 4;
#endif

// ---------------------------------------------------------------------------
// Non-coalescing / no-allocate global loads + stores for the data-path copies.
// ---------------------------------------------------------------------------

// On AMD: cross-GPU IPC reads use __builtin_nontemporal_load to bypass L2.
// Required because xGMI peer writes don't invalidate the local L2.
__device__ __forceinline__ int ld_nc_global(const int* ptr) {
#ifdef __HIP_PLATFORM_AMD__
  return __builtin_nontemporal_load(ptr);
#else
  int ret;
  asm volatile("ld.global.nc.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
  return ret;
#endif
}

__device__ __forceinline__ int4 ld_nc_global(const int4* ptr) {
#ifdef __HIP_PLATFORM_AMD__
  // No 16B nontemporal_load; emit 4×int32.
  const int* p = reinterpret_cast<const int*>(ptr);
  int4 ret;
  ret.x = __builtin_nontemporal_load(p + 0);
  ret.y = __builtin_nontemporal_load(p + 1);
  ret.z = __builtin_nontemporal_load(p + 2);
  ret.w = __builtin_nontemporal_load(p + 3);
  return ret;
#else
  int4 ret;
  asm volatile("ld.global.nc.v4.s32 {%0, %1, %2, %3}, [%4];"
               : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
               : "l"(ptr));
  return ret;
#endif
}

__device__ __forceinline__ int64_t ld_nc_global(const int64_t* ptr) {
#ifdef __HIP_PLATFORM_AMD__
  return __builtin_nontemporal_load(ptr);
#else
  int64_t ret;
  asm volatile("ld.global.nc.s64 %0, [%1];" : "=l"(ret) : "l"(ptr));
  return ret;
#endif
}

__device__ __forceinline__ float ld_nc_global(const float* ptr) {
#ifdef __HIP_PLATFORM_AMD__
  return __builtin_nontemporal_load(ptr);
#else
  float ret;
  asm volatile("ld.global.nc.f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
  return ret;
#endif
}

__device__ __forceinline__ void st_na_global(int* ptr, int val) {
#ifdef __HIP_PLATFORM_AMD__
  *ptr = val;
#else
  asm volatile("st.global.s32 [%0], %1;" : : "l"(ptr), "r"(val) : "memory");
#endif
}

__device__ __forceinline__ void st_na_global(int4* ptr, int4 val) {
#ifdef __HIP_PLATFORM_AMD__
  *ptr = val;
#else
  asm volatile("st.global.v4.s32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)
               : "memory");
#endif
}

// Cached read for LOCAL data (sender's own input). On AMD, plain `__ldg`
// suffices — local L1/L2 are coherent with the producer (CPU/torch). Using
// `ld_nc_global` (cache-bypass) here would force every read to round-trip
// to HBM and degrade dispatch sender throughput by 10-100x.
__device__ __forceinline__ int4 ld_cached_global(const int4* ptr) {
  return __ldg(ptr);
}
__device__ __forceinline__ int ld_cached_global(const int* ptr) {
  return __ldg(ptr);
}
__device__ __forceinline__ float ld_cached_global(const float* ptr) {
  return __ldg(ptr);
}
__device__ __forceinline__ int64_t ld_cached_global(const int64_t* ptr) {
  return __ldg(ptr);
}

} // namespace comms::prims::moe_ep::kernels

// ---------------------------------------------------------------------------
// UNROLLED_WARP_COPY — bulk warp-cooperative memcpy with N-way unrolling.
//
// Used by the data-path of dispatch / combine to copy `hidden_int4`-sized
// payloads from peer or local memory. LANE_ID is `threadIdx.x % kWarpSize`, N
// is the count, DST + SRC are typed pointers, LD_FUNC / ST_FUNC are the load /
// store primitives to use.
// ---------------------------------------------------------------------------

#ifndef MOE_EP_UNROLLED_WARP_COPY
#define MOE_EP_UNROLLED_WARP_COPY(                                           \
    UNROLL_FACTOR, LANE_ID, N, DST, SRC, LD_FUNC, ST_FUNC)                   \
  do {                                                                       \
    constexpr int _kLoopStride =                                             \
        ::comms::prims::moe_ep::kernels::kWarpSize * (UNROLL_FACTOR);        \
    typename std::remove_reference<decltype(LD_FUNC((SRC) + 0))>::type       \
        _unrolled_values[(UNROLL_FACTOR)];                                   \
    auto _src = (SRC);                                                       \
    auto _dst = (DST);                                                       \
    for (int _i = (LANE_ID); _i < ((N) / _kLoopStride) * _kLoopStride;       \
         _i += _kLoopStride) {                                               \
      _Pragma("unroll") for (int _j = 0; _j < (UNROLL_FACTOR); ++_j) {       \
        _unrolled_values[_j] = LD_FUNC(                                      \
            _src + _i + _j * ::comms::prims::moe_ep::kernels::kWarpSize);    \
      }                                                                      \
      _Pragma("unroll") for (int _j = 0; _j < (UNROLL_FACTOR); ++_j) {       \
        ST_FUNC(                                                             \
            _dst + _i + _j * ::comms::prims::moe_ep::kernels::kWarpSize,     \
            _unrolled_values[_j]);                                           \
      }                                                                      \
    }                                                                        \
    for (int _i = ((N) / _kLoopStride) * _kLoopStride + (LANE_ID); _i < (N); \
         _i += ::comms::prims::moe_ep::kernels::kWarpSize) {                 \
      ST_FUNC(_dst + _i, LD_FUNC(_src + _i));                                \
    }                                                                        \
  } while (0)
#endif
