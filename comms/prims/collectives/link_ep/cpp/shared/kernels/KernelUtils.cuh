// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>
#include <type_traits>

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

#include "comms/prims/collectives/link_ep/cpp/shared/kernels/Exception.cuh"
#include "comms/prims/collectives/link_ep/cpp/shared/kernels/KernelConfigs.cuh"

// Helpers used by intranode dispatch / combine kernels.
//
// Prims-specific notes:
//
//  - A `barrier_device<kNumRanks>(task_fifo_ptrs, head, rank)` star-pattern
//    atomic-add/sub barrier needs N peer-mapped FIFO arrays plus a rotating
//    `head` counter. We replace it everywhere with a thin `barrier_all_peers`
//    wrapper that calls into prims' `MultiPeerNvlTransport::barrier_sync`.
//    The kernel keeps the same call shape so
//    the body of `notify_dispatch` etc. doesn't have to change otherwise.

namespace comms::prims::link_ep::kernels {

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

// Lane 0's value, wave-uniform (SGPR on AMD). All warp lanes must be active.
__device__ __forceinline__ int broadcast_first_lane(int val) {
#ifdef __HIP_PLATFORM_AMD__
  return __builtin_amdgcn_readfirstlane(val);
#else
  return __shfl_sync(kFullWarpMask, val, 0);
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
  // AMD/xGMI: cross-GPU peer writes do NOT invalidate the local L2 (see the
  // ld_nc_global note below), so a plain volatile load can read a stale cached
  // value indefinitely. Every intranode use of this helper polls peer-written
  // FIFO head/tail/offset or barrier-completion state, so a stale read makes
  // the producer/consumer or barrier spin forever -> GPU hang (intermittent,
  // surfaces under the full model's L2 pressure). Read the coherent value with
  // a system-scoped ACQUIRE atomic load, pairing with the peers' system atomics
  // / st_release_sys_global writes.
  return __hip_atomic_load(
      const_cast<const int*>(ptr), __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
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
// WARP_COPY unroll factor. Unused by the AMD copy (see kWarpCopyStage).
// ---------------------------------------------------------------------------
#ifdef __HIP_PLATFORM_AMD__
constexpr int kIntranodeUnrollFactor = 2;
// 16 B elements per lane loaded before any store in the AMD warp copy.
constexpr int kWarpCopyStage = 8;
// 16 B vector in the global address space: without this cast, pointers from
// buffer_ptrs[] compile to FLAT ops with a full wait before every store.
typedef int GlobalInt4 __attribute__((ext_vector_type(4)));
#define LINK_EP_GLOBAL_INT4(p) \
  ((__attribute__((address_space(1))) GlobalInt4*)(p))
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
  const GlobalInt4 v = __builtin_nontemporal_load(
      LINK_EP_GLOBAL_INT4(const_cast<void*>(static_cast<const void*>(ptr))));
  return make_int4(v.x, v.y, v.z, v.w);
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
  *LINK_EP_GLOBAL_INT4(static_cast<void*>(ptr)) =
      GlobalInt4{val.x, val.y, val.z, val.w};
#else
  asm volatile("st.global.v4.s32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)
               : "memory");
#endif
}

// Nontemporal (cache-bypassing) 16B store for the sender's cross-device data
// copy: streams the payload past L2 toward the peer as it is produced. Use for
// sender->peer copies only; local receiver writes keep the plain st_na_global.
__device__ __forceinline__ void st_nt_global(int4* ptr, int4 val) {
#ifdef __HIP_PLATFORM_AMD__
  __builtin_nontemporal_store(
      GlobalInt4{val.x, val.y, val.z, val.w},
      LINK_EP_GLOBAL_INT4(static_cast<void*>(ptr)));
#else
  asm volatile("st.global.cs.v4.s32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)
               : "memory");
#endif
}

// Cached read for LOCAL data (sender's own input); L1/L2 are coherent with its
// producer. `ld_nc_global` (cache-bypass) would send every read to HBM and cut
// dispatch sender throughput by 10-100x.
__device__ __forceinline__ int4 ld_cached_global(const int4* ptr) {
#ifdef __HIP_PLATFORM_AMD__
  const GlobalInt4 v =
      *LINK_EP_GLOBAL_INT4(const_cast<void*>(static_cast<const void*>(ptr)));
  return make_int4(v.x, v.y, v.z, v.w);
#else
  return __ldg(ptr);
#endif
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

} // namespace comms::prims::link_ep::kernels

// ---------------------------------------------------------------------------
// UNROLLED_WARP_COPY — bulk warp-cooperative memcpy with N-way unrolling.
//
// Used by the data-path of dispatch / combine to copy `hidden_int4`-sized
// payloads from peer or local memory. LANE_ID is `threadIdx.x % kWarpSize`, N
// is the count, DST + SRC are typed pointers, LD_FUNC / ST_FUNC are the load /
// store primitives to use.
// ---------------------------------------------------------------------------

#if defined(__GFX9__)
// s_waitcnt vmcnt(0) in the gfx9 encoding; other targets encode it differently
// and get the compiler's waits.
#define LINK_EP_WAIT_VMCNT0() __builtin_amdgcn_s_waitcnt(0x0F70)
#else
#define LINK_EP_WAIT_VMCNT0() \
  do {                        \
  } while (0)
#endif

#ifndef LINK_EP_UNROLLED_WARP_COPY
#if defined(__HIP_PLATFORM_AMD__)
// AMD: load up to kWarpCopyStage elements per lane before storing any. gfx9
// retires vmcnt in order, so interleaving loads and stores makes every load
// wait on earlier remote-store acks. UNROLL_FACTOR is unused.
#define LINK_EP_UNROLLED_WARP_COPY(                                           \
    UNROLL_FACTOR, LANE_ID, N, DST, SRC, LD_FUNC, ST_FUNC)                    \
  do {                                                                        \
    constexpr int _kStage = ::comms::prims::link_ep::kernels::kWarpCopyStage; \
    constexpr int _kLanes = ::comms::prims::link_ep::kernels::kWarpSize;      \
    typename std::remove_reference<decltype(LD_FUNC((SRC) + 0))>::type        \
        _staged_values[_kStage];                                              \
    auto _src = (SRC);                                                        \
    auto _dst = (DST);                                                        \
    for (int _base = 0; _base < (N); _base += _kStage * _kLanes) {            \
      _Pragma("unroll") for (int _j = 0; _j < _kStage; ++_j) {                \
        const int _i = _base + _j * _kLanes + (LANE_ID);                      \
        if (_i < (N)) {                                                       \
          _staged_values[_j] = LD_FUNC(_src + _i);                            \
        }                                                                     \
      }                                                                       \
      /* One vmcnt(0) here: with predicated loads the compiler otherwise */   \
      /* emits a full wait before every store.                           */   \
      LINK_EP_WAIT_VMCNT0();                                                  \
      _Pragma("unroll") for (int _j = 0; _j < _kStage; ++_j) {                \
        const int _i = _base + _j * _kLanes + (LANE_ID);                      \
        if (_i < (N)) {                                                       \
          ST_FUNC(_dst + _i, _staged_values[_j]);                             \
        }                                                                     \
      }                                                                       \
    }                                                                         \
  } while (0)
#else
#define LINK_EP_UNROLLED_WARP_COPY(                                          \
    UNROLL_FACTOR, LANE_ID, N, DST, SRC, LD_FUNC, ST_FUNC)                   \
  do {                                                                       \
    constexpr int _kLoopStride =                                             \
        ::comms::prims::link_ep::kernels::kWarpSize * (UNROLL_FACTOR);       \
    typename std::remove_reference<decltype(LD_FUNC((SRC) + 0))>::type       \
        _unrolled_values[(UNROLL_FACTOR)];                                   \
    auto _src = (SRC);                                                       \
    auto _dst = (DST);                                                       \
    for (int _i = (LANE_ID); _i < ((N) / _kLoopStride) * _kLoopStride;       \
         _i += _kLoopStride) {                                               \
      _Pragma("unroll") for (int _j = 0; _j < (UNROLL_FACTOR); ++_j) {       \
        _unrolled_values[_j] = LD_FUNC(                                      \
            _src + _i + _j * ::comms::prims::link_ep::kernels::kWarpSize);   \
      }                                                                      \
      _Pragma("unroll") for (int _j = 0; _j < (UNROLL_FACTOR); ++_j) {       \
        ST_FUNC(                                                             \
            _dst + _i + _j * ::comms::prims::link_ep::kernels::kWarpSize,    \
            _unrolled_values[_j]);                                           \
      }                                                                      \
    }                                                                        \
    for (int _i = ((N) / _kLoopStride) * _kLoopStride + (LANE_ID); _i < (N); \
         _i += ::comms::prims::link_ep::kernels::kWarpSize) {                \
      ST_FUNC(_dst + _i, LD_FUNC(_src + _i));                                \
    }                                                                        \
  } while (0)
#endif // __HIP_PLATFORM_AMD__
#endif
