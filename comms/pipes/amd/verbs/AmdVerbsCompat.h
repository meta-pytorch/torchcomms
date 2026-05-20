// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// =============================================================================
// AMD GPU (HIP/ROCm) Compatibility Layer for IBGDA Verbs
// =============================================================================
//
// This header provides AMD GPU equivalents of the CUDA-specific intrinsics and
// device functions used by IBGDA verbs device-side code.
//
// The original NVIDIA implementation in AmdVerbsCompat.h uses:
// - CUDA PTX inline assembly for memory ordering (ld.relaxed, st.release,
// fence)
// - cuda::atomic_ref for lock-free atomic operations with scope control
// - __ldg() for read-only cache loads
// - __syncwarp() and __reduce_max_sync() for warp-level primitives
//
// This file maps those to AMD GCN/CDNA equivalents using:
// - __builtin_amdgcn_fence() for memory ordering
// - __hip_atomic_* builtins for scoped atomics
// - __builtin_nontemporal_load/store for non-cached I/O
// - AMD wavefront intrinsics for warp-level operations
// =============================================================================

#pragma once

#include <hip/hip_runtime.h>
#include <cstdint>

// =============================================================================
// Platform Detection
// =============================================================================

#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIPCC__)
#error \
    "This header requires AMD HIP (ROCm). For NVIDIA GPUs, use the original CUDA headers."
#endif

// AMD wavefront size: 64 on GCN/CDNA, 32 on RDNA (default to 64 for datacenter
// GPUs)
#ifndef AMD_WAVEFRONT_SIZE
#if defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || \
    defined(__gfx942__)
#define AMD_WAVEFRONT_SIZE 64
#else
#define AMD_WAVEFRONT_SIZE 64
#endif
#endif

// =============================================================================
// Memory Ordering / Fence Operations
// =============================================================================

// AMD equivalent of CUDA PTX: fence.acquire.gpu / fence.acquire.sys
// Uses __builtin_amdgcn_fence with appropriate scope.
// "" = system scope (GPU + CPU + PCIe devices like NICs)
// "agent" = GPU scope (all CUs within the GPU)
// "workgroup" = workgroup scope
__device__ __forceinline__ void amd_fence_acquire_system() {
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
}

__device__ __forceinline__ void amd_fence_acquire_device() {
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
}

__device__ __forceinline__ void amd_fence_acquire_workgroup() {
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}

__device__ __forceinline__ void amd_fence_release_system() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
}

__device__ __forceinline__ void amd_fence_release_device() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
}

__device__ __forceinline__ void amd_fence_release_workgroup() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
}

// Full system-scope memory fence (equivalent to __threadfence_system)
__device__ __forceinline__ void amd_fence_system() {
  __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "");
}

// =============================================================================
// Relaxed / Non-temporal Loads (replace CUDA PTX ld.relaxed.sys.global)
// =============================================================================

// System-scope relaxed loads for memory shared with PCIe devices (NICs).
// Must use __hip_atomic_load with __HIP_MEMORY_SCOPE_SYSTEM to generate
// loads with system coherence modifiers (sc0 sc1 on CDNA3), ensuring the
// GPU sees the latest values written by the NIC (e.g., CQE entries).
// Plain __atomic_load_n generates loads without coherence modifiers, which
// may return stale cached values.

__device__ __forceinline__ uint8_t amd_load_relaxed_sys(uint8_t* ptr) {
  return __hip_atomic_load(ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ uint32_t amd_load_relaxed_sys(uint32_t* ptr) {
  return __hip_atomic_load(ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ uint64_t amd_load_relaxed_sys(uint64_t* ptr) {
  return __hip_atomic_load(ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

// Device-scope relaxed loads (for QP metadata shared among GPU threads)
__device__ __forceinline__ uint64_t amd_load_relaxed_device(uint64_t* ptr) {
  return __atomic_load_n(ptr, __ATOMIC_RELAXED);
}

// =============================================================================
// MMIO Stores (replace CUDA PTX st.mmio.relaxed.sys.global)
// =============================================================================

// For writing to NIC doorbell registers mapped into GPU address space.
// On AMD, we use a volatile store + system fence to ensure visibility to the
// NIC.
__device__ __forceinline__ void amd_store_relaxed_mmio_u64(
    uint64_t* ptr,
    uint64_t val) {
  __hip_atomic_store(ptr, val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__ __forceinline__ void amd_store_relaxed_mmio_u32(
    uint32_t* ptr,
    uint32_t val) {
  __hip_atomic_store(ptr, val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

// =============================================================================
// Byte-swap (replace CUDA PTX prmt.b32 byte permute instructions)
// ============================================================================

__device__ __forceinline__ uint64_t amd_bswap64(uint64_t x) {
  return __builtin_bswap64(x);
}

__device__ __forceinline__ uint32_t amd_bswap32(uint32_t x) {
  return __builtin_bswap32(x);
}

__device__ __forceinline__ uint16_t amd_bswap16(uint16_t x) {
  return __builtin_bswap16(x);
}

// =============================================================================
// Lane ID (replace CUDA PTX: mov.u32 %0, %%laneid)
// =============================================================================

__device__ __forceinline__ unsigned int amd_get_lane_id() {
  return __lane_id();
}

// =============================================================================
// Global Timer (replace CUDA PTX: mov.u64 %0, %%globaltimer)
// =============================================================================

// AMD uses s_memtime for GPU clock counter (returns 64-bit cycle count)
__device__ __forceinline__ uint64_t amd_query_global_timer() {
  return wall_clock64();
}

// =============================================================================
// Read-only Cache Load (replace CUDA __ldg())
// =============================================================================

// On AMD, __ldg equivalent is a const-qualified load.
// The compiler + hardware will route through the scalar cache or texture cache.
template <typename T>
__device__ __forceinline__ T amd_ldg(const T* ptr) {
  return *ptr; // AMD GCN/CDNA handles caching at HW level
}

// Specialization for pointer-width loads (uintptr_t)
__device__ __forceinline__ uintptr_t amd_ldg(const uintptr_t* ptr) {
  return *ptr;
}

// =============================================================================
// Scoped Atomics (replace cuda::atomic_ref<T, cuda::thread_scope_*>)
// =============================================================================

// fetch_add with device scope
__device__ __forceinline__ uint64_t
amd_atomic_add_device(uint64_t* ptr, uint64_t val) {
  return __hip_atomic_fetch_add(
      ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ __forceinline__ int amd_atomic_add_device(int* ptr, int val) {
  return __hip_atomic_fetch_add(
      ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// fetch_max with device scope
__device__ __forceinline__ uint64_t
amd_atomic_max_device(uint64_t* ptr, uint64_t val) {
  return __hip_atomic_fetch_max(
      ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// fetch_max with device scope + acquire ordering
__device__ __forceinline__ uint64_t
amd_atomic_max_device_acquire(uint64_t* ptr, uint64_t val) {
  return __hip_atomic_fetch_max(
      ptr, val, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
}

// fetch_add with workgroup scope
__device__ __forceinline__ uint64_t
amd_atomic_add_workgroup(uint64_t* ptr, uint64_t val) {
  return __hip_atomic_fetch_add(
      ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
}

// fetch_max with workgroup scope
__device__ __forceinline__ uint64_t
amd_atomic_max_workgroup(uint64_t* ptr, uint64_t val) {
  return __hip_atomic_fetch_max(
      ptr, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
}

// compare-exchange with device scope (replace atomicCAS)
__device__ __forceinline__ int
amd_atomic_cas_device(int* ptr, int expected, int desired) {
  __hip_atomic_compare_exchange_strong(
      ptr,
      &expected,
      desired,
      __ATOMIC_RELAXED,
      __ATOMIC_RELAXED,
      __HIP_MEMORY_SCOPE_AGENT);
  return expected;
}

// compare-exchange with workgroup scope (replace atomicCAS_block)
__device__ __forceinline__ int
amd_atomic_cas_workgroup(int* ptr, int expected, int desired) {
  __hip_atomic_compare_exchange_strong(
      ptr,
      &expected,
      desired,
      __ATOMIC_RELAXED,
      __ATOMIC_RELAXED,
      __HIP_MEMORY_SCOPE_WORKGROUP);
  return expected;
}

// release store with device scope (for unlock operations)
__device__ __forceinline__ void amd_store_release_device(int* ptr, int val) {
  __hip_atomic_store(ptr, val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ __forceinline__ void amd_store_release_workgroup(int* ptr, int val) {
  __hip_atomic_store(ptr, val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_WORKGROUP);
}

// release store for doorbell record (32-bit, system/agent scope)
__device__ __forceinline__ void amd_store_release_sys_u32(
    uint32_t* ptr,
    uint32_t val) {
  __hip_atomic_store(ptr, val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

// Direct doorbell store to NIC UAR BlueFlame register (64-bit, system scope).
// Uses SEQ_CST ordering with system scope to ensure the PCIe posted write
// reaches the NIC's PCI BAR.
__device__ __forceinline__ void amd_store_doorbell_sys_u64(
    uint64_t* ptr,
    uint64_t val) {
  __hip_atomic_store(ptr, val, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
}

// =============================================================================
// Warp-level Reduction (replace __reduce_max_sync)
// =============================================================================

// AMD uses DPP (Data Parallel Primitives) or cross-lane shuffle for reductions.
// For CDNA (MI200/MI300), wavefront is 64 lanes.
__device__ __forceinline__ uint32_t amd_warp_reduce_max(uint32_t val) {
  // Butterfly reduction across wavefront using __shfl_xor
  for (int offset = AMD_WAVEFRONT_SIZE / 2; offset > 0; offset >>= 1) {
    uint32_t other = __shfl_xor(val, offset);
    val = val > other ? val : other;
  }
  return val;
}

// =============================================================================
// Funnel Shift (replace CUDA __funnelshift_r)
// =============================================================================

// __funnelshift_r(lo, hi, shift) = (lo, hi) >> shift, taking low 32 bits
// This is used for the div_ceil_aligned_pow2_32bits fast path.
__device__ __forceinline__ uint32_t
amd_funnelshift_r(uint32_t lo, uint32_t hi, int shift) {
  uint64_t combined = (static_cast<uint64_t>(hi) << 32) | lo;
  return static_cast<uint32_t>(combined >> (shift & 31));
}

// =============================================================================
// Utility: ceil division by power-of-2 (replacing CUDA version that uses
// __funnelshift_r)
// =============================================================================

__device__ __forceinline__ uint32_t
amd_div_ceil_aligned_pow2_32bits(uint64_t x, int denominator_shift) {
  return static_cast<uint32_t>(x >> denominator_shift) +
      (amd_funnelshift_r(0, static_cast<uint32_t>(x), denominator_shift) != 0
           ? 1
           : 0);
}

// =============================================================================
// WQE Segment Store (64-byte aligned store for WQE writes)
// =============================================================================

// Copy a 16-byte WQE segment using 64-bit stores for atomicity
__device__ __forceinline__ void amd_store_wqe_seg(
    uint64_t* dst,
    const uint64_t* src) {
  dst[0] = src[0];
  dst[1] = src[1];
}
