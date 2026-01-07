// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace comms::device {

// =============================================================================
// GPU Atomic Operations with PTX Assembly
// =============================================================================
//
// This file provides PTX assembly primitives for memory ordering.
// These are essential for correct GPU-CPU and GPU-GPU synchronization.
//
// MEMORY ORDERING OVERVIEW:
// =========================
//
// 1. VOLATILE (.volatile):
//    - Bypasses L1 cache, reads from L2 or memory
//    - No ordering guarantees with other operations
//    - Used for polling when cache line atomicity provides ordering
//
// 2. RELAXED (.relaxed.sys):
//    - Atomic but no ordering guarantees
//    - Used with explicit fences for ordering
//
// 3. ACQUIRE (.acquire.sys):
//    - Load-acquire: subsequent loads/stores won't be reordered before this
//    - Pairs with release store from another thread
//
// 4. RELEASE (.release.sys):
//    - Store-release: prior loads/stores won't be reordered after this
//    - Pairs with acquire load from another thread
//
// 5. FENCE (.fence.acq_rel.sys):
//    - Full fence: orders all prior operations before all subsequent ones
//    - Used between data writes and flag updates
//
// SCOPE MODIFIERS:
// ================
//
// - .gpu: Visible to GPU threads only
// - .sys: Visible to GPU + CPU (system-wide visibility for GDRCopy)
//
// For GPU<->CPU communication via GDRCopy, always use .sys scope.
//
// CUDA ARCH FALLBACKS:
// ====================
//
// - SM 7.0+ (Volta): Native acquire/release/fence support
// - SM < 7.0: Fall back to volatile + membar.sys

// =============================================================================
// Address Space Conversion
// =============================================================================

template <typename T>
__device__ __forceinline__ uintptr_t cvta_to_global(T* ptr) {
#ifdef __CUDA_ARCH__
  return (uintptr_t)__cvta_generic_to_global(ptr);
#else
  return (uintptr_t)ptr;
#endif
}

// =============================================================================
// Volatile Loads (bypass L1 cache)
// =============================================================================

__device__ __forceinline__ uint64_t ld_volatile_global(volatile uint64_t* ptr) {
#if defined(__HIP_PLATFORM_AMD__)
  uint64_t v = *ptr;
  return v;
#else
  uint64_t v;
  asm volatile("ld.volatile.global.u64 %0, [%1];"
               : "=l"(v)
               : "l"(ptr)
               : "memory");
  return v;
#endif
}

__device__ __forceinline__ uint32_t ld_volatile_global(volatile uint32_t* ptr) {
#if defined(__HIP_PLATFORM_AMD__)
  uint32_t v = *ptr;
  return v;
#else
  uint32_t v;
  asm volatile("ld.volatile.global.u32 %0, [%1];"
               : "=r"(v)
               : "l"(ptr)
               : "memory");
  return v;
#endif
}

__device__ __forceinline__ int32_t ld_volatile_global(volatile int32_t* ptr) {
#if defined(__HIP_PLATFORM_AMD__)
  int32_t v = *ptr;
  return v;
#else
  int32_t v;
  asm volatile("ld.volatile.global.s32 %0, [%1];"
               : "=r"(v)
               : "l"(ptr)
               : "memory");
  return v;
#endif
}

// =============================================================================
// Volatile Stores (bypass L1 cache)
// =============================================================================

__device__ __forceinline__ void st_volatile_global(
    volatile uint64_t* ptr,
    uint64_t val) {
#if defined(__HIP_PLATFORM_AMD__)
  *ptr = val;
#else
  asm volatile("st.volatile.global.u64 [%0], %1;"
               :
               : "l"(ptr), "l"(val)
               : "memory");
#endif
}

__device__ __forceinline__ void st_volatile_global(
    volatile uint32_t* ptr,
    uint32_t val) {
#if defined(__HIP_PLATFORM_AMD__)
  *ptr = val;
#else
  asm volatile("st.volatile.global.u32 [%0], %1;"
               :
               : "l"(ptr), "r"(val)
               : "memory");
#endif
}

__device__ __forceinline__ void st_volatile_global(
    volatile int32_t* ptr,
    int32_t val) {
#if defined(__HIP_PLATFORM_AMD__)
  *ptr = val;
#else
  asm volatile("st.volatile.global.s32 [%0], %1;"
               :
               : "l"(ptr), "r"(val)
               : "memory");
#endif
}

// =============================================================================
// Relaxed System-scope Operations
// =============================================================================
//
// Used with explicit fences for ordering. More efficient than volatile on
// modern GPUs but requires SM 7.0+.

__device__ __forceinline__ uint64_t ld_relaxed_sys_global(uint64_t* ptr) {
#if defined(__HIP_PLATFORM_AMD__)
  uint64_t v = __atomic_load_n(ptr, __ATOMIC_RELAXED);
  return v;
#else
  uint64_t v;
#if __CUDA_ARCH__ >= 700
  asm volatile("ld.relaxed.sys.global.u64 %0, [%1];"
               : "=l"(v)
               : "l"(ptr)
               : "memory");
#else
  asm volatile("ld.volatile.global.u64 %0, [%1];"
               : "=l"(v)
               : "l"(ptr)
               : "memory");
#endif
  return v;
#endif
}

__device__ __forceinline__ int32_t ld_relaxed_sys_global(int32_t* ptr) {
#if defined(__HIP_PLATFORM_AMD__)
  int32_t v = __atomic_load_n(ptr, __ATOMIC_RELAXED);
  return v;
#else
  int32_t v;
#if __CUDA_ARCH__ >= 700
  asm volatile("ld.relaxed.sys.global.s32 %0, [%1];"
               : "=r"(v)
               : "l"(ptr)
               : "memory");
#else
  asm volatile("ld.volatile.global.s32 %0, [%1];"
               : "=r"(v)
               : "l"(ptr)
               : "memory");
#endif
  return v;
#endif
}

__device__ __forceinline__ void st_relaxed_sys_global(
    uint64_t* ptr,
    uint64_t val) {
#if defined(__HIP_PLATFORM_AMD__)
  __atomic_store_n(ptr, val, __ATOMIC_RELAXED);
#else
#if __CUDA_ARCH__ >= 700
  asm volatile("st.relaxed.sys.global.u64 [%0], %1;"
               :
               : "l"(ptr), "l"(val)
               : "memory");
#else
  asm volatile("st.volatile.global.u64 [%0], %1;"
               :
               : "l"(ptr), "l"(val)
               : "memory");
#endif
#endif
}

__device__ __forceinline__ void st_relaxed_sys_global(
    int32_t* ptr,
    int32_t val) {
#if defined(__HIP_PLATFORM_AMD__)
  __atomic_store_n(ptr, val, __ATOMIC_RELAXED);
#else
#if __CUDA_ARCH__ >= 700
  asm volatile("st.relaxed.sys.global.s32 [%0], %1;"
               :
               : "l"(ptr), "r"(val)
               : "memory");
#else
  asm volatile("st.volatile.global.s32 [%0], %1;"
               :
               : "l"(ptr), "r"(val)
               : "memory");
#endif
#endif
}

// =============================================================================
// Acquire/Release System-scope Operations
// =============================================================================
//
// For paired synchronization:
// - Writer: st_release_sys_global(ptr, val)  -- prior writes visible
// - Reader: ld_acquire_sys_global(ptr)       -- subsequent reads see prior
// writes

__device__ __forceinline__ uint64_t ld_acquire_sys_global(uint64_t* ptr) {
#if defined(__HIP_PLATFORM_AMD__)
  uint64_t v = __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
  return v;
#else
  uint64_t v;
#if __CUDA_ARCH__ >= 700
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];"
               : "=l"(v)
               : "l"(ptr)
               : "memory");
#else
  asm volatile("ld.volatile.global.u64 %0, [%1]; membar.sys;"
               : "=l"(v)
               : "l"(ptr)
               : "memory");
#endif
  return v;
#endif
}

__device__ __forceinline__ int32_t ld_acquire_sys_global(int32_t* ptr) {
#if defined(__HIP_PLATFORM_AMD__)
  int32_t v = __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
  return v;
#else
  int32_t v;
#if __CUDA_ARCH__ >= 700
  asm volatile("ld.acquire.sys.global.s32 %0, [%1];"
               : "=r"(v)
               : "l"(ptr)
               : "memory");
#else
  asm volatile("ld.volatile.global.s32 %0, [%1]; membar.sys;"
               : "=r"(v)
               : "l"(ptr)
               : "memory");
#endif
  return v;
#endif
}

__device__ __forceinline__ void st_release_sys_global(
    uint64_t* ptr,
    uint64_t val) {
#if defined(__HIP_PLATFORM_AMD__)
  __atomic_store_n(ptr, val, __ATOMIC_RELEASE);
#else
#if __CUDA_ARCH__ >= 700
  asm volatile("st.release.sys.global.u64 [%0], %1;"
               :
               : "l"(ptr), "l"(val)
               : "memory");
#else
  asm volatile("membar.sys; st.volatile.global.u64 [%0], %1;"
               :
               : "l"(ptr), "l"(val)
               : "memory");
#endif
#endif
}

__device__ __forceinline__ void st_release_sys_global(
    int32_t* ptr,
    int32_t val) {
#if defined(__HIP_PLATFORM_AMD__)
  __atomic_store_n(ptr, val, __ATOMIC_RELEASE);
#else
#if __CUDA_ARCH__ >= 700
  asm volatile("st.release.sys.global.s32 [%0], %1;"
               :
               : "l"(ptr), "r"(val)
               : "memory");
#else
  asm volatile("membar.sys; st.volatile.global.s32 [%0], %1;"
               :
               : "l"(ptr), "r"(val)
               : "memory");
#endif
#endif
}

// =============================================================================
// Memory Fences
// =============================================================================

/**
 * fence_acq_rel_sys - System-scope acquire-release fence
 *
 * Ensures all prior loads/stores are visible to all observers (GPU + CPU)
 * before any subsequent loads/stores. Use this between data operations
 * and flag updates.
 *
 * SIMPLE PROTOCOL PATTERN (GPU→CPU signaling with GDRCopy):
 * =========================================================
 *
 *   // GPU side (sender notifying CPU)
 *   write_data_to_buffer();
 *   fence_acq_rel_sys();              // <-- Ensures data visible before flag
 *   st_relaxed_sys_global(flag, val); // <-- CPU sees this after data is
 * visible
 *
 *   // CPU side (via GDRCopy mapping)
 *   while (gdr_mapped_flag != expected) { spin; }
 *   // After seeing flag, data is guaranteed visible due to fence+relaxed store
 */
__device__ __forceinline__ void fence_acq_rel_sys() {
#if defined(__HIP_PLATFORM_AMD__)
#ifdef __HIP_ARCH__
  __threadfence_system();
#endif
#else
#if __CUDA_ARCH__ >= 700
  asm volatile("fence.acq_rel.sys;" ::: "memory");
#else
  asm volatile("membar.sys;" ::: "memory");
#endif
#endif
}

/**
 * fence_acq_rel_gpu - GPU-scope acquire-release fence
 *
 * Same as fence_acq_rel_sys but only visible to GPU threads (not CPU).
 * Use for GPU-GPU synchronization without CPU involvement.
 */
__device__ __forceinline__ void fence_acq_rel_gpu() {
#if defined(__HIP_PLATFORM_AMD__)
#ifdef __HIP_ARCH__
  __threadfence();
#endif
#else
#if __CUDA_ARCH__ >= 700
  asm volatile("fence.acq_rel.gpu;" ::: "memory");
#else
  asm volatile("membar.gl;" ::: "memory");
#endif
#endif
}

// =============================================================================
// 128-bit Volatile Load/Store (for LL128 cache line operations)
// =============================================================================

/**
 * load128_volatile - Volatile 128-bit load
 *
 * Loads 128 bits (two uint64_t values) atomically using v2.u64 instruction.
 * Cache line atomicity on GPU L2 ensures both values are from the same
 * cache line snapshot.
 *
 * Used in LL128 protocol to load data + flag from a 128-byte line.
 */
__device__ __forceinline__ void
load128_volatile_global(volatile uint64_t* ptr, uint64_t& v0, uint64_t& v1) {
#if defined(__HIP_PLATFORM_AMD__)
  v0 = ptr[0];
  v1 = ptr[1];
#else
  asm volatile("ld.volatile.global.v2.u64 {%0,%1}, [%2];"
               : "=l"(v0), "=l"(v1)
               : "l"(ptr)
               : "memory");
#endif
}

/**
 * store128_volatile - Volatile 128-bit store
 *
 * Stores 128 bits (two uint64_t values) atomically using v2.u64 instruction.
 * Cache line atomicity ensures both values appear atomically to readers.
 *
 * Used in LL128 protocol to store data + flag to a 128-byte line.
 */
__device__ __forceinline__ void
store128_volatile_global(volatile uint64_t* ptr, uint64_t v0, uint64_t v1) {
#if defined(__HIP_PLATFORM_AMD__)
  ptr[0] = v0;
  ptr[1] = v1;
#else
  asm volatile("st.volatile.global.v2.u64 [%2], {%0,%1};"
               :
               : "l"(v0), "l"(v1), "l"(ptr)
               : "memory");
#endif
}

} // namespace comms::device
