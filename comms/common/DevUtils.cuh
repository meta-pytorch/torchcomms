// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda.h>

namespace comms::device {

__device__ __forceinline__ int loadInt(volatile int* ptr) {
#if defined(__HIP_PLATFORM_AMD__)
  int v = *ptr;
  return v;
#else
  int v;
  asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(v) : "l"(ptr));
  return v;
#endif
}

__device__ __forceinline__ void storeInt(volatile int* ptr, int val) {
#if defined(__HIP_PLATFORM_AMD__)
  *ptr = val;
#else
  asm volatile("st.volatile.global.s32 [%0], %1;" ::"l"(ptr), "r"(val));
#endif
}

// =============================================================================
// Atomic operations with system-wide visibility for NVLink cross-GPU signaling
// =============================================================================
//
// WHY .global QUALIFIER?
// ======================
// Without explicit .global, the compiler uses generic addressing which adds:
//   1. Runtime address space detection (global vs shared vs local)
//   2. Extra instructions for address translation
//   3. Potential predicated branches in generated SASS
//
// With explicit .global:
//   1. Compiler knows memory space at compile time
//   2. Direct addressing with no runtime checks
//   3. Simpler, faster instruction encoding (~2% throughput improvement)
//
// WHY .sys SCOPE?
// ===============
// The .sys (system) scope is required for cross-GPU NVLink communication:
//   - .cta  = visible only within thread block
//   - .gpu  = visible only within same GPU
//   - .sys  = visible across all GPUs + CPU
//
// For P2P NVLink, sender writes to memory that receiver reads via NVLink peer
// mapping. The .sys scope ensures the NVLink coherence protocol propagates
// writes across GPU boundaries.

__device__ __forceinline__ int loadIntAcq(int* ptr) {
#if defined(__HIP_PLATFORM_AMD__)
  int v = __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
  return v;
#else
  int v;
  asm volatile("ld.global.acquire.sys.s32 %0, [%1];"
               : "=r"(v)
               : "l"(ptr)
               : "memory");
  return v;
#endif
}

__device__ __forceinline__ void storeIntRel(int* ptr, int val) {
#if defined(__HIP_PLATFORM_AMD__)
  __atomic_store_n(ptr, val, __ATOMIC_RELEASE);
#else
  asm volatile("st.global.release.sys.s32 [%0], %1;" ::"l"(ptr), "r"(val)
               : "memory");
#endif
}

__device__ __forceinline__ void storeIntRelax(int* ptr, int val) {
#if defined(__HIP_PLATFORM_AMD__)
  __atomic_store_n(ptr, val, __ATOMIC_RELAXED);
#else
  asm volatile("st.relaxed.sys.global.s32 [%0], %1;" ::"l"(ptr), "r"(val)
               : "memory");
#endif
}

__device__ __forceinline__ uint64_t loadUint64(volatile uint64_t* ptr) {
#if defined(__HIP_PLATFORM_AMD__)
  uint64_t v = *ptr;
  return v;
#else
  uint64_t v;
  asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(v) : "l"(ptr));
  return v;
#endif
}

} // namespace comms::device
