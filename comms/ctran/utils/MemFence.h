// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

// Copied from gdrwrap.h to avoid including dependency to NCCL header
//
// Naming follows the original gdrwrap.h convention ("wc" = write-combined),
// but the fences are applicable to any host-side ordering across coherent
// pinned memory shared with another agent (GPU / NIC).
//
// Pair semantics:
//   wcStoreFence()    — release. Use BEFORE storing a flag/doorbell so that
//                       any prior data writes are ordered before the flag.
//   wcAcquireFence()  — acquire. Use AFTER reading a flag so that any
//                       subsequent loads of data observed-by-the-flag see
//                       the producer's prior writes.
#if defined(__NVCC__)
#define wcStoreFence()
#define wcAcquireFence()
#else
#if defined(__PPC__)
static inline void wcStoreFence(void) {
  asm volatile("sync");
}
static inline void wcAcquireFence(void) {
  asm volatile("sync");
}
#elif defined(__x86_64__)
#include <immintrin.h>
static inline void wcStoreFence(void) {
  _mm_sfence();
}
// x86 is TSO: load->load is already ordered, so the read side never needs
// a hardware barrier. A compiler reordering barrier ("memory" clobber) is
// sufficient to mirror wcStoreFence as a logical pair.
static inline void wcAcquireFence(void) {
  asm volatile("" ::: "memory");
}
#elif defined(__aarch64__)
#ifdef __cplusplus
#include <atomic>
static inline void wcStoreFence(void) {
  std::atomic_thread_fence(std::memory_order_release);
}
// aarch64 permits load->load reordering. After polling a flag released by
// the GPU via st.release.sys, the host needs a load-acquire barrier
// (dmb ishld) before any subsequent dependent load/store.
static inline void wcAcquireFence(void) {
  std::atomic_thread_fence(std::memory_order_acquire);
}
#else
#include <stdatomic.h>
static inline void wcStoreFence(void) {
  atomic_thread_fence(memory_order_release);
}
static inline void wcAcquireFence(void) {
  atomic_thread_fence(memory_order_acquire);
}
#endif
#endif
#endif
