// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

// Copied from gdrwrap.h to avoid including dependency to NCCL header
#if defined(__NVCC__)
#define wcStoreFence()
#else
#if defined(__PPC__)
static inline void wcStoreFence(void) {
  asm volatile("sync");
}
#elif defined(__x86_64__)
#include <immintrin.h>
static inline void wcStoreFence(void) {
  _mm_sfence();
}
#elif defined(__aarch64__)
#ifdef __cplusplus
#include <atomic>
static inline void wcStoreFence(void) {
  std::atomic_thread_fence(std::memory_order_release);
}
#else
#include <stdatomic.h>
static inline void wcStoreFence(void) {
  atomic_thread_fence(memory_order_release);
}
#endif
#endif
#endif
