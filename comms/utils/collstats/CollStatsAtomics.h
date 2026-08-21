// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <cstdint>

#include "comms/utils/collstats/CollStatsTypes.h"

// Portable atomics for the finalizer and the bank's epoch read: CUDA/HIP
// intrinsics on device, compiler builtins on host. One home so the device path
// and the host-tested path can never diverge.
//
// Two widths, because CollStatValue mixes them: u64 for durations and byte
// counts, u32 for the histogram buckets and threshold counters. Overload
// resolution on the pointer type selects the right one.

namespace meta::comms::collstats {

COLLSTATS_HD inline uint64_t collStatAtomicLoad(const uint64_t* addr) {
#if defined(__CUDA_ARCH__)
  return *static_cast<const volatile uint64_t*>(addr);
#else
  return __atomic_load_n(addr, __ATOMIC_ACQUIRE);
#endif
}

COLLSTATS_HD inline void collStatAtomicAdd(uint64_t* addr, uint64_t val) {
#if defined(__CUDA_ARCH__)
  atomicAdd(
      reinterpret_cast<unsigned long long*>(addr),
      static_cast<unsigned long long>(val));
#else
  __atomic_fetch_add(addr, val, __ATOMIC_RELAXED);
#endif
}

COLLSTATS_HD inline void collStatAtomicAdd(uint32_t* addr, uint32_t val) {
#if defined(__CUDA_ARCH__)
  atomicAdd(reinterpret_cast<unsigned int*>(addr), val);
#else
  __atomic_fetch_add(addr, val, __ATOMIC_RELAXED);
#endif
}

COLLSTATS_HD inline void collStatAtomicInc(uint64_t* addr) {
  collStatAtomicAdd(addr, 1ull);
}

COLLSTATS_HD inline void collStatAtomicInc(uint32_t* addr) {
  collStatAtomicAdd(addr, 1u);
}

COLLSTATS_HD inline void collStatAtomicMax(uint64_t* addr, uint64_t val) {
#if defined(__CUDA_ARCH__)
  atomicMax(
      reinterpret_cast<unsigned long long*>(addr),
      static_cast<unsigned long long>(val));
#else
  uint64_t cur = __atomic_load_n(addr, __ATOMIC_RELAXED);
  while (val > cur &&
         !__atomic_compare_exchange_n(
             addr, &cur, val, true, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED)) {
  }
#endif
}

} // namespace meta::comms::collstats
