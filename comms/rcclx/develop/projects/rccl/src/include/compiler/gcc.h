/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_COMPILER_GCC_H
#define NCCL_COMPILER_GCC_H

#include <atomic>

// Helper macros to convert C++ memory ordering to GCC atomic ordering.
// std::memory_order and the __ATOMIC_* constants are specified with the same
// underlying values, so the conversion is a plain cast; the static_asserts
// below pin that down. Testing `order` against each enumerator instead is a
// self-comparison at every call site that passes a literal, and a constexpr
// helper is not folded at -O0, which emits a runtime switch on every atomic op.
#define NCCL_CONVERT_ORDER(order) (static_cast<int>(order))

static_assert(static_cast<int>(std::memory_order_relaxed) == __ATOMIC_RELAXED, "memory order mismatch");
static_assert(static_cast<int>(std::memory_order_consume) == __ATOMIC_CONSUME, "memory order mismatch");
static_assert(static_cast<int>(std::memory_order_acquire) == __ATOMIC_ACQUIRE, "memory order mismatch");
static_assert(static_cast<int>(std::memory_order_release) == __ATOMIC_RELEASE, "memory order mismatch");
static_assert(static_cast<int>(std::memory_order_acq_rel) == __ATOMIC_ACQ_REL, "memory order mismatch");
static_assert(static_cast<int>(std::memory_order_seq_cst) == __ATOMIC_SEQ_CST, "memory order mismatch");

#define COMPILER_ATOMIC_LOAD(ptr, order) __atomic_load_n((ptr), NCCL_CONVERT_ORDER(order))
#define COMPILER_ATOMIC_LOAD_DEST(ptr, dest, order) \
  do { \
    __atomic_load((ptr), (dest), NCCL_CONVERT_ORDER(order)); \
  } while (0)
#define COMPILER_ATOMIC_STORE(ptr, val, order) __atomic_store_n((ptr), (val), NCCL_CONVERT_ORDER(order))
#define COMPILER_ATOMIC_LOAD_32(ptr, order) __atomic_load_n((ptr), NCCL_CONVERT_ORDER(order))
#define COMPILER_ATOMIC_STORE_32(ptr, val, order) __atomic_store_n((ptr), (val), NCCL_CONVERT_ORDER(order))
#define COMPILER_ATOMIC_EXCHANGE(ptr, val, order) __atomic_exchange_n((ptr), (val), NCCL_CONVERT_ORDER(order))
#define COMPILER_ATOMIC_COMPARE_EXCHANGE(ptr, expected, desired, success_order, failure_order) \
  __atomic_compare_exchange_n((ptr), (expected), (desired), true, NCCL_CONVERT_ORDER(success_order), \
                              NCCL_CONVERT_ORDER(failure_order))

#define COMPILER_ATOMIC_FETCH_ADD(ptr, val, order) __atomic_fetch_add((ptr), (val), NCCL_CONVERT_ORDER(order))
#define COMPILER_ATOMIC_ADD_FETCH(ptr, val, order) __atomic_add_fetch((ptr), (val), NCCL_CONVERT_ORDER(order))
#define COMPILER_ATOMIC_SUB_FETCH(ptr, val, order) __atomic_sub_fetch((ptr), (val), NCCL_CONVERT_ORDER(order))

#define COMPILER_PREFETCH(addr) __builtin_prefetch((addr))

#define COMPILER_POPCOUNT32(x) __builtin_popcount(x)
#define COMPILER_POPCOUNT64(x) __builtin_popcountll(x)

#define COMPILER_EXPECT(x, v) __builtin_expect((x), (v))

// Find First Set (FFS) - returns index of first set bit (1-indexed), 0 if no bits set
#define COMPILER_FFS(x) __builtin_ffs(x)
#define COMPILER_FFSL(x) __builtin_ffsl(x)
#define COMPILER_FFSLL(x) __builtin_ffsll(x)

// Count Leading Zeros (CLZ) - undefined behavior if x == 0
#define COMPILER_CLZ(x) __builtin_clz(x)
#define COMPILER_CLZL(x) __builtin_clzl(x)
#define COMPILER_CLZLL(x) __builtin_clzll(x)

// Byte Swap
#define COMPILER_BSWAP16(x) __builtin_bswap16(x)
#define COMPILER_BSWAP32(x) __builtin_bswap32(x)
#define COMPILER_BSWAP64(x) __builtin_bswap64(x)

// Compiler hints
#define COMPILER_ASSUME_ALIGNED(ptr, alignment) __builtin_assume_aligned((ptr), (alignment))

#define COMPILER_ATTRIBUTE_UNUSED __attribute__((unused))

#endif // NCCL_COMPILER_GCC_H
